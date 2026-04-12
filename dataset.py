"""
HiC-Verse Dataset — v2
=======================
Supports the real data directory layout:

    matrices/
        X/                          ← RNA-seq signal (inputs)
            cond01_rep1/
                window_0000_chr1_0_2000000.npy    shape: (n_bins,)
                window_0001_chr1_2000000_4000000.npy
                ...
            cond01_rep2/            ← replicate of same condition
            cond02_rep1/
            ...
            cond06_rep2/            ← 6 conditions × 2 reps = 12 folders
        Y/                          ← Hi-C contact maps (targets)
            cond01_rep1/
                window_0000_chr1_0_2000000.npy    shape: (n_bins, n_bins)
            ...

Loading strategy:
  1. Scan Y/cond01_rep1/ to discover all window filenames.
  2. For each window, load its RNA signal from all X folders,
     group by condition prefix (cond01, cond02, …), and average
     the replicates within each condition.
     → rna_signal tensor: (n_conditions, n_bins)
  3. Load Hi-C targets from all Y folders, same grouping.
     → target tensor: (n_conditions, n_bins, n_bins)
  4. DNA sequence is fetched from the reference genome FASTA using
     the chr/start/end encoded in each window filename.

Design decision — why average replicates?
  Biological replicates measure the same underlying process with
  technical noise.  Averaging them before feeding to the model
  reduces noise without discarding any information about condition
  differences.  If you want the model to also learn replicate
  variability, set average_reps=False and double n_conditions.
"""

from __future__ import annotations
import csv
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────
#  Nucleotide helpers
# ─────────────────────────────────────────────────────────────────

NUC2IDX = {b'A': 0, b'C': 1, b'G': 2, b'T': 3, b'N': 4,
           b'a': 0, b'c': 1, b'g': 2, b't': 3, b'n': 4}
DNA_LOOKUP = np.full(256, 4, dtype=np.int64)
for _base, _idx in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3),
                    (b"a", 0), (b"c", 1), (b"g", 2), (b"t", 3),
                    (b"N", 4), (b"n", 4)):
    DNA_LOOKUP[_base[0]] = _idx


def synthetic_dna(length: int, gc_bias: float = 0.5,
                  rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Random DNA indices with mild GC bias."""
    if rng is None:
        rng = np.random.default_rng()
    p  = [(1 - gc_bias) / 2, gc_bias / 2, gc_bias / 2, (1 - gc_bias) / 2]
    return torch.from_numpy(
        rng.choice(4, size=length, p=p).astype(np.int64))


def encode_dna(sequence: bytes, expected_len: Optional[int] = None) -> torch.Tensor:
    """Encode ASCII DNA bytes to A/C/G/T/N indices."""
    arr = DNA_LOOKUP[np.frombuffer(sequence, dtype=np.uint8)].copy()
    if expected_len is not None and arr.shape[0] != expected_len:
        padded = np.full(expected_len, 4, dtype=np.int64)
        padded[:min(expected_len, arr.shape[0])] = arr[:expected_len]
        arr = padded
    return torch.from_numpy(arr)


class GenomeFastaReader:
    """
    Lightweight FASTA reader with a one-time byte-offset index.
    This avoids loading the full genome into memory while still allowing
    random access to genomic windows in __getitem__.
    """

    def __init__(self, fasta_path: str):
        self.path = Path(fasta_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Genome FASTA not found: {self.path}\n"
                "Set cfg.genome_fasta_path to a valid reference genome."
            )
        self._index = self._build_index()
        self._handle = None

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        return self._handle

    def _build_index(self) -> Dict[str, dict]:
        index: Dict[str, dict] = {}
        with self.path.open("rb") as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                if not header.startswith(b">"):
                    continue

                name = header[1:].strip().split()[0].decode("utf-8")
                seq_start = fh.tell()
                first_line = fh.readline()
                while first_line and not first_line.strip():
                    seq_start = fh.tell()
                    first_line = fh.readline()
                if not first_line:
                    index[name] = dict(
                        offset=seq_start,
                        line_bases=1,
                        line_bytes=1,
                        length=0,
                    )
                    break
                if first_line.startswith(b">"):
                    fh.seek(seq_start)
                    index[name] = dict(
                        offset=seq_start,
                        line_bases=1,
                        line_bytes=1,
                        length=0,
                    )
                    continue

                line_bases = len(first_line.rstrip(b"\r\n"))
                line_bytes = len(first_line)
                length = line_bases

                while True:
                    pos = fh.tell()
                    line = fh.readline()
                    if not line or line.startswith(b">"):
                        if line and line.startswith(b">"):
                            fh.seek(pos)
                        break
                    length += len(line.rstrip(b"\r\n"))

                index[name] = dict(
                    offset=seq_start,
                    line_bases=max(1, line_bases),
                    line_bytes=max(1, line_bytes),
                    length=length,
                )

        if not index:
            raise ValueError(f"No FASTA records found in {self.path}")
        return index

    def _resolve_chrom(self, chrom: str) -> str:
        if chrom in self._index:
            return chrom
        aliases = [
            chrom[3:] if chrom.startswith("chr") else f"chr{chrom}",
            chrom.replace("chrM", "MT"),
            chrom.replace("MT", "chrM"),
        ]
        for alias in aliases:
            if alias in self._index:
                return alias
        raise KeyError(
            f"Chromosome {chrom!r} not found in FASTA. "
            f"Available examples: {list(self._index)[:5]}"
        )

    def fetch(self, chrom: str, start: int, end: int) -> bytes:
        chrom_name = self._resolve_chrom(chrom)
        meta = self._index[chrom_name]
        req_len = max(0, end - start)
        if req_len == 0:
            return b""

        start = max(0, start)
        end = min(max(start, end), meta["length"])
        if start >= end:
            return b"N" * req_len

        fh = self._open()
        line_bases = meta["line_bases"]
        line_bytes = meta["line_bytes"]
        cur = start
        out = bytearray()

        while cur < end:
            byte_pos = meta["offset"] + (cur // line_bases) * line_bytes + (cur % line_bases)
            fh.seek(byte_pos)
            take = min(line_bases - (cur % line_bases), end - cur)
            out.extend(fh.read(take))
            cur += take

        if len(out) < req_len:
            out.extend(b"N" * (req_len - len(out)))
        return bytes(out).upper()


# ─────────────────────────────────────────────────────────────────
#  Normalisation helpers
# ─────────────────────────────────────────────────────────────────

def normalise_hic(M: np.ndarray) -> np.ndarray:
    """
    Log₁₀(x+1) normalisation followed by min-max scaling to [0,1].
    Decision: log scale compresses the wide dynamic range of Hi-C
    counts (0–10,000+) into a range that MSE loss can handle.
    """
    M = np.nan_to_num(M.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    M = np.clip(M, 0, None)
    M = np.log10(M + 1.0)
    lo, hi = M.min(), M.max()
    if hi > lo:
        M = (M - lo) / (hi - lo)
    return M.astype(np.float32)


def normalise_rna(arr: np.ndarray) -> np.ndarray:
    """
    Log₁₀(x+1) followed by z-score per sample.
    Decision: RNA-seq counts also have heavy-tailed distributions.
    Z-scoring makes the gating signal scale-invariant across
    conditions with very different sequencing depths.
    """
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    arr = np.log10(arr + 1.0)
    mu, sigma = arr.mean(), arr.std()
    if sigma > 1e-8:
        arr = (arr - mu) / sigma
    return arr.astype(np.float32)


# ═════════════════════════════════════════════════════════════════
#  REAL-DATA DATASET  (X/ + Y/ layout)
# ═════════════════════════════════════════════════════════════════

class HiCWindowDataset(Dataset):
    """
    Loads RNA-seq (X/) and Hi-C (Y/) windows from the paired directory
    layout described at the top of this file.

    Args:
        x_dir        : path to X/ (RNA-seq inputs)
        y_dir        : path to Y/ (Hi-C targets)
        n_bins       : expected bins per window (must match file shape)
        n_conditions : number of conditions to expose to the model.
                       Must equal the number of condXX prefixes found.
                       If the data has fewer, a warning is printed.
        bin_size     : bp per bin (determines synthetic DNA length)
        genome_fasta_path : reference genome FASTA used for real DNA windows
        chroms       : restrict to these chromosomes; None = all
        average_reps : if True, average replicates within each condition
                       before returning; if False, treat each rep as its
                       own condition (doubles n_conditions)
        augment      : random horizontal flip augmentation
        normalize_rna : apply log/z-score scaling to RNA windows
        normalize_hic : apply log/min-max scaling to Hi-C targets
        ctcf_fimo_tsv_path : path to FIMO TSV file for CTCF priors
        use_ctcf_fimo_prior : enable convergent CTCF 2D prior generation
        tad_priors_dir : directory containing per-window TAD prior .npy files
        use_tad_priors : enable TAD prior loading
        seed         : RNG seed
    """

    def __init__(
        self,
        x_dir:        str,
        y_dir:        str,
        n_bins:       int   = 200,
        n_conditions: int   = 6,
        bin_size:     int   = 10_000,
        genome_fasta_path: Optional[str] = None,
        chroms:       Optional[List[str]] = None,
        average_reps: bool  = True,
        augment:      bool  = False,
        normalize_rna: bool = True,
        normalize_hic: bool = True,
        ctcf_fimo_tsv_path: Optional[str] = None,
        use_ctcf_fimo_prior: bool = True,
        tad_priors_dir: Optional[str] = None,
        use_tad_priors: bool = True,
        seed:         int   = 42,
    ):
        super().__init__()
        self.n_bins       = n_bins
        self.bin_size     = bin_size
        self.dna_len      = n_bins * bin_size
        self.average_reps = average_reps
        self.augment      = augment
        self.normalize_rna = normalize_rna
        self.normalize_hic = normalize_hic
        self.use_ctcf_fimo_prior = bool(use_ctcf_fimo_prior)
        self.ctcf_fimo_tsv_path = str(ctcf_fimo_tsv_path) if ctcf_fimo_tsv_path else None
        self.use_tad_priors = bool(use_tad_priors)
        self.tad_priors_dir = Path(tad_priors_dir) if tad_priors_dir else None
        self.rng          = np.random.default_rng(seed)
        self._ctcf_sites_by_chrom: Dict[str, Dict[str, np.ndarray]] = {}
        self._tad_prior_index: Dict[Tuple[int, str, int, int], Path] = {}
        self._warned_missing_tad = False

        self.x_dir = Path(x_dir)
        self.y_dir = Path(y_dir)
        for d in (self.x_dir, self.y_dir):
            if not d.exists():
                raise FileNotFoundError(
                    f"Directory not found: {d}\n"
                    "Set x_dir / y_dir in config.py to your matrices folder."
                )
        if genome_fasta_path is None:
            raise ValueError(
                "Real-data mode requires a reference genome FASTA. "
                "Set cfg.genome_fasta_path in config.py."
            )
        self.genome_fasta_path = str(genome_fasta_path)
        self.genome = GenomeFastaReader(self.genome_fasta_path)

        # ── Discover condition/replicate structure ─────────────
        # Returns list-of-lists: outer=conditions, inner=rep dirs
        self.x_groups = self._group_by_condition(self.x_dir)
        self.y_groups = self._group_by_condition(self.y_dir)

        # How many conditions will we actually expose?
        actual = len(self.x_groups)
        if actual != n_conditions:
            warnings.warn(
                f"[HiCWindowDataset] Found {actual} conditions in X/ "
                f"but n_conditions={n_conditions}.  Using {actual}."
            )
        self.n_conditions = actual

        # Optional CTCF prior index from FIMO motifs.
        if self.use_ctcf_fimo_prior and self.ctcf_fimo_tsv_path:
            self._ctcf_sites_by_chrom = self._load_ctcf_sites(self.ctcf_fimo_tsv_path)
            if not self._ctcf_sites_by_chrom:
                warnings.warn(
                    f"[HiCWindowDataset] No valid CTCF sites loaded from {self.ctcf_fimo_tsv_path}. "
                    "CTCF prior will be zeros."
                )
        elif self.use_ctcf_fimo_prior:
            warnings.warn(
                "[HiCWindowDataset] use_ctcf_fimo_prior=True but no ctcf_fimo_tsv_path was set. "
                "CTCF prior will be zeros."
            )

        if self.use_tad_priors:
            if self.tad_priors_dir is None:
                for candidate in (
                    Path("TAD_Priors"),
                    Path(__file__).resolve().parent / "TAD_Priors",
                ):
                    if candidate.exists() and candidate.is_dir():
                        self.tad_priors_dir = candidate
                        break
            if self.tad_priors_dir is not None and self.tad_priors_dir.exists():
                self._tad_prior_index = self._build_tad_prior_index(self.tad_priors_dir)
            else:
                warnings.warn(
                    "[HiCWindowDataset] use_tad_priors=True but no TAD prior directory found. "
                    "TAD priors will be zeros."
                )

        # ── Discover window filenames from first Y condition ───
        anchor_dir = self.y_groups[0][0]   # Y/cond01_rep1/
        self.windows = self._scan_windows(anchor_dir, chroms)
        if not self.windows:
            raise ValueError(
                f"No window_*.npy files found in {anchor_dir} "
                f"(chroms={chroms}).\n"
                "Expected: window_NNNN_chrXX_START_END.npy"
            )

        n_reps = [len(g) for g in self.x_groups]
        print(
            f"[HiCWindowDataset] {len(self.windows)} windows | "
            f"{self.n_conditions} conditions | "
            f"reps per condition: {n_reps} | "
            f"average_reps={average_reps} | "
            f"ctcf_prior={'on' if self._ctcf_sites_by_chrom else 'off'} | "
            f"tad_prior={'on' if self._tad_prior_index else 'off'}"
        )

    # ── Directory scanning helpers ────────────────────────────────

    def _group_by_condition(self, base: Path) -> List[List[Path]]:
        """
        Find all subdirectories matching 'cond##_rep##' and group
        them by condition prefix.

        Returns: [[cond01_rep1, cond01_rep2], [cond02_rep1, ...], ...]
        """
        groups: dict = {}
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            m = re.match(r'(cond\d+)', d.name)
            if m:
                groups.setdefault(m.group(1), []).append(d)
        if not groups:
            raise ValueError(
                f"No cond*/rep* subdirectories found in {base}.\n"
                "Expected names like: cond01_rep1, cond02_rep1, …"
            )
        return [groups[k] for k in sorted(groups.keys())]

    def _scan_windows(self, directory: Path,
                      chroms: Optional[List[str]]) -> List[dict]:
        """List all window_*.npy files, optionally filtered by chrom."""
        windows = []
        for fp in sorted(directory.glob("window_*.npy")):
            m = re.search(r'window_(\d+)_(chr\w+)_(\d+)_(\d+)', fp.stem)
            if not m:
                continue
            chrom = m.group(2)
            if chroms is not None and chrom not in chroms:
                continue
            windows.append(dict(
                fname = fp.name,
                idx   = int(m.group(1)),
                chrom = chrom,
                start = int(m.group(3)),
                end   = int(m.group(4)),
            ))
        return windows

    @staticmethod
    def _chrom_aliases(chrom: str) -> List[str]:
        aliases = [chrom]
        if chrom.startswith("chr"):
            aliases.append(chrom[3:])
        else:
            aliases.append(f"chr{chrom}")
        aliases.append(chrom.replace("chrM", "MT"))
        aliases.append(chrom.replace("MT", "chrM"))
        # preserve order, remove dups
        out = []
        seen = set()
        for c in aliases:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _load_ctcf_sites(self, tsv_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build chromosome-wise sorted start-coordinate arrays for '+' and '-'
        strands from a FIMO TSV file.
        """
        path = Path(tsv_path)
        if not path.exists():
            warnings.warn(f"[HiCWindowDataset] CTCF FIMO TSV not found: {path}")
            return {}

        by_chrom: Dict[str, Dict[str, List[int]]] = {}
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            required = {"sequence_name", "start", "strand"}
            if not required.issubset(set(reader.fieldnames or [])):
                warnings.warn(
                    f"[HiCWindowDataset] {path} missing required columns {required}. "
                    "CTCF prior will be disabled."
                )
                return {}
            for row in reader:
                chrom = str(row.get("sequence_name", "")).strip()
                strand = str(row.get("strand", "")).strip()
                if not chrom or strand not in {"+", "-"}:
                    continue
                try:
                    start = int(float(row["start"]))
                except Exception:
                    continue
                by_chrom.setdefault(chrom, {"+": [], "-": []})[strand].append(start)

        out: Dict[str, Dict[str, np.ndarray]] = {}
        for chrom, strands in by_chrom.items():
            out[chrom] = {
                "+": np.sort(np.asarray(strands["+"], dtype=np.int64)),
                "-": np.sort(np.asarray(strands["-"], dtype=np.int64)),
            }
        return out

    def _resolve_ctcf_chrom(self, chrom: str) -> Optional[str]:
        if not self._ctcf_sites_by_chrom:
            return None
        for alias in self._chrom_aliases(chrom):
            if alias in self._ctcf_sites_by_chrom:
                return alias
        return None

    def _build_ctcf_prior_2d(self, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Orientation-aware convergent CTCF prior:
          P[i, j] = bin_i(+) * bin_j(-), i < j
        Symmetrized for Hi-C map compatibility.
        """
        prior = np.zeros((self.n_bins, self.n_bins), dtype=np.float32)
        if not self._ctcf_sites_by_chrom:
            return prior

        key = self._resolve_ctcf_chrom(chrom)
        if key is None:
            return prior

        chrom_sites = self._ctcf_sites_by_chrom[key]
        fwd = np.zeros(self.n_bins, dtype=np.float32)
        rev = np.zeros(self.n_bins, dtype=np.float32)

        for strand, arr, out in (("+", chrom_sites["+"], fwd), ("-", chrom_sites["-"], rev)):
            if arr.size == 0:
                continue
            lo = int(np.searchsorted(arr, start, side="left"))
            hi = int(np.searchsorted(arr, end, side="left"))
            if hi <= lo:
                continue
            pos = arr[lo:hi]
            bins = ((pos - start) // self.bin_size).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < self.n_bins)]
            if bins.size == 0:
                continue
            out[np.unique(bins)] = 1.0

        prior = np.outer(fwd, rev).astype(np.float32)
        prior = np.triu(prior, k=1)
        prior = prior + prior.T
        return prior

    @staticmethod
    def _zero_tad_prior(n_bins: int) -> np.ndarray:
        return np.zeros((4, n_bins, n_bins), dtype=np.float32)

    def _build_tad_prior_index(self, tad_dir: Path) -> Dict[Tuple[int, str, int, int], Path]:
        """
        Index TAD prior files by window signature:
          (idx, chrom, start, end) -> filepath
        Expected filename pattern:
          window_<idx>_<chrom>_<start>_<end>_tad.npy
        """
        out: Dict[Tuple[int, str, int, int], Path] = {}
        for fp in sorted(tad_dir.glob("*.npy")):
            stem = fp.stem
            m = re.match(r'window_(\d+)_(chr\w+)_(\d+)_(\d+)(?:_tad)?$', stem)
            if not m:
                continue
            key = (int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)))
            out[key] = fp
        if not out:
            warnings.warn(
                f"[HiCWindowDataset] No valid TAD prior files found in {tad_dir}. "
                "Expected names like window_0000_chr1_0_2000000_tad.npy"
            )
        return out

    def _resize_2d(self, arr: np.ndarray) -> np.ndarray:
        """Resize 2D matrix to (n_bins, n_bins) via bilinear interpolation."""
        t = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t,
            size=(self.n_bins, self.n_bins),
            mode='bilinear',
            align_corners=False,
        )
        return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    def _load_tad_prior(self, win: dict) -> np.ndarray:
        if not self._tad_prior_index:
            return self._zero_tad_prior(self.n_bins)

        key = (int(win['idx']), str(win['chrom']), int(win['start']), int(win['end']))
        fp = self._tad_prior_index.get(key)
        if fp is None:
            if not self._warned_missing_tad:
                warnings.warn(
                    f"[HiCWindowDataset] No TAD prior found for window key {key}. "
                    "Using zeros for missing windows."
                )
                self._warned_missing_tad = True
            return self._zero_tad_prior(self.n_bins)

        try:
            arr = np.load(fp).astype(np.float32)
        except Exception as exc:
            warnings.warn(f"[HiCWindowDataset] Failed loading TAD prior {fp}: {exc}. Using zeros.")
            return self._zero_tad_prior(self.n_bins)

        # Normalize possible layouts to (4, H, W).
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 4:
            pass
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            single = self._resize_2d(arr)
            out = self._zero_tad_prior(self.n_bins)
            out[0] = single
            return out
        else:
            warnings.warn(
                f"[HiCWindowDataset] Unexpected TAD prior shape {arr.shape} in {fp}. "
                "Using zeros."
            )
            return self._zero_tad_prior(self.n_bins)

        if arr.shape[0] != 4:
            warnings.warn(
                f"[HiCWindowDataset] TAD prior channel count must be 4, got {arr.shape[0]} in {fp}. "
                "Using zeros."
            )
            return self._zero_tad_prior(self.n_bins)

        if arr.shape[1] != self.n_bins or arr.shape[2] != self.n_bins:
            arr = np.stack([self._resize_2d(arr[c]) for c in range(4)], axis=0)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    # ── Per-file loaders ─────────────────────────────────────────

    def _load_rna_signal(self, fname: str) -> np.ndarray:
        """
        Load RNA signal for all conditions.

        If average_reps=True:
          For each condition, average across all replicates that have
          the file.  Missing files → zero vector with a warning.
          Returns (n_conditions, n_bins) float32.

        If average_reps=False:
          Each replicate becomes a separate "condition" row.
          Returns (total_reps_across_all_conditions, n_bins) float32.
        """
        out_rows = []
        for cond_idx, rep_dirs in enumerate(self.x_groups):
            rep_signals = []
            for rep_dir in rep_dirs:
                fpath = rep_dir / fname
                if not fpath.exists():
                    warnings.warn(
                        f"RNA file missing: {fpath}  → using zeros")
                    continue
                arr = np.load(fpath).astype(np.float32).squeeze()
                # Handle 2D array (n_bins, n_features) — take mean over features
                if arr.ndim == 2:
                    arr = arr.mean(axis=-1)
                # Resize if needed
                if arr.shape[0] != self.n_bins:
                    arr = np.interp(
                        np.linspace(0, 1, self.n_bins),
                        np.linspace(0, 1, arr.shape[0]),
                        arr,
                    ).astype(np.float32)
                rep_signals.append(
                    normalise_rna(arr) if self.normalize_rna
                    else arr.astype(np.float32)
                )

            if not rep_signals:
                row = np.zeros(self.n_bins, dtype=np.float32)
                if self.average_reps:
                    out_rows.append(row)
                else:
                    for _ in rep_dirs:
                        out_rows.append(row)
                continue

            if self.average_reps:
                out_rows.append(
                    np.mean(rep_signals, axis=0).astype(np.float32))
            else:
                # Pad missing reps with the mean so row count stays constant
                mean_sig = np.mean(rep_signals, axis=0).astype(np.float32)
                padded = rep_signals + [mean_sig] * (len(rep_dirs) - len(rep_signals))
                out_rows.extend(padded)

        return np.stack(out_rows, axis=0)   # (N_cond, n_bins)

    def _load_hic_target(self, fname: str) -> np.ndarray:
        """
        Load Hi-C target for all conditions, same grouping as RNA.
        Returns (n_conditions, n_bins, n_bins) float32.
        """
        out_rows = []
        for cond_idx, rep_dirs in enumerate(self.y_groups):
            rep_mats = []
            for rep_dir in rep_dirs:
                fpath = rep_dir / fname
                if not fpath.exists():
                    warnings.warn(
                        f"Hi-C file missing: {fpath}  → using zeros")
                    continue
                M = np.load(fpath).astype(np.float32)
                # Allow (n_bins, n_bins) or (1, n_bins, n_bins)
                if M.ndim == 3:
                    M = M[0]
                if M.shape[0] != self.n_bins:
                    try:
                        from scipy.ndimage import zoom
                        fac = self.n_bins / M.shape[0]
                        M = zoom(M, fac, order=1).astype(np.float32)
                    except ImportError:
                        M = np.zeros(
                            (self.n_bins, self.n_bins), dtype=np.float32)
                        warnings.warn(
                            "scipy not found; cannot resize Hi-C matrix. "
                            "Install with: pip install scipy"
                        )
                rep_mats.append(
                    normalise_hic(M) if self.normalize_hic
                    else M.astype(np.float32)
                )

            if not rep_mats:
                row = np.zeros((self.n_bins, self.n_bins), dtype=np.float32)
            else:
                row = np.mean(rep_mats, axis=0).astype(np.float32)

            if self.average_reps:
                out_rows.append(row)
            else:
                mean_mat = row
                padded = rep_mats + [mean_mat] * (len(rep_dirs) - len(rep_mats))
                out_rows.extend(padded)

        return np.stack(out_rows, axis=0)   # (N_cond, n_bins, n_bins)

    # ── Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        win   = self.windows[idx]
        fname = win['fname']
        rng   = np.random.default_rng(
            int(self.rng.integers(0, 2**31)) + idx)

        rna    = self._load_rna_signal(fname)   # (N_cond, n_bins)
        target = self._load_hic_target(fname)   # (N_cond, n_bins, n_bins)
        ctcf_prior = self._build_ctcf_prior_2d(
            chrom=win['chrom'],
            start=win['start'],
            end=win['end'],
        )
        tad_priors = self._load_tad_prior(win)  # (4, n_bins, n_bins)

        dna_seq = encode_dna(
            self.genome.fetch(win['chrom'], win['start'], win['end']),
            expected_len=self.dna_len,
        )

        # Augmentation: random horizontal flip of the genomic window
        if self.augment and rng.random() > 0.5:
            dna_seq = dna_seq.flip(0)
            rna     = rna[:, ::-1].copy()
            target  = target[:, ::-1, ::-1].copy()
            ctcf_prior = ctcf_prior[::-1, ::-1].copy()
            tad_priors = tad_priors[:, ::-1, ::-1].copy()

        return {
            'dna_seq'   : dna_seq,
            'rna_signal': torch.from_numpy(rna),
            'target'    : torch.from_numpy(target),
            'ctcf_prior_2d': torch.from_numpy(ctcf_prior),
            'tad_priors': torch.from_numpy(tad_priors),
            'meta'      : {k: win[k]
                           for k in ('idx', 'chrom', 'start', 'end')},
        }


# ═════════════════════════════════════════════════════════════════
#  SYNTHETIC DATASET  (no files required — for fast testing)
# ═════════════════════════════════════════════════════════════════

def _make_synthetic_matrix(n_bins: int, seed: int) -> np.ndarray:
    """
    Vectorised biologically plausible Hi-C matrix with:
      - Power-law distance decay (dominant feature of all Hi-C data)
      - 3–5 TADs (block-diagonal enrichments)
      - 2–4 chromatin loops (off-diagonal point enrichments)
    """
    rng  = np.random.default_rng(seed)
    idx  = np.arange(n_bins, dtype=np.float32)
    dist = np.abs(idx[:, None] - idx[None, :])
    M    = (5.0 * np.exp(-dist / (n_bins * 0.08))
            + rng.exponential(0.05, (n_bins, n_bins)).astype(np.float32))

    # TADs
    edges = sorted(set([0, n_bins] + rng.integers(0, n_bins,
                   rng.integers(3, 6)).tolist()))
    for s, e in zip(edges[:-1], edges[1:]):
        if e - s >= 3:
            M[s:e, s:e] += rng.uniform(1.5, 3.5)

    # Loops
    for _ in range(rng.integers(2, 5)):
        a = rng.integers(5, n_bins - 15)
        b = a + rng.integers(10, min(40, n_bins - a))
        M[a-2:a+2, b-2:b+2] += rng.uniform(1.5, 3.0)
        M[b-2:b+2, a-2:a+2] += rng.uniform(1.5, 3.0)

    return normalise_hic((M + M.T) / 2.0)


def _make_rna_signal(M: np.ndarray, n_conditions: int,
                     rng: np.random.Generator,
                     noise: float = 0.15) -> np.ndarray:
    """Derive plausible RNA proxy from diagonal contact sum."""
    base = M.sum(axis=1)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    sigs = []
    for _ in range(n_conditions):
        s = np.clip(base * rng.uniform(0.7, 1.3)
                    + rng.normal(0, noise, base.shape[0]), 0, None)
        sigs.append(normalise_rna(s.astype(np.float32)))
    return np.stack(sigs, axis=0)


def _make_condition_specific_targets(
    base_map: np.ndarray,
    rna_signal: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build per-condition Hi-C targets that depend on per-condition RNA.
    This prevents the synthetic task from teaching the model that RNA is ignorable.
    """
    n_conditions, n_bins = rna_signal.shape
    out = []
    base = np.asarray(base_map, dtype=np.float32)
    for c in range(n_conditions):
        sig = rna_signal[c].astype(np.float32)
        sig = sig - sig.mean()
        sig_std = float(sig.std()) + 1e-6
        sig = sig / sig_std

        # Condition-specific smooth modulation over bin-pairs.
        outer = np.outer(sig, sig).astype(np.float32)
        outer = outer / (np.max(np.abs(outer)) + 1e-6)

        # Add structured perturbation + mild noise, then symmetrise.
        alpha = float(rng.uniform(0.08, 0.22))
        cond_map = base + alpha * outer
        cond_map += rng.normal(0.0, 0.02, size=base.shape).astype(np.float32)
        cond_map = np.clip((cond_map + cond_map.T) * 0.5, 0.0, None)

        out.append(normalise_hic(cond_map))
    return np.stack(out, axis=0).astype(np.float32)


class SyntheticHiCDataset(Dataset):
    """
    Synthetic dataset — generates Hi-C + RNA + DNA on the fly.
    No files required.  Used for test_run.py and CI checks.
    """

    def __init__(
        self,
        n_samples:    int = 64,
        n_bins:       int = 200,
        n_conditions: int = 6,
        bin_size:     int = 200,
        seed:         int = 0,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        self.n_bins       = n_bins
        self.dna_len      = n_bins * bin_size
        self.seed         = seed

        print(f"[SyntheticHiCDataset] Generating {n_samples} samples "
              f"(n_bins={n_bins}, N_cond={n_conditions})…", flush=True)
        self._matrices = [_make_synthetic_matrix(n_bins, seed + i)
                          for i in range(n_samples)]
        print("[SyntheticHiCDataset] Done.", flush=True)

    def __len__(self) -> int:
        return len(self._matrices)

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        M   = self._matrices[idx]
        rna = _make_rna_signal(M, self.n_conditions, rng)
        target = _make_condition_specific_targets(M, rna, rng)
        return {
            'dna_seq'   : synthetic_dna(self.dna_len, rng=rng),
            'rna_signal': torch.from_numpy(rna),
            'target'    : torch.from_numpy(target),
            'ctcf_prior_2d': torch.zeros(
                (self.n_bins, self.n_bins), dtype=torch.float32
            ),
            'tad_priors': torch.zeros(
                (4, self.n_bins, self.n_bins), dtype=torch.float32
            ),
            'meta'      : {'idx': idx, 'chrom': 'synth',
                           'start': idx * 200_000,
                           'end':  (idx + 1) * 200_000},
        }


# ─────────────────────────────────────────────────────────────────
#  DataLoader builder
# ─────────────────────────────────────────────────────────────────

def build_dataloaders(
    dataset_train: Dataset,
    dataset_val:   Optional[Dataset] = None,
    batch_size:    int   = 2,
    num_workers:   int   = 2,
    pin_memory:    bool  = True,
    val_split:     float = 0.1,
    seed:          int   = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optionally val) DataLoaders.
    If dataset_val is None, val_split fraction is split from train.
    """
    if dataset_val is None and val_split > 0.0:
        n_val   = max(1, int(len(dataset_train) * val_split))
        n_train = len(dataset_train) - n_val
        gen = torch.Generator().manual_seed(seed)
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train, [n_train, n_val], generator=gen)

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=True,
        )

    return (
        _loader(dataset_train, shuffle=True),
        _loader(dataset_val, shuffle=False) if dataset_val else None,
    )
