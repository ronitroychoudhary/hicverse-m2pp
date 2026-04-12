"""
HiC-Verse M2++ Testing Script
================================
Comprehensive evaluation of saved checkpoints.

Features:
  - Per-condition metrics
  - Per-chromosome analysis
  - Distance-stratified correlation
  - TAD boundary detection accuracy
  - Loop calling metrics
  - Visualization outputs
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_OK = True
except ImportError:
    PLOT_OK = False
    warnings.warn("Matplotlib/seaborn not available. Skipping plots.")

from config_m2pp import HiCVerseConfigM2PP
from model_m2pp import HiCVerseModelM2PP, GRUFallbackSSM
from dataset import SyntheticHiCDataset, HiCWindowDataset


# ═════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════

def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation."""
    p = pred.flatten()
    t = target.flatten()
    if len(p) < 2:
        return 0.0
    p_c = p - p.mean()
    t_c = t - t.mean()
    num = (p_c * t_c).sum()
    denom = np.sqrt((p_c ** 2).sum() * (t_c ** 2).sum()) + 1e-8
    return float(num / denom)


def spearman_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation."""
    p = np.asarray(pred, dtype=float).flatten()
    t = np.asarray(target, dtype=float).flatten()
    if p.size < 2 or t.size < 2:
        return 0.0
    if (np.nanstd(p) < 1e-12) or (np.nanstd(t) < 1e-12):
        return 0.0
    try:
        from scipy.stats import spearmanr
        val = float(spearmanr(p, t, nan_policy='omit')[0])
        if np.isfinite(val):
            return val
    except:
        pass

    # Fallback: Pearson on ranks
    p_rank = p.argsort().argsort().astype(float)
    t_rank = t.argsort().argsort().astype(float)
    return pearson_r(p_rank, t_rank)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""
    return float(((pred - target) ** 2).mean())


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error (L1)."""
    return float(np.abs(pred - target).mean())


def distance_stratified_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute correlation stratified by genomic distance.
    Returns dict: {distance_bin → correlation}
    """
    L = pred.shape[0]
    results = {}
    for d_min, d_max in [(0, L // 10), (L // 10, L // 4),
                          (L // 4, L // 2), (L // 2, L)]:
        mask = np.zeros((L, L), dtype=bool)
        for i in range(L):
            for j in range(i + d_min, min(i + d_max, L)):
                mask[i, j] = True
                mask[j, i] = True
        if mask.sum() > 0:
            r = pearson_r(pred[mask], target[mask])
            results[f"{d_min}-{d_max}"] = r
    return results


def compute_insulation_score(
    mat: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute insulation score for TAD boundary detection.
    Lower values = stronger boundary.
    """
    L = mat.shape[0]
    ins = np.zeros(L)
    for i in range(L):
        left = max(0, i - window)
        right = min(L, i + window)
        left_block = mat[left:i, i:right]
        right_block = mat[i:right, left:i]
        vals = []
        if left_block.size > 0:
            vals.append(float(left_block.mean()))
        if right_block.size > 0:
            vals.append(float(right_block.mean()))
        ins[i] = float(np.mean(vals)) if vals else 0.0
    return ins


def detect_loops(
    mat: np.ndarray,
    min_dist: int = 10,
    threshold_pct: float = 95.0,
) -> List[tuple]:
    """
    Simple loop detection via local maxima.
    Returns list of (i, j) loop anchors.
    """
    L = mat.shape[0]
    loops = []
    # Mask diagonal
    mask = np.ones((L, L), dtype=bool)
    for d in range(min_dist):
        idx = np.arange(L - d)
        mask[idx, idx + d] = False
        mask[idx + d, idx] = False
    # Threshold
    vals = mat[mask]
    if len(vals) == 0:
        return loops
    thresh = np.percentile(vals, threshold_pct)
    # Local maxima
    for i in range(min_dist, L - min_dist):
        for j in range(i + min_dist, L - min_dist):
            if mat[i, j] >= thresh:
                # Check if local max
                neighbors = mat[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
                if mat[i, j] == neighbors.max():
                    loops.append((i, j))
    return loops


def loop_precision_recall(
    pred_loops: List[tuple],
    target_loops: List[tuple],
    tolerance: int = 5,
) -> Dict[str, float]:
    """
    Compute precision/recall for loop calling.
    tolerance: pixels allowed for matching.
    """
    if len(pred_loops) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    if len(target_loops) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Precision: fraction of predicted loops near a target loop
    true_pos = 0
    for pi, pj in pred_loops:
        for ti, tj in target_loops:
            if abs(pi - ti) <= tolerance and abs(pj - tj) <= tolerance:
                true_pos += 1
                break
    precision = true_pos / len(pred_loops) if len(pred_loops) > 0 else 0.0

    # Recall: fraction of target loops near a predicted loop
    true_pos = 0
    for ti, tj in target_loops:
        for pi, pj in pred_loops:
            if abs(pi - ti) <= tolerance and abs(pj - tj) <= tolerance:
                true_pos += 1
                break
    recall = true_pos / len(target_loops) if len(target_loops) > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    return {'precision': precision, 'recall': recall, 'f1': f1}


# ═════════════════════════════════════════════════════════════════
#  TESTER
# ═════════════════════════════════════════════════════════════════

class TesterM2PP:
    """
    Comprehensive testing for M2++ checkpoints.

    Args:
        checkpoint_path : path to .pt checkpoint
        mode            : 'synthetic' or 'real'
        chroms          : chromosome filter for real mode
        out_dir         : output directory for results
        n_samples       : total windows to test (overrides samples_per_condition)
        samples_per_condition : windows to test per selected condition
        condition_tokens: condition indices/names to evaluate (None = all)
        sample_idx      : specific sample index to test (overrides n_samples)
    """

    def __init__(
        self,
        checkpoint_path: str,
        mode: str = 'real',
        chroms: Optional[List[str]] = None,
        out_dir: str = './test_results_m2pp',
        n_samples: Optional[int] = None,
        samples_per_condition: int = 10,
        condition_tokens: Optional[List[str]] = None,
        sample_idx: Optional[int] = None,
        force_cpu: bool = False,
        require_gpu: bool = False,
        allow_cpu_fallback: bool = False,
        align_conditions: bool = True,
        strict_ckpt_load: bool = True,
        auto_fix_cudnn_path: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.mode = mode
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.requested_n_samples = n_samples
        self.samples_per_condition = int(samples_per_condition)
        self.n_samples = n_samples
        self.condition_tokens = condition_tokens
        self.sample_idx = sample_idx
        self.require_gpu = require_gpu
        self.allow_cpu_fallback = allow_cpu_fallback
        self.align_conditions = align_conditions
        self.strict_ckpt_load = strict_ckpt_load
        self.auto_fix_cudnn_path = auto_fix_cudnn_path
        self._signal_cache: Dict[int, bool] = {}
        self._warned_condition_mismatch = False
        self.eval_condition_indices: List[int] = []
        self.eval_condition_labels: List[str] = []
        self._condition_label_by_index: Dict[int, str] = {}

        print(f"\n{'=' * 60}")
        print(f"  HiC-Verse M2++ Tester")
        print(f"  Checkpoint: {self.checkpoint_path.name}")
        print(f"  Mode: {mode}")
        print(f"{'=' * 60}\n")

        # Load checkpoint
        ckpt = self._load_checkpoint(checkpoint_path)
        self.cfg: HiCVerseConfigM2PP = ckpt['cfg']
        self._sync_cfg_with_checkpoint(ckpt['model'])
        if force_cpu:
            self.cfg.device = 'cpu'
        else:
            if torch.cuda.is_available():
                self.cfg.device = 'cuda'
            else:
                if require_gpu:
                    raise RuntimeError(
                        "GPU was requested (--require_gpu) but CUDA is not available."
                    )
                if str(self.cfg.device).startswith('cuda'):
                    warnings.warn("Checkpoint config requested CUDA but CUDA is unavailable. Using CPU.")
                self.cfg.device = 'cpu'
        self.device = torch.device(self.cfg.device)

        # Model
        self.model = self._build_model_with_device_fallback()
        self._load_state_dict_with_fallback(ckpt['model'])
        if self.device.type == 'cpu':
            self._replace_mamba_ssm_with_gru_for_cpu()
        self.model.eval()
        print(self.model.summary())

        if self.sample_idx is not None and self.n_samples is not None:
            warnings.warn("--sample_idx is set; --n_samples will be ignored.")
        if self.sample_idx is None and self.n_samples is None:
            self.n_samples = self.samples_per_condition

        # Dataset
        self.dataset = self._build_dataset(mode, chroms)
        self.eval_condition_indices, self.eval_condition_labels = self._resolve_eval_conditions()
        self._condition_label_by_index = {
            idx: label for idx, label in zip(self.eval_condition_indices, self.eval_condition_labels)
        }
        self.eval_indices = self._resolve_eval_indices()
        self.eval_dataset = Subset(self.dataset, self.eval_indices)
        self.dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        print(f"Dataset: {len(self.dataset)} samples\n")
        print(f"Evaluating: {len(self.eval_indices)} sample(s)")
        print(
            "Evaluating conditions: "
            + ", ".join(
                f"{label}(idx={idx})"
                for idx, label in zip(self.eval_condition_indices, self.eval_condition_labels)
            )
        )
        if self.sample_idx is not None:
            print(f"Selection mode: fixed index = {self.sample_idx}\n")
        elif self.requested_n_samples is not None:
            print("Selection mode: auto-selected evenly across dataset (--n_samples)\n")
        else:
            print(
                f"Selection mode: auto-selected evenly across dataset "
                f"({self.samples_per_condition} per condition)\n"
            )

        self._print_selected_windows()

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint across PyTorch versions.
        PyTorch 2.6 changed torch.load default to weights_only=True, which
        fails for checkpoints that store custom classes (e.g. cfg dataclass).
        """
        try:
            # Preferred path for PyTorch>=2.6 when checkpoint is trusted.
            return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch versions without weights_only kwarg.
            return torch.load(checkpoint_path, map_location='cpu')
        except pickle.UnpicklingError as e:
            # Compatibility fallback if a runtime still defaulted to weights_only=True.
            msg = str(e)
            if "Weights only load failed" in msg:
                warnings.warn(
                    "Checkpoint requires full unpickling. Retrying with "
                    "weights_only=False (trusted checkpoint assumed)."
                )
                return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            raise

    def _build_model_with_device_fallback(self) -> HiCVerseModelM2PP:
        """Create model on requested device; optionally fallback to CPU if explicitly allowed."""
        try:
            return HiCVerseModelM2PP(self.cfg).to(self.device)
        except RuntimeError as e:
            msg = str(e)
            cudnn_mismatch = (
                self.device.type == 'cuda'
                and ('cudnn' in msg.lower() or 'cuDNN version incompatibility' in msg)
            )
            if not cudnn_mismatch:
                raise

            if self.auto_fix_cudnn_path and self.device.type == 'cuda':
                retried = os.environ.get("HICVERSE_CUDNN_RELAUNCHED", "0") == "1"
                if not retried:
                    warnings.warn(
                        "CUDA/cuDNN mismatch detected. Relaunching once with "
                        "LD_LIBRARY_PATH cleared so PyTorch can use bundled cuDNN."
                    )
                    self._relaunch_without_ld_library_path()

            if not self.allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA/cuDNN runtime mismatch detected while creating the model. "
                    "GPU inference is required here, so CPU fallback is disabled.\n"
                    "Fix environment and rerun, or pass --allow_cpu_fallback to permit CPU fallback.\n"
                    "Likely fix: remove incompatible cuDNN from LD_LIBRARY_PATH so PyTorch uses bundled cuDNN."
                ) from e

            warnings.warn(
                "CUDA/cuDNN runtime mismatch detected. Falling back to CPU because "
                "--allow_cpu_fallback was set."
            )
            self.cfg.device = 'cpu'
            self.device = torch.device('cpu')
            torch.backends.cudnn.enabled = False
            return HiCVerseModelM2PP(self.cfg).to(self.device)

    @staticmethod
    def _relaunch_without_ld_library_path() -> None:
        env = os.environ.copy()
        old_ld = env.pop("LD_LIBRARY_PATH", None)
        env["HICVERSE_CUDNN_RELAUNCHED"] = "1"
        if old_ld is not None:
            env["HICVERSE_OLD_LD_LIBRARY_PATH"] = old_ld
        argv = [sys.executable] + sys.argv
        os.execvpe(sys.executable, argv, env)

    def _sync_cfg_with_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Bring cfg in line with checkpoint tensor shapes when serialized cfg is stale.
        """
        changed: List[str] = []

        d_model_key = 'fusion.gate_proj.weight'
        if d_model_key in state_dict:
            inferred_d_model = int(state_dict[d_model_key].shape[0])
            if inferred_d_model != int(self.cfg.d_model):
                self.cfg.d_model = inferred_d_model
                if self.cfg.dna_cnn_channels:
                    self.cfg.dna_cnn_channels[-1] = inferred_d_model
                if self.cfg.rna_cnn_channels:
                    self.cfg.rna_cnn_channels[-1] = inferred_d_model
                changed.append(f"d_model={inferred_d_model}")

        d_outer_key = 'outer_proj.proj.weight'
        if d_outer_key in state_dict:
            inferred_d_outer = int(state_dict[d_outer_key].shape[0])
            if inferred_d_outer != int(self.cfg.d_outer):
                self.cfg.d_outer = inferred_d_outer
                changed.append(f"d_outer={inferred_d_outer}")

        # Infer number of Hi-C refinement blocks from checkpoint key layout.
        extra_block_indices = []
        prefix = "hic_head.extra_refine_layers."
        for k in state_dict.keys():
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            idx_str = rest.split(".", 1)[0]
            if idx_str.isdigit():
                extra_block_indices.append(int(idx_str))
        inferred_hic_blocks = 1 + (max(extra_block_indices) + 1 if extra_block_indices else 0)
        if inferred_hic_blocks != int(getattr(self.cfg, "n_hic_blocks", 1)):
            self.cfg.n_hic_blocks = inferred_hic_blocks
            changed.append(f"n_hic_blocks={inferred_hic_blocks}")

        d_inner = int(self.cfg.d_model) * int(self.cfg.expand)
        if d_inner % int(self.cfg.headdim) != 0:
            for hd in [128, 64, 32, 16, 8, 4, 2, 1]:
                if d_inner % hd == 0:
                    self.cfg.headdim = hd
                    changed.append(f"headdim={hd}")
                    break

        if changed:
            warnings.warn(
                "Adjusted checkpoint cfg from state_dict shapes: " + ", ".join(changed)
            )

    def _load_state_dict_with_fallback(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load checkpoint strictly first; if that fails, load matching keys only.
        """
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError as e:
            if self.strict_ckpt_load:
                raise RuntimeError(
                    "Strict checkpoint load failed. This usually means model/config mismatch. "
                    "Recheck checkpoint compatibility or rerun with --non_strict_ckpt_load."
                ) from e
            warnings.warn(
                "Strict checkpoint load failed; retrying with shape-matched keys only. "
                "Unmatched layers will use random initialization.\n"
                f"Original error: {e}"
            )

        model_sd = self.model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
        load_info = self.model.load_state_dict(filtered, strict=False)
        missing_n = len(load_info.missing_keys)
        unexpected_n = len(load_info.unexpected_keys)
        dropped_n = len(state_dict) - len(filtered)
        warnings.warn(
            f"Loaded {len(filtered)} keys, dropped {dropped_n}; "
            f"missing={missing_n}, unexpected={unexpected_n}."
        )

    def _replace_mamba_ssm_with_gru_for_cpu(self) -> None:
        """
        CPU-safe mode: keep non-SSM checkpoint weights, replace Mamba SSM blocks
        with GRU fallback modules to avoid CUDA-only causal_conv1d runtime errors.
        """
        replaced = 0
        blocks = getattr(self.model.dna_encoder.hybrid_stack, 'blocks', [])
        for blk in blocks:
            if hasattr(blk, 'ssm_fwd'):
                blk.ssm_fwd = GRUFallbackSSM(d_model=self.cfg.d_model).to(self.device)
                replaced += 1
            if hasattr(blk, 'ssm_bwd'):
                blk.ssm_bwd = GRUFallbackSSM(d_model=self.cfg.d_model).to(self.device)
        if replaced > 0:
            warnings.warn(
                f"CPU mode active: replaced {replaced} Mamba SSM block(s) with GRU fallback. "
                "Evaluation will run, but metrics may differ from true Mamba checkpoint behavior."
            )

    def _derive_dataset_condition_labels(self) -> List[str]:
        # Real dataset: prefer condXX names from directory groups.
        if hasattr(self.dataset, 'x_groups'):
            labels: List[str] = []
            for rep_dirs in getattr(self.dataset, 'x_groups', []):
                if not rep_dirs:
                    labels.append(f"cond{len(labels) + 1:02d}")
                    continue
                m = re.match(r'(cond\d+)', rep_dirs[0].name)
                labels.append(m.group(1) if m else f"cond{len(labels) + 1:02d}")
            if labels:
                return labels

        n_dataset = int(getattr(self.dataset, 'n_conditions', self.cfg.n_conditions))
        return [f"cond{i + 1:02d}" for i in range(n_dataset)]

    def _resolve_eval_conditions(self) -> tuple[List[int], List[str]]:
        labels_all = self._derive_dataset_condition_labels()
        n_dataset = len(labels_all)
        n_ckpt = int(self.cfg.n_conditions)

        if self.align_conditions:
            if n_dataset < n_ckpt:
                raise ValueError(
                    f"Dataset provides {n_dataset} conditions but checkpoint expects {n_ckpt}. "
                    "Cannot evaluate reliably."
                )
            n_eval_max = min(n_dataset, n_ckpt)
        else:
            n_eval_max = n_dataset

        labels_eval = labels_all[:n_eval_max]
        available_by_name = {name.lower(): i for i, name in enumerate(labels_eval)}

        if not self.condition_tokens:
            indices = list(range(n_eval_max))
            return indices, labels_eval

        indices: List[int] = []
        seen = set()
        for token in self.condition_tokens:
            token_l = str(token).strip().lower()
            idx: Optional[int] = None
            if token_l.isdigit():
                idx = int(token_l)
            elif token_l in available_by_name:
                idx = available_by_name[token_l]

            if idx is None:
                raise ValueError(
                    f"Unknown condition {token!r}. Available: "
                    + ", ".join(f"{i}:{name}" for i, name in enumerate(labels_eval))
                )
            if idx < 0 or idx >= n_eval_max:
                raise ValueError(
                    f"Condition index {idx} out of range [0, {n_eval_max - 1}]"
                )
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)

        labels = [labels_eval[i] for i in indices]
        return indices, labels

    def _window_info(self, dataset_idx: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {'dataset_idx': int(dataset_idx)}
        if hasattr(self.dataset, 'windows'):
            windows = getattr(self.dataset, 'windows')
            if 0 <= dataset_idx < len(windows):
                win = windows[dataset_idx]
                info.update({
                    'window_idx': int(win.get('idx', dataset_idx)),
                    'chrom': win.get('chrom'),
                    'start': int(win.get('start', -1)),
                    'end': int(win.get('end', -1)),
                    'fname': win.get('fname'),
                })
                return info
        info.update({
            'window_idx': int(dataset_idx),
            'chrom': 'synth',
            'start': None,
            'end': None,
            'fname': None,
        })
        return info

    def _print_selected_windows(self) -> None:
        if not self.eval_indices:
            return
        print("Selected windows:")
        for idx in self.eval_indices:
            w = self._window_info(idx)
            if w.get('chrom') is not None and w.get('start') is not None:
                region = f"{w['chrom']}:{w['start']}-{w['end']}"
            else:
                region = "synth"
            print(
                f"  dataset_idx={w['dataset_idx']:04d} | window={w['window_idx']:04d} "
                f"| region={region} | file={w.get('fname')}"
            )
        print("")

    def _resolve_eval_indices(self) -> List[int]:
        dataset_len = len(self.dataset)
        if dataset_len <= 0:
            return []

        if self.sample_idx is not None:
            if self.sample_idx < 0 or self.sample_idx >= dataset_len:
                raise ValueError(
                    f"sample_idx={self.sample_idx} out of range for dataset "
                    f"of size {dataset_len}"
                )
            return [self.sample_idx]

        if self.n_samples is None or self.n_samples >= dataset_len:
            return list(range(dataset_len))
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided.")

        # Deterministic automatic selection spread across the dataset.
        # Use segment centers (not endpoints) so n_samples=1 does not always pick 0.
        edges = np.linspace(0, dataset_len - 1, num=self.n_samples + 2, dtype=int)
        indices = edges[1:-1].tolist()
        unique = []
        seen = set()
        for i in indices:
            if i not in seen:
                unique.append(i)
                seen.add(i)

        if len(unique) < self.n_samples:
            needed = self.n_samples - len(unique)
            extras = [i for i in range(dataset_len) if i not in seen][:needed]
            unique.extend(extras)
            seen.update(extras)

        if self.mode == 'real':
            unique = self._prefer_informative_indices(unique, desired=self.n_samples)
        return unique[:self.n_samples]

    @staticmethod
    def _is_informative_map(arr: np.ndarray, eps: float = 1e-6) -> bool:
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return False
        return (np.nanmax(np.abs(vals)) > eps) and (np.nanstd(vals) > eps)

    def _sample_has_signal(self, idx: int) -> bool:
        if idx in self._signal_cache:
            return self._signal_cache[idx]
        ok = True
        try:
            ds = self.dataset
            if hasattr(ds, 'windows') and hasattr(ds, '_load_hic_target'):
                fname = ds.windows[idx]['fname']
                target = ds._load_hic_target(fname)  # (N, L, L)
            else:
                target = ds[idx]['target']
                if isinstance(target, torch.Tensor):
                    target = target.detach().cpu().numpy()
            ok = self._is_informative_map(target)
        except Exception:
            ok = True
        self._signal_cache[idx] = ok
        return ok

    def _prefer_informative_indices(self, base: List[int], desired: int) -> List[int]:
        if desired <= 0:
            return []
        dataset_len = len(self.dataset)
        used = set()
        chosen: List[int] = []

        # First pass: keep base indices if informative, otherwise search nearby.
        local_radius = max(25, dataset_len // max(1, desired * 10))
        for idx in base:
            idx = int(idx)
            if idx in used:
                continue
            if self._sample_has_signal(idx):
                chosen.append(idx)
                used.add(idx)
                continue
            found = None
            for r in range(1, local_radius + 1):
                left = idx - r
                right = idx + r
                if left >= 0 and left not in used and self._sample_has_signal(left):
                    found = left
                    break
                if right < dataset_len and right not in used and self._sample_has_signal(right):
                    found = right
                    break
            if found is not None:
                chosen.append(found)
                used.add(found)

        # Second pass: fill remaining slots with informative samples.
        if len(chosen) < desired:
            for idx in range(dataset_len):
                if idx in used:
                    continue
                if self._sample_has_signal(idx):
                    chosen.append(idx)
                    used.add(idx)
                    if len(chosen) >= desired:
                        break

        # Last resort: fill from any remaining indices.
        if len(chosen) < desired:
            for idx in range(dataset_len):
                if idx in used:
                    continue
                chosen.append(idx)
                used.add(idx)
                if len(chosen) >= desired:
                    break

        return sorted(chosen[:desired])

    def _build_dataset(self, mode, chroms):
        cfg = self.cfg
        if mode == 'synthetic':
            synthetic_n = self.n_samples if self.n_samples is not None else 64
            if self.sample_idx is not None:
                synthetic_n = max(synthetic_n, self.sample_idx + 1)
            return SyntheticHiCDataset(
                n_samples=synthetic_n,
                n_bins=cfg.n_bins,
                n_conditions=cfg.n_conditions,
                bin_size=cfg.test_bin_size,
                seed=12345,
            )
        elif mode == 'real':
            return HiCWindowDataset(
                x_dir=cfg.x_dir,
                y_dir=cfg.y_dir,
                n_bins=cfg.n_bins,
                n_conditions=cfg.n_conditions,
                bin_size=cfg.bin_size,
                genome_fasta_path=cfg.genome_fasta_path,
                chroms=chroms,
                augment=False,
                normalize_rna=cfg.normalize_rna_inputs,
                normalize_hic=cfg.normalize_hic_targets,
                ctcf_fimo_tsv_path=getattr(cfg, 'ctcf_fimo_tsv_path', None),
                use_ctcf_fimo_prior=getattr(cfg, 'use_ctcf_fimo_prior', True),
                tad_priors_dir=getattr(cfg, 'tad_priors_dir', None),
                use_tad_priors=getattr(cfg, 'use_tad_priors', True),
            )
        raise ValueError(f"mode must be 'synthetic' or 'real', got {mode!r}")

    # ── Per-sample evaluation ────────────────────────────────────

    @torch.no_grad()
    def _evaluate_sample(self, batch: dict) -> Dict:
        """Run model and compute metrics for one sample."""
        dna = batch['dna_seq'].to(self.device)
        rna = batch['rna_signal'].to(self.device)
        ctcf_prior = batch.get('ctcf_prior_2d')
        tad_priors = batch.get('tad_priors')
        if ctcf_prior is not None:
            ctcf_prior = ctcf_prior.to(self.device)
        if tad_priors is not None:
            tad_priors = tad_priors.to(self.device)
        target = batch['target'].cpu().numpy()[0]  # (N, L, L)
        rna, target = self._align_sample_conditions(rna, target)

        # Predict
        preds = self.model(
            dna,
            rna,
            ctcf_prior_2d=ctcf_prior,
            tad_priors=tad_priors,
            return_loops=False,
        )
        pred_maps = preds['contact_maps'].cpu().numpy()[0]  # (N, L, L)

        # Metrics
        results = {
            'meta': self._to_python(batch['meta']),
            'per_condition': [],
            'aggregate': {},
        }

        # Per-condition metrics
        n_conditions_eval = min(pred_maps.shape[0], target.shape[0])
        cond_indices = [
            i for i in self.eval_condition_indices
            if 0 <= i < n_conditions_eval
        ]
        if not cond_indices:
            raise ValueError(
                f"No selected conditions available for this sample. "
                f"Selected={self.eval_condition_indices}, available=[0..{n_conditions_eval - 1}]"
            )

        for cond_idx in cond_indices:
            pred = pred_maps[cond_idx]
            tgt = target[cond_idx]

            metrics = {
                'condition': cond_idx,
                'condition_name': self._condition_label_by_index.get(cond_idx, f"cond{cond_idx + 1:02d}"),
                'pearson': pearson_r(pred, tgt),
                'spearman': spearman_r(pred, tgt),
                'mse': mse(pred, tgt),
                'mae': mae(pred, tgt),
            }

            # Distance-stratified
            dist_corr = distance_stratified_correlation(pred, tgt)
            metrics['distance_stratified'] = dist_corr

            # TAD boundaries
            ins_pred = compute_insulation_score(pred)
            ins_tgt = compute_insulation_score(tgt)
            metrics['insulation_r'] = pearson_r(ins_pred, ins_tgt)

            # Loop detection
            loops_pred = detect_loops(pred)
            loops_tgt = detect_loops(tgt)
            metrics['loop_metrics'] = loop_precision_recall(loops_pred, loops_tgt)
            metrics['n_loops_pred'] = len(loops_pred)
            metrics['n_loops_target'] = len(loops_tgt)

            results['per_condition'].append(metrics)

        # Aggregate across conditions
        results['aggregate'] = {
            'pearson': np.mean([m['pearson'] for m in results['per_condition']]),
            'spearman': np.mean([m['spearman'] for m in results['per_condition']]),
            'mse': np.mean([m['mse'] for m in results['per_condition']]),
            'mae': np.mean([m['mae'] for m in results['per_condition']]),
            'insulation_r': np.mean([m['insulation_r'] for m in results['per_condition']]),
            'loop_precision': np.mean([m['loop_metrics']['precision']
                                       for m in results['per_condition']]),
            'loop_recall': np.mean([m['loop_metrics']['recall']
                                    for m in results['per_condition']]),
            'loop_f1': np.mean([m['loop_metrics']['f1']
                                for m in results['per_condition']]),
        }

        return results, pred_maps, target

    def _align_sample_conditions(
        self,
        rna: torch.Tensor,
        target: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Align condition dimension between input data and checkpoint expectation.
        """
        n_rna = int(rna.shape[1])
        n_tgt = int(target.shape[0])
        n_ckpt = int(self.cfg.n_conditions)

        n_common = min(n_rna, n_tgt)
        if n_rna != n_tgt:
            if not self._warned_condition_mismatch:
                warnings.warn(
                    f"Condition mismatch in batch: rna={n_rna}, target={n_tgt}. "
                    f"Using first {n_common} shared conditions."
                )
                self._warned_condition_mismatch = True
            rna = rna[:, :n_common, :]
            target = target[:n_common]

        if self.align_conditions:
            if n_common > n_ckpt:
                if not self._warned_condition_mismatch:
                    warnings.warn(
                        f"Dataset has {n_common} conditions but checkpoint expects {n_ckpt}. "
                        f"Using first {n_ckpt} conditions for evaluation."
                    )
                    self._warned_condition_mismatch = True
                rna = rna[:, :n_ckpt, :]
                target = target[:n_ckpt]
            elif n_common < n_ckpt:
                raise ValueError(
                    f"Dataset provides {n_common} conditions but checkpoint expects {n_ckpt}. "
                    "Cannot evaluate reliably."
                )
        return rna, target

    @staticmethod
    def _to_python(value: Any) -> Any:
        """Convert tensors/ndarrays in batched metadata to JSON-serializable python types."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: TesterM2PP._to_python(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return TesterM2PP._to_python(value[0])
            return [TesterM2PP._to_python(v) for v in value]
        return value

    # ── Visualization ────────────────────────────────────────────

    @staticmethod
    def _sanitize_token(text: str) -> str:
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_') or "na"

    def _next_available_path(self, base_path: Path) -> Path:
        """
        Return a non-overwriting path by appending _vNN when needed.
        """
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        for i in range(1, 10000):
            cand = parent / f"{stem}_v{i:02d}{suffix}"
            if not cand.exists():
                return cand
        raise RuntimeError(f"Could not allocate unique output filename for {base_path}")

    def _plot_comparison(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        sample_idx: int,
        cond_idx: int,
    ) -> Optional[Path]:
        """Plot predicted vs target Hi-C map via model helper."""
        if not PLOT_OK:
            return None
        cond_name = self._sanitize_token(
            self._condition_label_by_index.get(cond_idx, f"cond{cond_idx + 1:02d}")
        )
        base_path = self.out_dir / (
            f"sample_{sample_idx:04d}_cond_{cond_idx:02d}_{cond_name}.png"
        )
        out_path = self._next_available_path(base_path)
        self.model.visualize_contact_maps(
            pred_maps=pred,
            target_maps=target,
            sample_idx=0,
            condition_idx=0,
            out_path=str(out_path),
            title_prefix=f"S{sample_idx:04d} C{cond_idx:02d} ",
        )
        return out_path

    def _plot_summary(self, all_results: List[Dict]):
        """Plot summary statistics."""
        if not PLOT_OK:
            return

        # Collect aggregate metrics
        metrics = {
            'Pearson r': [r['aggregate']['pearson'] for r in all_results],
            'Spearman r': [r['aggregate']['spearman'] for r in all_results],
            'MAE': [r['aggregate']['mae'] for r in all_results],
            'MSE': [r['aggregate']['mse'] for r in all_results],
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, (name, vals) in zip(axes, metrics.items()):
            vals_arr = np.asarray(vals, dtype=float)
            vals_finite = vals_arr[np.isfinite(vals_arr)]
            if vals_finite.size == 0:
                ax.text(0.5, 0.5, 'No finite values', ha='center', va='center')
                ax.set_title(f'{name} Distribution')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.hist(vals_finite, bins=20, edgecolor='black', alpha=0.7)
            mean_val = float(np.mean(vals_finite))
            ax.axvline(mean_val, color='red', linestyle='--',
                       label=f'Mean: {mean_val:.4f}')
            ax.set_xlabel(name)
            ax.set_ylabel('Count')
            ax.set_title(f'{name} Distribution')
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.out_dir / 'summary_metrics.png', dpi=150)
        plt.close()

    # ── Main test loop ───────────────────────────────────────────

    def run(
        self,
        save_plots: bool = True,
        compare_maps: bool = False,
        compare_max_samples: int = 10,
        compare_n_conditions: int = 2,
    ):
        """Run comprehensive testing."""
        all_results = []

        if compare_maps and not save_plots:
            warnings.warn("compare_maps requested but plots are disabled (--no_plots). Skipping map comparisons.")
            compare_maps = False
        if compare_maps and save_plots:
            max_samples = min(compare_max_samples, len(self.eval_indices))
            max_conds = min(compare_n_conditions, len(self.eval_condition_indices))
            print(
                f"Map export enabled: up to {max_samples} sample(s) x {max_conds} condition(s) "
                f"= {max_samples * max_conds} image(s)."
            )

        saved_map_paths: List[str] = []

        print("Running evaluation...\n")
        for idx, batch in enumerate(self.dataloader):
            source_sample_idx = self.eval_indices[idx]

            results, pred_maps, target = self._evaluate_sample(batch)
            all_results.append(results)

            # Print progress
            agg = results['aggregate']
            meta = results.get('meta', {})
            win_idx = meta.get('idx', source_sample_idx)
            chrom = meta.get('chrom', 'NA')
            start = meta.get('start', 'NA')
            end = meta.get('end', 'NA')
            print(
                f"Sample {source_sample_idx:04d}  |  "
                f"Window {int(win_idx):04d}  |  "
                f"{chrom}:{start}-{end}  |  "
                f"Pearson: {agg['pearson']:.4f}  "
                f"Spearman: {agg['spearman']:.4f}  "
                f"MAE: {agg['mae']:.4f}  "
                f"Loop F1: {agg['loop_f1']:.4f}"
            )
            for m in results['per_condition']:
                print(
                    f"    Condition {m['condition_name']} (idx={m['condition']})  |  "
                    f"Pearson: {m['pearson']:.4f}  "
                    f"Spearman: {m['spearman']:.4f}  "
                    f"MAE: {m['mae']:.4f}  "
                    f"Loop F1: {m['loop_metrics']['f1']:.4f}"
                )

            # Save side-by-side Actual vs Predicted maps when requested.
            if save_plots and compare_maps and idx < compare_max_samples:
                plot_cond_indices = [
                    cond_idx for cond_idx in self.eval_condition_indices
                    if 0 <= cond_idx < pred_maps.shape[0]
                ]
                for cond_idx in plot_cond_indices[:compare_n_conditions]:
                    out_path = self._plot_comparison(
                        pred_maps[cond_idx],
                        target[cond_idx],
                        source_sample_idx,
                        cond_idx,
                    )
                    if out_path is not None:
                        saved_map_paths.append(str(out_path))

        # Aggregate statistics
        print(f"\n{'=' * 60}")
        print("  Aggregate Statistics")
        print(f"{'=' * 60}")
        if not all_results:
            print("No samples were evaluated.")
            return

        agg_keys = all_results[0]['aggregate'].keys()
        summary = {}
        for key in agg_keys:
            vals = [r['aggregate'][key] for r in all_results]
            summary[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }
            print(f"{key:20s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

        # Save results
        results_path = self.out_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(
                {
                    'summary': summary,
                    'per_sample': all_results,
                    'checkpoint': str(self.checkpoint_path),
                    'n_samples': len(all_results),
                    'selected_sample_indices': self.eval_indices,
                    'selected_conditions': [
                        {'index': idx, 'name': name}
                        for idx, name in zip(self.eval_condition_indices, self.eval_condition_labels)
                    ],
                    'saved_map_images': saved_map_paths,
                },
                f,
                indent=2,
            )
        print(f"\n✓ Results saved to {results_path}")
        if saved_map_paths:
            print(f"✓ Saved {len(saved_map_paths)} map image(s) in {self.out_dir}")

        # Summary plot
        if save_plots:
            self._plot_summary(all_results)
            print(f"✓ Plots saved to {self.out_dir}")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test HiC-Verse M2++ checkpoint'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint (.pt file)',
    )
    parser.add_argument(
        '--mode',
        default='real',
        choices=['synthetic', 'real'],
        help='Dataset mode',
    )
    parser.add_argument(
        '--chroms',
        nargs='+',
        default=None,
        help='Chromosomes to test (real mode only)',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Total number of windows to test (overrides --samples_per_condition)',
    )
    parser.add_argument(
        '--samples_per_condition',
        type=int,
        default=10,
        help='How many windows to test per selected condition (default: 10)',
    )
    parser.add_argument(
        '--conditions',
        nargs='+',
        default=None,
        help='Condition indices or names to evaluate (e.g. 0 2 cond03). Default: all',
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=None,
        help='Specific sample index to test (overrides --n_samples/--samples_per_condition)',
    )
    parser.add_argument(
        '--out_dir',
        default='./test_results_m2pp',
        help='Output directory',
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip plot generation',
    )
    parser.add_argument(
        '--compare_maps',
        action='store_true',
        help='Save side-by-side Actual vs Predicted contact-map visualizations',
    )
    parser.add_argument(
        '--compare_max_samples',
        type=int,
        default=10,
        help='Maximum number of samples for contact-map comparisons',
    )
    parser.add_argument(
        '--compare_n_conditions',
        type=int,
        default=2,
        help='Number of conditions to visualize per sample for comparisons',
    )
    parser.add_argument(
        '--force_cpu',
        action='store_true',
        help='Force CPU inference (useful when CUDA/cuDNN runtime is mismatched)',
    )
    parser.add_argument(
        '--require_gpu',
        action='store_true',
        help='Fail immediately if CUDA is unavailable or unusable',
    )
    parser.add_argument(
        '--allow_cpu_fallback',
        action='store_true',
        help='Allow fallback to CPU if CUDA/cuDNN runtime mismatch occurs',
    )
    parser.add_argument(
        '--no_align_conditions',
        action='store_true',
        help='Do not auto-align dataset conditions to checkpoint n_conditions',
    )
    parser.add_argument(
        '--non_strict_ckpt_load',
        action='store_true',
        help='Allow partial shape-matched checkpoint loading when strict load fails',
    )
    parser.add_argument(
        '--no_auto_cudnn_retry',
        action='store_true',
        help='Disable automatic one-time relaunch with cleared LD_LIBRARY_PATH',
    )

    args = parser.parse_args()
    if args.n_samples is not None and args.n_samples <= 0:
        parser.error("--n_samples must be > 0 when provided.")
    if args.samples_per_condition <= 0:
        parser.error("--samples_per_condition must be > 0.")
    if args.sample_idx is not None and args.sample_idx < 0:
        parser.error("--sample_idx must be >= 0.")
    if args.compare_max_samples <= 0:
        parser.error("--compare_max_samples must be > 0.")
    if args.compare_n_conditions <= 0:
        parser.error("--compare_n_conditions must be > 0.")

    tester = TesterM2PP(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        chroms=args.chroms,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        samples_per_condition=args.samples_per_condition,
        condition_tokens=args.conditions,
        sample_idx=args.sample_idx,
        force_cpu=args.force_cpu,
        require_gpu=args.require_gpu,
        allow_cpu_fallback=args.allow_cpu_fallback,
        align_conditions=not args.no_align_conditions,
        strict_ckpt_load=not args.non_strict_ckpt_load,
        auto_fix_cudnn_path=not args.no_auto_cudnn_retry,
    )
    tester.run(
        save_plots=not args.no_plots,
        compare_maps=args.compare_maps,
        compare_max_samples=args.compare_max_samples,
        compare_n_conditions=args.compare_n_conditions,
    )
