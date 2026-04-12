"""
HiC-Verse M2++ Configuration
================================
Enhanced configuration for M2++ model with:
  - SS2D parameters
  - LEFN parameters  
  - Auxiliary feature settings
  - L1 loss weights
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional
import warnings
import torch


def _resolve_chunk_size(n_bins: int, preferred: int) -> int:
    preferred = max(1, int(preferred))
    candidate = 1
    while candidate * 2 <= preferred:
        candidate *= 2
    if n_bins >= 16:
        candidate = max(candidate, 16)
        while candidate > preferred:
            candidate //= 2
        return max(candidate, 16)
    return max(1, candidate)


def _resolve_padded_n_bins(n_bins: int, chunk_size: int) -> int:
    n_bins = int(n_bins)
    chunk_size = max(1, int(chunk_size))
    return ((n_bins + chunk_size - 1) // chunk_size) * chunk_size


@dataclass
class HiCVerseConfigM2PP:
    """
    M2++ configuration with enhanced parameters.
    """

    # ─────────────────────────────────────────────
    # Genomic / data parameters
    # ─────────────────────────────────────────────
    n_bins: int = 200
    bin_size: int = 10_000
    n_conditions: int = 6
    vocab_size: int = 5
    test_bin_size: int = 200
    padded_n_bins: int = field(init=False)

    # ─────────────────────────────────────────────
    # Paths
    # ─────────────────────────────────────────────
    x_dir: str = "./matrices/X"
    y_dir: str = "./matrices/Y"
    genome_fasta_path: Optional[str] = None
    out_dir: str = "./hicverse_m2pp_output"

    # ─────────────────────────────────────────────
    # DNA encoder
    # ─────────────────────────────────────────────
    dna_embed_dim: int = 128
    dna_cnn_channels: List[int] = field(
        default_factory=lambda: [128, 256, 256]
    )
    dna_cnn_kernels: List[int] = field(
        default_factory=lambda: [7, 5, 3]
    )

    # ─────────────────────────────────────────────
    # Core model dimensions
    # ─────────────────────────────────────────────
    d_model: int = 256
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1

    # ─────────────────────────────────────────────
    # SSM / Mamba parameters
    # ─────────────────────────────────────────────
    prefer_mamba2: bool = True
    allow_mamba3_fallback: bool = False
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = False
    mamba3_bidirectional: bool = True

    # ─────────────────────────────────────────────
    # Hybrid stack
    # ─────────────────────────────────────────────
    n_mamba_per_attn: int = 5
    n_hybrid_stacks: int = 4
    n_attn_heads: int = 8
    attn_dropout: float = 0.1

    # ─────────────────────────────────────────────
    # RNA encoder
    # ─────────────────────────────────────────────
    rna_cnn_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    rna_dilations: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16]
    )

    # ─────────────────────────────────────────────
    # Outer product
    # ─────────────────────────────────────────────
    d_outer: int = 64

    # ─────────────────────────────────────────────
    # NEW: SS2D parameters
    # ─────────────────────────────────────────────
    ss2d_d_state: int = 64           # state size for row/col scans
    ss2d_merge_method: str = "conv"  # 'conv' or 'learned_weights'

    # ─────────────────────────────────────────────
    # NEW: LEFN parameters
    # ─────────────────────────────────────────────
    lefn_expansion: int = 4          # hidden expansion factor
    n_hic_blocks: int = 3            # stacked SS2D+LEFN refinement blocks

    # ─────────────────────────────────────────────
    # NEW: Auxiliary features
    # ─────────────────────────────────────────────
    use_aux_features: bool = True    # enable distance/TAD/CTCF features
    aux_embed_dim: int = 32          # embedding dim for aux features
    aux_feature_dropout_prob: float = 0.5   # drop aux priors on random samples
    use_ctcf_fimo_prior: bool = True
    ctcf_fimo_tsv_path: Optional[str] = None
    use_tad_priors: bool = True
    tad_priors_dir: Optional[str] = None

    # ─────────────────────────────────────────────
    # Loop detection head
    # ─────────────────────────────────────────────
    enable_loop_head: bool = False
    loop_head_dim: int = 64

    # ─────────────────────────────────────────────
    # Regularization
    # ─────────────────────────────────────────────
    dropout: float = 0.1

    # ─────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────
    normalize_rna_inputs: bool = True
    normalize_hic_targets: bool = True

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 2
    max_epochs: int = 100
    grad_clip: float = 1.0
    warmup_steps: int = 500
    val_every_n_epochs: int = 5
    scheduler_type: str = "onecycle"  # onecycle or warmup_cosine
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 10.0
    onecycle_final_div_factor: float = 1_000.0

    # ─────────────────────────────────────────────
    # NEW: L1 loss weights (replaces MSE)
    # ─────────────────────────────────────────────
    loss_map_weight: float = 10.0     # L1 weight
    loss_pearson_weight: float = 1.0
    loss_loop_weight: float = 0.5

    # ─────────────────────────────────────────────
    # Device / precision
    # ─────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    # ─────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "./checkpoints_m2pp"

    # ─────────────────────────────────────────────
    def validate(self) -> None:
        resolved_chunk = _resolve_chunk_size(self.n_bins, self.chunk_size)
        if resolved_chunk != self.chunk_size:
            warnings.warn(
                f"chunk_size adjusted from {self.chunk_size} to {resolved_chunk}"
            )
            self.chunk_size = resolved_chunk
        self.padded_n_bins = _resolve_padded_n_bins(self.n_bins, self.chunk_size)

        d_inner = self.d_model * self.expand
        assert d_inner % self.headdim == 0, (
            f"d_model={self.d_model} × expand={self.expand} = {d_inner} "
            f"must be divisible by headdim={self.headdim}"
        )
        assert self.rope_fraction in (0.5, 1.0), (
            f"rope_fraction must be 0.5 or 1.0, got {self.rope_fraction}"
        )
        split = int(self.d_state * self.rope_fraction)
        if split % 2 != 0:
            split -= 1
        assert split // 2 > 0, (
            f"d_state={self.d_state} × rope_fraction={self.rope_fraction} "
            f"→ num_rope_angles={split // 2}. Increase d_state."
        )
        assert self.d_model % self.n_attn_heads == 0, (
            f"d_model={self.d_model} must be divisible by "
            f"n_attn_heads={self.n_attn_heads}"
        )
        assert self.ss2d_d_state > 0, "ss2d_d_state must be > 0"
        assert self.lefn_expansion > 0, "lefn_expansion must be > 0"
        assert self.n_hic_blocks > 0, "n_hic_blocks must be > 0"
        if self.use_aux_features:
            assert self.aux_embed_dim > 0, "aux_embed_dim must be > 0"
            assert 0.0 <= self.aux_feature_dropout_prob < 1.0, (
                "aux_feature_dropout_prob must be in [0.0, 1.0)"
            )
            if self.use_ctcf_fimo_prior and self.ctcf_fimo_tsv_path is not None:
                if not Path(self.ctcf_fimo_tsv_path).exists():
                    warnings.warn(
                        f"ctcf_fimo_tsv_path does not exist: {self.ctcf_fimo_tsv_path}. "
                        "CTCF priors will fall back to zeros."
                    )
            if self.use_tad_priors and self.tad_priors_dir is not None:
                if not Path(self.tad_priors_dir).exists():
                    warnings.warn(
                        f"tad_priors_dir does not exist: {self.tad_priors_dir}. "
                        "TAD priors will fall back to zeros."
                    )
        assert self.scheduler_type in ("onecycle", "warmup_cosine"), (
            f"scheduler_type must be 'onecycle' or 'warmup_cosine', got {self.scheduler_type!r}"
        )
        assert 0.0 < self.onecycle_pct_start < 1.0, (
            "onecycle_pct_start must be in (0.0, 1.0)"
        )
        assert self.onecycle_div_factor > 1.0, "onecycle_div_factor must be > 1.0"
        assert self.onecycle_final_div_factor > 1.0, "onecycle_final_div_factor must be > 1.0"
        assert len(self.dna_cnn_channels) == len(self.dna_cnn_kernels)
        assert self.dna_cnn_channels[-1] == self.d_model
        assert self.rna_cnn_channels[-1] == self.d_model

    def __post_init__(self):
        if self.genome_fasta_path is None:
            env_fasta = os.environ.get("HICVERSE_GENOME_FASTA")
            candidates = []
            if env_fasta:
                candidates.append(Path(env_fasta))
            candidates.extend((
                Path("./reference/genome.fa"),
                Path("./reference/genome.fasta"),
                Path("./genome.fa"),
                Path("./genome.fasta"),
            ))
            for candidate in candidates:
                if candidate.exists():
                    self.genome_fasta_path = str(candidate.resolve())
                    break

        if self.is_mimo and self.chunk_size == 64:
            self.chunk_size = max(1, 64 // self.mimo_rank)
        self.chunk_size = _resolve_chunk_size(self.n_bins, self.chunk_size)
        self.padded_n_bins = _resolve_padded_n_bins(self.n_bins, self.chunk_size)

        if self.ctcf_fimo_tsv_path is None:
            local_fimo = Path("fimo_ctcf_output/fimo.tsv")
            if local_fimo.exists():
                self.ctcf_fimo_tsv_path = str(local_fimo.resolve())
        if self.tad_priors_dir is None:
            local_tad = Path("TAD_Priors")
            if local_tad.exists() and local_tad.is_dir():
                self.tad_priors_dir = str(local_tad.resolve())

        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.genome_fasta_path is None:
            warnings.warn(
                "No genome FASTA resolved. Real-data mode will raise error."
            )


# ─────────────────────────────────────────────────────────────────
#  Preset configs
# ─────────────────────────────────────────────────────────────────

def get_test_config_m2pp(n_conditions: int = 6) -> HiCVerseConfigM2PP:
    """Test config: medium-size model with stronger structural capacity."""
    cfg = HiCVerseConfigM2PP(
        d_model=256,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        chunk_size=64,
        is_mimo=False,
        n_attn_heads=8,
        dna_embed_dim=64,
        dna_cnn_channels=[64, 128, 256],
        dna_cnn_kernels=[7, 5, 3],
        rna_cnn_channels=[32, 64, 128, 256],
        rna_dilations=[1, 2, 4, 8],
        d_outer=64,
        ss2d_d_state=32,
        lefn_expansion=4,
        n_hic_blocks=3,
        aux_feature_dropout_prob=0.5,
        enable_loop_head=False,
        loop_head_dim=32,
        n_mamba_per_attn=3,
        n_hybrid_stacks=2,
        mamba3_bidirectional=False,
        n_conditions=n_conditions,
        test_bin_size=200,
        batch_size=2,
        learning_rate=1e-3,
        max_epochs=5,
        mixed_precision=False,
        scheduler_type="onecycle",
    )
    cfg.dna_cnn_channels[-1] = cfg.d_model
    cfg.rna_cnn_channels[-1] = cfg.d_model
    cfg.validate()
    return cfg


def get_full_config_m2pp(n_conditions: int = 6) -> HiCVerseConfigM2PP:
    """Full M2++ config with all enhancements."""
    cfg = HiCVerseConfigM2PP(
        n_conditions=n_conditions,
        is_mimo=False,
        enable_loop_head=False,
        use_aux_features=True,
        learning_rate=1e-4,
        mixed_precision=True,
        lefn_expansion=4,
        n_hic_blocks=3,
        aux_feature_dropout_prob=0.5,
        scheduler_type="onecycle",
    )
    cfg.validate()
    return cfg
