"""
HiC-Verse M2++ Model — Enhanced Mamba-2/3 + Transformer Architecture
========================================================================

Major enhancements over v2:
  1. SS2D (2D Selective Scan) — replaces Symmetric2DCNN
  2. LEFN (Locally Enhanced Feedforward) — restores local detail
  3. Auxiliary biological features — TAD, distance, CTCF orientation
  4. L1 loss — sharper Hi-C maps vs MSE

Architecture flow:
  DNA → CNN → Hybrid Mamba-Transformer → DNA features
  RNA → Dilated CNN → RNA features
  DNA ⊙ RNA gate → Fused features
  Outer Product → 2D tensor
  + Auxiliary features (distance, TAD, CTCF)
  → Holistic Scan Block (SS2D + LEFN)
  → Symmetric Hi-C map + Loop logits
"""

from __future__ import annotations
import math
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═════════════════════════════════════════════════════════════════
#  Mamba import chain (Mamba-2 preferred)
# ═════════════════════════════════════════════════════════════════

MAMBA3_OK = False
MAMBA2_OK = False
MAMBA1_OK = False
_MAMBA3_CLASS = None
_MAMBA2_CLASS = None
_MAMBA1_CLASS = None
_MAMBA3_NAME = "unavailable"
_MAMBA2_NAME = "unavailable"
_SSM_NAME = "GRU fallback"

try:
    from mamba_ssm import Mamba2 as _M2
    _MAMBA2_CLASS = _M2
    MAMBA2_OK = True
    _MAMBA2_NAME = "Mamba2 (mamba_ssm package)"
except ImportError:
    try:
        from mamba_ssm.modules.mamba2 import Mamba2 as _M2
        _MAMBA2_CLASS = _M2
        MAMBA2_OK = True
        _MAMBA2_NAME = "Mamba2 (mamba_ssm.modules.mamba2)"
    except ImportError:
        pass

try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _M3
    _MAMBA3_CLASS = _M3
    MAMBA3_OK = True
    _MAMBA3_NAME = "Mamba3 (mamba_ssm package)"
except ImportError:
    try:
        from mamba3 import Mamba3 as _M3
        _MAMBA3_CLASS = _M3
        MAMBA3_OK = True
        _MAMBA3_NAME = "Mamba3 (local mamba3.py)"
    except ImportError:
        pass

try:
    from mamba_ssm import Mamba as _M1
    _MAMBA1_CLASS = _M1
    MAMBA1_OK = True
except ImportError:
    pass

if MAMBA2_OK:
    print(f"[HiCVerse M2++] ✓ {_MAMBA2_NAME} (preferred backend)")
elif MAMBA3_OK:
    print(f"[HiCVerse M2++] ⚠  {_MAMBA3_NAME} available; Mamba2 not found")
elif MAMBA1_OK:
    print("[HiCVerse M2++] ⚠  Mamba1 available; Mamba2/Mamba3 not found")
else:
    print("[HiCVerse M2++] ⚠  No mamba_ssm found — using GRU fallback")


# ═════════════════════════════════════════════════════════════════
#  GRU FALLBACK (same as v2)
# ═════════════════════════════════════════════════════════════════

class GRUFallbackSSM(nn.Module):
    def __init__(self, d_model: int, **_kwargs):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.norm(self.proj(out))


# ═════════════════════════════════════════════════════════════════
#  SSM FACTORY (same as v2)
# ═════════════════════════════════════════════════════════════════

def _find_valid_headdim(d_inner: int) -> int:
    for hd in [128, 64, 32, 16, 8]:
        if d_inner % hd == 0:
            return hd
    return 1


def _make_ssm(
    d_model: int,
    d_state: int,
    expand: int,
    headdim: int,
    ngroups: int,
    rope_fraction: float = 0.5,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init_floor: float = 1e-4,
    A_floor: float = 1e-4,
    chunk_size: int = 64,
    is_mimo: bool = False,
    mimo_rank: int = 4,
    is_outproj_norm: bool = False,
    layer_idx: Optional[int] = None,
    prefer_mamba2: bool = True,
    allow_mamba3: bool = False,
) -> nn.Module:
    global _SSM_NAME
    d_inner = d_model * expand
    if d_inner % headdim != 0:
        headdim = _find_valid_headdim(d_inner)
        warnings.warn(
            f"headdim adjusted to {headdim} so that d_inner={d_inner} is divisible"
        )

    def _try_mamba2() -> Optional[nn.Module]:
        global _SSM_NAME
        if not MAMBA2_OK or _MAMBA2_CLASS is None:
            return None
        try:
            kw2 = dict(
                d_model=d_model, d_state=d_state,
                d_conv=4, expand=expand,
                headdim=headdim, ngroups=ngroups
            )
            if layer_idx is not None:
                kw2["layer_idx"] = layer_idx
            _SSM_NAME = _MAMBA2_NAME
            return _MAMBA2_CLASS(**kw2)
        except Exception as exc2:
            warnings.warn(f"Mamba-2 init failed: {exc2}.")
            return None

    def _try_mamba3() -> Optional[nn.Module]:
        global _SSM_NAME
        if not allow_mamba3 or not MAMBA3_OK or _MAMBA3_CLASS is None:
            return None
        try:
            kw = dict(
                d_model=d_model, d_state=d_state, expand=expand,
                headdim=headdim, ngroups=ngroups,
                rope_fraction=rope_fraction, dt_min=dt_min, dt_max=dt_max,
                dt_init_floor=dt_init_floor, A_floor=A_floor,
                chunk_size=chunk_size, is_mimo=is_mimo,
                is_outproj_norm=is_outproj_norm,
            )
            if is_mimo:
                kw["mimo_rank"] = mimo_rank
            if layer_idx is not None:
                kw["layer_idx"] = layer_idx
            _SSM_NAME = _MAMBA3_NAME
            return _MAMBA3_CLASS(**kw)
        except Exception as exc:
            warnings.warn(f"Mamba-3 init failed: {exc}.")
            return None

    if prefer_mamba2:
        ssm = _try_mamba2() or _try_mamba3()
    else:
        ssm = _try_mamba3() or _try_mamba2()
    if ssm is not None:
        return ssm

    if MAMBA1_OK:
        try:
            _SSM_NAME = "Mamba1 (mamba_ssm)"
            return _MAMBA1_CLASS(d_model=d_model, d_state=d_state,
                                 d_conv=4, expand=expand)
        except Exception as exc3:
            warnings.warn(f"Mamba-1 init failed: {exc3}. Using GRU.")

    _SSM_NAME = "GRU fallback"
    return GRUFallbackSSM(d_model=d_model)


# ═════════════════════════════════════════════════════════════════
#  MAMBA-3 BLOCK (same as v2)
# ═════════════════════════════════════════════════════════════════

class Mamba3Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        chunk_size: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        is_outproj_norm: bool = False,
        bidirectional: bool = True,
        dropout: float = 0.1,
        layer_idx: Optional[int] = None,
        prefer_mamba2: bool = True,
        allow_mamba3: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        _ssm_kw = dict(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, ngroups=ngroups,
            rope_fraction=rope_fraction, dt_min=dt_min, dt_max=dt_max,
            dt_init_floor=dt_init_floor, A_floor=A_floor,
            chunk_size=chunk_size, is_mimo=is_mimo, mimo_rank=mimo_rank,
            is_outproj_norm=is_outproj_norm,
            prefer_mamba2=prefer_mamba2, allow_mamba3=allow_mamba3,
        )
        self.ssm_fwd = _make_ssm(**_ssm_kw, layer_idx=layer_idx)
        if bidirectional:
            bwd_idx = None if layer_idx is None else layer_idx + 1000
            self.ssm_bwd = _make_ssm(**_ssm_kw, layer_idx=bwd_idx)
            self.bidi_mix = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        y_fwd = self.ssm_fwd(x)
        if self.bidirectional:
            y_bwd = self.ssm_bwd(x.flip(1)).flip(1)
            y = self.bidi_mix(torch.cat([y_fwd, y_bwd], dim=-1))
        else:
            y = y_fwd
        return residual + self.dropout(y)


# ═════════════════════════════════════════════════════════════════
#  SELF-ATTENTION BLOCK (same as v2)
# ═════════════════════════════════════════════════════════════════

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        _ff = d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, _ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(_ff, d_model), nn.Dropout(dropout),
        )
        self.rel_bias = nn.Embedding(max_len, n_heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.n_heads = n_heads

    def _make_rel_bias(self, L: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(L, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().clamp(max=L - 1)
        bias = self.rel_bias(dist)
        return bias.permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        rb = self._make_rel_bias(L, x.device)
        rb = rb.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.n_heads, L, L)
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2, attn_mask=rb, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ═════════════════════════════════════════════════════════════════
#  HYBRID STACK (same as v2)
# ═════════════════════════════════════════════════════════════════

class HybridMamba3AttentionStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        blocks: List[nn.Module] = []
        for stack_i in range(cfg.n_hybrid_stacks):
            base_idx = stack_i * (cfg.n_mamba_per_attn + 1)
            for m_i in range(cfg.n_mamba_per_attn):
                blocks.append(Mamba3Block(
                    d_model=cfg.d_model, d_state=cfg.d_state,
                    expand=cfg.expand, headdim=cfg.headdim,
                    ngroups=cfg.ngroups,
                    rope_fraction=cfg.rope_fraction,
                    dt_min=cfg.dt_min, dt_max=cfg.dt_max,
                    dt_init_floor=cfg.dt_init_floor,
                    A_floor=cfg.A_floor, chunk_size=cfg.chunk_size,
                    is_mimo=cfg.is_mimo, mimo_rank=cfg.mimo_rank,
                    is_outproj_norm=cfg.is_outproj_norm,
                    bidirectional=cfg.mamba3_bidirectional,
                    dropout=cfg.dropout,
                    layer_idx=base_idx + m_i,
                    prefer_mamba2=cfg.prefer_mamba2,
                    allow_mamba3=cfg.allow_mamba3_fallback,
                ))
            blocks.append(SelfAttentionBlock(
                d_model=cfg.d_model, n_heads=cfg.n_attn_heads,
                dropout=cfg.attn_dropout,
                max_len=cfg.padded_n_bins + 16,
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ═════════════════════════════════════════════════════════════════
#  DNA ENCODER (same as v2)
# ═════════════════════════════════════════════════════════════════

class NucleotideCNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dna_embed_dim,
                                  padding_idx=4)
        layers: List[nn.Module] = []
        in_ch = cfg.dna_embed_dim
        for out_ch, k in zip(cfg.dna_cnn_channels, cfg.dna_cnn_kernels):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch), nn.GELU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.proj = (nn.Identity() if in_ch == cfg.d_model
                     else nn.Linear(in_ch, cfg.d_model, bias=False))
        self.n_bins = cfg.n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).transpose(1, 2)
        h = self.cnn(h)
        B, C, L = h.shape
        n = self.n_bins
        bin_len = L // n
        if bin_len == 0:
            h = F.adaptive_avg_pool1d(h, n)
        else:
            h = h[:, :, :bin_len * n]
            h = h.reshape(B, C, n, bin_len).mean(-1)
        h = h.transpose(1, 2)
        return self.proj(h)


class DNAEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cnn_encoder = NucleotideCNNEncoder(cfg)
        self.hybrid_stack = HybridMamba3AttentionStack(cfg)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.target_bins = cfg.padded_n_bins

    @staticmethod
    def _pad_tokens(x: torch.Tensor, target_len: int) -> torch.Tensor:
        pad = target_len - x.shape[1]
        if pad <= 0:
            return x
        return F.pad(x, (0, 0, 0, pad))

    def forward(self, dna_seq: torch.Tensor) -> torch.Tensor:
        h = self.cnn_encoder(dna_seq)
        h = self._pad_tokens(h, self.target_bins)
        h = self.hybrid_stack(h)
        return self.out_norm(h)


# ═════════════════════════════════════════════════════════════════
#  RNA ENCODER (same as v2)
# ═════════════════════════════════════════════════════════════════

class DilatedRNAEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_ch = 1
        layers: List[nn.Module] = []
        for out_ch, dil in zip(cfg.rna_cnn_channels, cfg.rna_dilations):
            pad = dil * (3 - 1) // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3,
                          padding=pad, dilation=dil, bias=False),
                nn.BatchNorm1d(out_ch), nn.GELU(),
            ]
            in_ch = out_ch
        for dil in cfg.rna_dilations[len(cfg.rna_cnn_channels):]:
            pad = dil * (3 - 1) // 2
            layers += [
                nn.Conv1d(in_ch, in_ch, kernel_size=3,
                          padding=pad, dilation=dil, bias=False),
                nn.BatchNorm1d(in_ch), nn.GELU(),
            ]
        self.dilated_cnn = nn.Sequential(*layers)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(1),
            nn.Linear(in_ch, in_ch, bias=False), nn.GELU(),
        )
        self.gate = nn.Sigmoid()
        self.d_model = cfg.d_model

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        B, N, L = rna.shape
        x = rna.reshape(B * N, 1, L)
        h = self.dilated_cnn(x)
        ctx = self.context(h)
        h = h * self.gate(ctx).unsqueeze(-1)
        h = h.transpose(1, 2)
        return h.reshape(B, N, L, self.d_model)


# ═════════════════════════════════════════════════════════════════
#  FEATURE GATING (same as v2)
# ═════════════════════════════════════════════════════════════════

class FeatureGating(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dna_feat: torch.Tensor,
                rna_feat: torch.Tensor) -> torch.Tensor:
        B, N, L, D = rna_feat.shape
        dna_flat = dna_feat.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, L, D)
        rna_flat = rna_feat.reshape(B * N, L, D)
        gate = torch.sigmoid(self.gate_proj(rna_flat))
        dna_gated = self.value_proj(dna_flat) * gate
        out = self.out_proj(torch.cat([dna_gated, rna_flat], dim=-1))
        return self.norm(out).reshape(B, N, L, D)


# ═════════════════════════════════════════════════════════════════
#  OUTER PRODUCT PROJECTION (same as v2 — KEPT, critical component)
# ═════════════════════════════════════════════════════════════════

class OuterProductProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_outer = cfg.d_outer
        self.proj = nn.Linear(cfg.d_model, cfg.d_outer, bias=False)
        self.d_2d = cfg.d_outer * 2
        self.mix = nn.Conv2d(self.d_2d, self.d_2d, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, self.d_2d),
                                 num_channels=self.d_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, N, L, _ = x.shape
        p = self.proj(x)
        p_f = p.reshape(B * N, L, self.d_outer)
        fi = p_f.unsqueeze(2).expand(-1, -1, L, -1)
        fj = p_f.unsqueeze(1).expand(-1, L, -1, -1)
        z = torch.cat([fi * fj, (fi + fj) * 0.5], dim=-1)
        z = z.permute(0, 3, 1, 2)
        z = F.gelu(self.norm(self.mix(z)))
        return z.reshape(B, N, self.d_2d, L, L)


# ═════════════════════════════════════════════════════════════════
#  NEW: AUXILIARY BIOLOGICAL FEATURES
#  Decision: Inject biological priors (distance decay, TAD boundaries,
#  CTCF orientation) as explicit features. These are simple to compute
#  and provide strong inductive biases.
# ═════════════════════════════════════════════════════════════════

class AuxiliaryBiologicalFeatures(nn.Module):
    """
    Generates auxiliary feature maps:
      - distance prior: log₂(|i-j|+1), normalized
      - 4-channel TAD priors from precomputed matrices

    Output: (B, n_aux_features, L, L) to concatenate with z
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_bins = cfg.n_bins
        self.out_channels = max(1, int(cfg.aux_embed_dim))
        # 1 distance channel + 4 TAD channels
        self.aux_input_channels = 5
        hidden = max(16, self.out_channels)
        self.aux_embed = nn.Sequential(
            nn.Conv2d(self.aux_input_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
        )

    def _compute_distance_map(self, L: int, device: torch.device) -> torch.Tensor:
        """(L, L) distance map: log₂(|i-j|+1)"""
        idx = torch.arange(L, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()
        return torch.log2(dist + 1.0)

    @staticmethod
    def _zero_tad_priors(B: int, L: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, 4, L, L, device=device)

    def forward(
        self,
        B: int,
        L: int,
        device: torch.device,
        tad_priors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: (B, aux_embed_dim, L, L) auxiliary feature tensor
        """
        param_device = next(self.aux_embed.parameters()).device
        compute_device = param_device

        dist_map = self._compute_distance_map(L, compute_device)
        dist_batch = dist_map.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)  # (B,1,L,L)

        if tad_priors is None:
            tad_batch = self._zero_tad_priors(B, L, compute_device)
        else:
            tad_batch = tad_priors.to(compute_device).float()
            if tad_batch.dim() == 3 and tad_batch.shape[0] == B:
                tad_batch = tad_batch.unsqueeze(1).expand(-1, 4, -1, -1)
            elif tad_batch.dim() == 3 and tad_batch.shape[0] == 4:
                tad_batch = tad_batch.unsqueeze(0).expand(B, -1, -1, -1)
            elif tad_batch.dim() == 4:
                pass
            else:
                raise ValueError(
                    f"tad_priors has unsupported shape {tuple(tad_batch.shape)}"
                )
            if tad_batch.shape[0] != B:
                raise ValueError(
                    f"tad_priors batch mismatch: expected B={B}, got {tad_batch.shape[0]}"
                )
            if tad_batch.shape[1] != 4:
                raise ValueError(
                    f"tad_priors channel mismatch: expected 4, got {tad_batch.shape[1]}"
                )
            if tad_batch.shape[-2:] != (L, L):
                raise ValueError(
                    f"tad_priors spatial mismatch: expected {(L, L)}, got {tuple(tad_batch.shape[-2:])}"
                )
            tad_batch = torch.nan_to_num(tad_batch, nan=0.0, posinf=0.0, neginf=0.0)
            tad_batch = tad_batch.clamp(0.0, 1.0)

        # Normalize distance to ~[0,1]
        denom = torch.log2(
            torch.tensor(float(L), device=compute_device, dtype=dist_batch.dtype)
        ) + 1e-8
        dist_batch = dist_batch / denom

        # Concatenate distance + 4 TAD priors -> 5 channels.
        aux_stack = torch.cat([dist_batch, tad_batch], dim=1)  # (B, 5, L, L)
        out = self.aux_embed(aux_stack)  # (B, d_aux, L, L)
        if out.device != device:
            out = out.to(device)
        return out


# ═════════════════════════════════════════════════════════════════
#  NEW: SS2D (2D SELECTIVE SCAN)
#  Decision: Replaces stacked 2D CNNs with a global-receptive-field
#  state-space scan. Four directional scans (TL→BR, TR→BL, BL→TR, BR→TL)
#  are merged to capture long-range dependencies in 2D Hi-C maps.
#
#  Implementation: For efficiency, we use a separable approximation:
#    - Row-wise 1D scan (left→right, right→left)
#    - Col-wise 1D scan (top→bottom, bottom→top)
#  This is O(L²) instead of full 2D traversal.
# ═════════════════════════════════════════════════════════════════

class SS2D(nn.Module):
    """
    2D Selective Scan via separable row/col scans.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)

    Four directions:
      1. Row L→R, then Col T→B
      2. Row R→L, then Col T→B
      3. Row L→R, then Col B→T
      4. Row R→L, then Col B→T
    Outputs are merged via cross-merge (learned weighted sum).
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        # Lightweight SSM for row and col scans
        # We use simple GRU for stability (Mamba-3 can replace if needed)
        self.row_scan_lr = nn.GRU(d_model, d_model, batch_first=True)
        self.row_scan_rl = nn.GRU(d_model, d_model, batch_first=True)
        self.col_scan_tb = nn.GRU(d_model, d_model, batch_first=True)
        self.col_scan_bt = nn.GRU(d_model, d_model, batch_first=True)

        # Cross-merge: learn to combine 4 directions
        self.merge = nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False)

    def _scan_rows(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """Scan each row independently. x: (B, C, H, W)"""
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        if reverse:
            x_flat = x_flat.flip(1)
        out, _ = self.row_scan_rl(x_flat) if reverse else self.row_scan_lr(x_flat)
        if reverse:
            out = out.flip(1)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def _scan_cols(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """Scan each column independently. x: (B, C, H, W)"""
        B, C, H, W = x.shape
        x_flat = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        if reverse:
            x_flat = x_flat.flip(1)
        out, _ = self.col_scan_bt(x_flat) if reverse else self.col_scan_tb(x_flat)
        if reverse:
            out = out.flip(1)
        return out.reshape(B, W, H, C).permute(0, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Four scan directions
        s1 = self._scan_cols(self._scan_rows(x, reverse=False), reverse=False)  # L→R, T→B
        s2 = self._scan_cols(self._scan_rows(x, reverse=True), reverse=False)   # R→L, T→B
        s3 = self._scan_cols(self._scan_rows(x, reverse=False), reverse=True)   # L→R, B→T
        s4 = self._scan_cols(self._scan_rows(x, reverse=True), reverse=True)    # R→L, B→T

        # Merge
        merged = torch.cat([s1, s2, s3, s4], dim=1)
        return self.merge(merged)


# ═════════════════════════════════════════════════════════════════
#  NEW: LEFN (LOCALLY ENHANCED FEEDFORWARD NETWORK)
#  Decision: After global SS2D, restore local detail with a small
#  CNN. This is critical for sharp loop anchors and TAD boundaries.
#
#  Structure: 1×1 Conv → 3×3 Conv → 1×1 Conv + GELU
#  This is a depthwise-separable bottleneck with local receptive field.
# ═════════════════════════════════════════════════════════════════

class LEFN(nn.Module):
    """
    Locally Enhanced Feedforward Network.
    Restores local spatial detail after global SS2D scan.
    """

    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        hidden = d_model * expansion
        self.conv1 = nn.Conv2d(d_model, hidden, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(hidden, hidden, kernel_size=3,
                                 padding=1, groups=hidden, bias=False)
        self.conv2 = nn.Conv2d(hidden, d_model, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.conv2(x)
        return self.norm(x + residual)


# ═════════════════════════════════════════════════════════════════
#  NEW: HOLISTIC SCAN BLOCK (combines SS2D + LEFN)
#  Decision: Replaces Symmetric2DCNN head. This is the core M2++
#  enhancement inspired by HiCMamba.
#
#  Structure: Input → Norm → SS2D → Norm → LEFN → Output
#  Applied per-condition, then averaged for symmetry.
# ═════════════════════════════════════════════════════════════════

class HolisticScanBlock(nn.Module):
    """
    Holistic Scan Block: SS2D + LEFN.

    Input:  (B, N, C, L, L)  — N conditions, C channels
    Output: (B, N, L, L)     — symmetric Hi-C maps per condition
    """

    class _RefineLayer(nn.Module):
        def __init__(self, d_in: int, ss2d_d_state: int, lefn_expansion: int):
            super().__init__()
            self.norm1 = nn.GroupNorm(num_groups=min(8, d_in), num_channels=d_in)
            self.ss2d = SS2D(d_model=d_in, d_state=ss2d_d_state)
            self.norm2 = nn.GroupNorm(num_groups=min(8, d_in), num_channels=d_in)
            self.lefn = LEFN(d_model=d_in, expansion=lefn_expansion)

        def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
            h = self.norm1(z_flat)
            h = self.ss2d(h) + z_flat  # residual
            h = self.norm2(h)
            h = self.lefn(h)
            return h

    def __init__(self, cfg):
        super().__init__()
        d_in = cfg.d_outer * 2 + (cfg.aux_embed_dim if cfg.use_aux_features else 0)
        self.n_hic_blocks = max(1, int(getattr(cfg, "n_hic_blocks", 1)))
        # Keep first layer names compatible with existing checkpoints.
        self.norm1 = nn.GroupNorm(num_groups=min(8, d_in), num_channels=d_in)
        self.ss2d = SS2D(d_model=d_in, d_state=cfg.ss2d_d_state)
        self.norm2 = nn.GroupNorm(num_groups=min(8, d_in), num_channels=d_in)
        self.lefn = LEFN(d_model=d_in, expansion=cfg.lefn_expansion)
        self.extra_refine_layers = nn.ModuleList([
            HolisticScanBlock._RefineLayer(
                d_in=d_in,
                ss2d_d_state=cfg.ss2d_d_state,
                lefn_expansion=cfg.lefn_expansion,
            )
            for _ in range(max(0, self.n_hic_blocks - 1))
        ])
        self.final_conv = nn.Conv2d(d_in, 1, kernel_size=1)

    @staticmethod
    def _symmetrize(x: torch.Tensor) -> torch.Tensor:
        return (x + x.transpose(-1, -2)) * 0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, C, L, L)
        Returns: (B, N, L, L)
        """
        B, N, C, L1, L2 = z.shape
        z_flat = z.reshape(B * N, C, L1, L2)
        h = self.norm1(z_flat)
        h = self.ss2d(h) + z_flat  # residual
        h = self.norm2(h)
        h = self.lefn(h)
        for layer in self.extra_refine_layers:
            h = layer(h)

        # Project to Hi-C map
        out = self.final_conv(h).squeeze(1)  # (B*N, L, L)
        out = torch.sigmoid(self._symmetrize(out))
        return out.reshape(B, N, L1, L2)


# ═════════════════════════════════════════════════════════════════
#  LOOP DETECTION HEAD (same as v2)
# ═════════════════════════════════════════════════════════════════

class LoopDetectionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid = cfg.loop_head_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.d_outer * 2, hid,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hid), nn.GELU(),
            nn.Conv2d(hid, hid // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hid // 2), nn.GELU(),
            nn.Conv2d(hid // 2, 1, kernel_size=1),
        )

    @staticmethod
    def _symmetrize(x: torch.Tensor) -> torch.Tensor:
        return (x + x.transpose(-1, -2)) * 0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, C, L1, L2 = z.shape
        out = self.cnn(z.mean(dim=1)).squeeze(1)
        return self._symmetrize(out)


# ═════════════════════════════════════════════════════════════════
#  FULL M2++ MODEL
# ═════════════════════════════════════════════════════════════════

class HiCVerseModelM2PP(nn.Module):
    """
    HiC-Verse M2++ — Enhanced Architecture

    Key improvements:
      1. SS2D (2D Selective Scan) for global receptive field
      2. LEFN (Locally Enhanced FFN) for sharp structural detail
      3. Auxiliary biological features (distance, TAD, CTCF)
      4. L1 loss (handled in losses_m2pp.py)

    Inputs:
        dna_seq    : (B, L_dna)        long   — nucleotide indices
        rna_signal : (B, N_cond, L)    float  — RNA-seq coverage

    Outputs:
        contact_maps : (B, N, L, L)   — predicted Hi-C maps
        loop_logits  : (B, L, L)      — optional loop detection
        (+ debug features)
    """

    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.use_aux_features = bool(cfg.use_aux_features)
        self.aux_feature_dropout_prob = float(
            getattr(cfg, "aux_feature_dropout_prob", 0.0)
        )

        self.dna_encoder = DNAEncoder(cfg)
        self.rna_encoder = DilatedRNAEncoder(cfg)
        self.fusion = FeatureGating(d_model=cfg.d_model)
        self.outer_proj = OuterProductProjection(cfg)
        self.aux_features = (AuxiliaryBiologicalFeatures(cfg)
                             if self.use_aux_features else None)
        self.hic_head = HolisticScanBlock(cfg)
        self.loop_head = LoopDetectionHead(cfg) if cfg.enable_loop_head else None

        self.target_bins = cfg.n_bins
        self.padded_bins = cfg.padded_n_bins

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def encode_dna(self, dna_seq: torch.Tensor) -> torch.Tensor:
        return self.dna_encoder(dna_seq)

    def encode_rna(self, rna_signal: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(rna_signal)

    @staticmethod
    def _pad_seq_dim(x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur = x.shape[-2] if x.dim() == 4 else x.shape[1]
        pad = target_len - cur
        if pad <= 0:
            return x
        if x.dim() == 3:
            return F.pad(x, (0, 0, 0, pad))
        if x.dim() == 4:
            return F.pad(x, (0, 0, 0, pad, 0, 0))
        raise ValueError(f"Unsupported tensor rank: {x.dim()}")

    def _crop_seq_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x[:, :self.target_bins]
        if x.dim() == 4:
            return x[:, :, :self.target_bins]
        raise ValueError(f"Unsupported tensor rank: {x.dim()}")

    def _crop_contact_map(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.target_bins, :self.target_bins]

    def forward(
        self,
        dna_seq: torch.Tensor,
        rna_signal: torch.Tensor,
        ctcf_prior_2d: Optional[torch.Tensor] = None,
        tad_priors: Optional[torch.Tensor] = None,
        return_loops: Optional[bool] = None,
    ) -> dict:
        if return_loops is None:
            return_loops = self.loop_head is not None

        # Encode DNA and RNA
        dna_feat = self.encode_dna(dna_seq)  # (B, L, D)
        rna_feat = self._pad_seq_dim(
            self.encode_rna(rna_signal), self.padded_bins
        )  # (B, N, L, D)

        # Fusion
        fused = self.fusion(dna_feat, rna_feat)  # (B, N, L, D)

        # Outer product
        z_outer = self.outer_proj(fused)  # (B, N, d_2d, L, L)
        z = z_outer

        # Auxiliary features
        if self.use_aux_features and self.aux_features is not None:
            B, N, C, L, _ = z.shape
            tad_bn = None
            if tad_priors is not None:
                prior = tad_priors
                if prior.dim() == 4 and prior.shape[:2] == (B, 4):
                    prior = prior.unsqueeze(1).expand(-1, N, -1, -1, -1)
                elif prior.dim() == 5 and prior.shape[:2] == (B, N):
                    pass
                else:
                    raise ValueError(
                        "tad_priors must have shape (B, 4, L, L) or (B, N, 4, L, L), "
                        f"got {tuple(prior.shape)}"
                    )
                if prior.shape[-3] != 4:
                    raise ValueError(
                        f"tad_priors must have 4 channels, got {prior.shape[-3]}"
                    )
                h, w = prior.shape[-2], prior.shape[-1]
                if h != w:
                    raise ValueError(f"tad_priors must be square, got {(h, w)}")
                if h < L:
                    pad = L - h
                    prior = F.pad(prior, (0, pad, 0, pad))
                elif h > L:
                    prior = prior[..., :L, :L]
                tad_bn = prior.reshape(B * N, 4, L, L)

            aux = self.aux_features(
                B=B * N,
                L=L,
                device=z.device,
                tad_priors=tad_bn,
            )  # (B*N, d_aux, L, L)
            aux = aux.reshape(B, N, -1, L, L)
            if self.training and self.aux_feature_dropout_prob > 0.0:
                keep = (
                    torch.rand(B, N, 1, 1, 1, device=z.device)
                    > self.aux_feature_dropout_prob
                ).float()
                aux = aux * keep
            z = torch.cat([z, aux], dim=2)  # (B, N, d_2d+d_aux, L, L)

        # Holistic Scan Block (SS2D + LEFN)
        contact_maps = self._crop_contact_map(self.hic_head(z))

        # Loop head (optional)
        loop_logits = None
        if return_loops and self.loop_head is not None:
            # Use only outer product features for loop detection
            loop_logits = self._crop_contact_map(self.loop_head(z_outer))

        return {
            'contact_maps': contact_maps,
            'loop_logits': loop_logits,
            'dna_feat': self._crop_seq_dim(dna_feat),
            'rna_feat': self._crop_seq_dim(rna_feat),
            'fused': self._crop_seq_dim(fused),
        }

    def predict_contact_map(
        self,
        dna_seq: torch.Tensor,
        rna_signal: torch.Tensor,
        ctcf_prior_2d: Optional[torch.Tensor] = None,
        tad_priors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(dna_seq, rna_signal,
                                ctcf_prior_2d=ctcf_prior_2d,
                                tad_priors=tad_priors,
                                return_loops=False)['contact_maps']

    @staticmethod
    def _select_map_for_plot(
        maps: torch.Tensor | np.ndarray,
        sample_idx: int = 0,
        condition_idx: int = 0,
    ) -> np.ndarray:
        arr = maps.detach().float().cpu().numpy() if isinstance(maps, torch.Tensor) else np.asarray(maps)
        if arr.ndim == 4:
            return arr[sample_idx, condition_idx]
        if arr.ndim == 3:
            if condition_idx < arr.shape[0]:
                return arr[condition_idx]
            return arr[sample_idx]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Unsupported map shape for plotting: {arr.shape}")

    def visualize_contact_maps(
        self,
        pred_maps: torch.Tensor | np.ndarray,
        target_maps: Optional[torch.Tensor | np.ndarray] = None,
        sample_idx: int = 0,
        condition_idx: int = 0,
        out_path: Optional[str] = None,
        title_prefix: str = "",
    ) -> Optional[str]:
        """Save a contact-map visualization. Returns saved path or None."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib is unavailable; skipping visualization.")
            return None

        pred = self._select_map_for_plot(pred_maps, sample_idx, condition_idx)
        tgt = (self._select_map_for_plot(target_maps, sample_idx, condition_idx)
               if target_maps is not None else None)

        pred_vmax = float(np.nanmax(pred)) if np.size(pred) else 1.0
        if tgt is None:
            vmax = max(pred_vmax, 1e-6)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(pred, cmap='Reds', vmin=0.0, vmax=vmax)
            ax.set_title(f"{title_prefix}Predicted".strip())
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            tgt_vmax = float(np.nanmax(tgt)) if np.size(tgt) else 1.0
            vmax = max(pred_vmax, tgt_vmax, 1e-6)
            diff = pred - tgt
            diff_abs = float(np.nanmax(np.abs(diff))) if np.size(diff) else 1.0
            diff_abs = max(diff_abs, 1e-6)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes[0].imshow(tgt, cmap='Reds', vmin=0.0, vmax=vmax)
            axes[0].set_title(f"{title_prefix}Target".strip())
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(pred, cmap='Reds', vmin=0.0, vmax=vmax)
            axes[1].set_title(f"{title_prefix}Predicted".strip())
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff_abs, vmax=diff_abs)
            axes[2].set_title(f"{title_prefix}Pred - Target".strip())
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        save_path = (Path(out_path) if out_path is not None else
                     Path(self.cfg.out_dir) / "visualizations"
                     / f"sample_{sample_idx:04d}_cond_{condition_idx:02d}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(save_path)

    def count_parameters(self) -> dict:
        def _c(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'dna_encoder': _c(self.dna_encoder),
            'rna_encoder': _c(self.rna_encoder),
            'fusion': _c(self.fusion),
            'outer_proj': _c(self.outer_proj),
            'aux_features': _c(self.aux_features),
            'hic_head': _c(self.hic_head),
            'loop_head': _c(self.loop_head) if self.loop_head else 0,
            'total': _c(self),
        }

    def summary(self) -> str:
        counts = self.count_parameters()
        lines = ["=" * 60, "  HiC-Verse M2++ Model Summary", "=" * 60]
        for k, v in counts.items():
            lines.append(f"  {k:<22s}  {v / 1e6:>7.3f} M params")
        lines.append("=" * 60)
        lines.append(f"  SSM backend: {_SSM_NAME}")
        lines.append(f"  prefer_mamba2={self.cfg.prefer_mamba2} "
                     f"allow_mamba3_fallback={self.cfg.allow_mamba3_fallback}")
        lines.append(f"  Hi-C head blocks: {getattr(self.cfg, 'n_hic_blocks', 1)}")
        lines.append(f"  Aux-feature dropout: {getattr(self.cfg, 'aux_feature_dropout_prob', 0.0):.2f}")
        lines.append(f"  Loop head: {'enabled' if self.loop_head else 'disabled'}")
        lines.append("  Enhancements: SS2D + LEFN + Aux Features")
        lines.append("=" * 60)
        return "\n".join(lines)
