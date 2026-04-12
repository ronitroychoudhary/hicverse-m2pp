"""
HiC-Verse M2++ Loss Functions
================================
Key change from v2: MSE → L1 loss for sharper Hi-C maps.

Loss components:
  1. L1          — mean absolute error (sharper than MSE)
  2. Pearson     — structural correlation
  3. Focal-BCE   — loop detection (optional)

Rationale for L1 over MSE:
  MSE = mean((ŷ - y)²) → penalizes large errors quadratically
    → optimizer produces smooth, blurry outputs to minimize variance
  L1 = mean(|ŷ - y|)   → linear penalty on errors
    → preserves sharp features (loop anchors, TAD boundaries)
    → better matches biology (Hi-C has sharp structural transitions)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonCorrelationLoss(nn.Module):
    """
    Pearson correlation loss: 1 - r
    Captures structural similarity independent of absolute scale.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Stable Pearson: epsilon is inside variance/std to avoid singular grads
        # when either tensor has near-zero variance.
        p = pred.reshape(pred.shape[0], -1).float()
        t = target.reshape(target.shape[0], -1).float()
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        p_c = p - p.mean(1, keepdim=True)
        t_c = t - t.mean(1, keepdim=True)
        p_std = torch.sqrt(p_c.pow(2).mean(1, keepdim=True) + self.eps)
        t_std = torch.sqrt(t_c.pow(2).mean(1, keepdim=True) + self.eps)
        p_n = p_c / p_std
        t_n = t_c / t_std
        r = (p_n * t_n).mean(1).clamp(-1.0, 1.0)
        return (1.0 - r).mean()


class FocalBCELoss(nn.Module):
    """
    Focal binary cross-entropy for imbalanced loop detection.
    Down-weights easy negatives (many zero-contact pairs).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 15.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p_t = torch.sigmoid(logits)
        focal = (1 - p_t * targets - (1 - p_t) * (1 - targets)).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none',
            pos_weight=logits.new_tensor(self.pos_weight)
        )
        return (focal * bce).mean()


class CompositeHiCLossM2PP(nn.Module):
    """
    M2++ composite loss:
      L_total = w_map × L1(pred, target)
              + w_pearson × (1 - Pearson_r)
              + w_loop × FocalBCE(loop_logits, loop_targets)

    Key change: L1 replaces MSE for sharper predictions.
    """

    def __init__(self, cfg):
        super().__init__()
        self.w_map = cfg.loss_map_weight
        self.w_pearson = cfg.loss_pearson_weight
        self.w_loop = cfg.loss_loop_weight if cfg.enable_loop_head else 0.0
        self.l1 = nn.L1Loss()
        self.pearson = PearsonCorrelationLoss()
        self.focal = FocalBCELoss()

    def _make_loop_targets(
        self,
        target: torch.Tensor,
        min_dist: int = 10,
        top_pct: float = 0.95,
    ) -> torch.Tensor:
        """Auto-generate loop labels from Hi-C target."""
        t = target.mean(dim=1)  # (B, L, L)
        B, L, _ = t.shape
        diag_mask = torch.ones(L, L, device=t.device, dtype=torch.bool)
        for d in range(min_dist):
            idx = torch.arange(L - d, device=t.device)
            diag_mask[idx, idx + d] = False
            diag_mask[idx + d, idx] = False
        loop_tgt = torch.zeros_like(t)
        for b in range(B):
            vals = t[b][diag_mask]
            if vals.numel():
                loop_tgt[b][(t[b] >= vals.quantile(top_pct)) & diag_mask] = 1.0
        return loop_tgt

    def forward(
        self,
        predictions: dict,
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        cm = predictions['contact_maps']
        ll = predictions.get('loop_logits')
        tgt = batch['target'].to(cm.device)
        info = {}

        # L1 loss (NEW: replaces MSE)
        l_l1 = self.l1(cm, tgt)
        info['l1'] = l_l1.item()

        # Pearson correlation
        l_prs = self.pearson(cm, tgt)
        info['pearson'] = l_prs.item()

        total = self.w_map * l_l1 + self.w_pearson * l_prs

        # Loop detection (optional)
        if ll is not None and self.w_loop > 0:
            l_loop = self.focal(ll, self._make_loop_targets(tgt))
            info['loop_bce'] = l_loop.item()
            total = total + self.w_loop * l_loop

        info['total'] = total.item()
        return total, info
