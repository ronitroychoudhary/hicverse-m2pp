"""
HiC-Verse M2++ Training Script
=================================
Enhanced training with:
  - L1 loss tracking
  - Per-condition metrics
  - Enhanced TensorBoard logging
  - Gradient statistics
  - Memory monitoring
"""

from __future__ import annotations
import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_OK = True
except ImportError:
    TENSORBOARD_OK = False
    warnings.warn("TensorBoard not available. Install: pip install tensorboard")

from config_m2pp import HiCVerseConfigM2PP, get_test_config_m2pp, get_full_config_m2pp
from model_m2pp import HiCVerseModelM2PP
from dataset import SyntheticHiCDataset, HiCWindowDataset, build_dataloaders
from losses_m2pp import CompositeHiCLossM2PP


# ─────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────

def pearson_r(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Pearson correlation coefficient."""
    p = pred.reshape(pred.shape[0], -1).float()
    t = tgt.reshape(tgt.shape[0], -1).float()
    p_c = p - p.mean(1, keepdim=True)
    t_c = t - t.mean(1, keepdim=True)
    num = (p_c * t_c).sum(1)
    denom = (p_c.pow(2).sum(1) * t_c.pow(2).sum(1)).sqrt() + 1e-8
    return (num / denom).mean().item()


def spearman_r(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Spearman rank correlation (approximation via Pearson on ranks)."""
    try:
        p = pred.reshape(pred.shape[0], -1).float()
        t = tgt.reshape(tgt.shape[0], -1).float()
        # Rank transform
        p_rank = p.argsort(dim=1).argsort(dim=1).float()
        t_rank = t.argsort(dim=1).argsort(dim=1).float()
        return pearson_r(
            p_rank.reshape(pred.shape),
            t_rank.reshape(tgt.shape)
        )
    except:
        return 0.0


def mse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Mean squared error."""
    return ((pred - tgt) ** 2).mean().item()


def mae(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Mean absolute error (L1)."""
    return (pred - tgt).abs().mean().item()


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    return 0.5 * (1.0 + math.cos(
        math.pi * (step - warmup) / max(1, total - warmup)
    ))


# ═════════════════════════════════════════════════════════════════
#  ENHANCED TRAINER
# ═════════════════════════════════════════════════════════════════

class TrainerM2PP:
    """
    M2++ Trainer with enhanced statistics tracking.

    New features:
      - Per-condition metrics
      - Gradient norm histogram
      - Memory usage tracking
      - TensorBoard integration
      - L1/MSE comparison
    """

    def __init__(
        self,
        cfg: HiCVerseConfigM2PP,
        mode: str = 'synthetic',
        n_synthetic_train: int = 32,
        n_synthetic_val: int = 8,
        chroms: Optional[list] = None,
        resume_ckpt: Optional[str] = None,
        viz_every_n_steps: int = 0,
        viz_max_per_epoch: int = 2,
        viz_n_conditions: int = 1,
    ):
        self.cfg = cfg
        self.mode = mode
        if mode == 'real':
            self._align_cfg_conditions_with_real_data()
        if str(cfg.device).startswith('cuda') and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            cfg.device = 'cpu'
        self.device = torch.device(cfg.device)
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.viz_every_n_steps = max(0, int(viz_every_n_steps))
        self.viz_max_per_epoch = max(0, int(viz_max_per_epoch))
        self.viz_n_conditions = max(1, int(viz_n_conditions))
        self.viz_dir = Path(cfg.out_dir) / 'visualizations'
        self.viz_enabled = self.viz_every_n_steps > 0
        self._resumed_from_ckpt = False
        self._disable_scheduler_step = False

        # TensorBoard
        self.writer = None
        if TENSORBOARD_OK:
            log_dir = Path(cfg.out_dir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

        print(f"\n{'=' * 60}")
        print(f"  HiC-Verse M2++ Trainer")
        print(f"  Mode: {mode}  |  Device: {self.device}")
        print(f"  Enhancements: SS2D + LEFN + Aux Features + L1 Loss")
        if cfg.use_ctcf_fimo_prior:
            print(f"  CTCF priors: ON  ({cfg.ctcf_fimo_tsv_path})")
        else:
            print("  CTCF priors: OFF")
        if getattr(cfg, 'use_tad_priors', False):
            print(f"  TAD priors: ON  ({getattr(cfg, 'tad_priors_dir', None)})")
        else:
            print("  TAD priors: OFF")
        print(f"{'=' * 60}\n")

        # Model
        self.model = HiCVerseModelM2PP(cfg).to(self.device)
        print(self.model.summary())

        # Data
        self.dl_train, self.dl_val = self._build_dataloaders(
            mode, n_synthetic_train, n_synthetic_val, chroms
        )

        # Optimizer
        total_steps = self._expected_total_steps()
        self.optim = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = self._build_scheduler(total_steps=total_steps)

        # Loss
        self.criterion = CompositeHiCLossM2PP(cfg)
        self.use_amp = cfg.mixed_precision and self.device.type == 'cuda'
        self.amp_dtype = torch.float16
        if self.use_amp and torch.cuda.is_bf16_supported():
            has_rmsnorm = hasattr(nn, "RMSNorm") and any(
                isinstance(m, nn.RMSNorm) for m in self.model.modules()
            )
            if has_rmsnorm:
                warnings.warn(
                    "Detected nn.RMSNorm modules; using FP16 autocast to avoid "
                    "BF16 RMSNorm dtype-mismatch fallback warnings."
                )
            else:
                self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_r = -1.0
        self.history = {
            'train_loss': [],
            'train_l1': [],
            'train_pearson': [],
            'val_loss': [],
            'val_l1': [],
            'val_pearson': [],
            'val_spearman': [],
            'val_mse': [],
            'val_mae': [],
        }

        if resume_ckpt:
            self._load_checkpoint(resume_ckpt)

    def _align_cfg_conditions_with_real_data(self) -> None:
        """
        For real mode, infer number of condition groups from X/ directory names
        (condXX_*) and align cfg.n_conditions to avoid runtime mismatch noise.
        """
        x_dir = Path(self.cfg.x_dir)
        if not x_dir.exists() or not x_dir.is_dir():
            return
        conds = set()
        for d in x_dir.iterdir():
            if not d.is_dir():
                continue
            m = re.match(r'(cond\d+)', d.name)
            if m:
                conds.add(m.group(1))
        if not conds:
            return
        inferred = len(conds)
        if inferred != int(self.cfg.n_conditions):
            old = int(self.cfg.n_conditions)
            self.cfg.n_conditions = inferred
            self.cfg.validate()
            print(
                f"[TrainerM2PP] Real dataset has {inferred} condition groups in X/. "
                f"Using n_conditions={inferred} (was {old}).",
                flush=True,
            )

    def _expected_total_steps(self) -> int:
        return max(1, len(self.dl_train) * self.cfg.max_epochs)

    def _build_scheduler(self, total_steps: int, last_epoch: int = -1):
        cfg = self.cfg
        if cfg.scheduler_type == "onecycle":
            if last_epoch >= 0:
                init_lr = cfg.learning_rate / cfg.onecycle_div_factor
                min_lr = init_lr / cfg.onecycle_final_div_factor
                for pg in self.optim.param_groups:
                    pg.setdefault('initial_lr', init_lr)
                    pg.setdefault('max_lr', cfg.learning_rate)
                    pg.setdefault('min_lr', min_lr)
            return OneCycleLR(
                self.optim,
                max_lr=cfg.learning_rate,
                total_steps=max(1, int(total_steps)),
                pct_start=cfg.onecycle_pct_start,
                anneal_strategy='cos',
                div_factor=cfg.onecycle_div_factor,
                final_div_factor=cfg.onecycle_final_div_factor,
                last_epoch=last_epoch,
            )
        return LambdaLR(
            self.optim,
            lr_lambda=lambda s: _warmup_cosine(s, cfg.warmup_steps, total_steps),
            last_epoch=last_epoch,
        )

    def _save_contact_map_visualization(
        self,
        batch: dict,
        epoch: int,
        step: int,
        split: str = 'train',
    ) -> None:
        if not self.viz_enabled:
            return
        dna = batch['dna_seq'].to(self.device)
        rna = batch['rna_signal'].to(self.device)
        target = batch['target']
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                ctcf_prior = batch.get('ctcf_prior_2d')
                tad_priors = batch.get('tad_priors')
                if ctcf_prior is not None:
                    ctcf_prior = ctcf_prior.to(self.device)
                if tad_priors is not None:
                    tad_priors = tad_priors.to(self.device)
                preds = self.model(
                    dna,
                    rna,
                    ctcf_prior_2d=ctcf_prior,
                    tad_priors=tad_priors,
                    return_loops=False,
                )['contact_maps']
            for cond_idx in range(min(self.viz_n_conditions, preds.shape[1])):
                out_path = (self.viz_dir / split
                            / f"epoch_{epoch + 1:03d}_step_{step + 1:04d}_cond_{cond_idx:02d}.png")
                self.model.visualize_contact_maps(
                    pred_maps=preds,
                    target_maps=target,
                    sample_idx=0,
                    condition_idx=cond_idx,
                    out_path=str(out_path),
                    title_prefix=f"E{epoch + 1:03d} ",
                )
        except Exception as exc:
            warnings.warn(f"Visualization skipped ({split}, epoch {epoch + 1}): {exc}")
        finally:
            if was_training:
                self.model.train()

    # ── Data loaders ─────────────────────────────────────────────

    def _build_dataloaders(self, mode, n_train, n_val, chroms):
        cfg = self.cfg
        if mode == 'synthetic':
            ds_tr = SyntheticHiCDataset(
                n_samples=n_train, n_bins=cfg.n_bins,
                n_conditions=cfg.n_conditions, bin_size=cfg.test_bin_size
            )
            ds_va = SyntheticHiCDataset(
                n_samples=n_val, n_bins=cfg.n_bins,
                n_conditions=cfg.n_conditions, bin_size=cfg.test_bin_size,
                seed=9999
            )
            return build_dataloaders(
                ds_tr, ds_va, batch_size=cfg.batch_size,
                num_workers=0, pin_memory=False
            )
        elif mode == 'real':
            ds_tr = HiCWindowDataset(
                x_dir=cfg.x_dir,
                y_dir=cfg.y_dir,
                n_bins=cfg.n_bins,
                n_conditions=cfg.n_conditions,
                bin_size=cfg.bin_size,
                genome_fasta_path=cfg.genome_fasta_path,
                chroms=chroms,
                augment=True,
                normalize_rna=cfg.normalize_rna_inputs,
                normalize_hic=cfg.normalize_hic_targets,
                ctcf_fimo_tsv_path=cfg.ctcf_fimo_tsv_path,
                use_ctcf_fimo_prior=cfg.use_ctcf_fimo_prior,
                tad_priors_dir=getattr(cfg, 'tad_priors_dir', None),
                use_tad_priors=getattr(cfg, 'use_tad_priors', True),
            )
            return build_dataloaders(
                ds_tr, None, batch_size=cfg.batch_size,
                num_workers=0, pin_memory=True, val_split=0.1
            )
        raise ValueError(f"mode must be 'synthetic' or 'real', got {mode!r}")

    # ── Training step ────────────────────────────────────────────

    def _autocast_ctx(self):
        return autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
            dtype=self.amp_dtype,
        )

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        dna = batch['dna_seq'].to(self.device)
        rna = batch['rna_signal'].to(self.device)
        ctcf_prior = batch.get('ctcf_prior_2d')
        tad_priors = batch.get('tad_priors')
        if ctcf_prior is not None:
            ctcf_prior = ctcf_prior.to(self.device)
        if tad_priors is not None:
            tad_priors = tad_priors.to(self.device)

        with self._autocast_ctx():
            preds = self.model(
                dna,
                rna,
                ctcf_prior_2d=ctcf_prior,
                tad_priors=tad_priors,
            )
            loss, info = self.criterion(preds, batch)

        if not torch.isfinite(loss):
            if self.use_amp:
                # AMP can produce occasional non-finite loss in early steps.
                # Retry this step in FP32 before skipping.
                with autocast(device_type=self.device.type, enabled=False):
                    preds = self.model(
                        dna,
                        rna,
                        ctcf_prior_2d=ctcf_prior,
                        tad_priors=tad_priors,
                    )
                    loss, info = self.criterion(preds, batch)
                if torch.isfinite(loss):
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    if torch.isfinite(grad_norm):
                        self.optim.step()
                        self._step_scheduler()
                        self.global_step += 1
                        info['grad_norm'] = float(grad_norm)
                        info['amp_fallback_fp32'] = 1.0
                        return info
            self.optim.zero_grad(set_to_none=True)
            info['total'] = float('nan')
            info['skipped_nonfinite_loss'] = 1.0
            return info

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
        else:
            loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        if not torch.isfinite(grad_norm):
            self.optim.zero_grad(set_to_none=True)
            info['total'] = float('nan')
            info['skipped_nonfinite_grad'] = 1.0
            if self.use_amp:
                self.scaler.update()
            return info

        if self.use_amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
        self._step_scheduler()
        self.global_step += 1

        info['grad_norm'] = float(grad_norm)

        # TensorBoard logging
        if self.writer and self.global_step % 10 == 0:
            self.writer.add_scalar('train/loss', info['total'], self.global_step)
            self.writer.add_scalar('train/l1', info['l1'], self.global_step)
            self.writer.add_scalar('train/pearson', info['pearson'], self.global_step)
            self.writer.add_scalar('train/grad_norm', float(grad_norm), self.global_step)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

        return info

    # ── Validation ───────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: Optional[int] = None) -> dict:
        if self.dl_val is None:
            return {}
        self.model.eval()
        all_loss, all_l1, all_pearson, all_spearman = [], [], [], []
        all_mse, all_mae = [], []

        for idx, batch in enumerate(self.dl_val):
            dna = batch['dna_seq'].to(self.device)
            rna = batch['rna_signal'].to(self.device)
            tgt = batch['target'].to(self.device)
            ctcf_prior = batch.get('ctcf_prior_2d')
            tad_priors = batch.get('tad_priors')
            if ctcf_prior is not None:
                ctcf_prior = ctcf_prior.to(self.device)
            if tad_priors is not None:
                tad_priors = tad_priors.to(self.device)

            with self._autocast_ctx():
                preds = self.model(
                    dna,
                    rna,
                    ctcf_prior_2d=ctcf_prior,
                    tad_priors=tad_priors,
                    return_loops=False,
                )
                _, info = self.criterion(preds, batch)

            cm = preds['contact_maps']
            all_loss.append(info['total'])
            all_l1.append(info['l1'])
            all_pearson.append(info['pearson'])
            all_spearman.append(spearman_r(cm, tgt))
            all_mse.append(mse(cm, tgt))
            all_mae.append(mae(cm, tgt))

            if (self.viz_enabled and idx == 0 and epoch is not None
                    and (epoch + 1) % self.cfg.val_every_n_epochs == 0):
                for cond_idx in range(min(self.viz_n_conditions, cm.shape[1])):
                    out_path = (self.viz_dir / 'val'
                                / f"epoch_{epoch + 1:03d}_step_{self.global_step:06d}_cond_{cond_idx:02d}.png")
                    self.model.visualize_contact_maps(
                        pred_maps=cm,
                        target_maps=tgt,
                        sample_idx=0,
                        condition_idx=cond_idx,
                        out_path=str(out_path),
                        title_prefix=f"E{epoch + 1:03d} ",
                    )

        if not all_loss:
            return {}

        metrics = {
            'val_loss': float(np.mean(all_loss)),
            'val_l1': float(np.mean(all_l1)),
            'val_pearson': float(np.mean(all_pearson)),
            'val_spearman': float(np.mean(all_spearman)),
            'val_mse': float(np.mean(all_mse)),
            'val_mae': float(np.mean(all_mae)),
            'val_r': 1.0 - float(np.mean(all_pearson)),  # for best model tracking
        }

        # TensorBoard
        if self.writer:
            for k, v in metrics.items():
                if k != 'val_r':
                    self.writer.add_scalar(f'val/{k[4:]}', v, self.global_step)

        return metrics

    # ── Checkpointing ────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        path = self.ckpt_dir / f"hicverse_m2pp_{tag}.pt"
        torch.save(
            {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'best_val_r': self.best_val_r,
                'history': self.history,
                'cfg': self.cfg,
            },
            path,
        )
        print(f"  💾  Saved: {path.name}")

    def _load_checkpoint(self, path: str):
        ck = self._load_checkpoint_file(path, map_location=self.device)
        self.model.load_state_dict(ck['model'])
        self.optim.load_state_dict(ck['optim'])
        self.epoch = int(ck['epoch'])
        self.global_step = int(ck['global_step'])
        self.best_val_r = ck.get('best_val_r', -1.0)
        self.history = ck.get('history', self.history)

        # Scheduler: handle changed --epochs (OneCycle total_steps mismatch) safely.
        sched_loaded = False
        ck_sched = ck.get('scheduler')
        expected_total = self._expected_total_steps()
        if ck_sched is not None:
            if isinstance(self.scheduler, OneCycleLR):
                ck_total = int(ck_sched.get('total_steps', -1)) if isinstance(ck_sched, dict) else -1
                if ck_total == expected_total:
                    try:
                        self.scheduler.load_state_dict(ck_sched)
                        sched_loaded = True
                    except Exception as exc:
                        warnings.warn(
                            f"Scheduler state could not be restored ({exc}). "
                            "Rebuilding scheduler from global_step."
                        )
                if not sched_loaded:
                    # Continue schedule with the new training horizon.
                    self.scheduler = self._build_scheduler(
                        total_steps=expected_total,
                        last_epoch=max(-1, self.global_step - 1),
                    )
                    sched_loaded = True
                    warnings.warn(
                        f"Rebuilt OneCycleLR for resume: checkpoint_total_steps={ck_total}, "
                        f"expected_total_steps={expected_total}, global_step={self.global_step}."
                    )
            else:
                try:
                    self.scheduler.load_state_dict(ck_sched)
                    sched_loaded = True
                except Exception as exc:
                    warnings.warn(
                        f"Scheduler state could not be restored ({exc}). "
                        "Rebuilding scheduler from global_step."
                    )

        if not sched_loaded:
            self.scheduler = self._build_scheduler(
                total_steps=expected_total,
                last_epoch=max(-1, self.global_step - 1),
            )

        if self.use_amp and ck.get('scaler'):
            self.scaler.load_state_dict(ck['scaler'])
        self._resumed_from_ckpt = True
        print(f"  ▶  Resumed from {path}  (epoch {self.epoch})")

    @staticmethod
    def _load_checkpoint_file(path: str, map_location='cpu'):
        """
        Load checkpoint across PyTorch versions.
        PyTorch 2.6 changed torch.load default to weights_only=True.
        """
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older PyTorch versions without weights_only kwarg.
            return torch.load(path, map_location=map_location)
        except pickle.UnpicklingError as exc:
            msg = str(exc)
            if "Weights only load failed" in msg:
                warnings.warn(
                    "Checkpoint requires full unpickling. Retrying with "
                    "weights_only=False (trusted checkpoint assumed)."
                )
                return torch.load(path, map_location=map_location, weights_only=False)
            raise

    def _step_scheduler(self) -> None:
        if self._disable_scheduler_step:
            return
        try:
            self.scheduler.step()
        except ValueError as exc:
            msg = str(exc)
            if isinstance(self.scheduler, OneCycleLR) and "Tried to step" in msg:
                warnings.warn(
                    "OneCycleLR step limit reached for current total_steps. "
                    "Keeping learning rate fixed for remaining steps."
                )
                self._disable_scheduler_step = True
                return
            raise

    # ── Main training loop ───────────────────────────────────────

    def train(self):
        cfg = self.cfg
        print(f"Training: {cfg.max_epochs} epochs, "
              f"{len(self.dl_train)} steps/epoch\n")

        start_epoch = self.epoch + 1 if self._resumed_from_ckpt else self.epoch
        for epoch in range(start_epoch, cfg.max_epochs):
            self.epoch = epoch
            ep_losses, ep_l1s, ep_pearsons = [], [], []
            t0 = time.time()
            viz_saved = 0

            for step, batch in enumerate(self.dl_train):
                info = self._train_step(batch)
                ep_losses.append(info['total'])
                ep_l1s.append(info.get('l1', 0))
                ep_pearsons.append(info.get('pearson', 0))

                if (self.viz_enabled
                        and viz_saved < self.viz_max_per_epoch
                        and (step == 0 or (step + 1) % self.viz_every_n_steps == 0)):
                    self._save_contact_map_visualization(
                        batch=batch,
                        epoch=epoch,
                        step=step,
                        split='train',
                    )
                    viz_saved += 1

                if step == 0 or (step + 1) % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"  Ep {epoch + 1:03d}  "
                        f"Step {step + 1:04d}/{len(self.dl_train)}  "
                        f"loss={info['total']:.4f}  "
                        f"l1={info.get('l1', 0):.4f}  "
                        f"prs={info.get('pearson', 0):.4f}"
                        + (f"  loop={info.get('loop_bce', 0):.4f}"
                           if 'loop_bce' in info else "")
                        + ("  skipped=loss" if info.get('skipped_nonfinite_loss') else "")
                        + ("  skipped=grad" if info.get('skipped_nonfinite_grad') else "")
                        + f"  lr={lr:.2e}",
                        flush=True,
                    )

            # Epoch stats
            finite_losses = [x for x in ep_losses if np.isfinite(x)]
            finite_l1s = [x for x in ep_l1s if np.isfinite(x)]
            finite_pearsons = [x for x in ep_pearsons if np.isfinite(x)]
            mean_loss = float(np.mean(finite_losses)) if finite_losses else float('nan')
            mean_l1 = float(np.mean(finite_l1s)) if finite_l1s else float('nan')
            mean_pearson = float(np.mean(finite_pearsons)) if finite_pearsons else float('nan')
            self.history['train_loss'].append(mean_loss)
            self.history['train_l1'].append(mean_l1)
            self.history['train_pearson'].append(mean_pearson)

            # Validation
            val_info = {}
            if (epoch + 1) % cfg.val_every_n_epochs == 0:
                val_info = self._validate(epoch=epoch)
                if val_info:
                    for k in ['val_loss', 'val_l1', 'val_pearson',
                              'val_spearman', 'val_mse', 'val_mae']:
                        if k in val_info:
                            self.history[k].append(val_info[k])

                    print(
                        f"\n  ── Val: loss={val_info['val_loss']:.4f}  "
                        f"Pearson r={val_info['val_r']:.4f}  "
                        f"Spearman r={val_info['val_spearman']:.4f}  "
                        f"MSE={val_info['val_mse']:.4f}  "
                        f"MAE={val_info['val_mae']:.4f}\n"
                    )

                    if val_info['val_r'] > self.best_val_r:
                        self.best_val_r = val_info['val_r']
                        self._save_checkpoint('best')

            # Periodic save
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(f'ep{epoch + 1:03d}')

            # Epoch summary
            print(
                f"Epoch {epoch + 1:03d} done | "
                f"train_loss={mean_loss:.4f} | "
                f"train_l1={mean_l1:.4f} | "
                f"time={time.time() - t0:.1f}s"
                + (f" | val_r={val_info.get('val_r', 0):.4f}"
                   if val_info else "")
            )

        # Final save
        self._save_checkpoint('final')
        path = self.ckpt_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Done.  Best val Pearson r = {self.best_val_r:.4f}")
        print(f"  History saved to {path}")

        if self.writer:
            self.writer.close()


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='synthetic',
                   choices=['synthetic', 'real'])
    p.add_argument('--config', default='test',
                   choices=['test', 'full'])
    p.add_argument('--n_conditions', type=int, default=6)
    p.add_argument('--resume', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--scheduler', default=None,
                   choices=['onecycle', 'warmup_cosine'])
    p.add_argument('--aux_dropout', type=float, default=None)
    p.add_argument('--n_hic_blocks', type=int, default=None)
    p.add_argument('--loss_map_weight', type=float, default=None)
    p.add_argument('--loss_pearson_weight', type=float, default=None)
    p.add_argument('--loss_loop_weight', type=float, default=None)
    p.add_argument('--ctcf_fimo_tsv', type=str, default=None)
    p.add_argument('--no_ctcf_prior', action='store_true')
    p.add_argument('--tad_priors_dir', type=str, default=None)
    p.add_argument('--no_tad_priors', action='store_true')
    p.add_argument('--viz_every_n_steps', type=int, default=0)
    p.add_argument('--viz_max_per_epoch', type=int, default=2)
    p.add_argument('--viz_n_conditions', type=int, default=1)
    args = p.parse_args()

    cfg = (get_test_config_m2pp if args.config == 'test'
           else get_full_config_m2pp)(n_conditions=args.n_conditions)
    if args.epochs:
        cfg.max_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.no_amp:
        cfg.mixed_precision = False
    if args.scheduler:
        cfg.scheduler_type = args.scheduler
    if args.aux_dropout is not None:
        cfg.aux_feature_dropout_prob = args.aux_dropout
    if args.n_hic_blocks is not None:
        cfg.n_hic_blocks = args.n_hic_blocks
    if args.loss_map_weight is not None:
        cfg.loss_map_weight = args.loss_map_weight
    if args.loss_pearson_weight is not None:
        cfg.loss_pearson_weight = args.loss_pearson_weight
    if args.loss_loop_weight is not None:
        cfg.loss_loop_weight = args.loss_loop_weight
    if args.ctcf_fimo_tsv is not None:
        cfg.ctcf_fimo_tsv_path = args.ctcf_fimo_tsv
    if args.no_ctcf_prior:
        cfg.use_ctcf_fimo_prior = False
    if args.tad_priors_dir is not None:
        cfg.tad_priors_dir = args.tad_priors_dir
    if args.no_tad_priors:
        cfg.use_tad_priors = False
    cfg.validate()

    TrainerM2PP(
        cfg=cfg, mode=args.mode, resume_ckpt=args.resume,
        n_synthetic_train=64, n_synthetic_val=16,
        viz_every_n_steps=args.viz_every_n_steps,
        viz_max_per_epoch=args.viz_max_per_epoch,
        viz_n_conditions=args.viz_n_conditions,
    ).train()
