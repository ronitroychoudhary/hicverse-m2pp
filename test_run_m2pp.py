"""
HiC-Verse M2++ · Validation Test Script
==========================================
Tests all M2++ components with synthetic data.
No real files needed.

Usage:
    python test_run_m2pp.py
    python test_run_m2pp.py --n_conditions 6 --n_bins 50 --full_model
"""

from __future__ import annotations
import sys
import time
import argparse
import traceback
import tempfile
from pathlib import Path
from typing import Optional

import torch
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

PASS = "  ✓"
FAIL = "  ✗"
SKIP = "  ─"


def banner(msg: str):
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def check(cond: bool, msg: str) -> bool:
    print(f"{'  ✓' if cond else '  ✗'}  {msg}")
    return cond


# ── Test 1: imports ──────────────────────────────────────────────

def test_imports() -> bool:
    banner("1 / 9  Import chain")
    ok = True
    ok &= check(torch.__version__ is not None, f"PyTorch {torch.__version__}")
    cuda = torch.cuda.is_available()
    check(cuda, f"CUDA available: {cuda}")
    if cuda:
        check(True, f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import mamba_ssm
        ok &= check(True, f"mamba_ssm {mamba_ssm.__version__}")
    except ImportError:
        check(False, "mamba_ssm not installed — GRU fallback will be used")

    for mod in ('config_m2pp', 'model_m2pp', 'dataset', 'losses_m2pp', 'train_m2pp'):
        try:
            __import__(mod)
            ok &= check(True, f"'{mod}' importable")
        except Exception as e:
            check(False, f"'{mod}' import failed: {e}")
            traceback.print_exc()
            ok = False
    return ok


# ── Test 2: config ───────────────────────────────────────────────

def test_config(n_conditions, n_bins, full_model) -> Optional[object]:
    banner("2 / 9  M2++ Config creation & validation")
    from config_m2pp import get_test_config_m2pp, get_full_config_m2pp
    try:
        cfg = (get_full_config_m2pp if full_model
               else get_test_config_m2pp)(n_conditions=n_conditions)
        cfg.n_bins = n_bins
        cfg.validate()
        check(True, f"Config OK: d_model={cfg.d_model}, d_state={cfg.d_state}, "
              f"headdim={cfg.headdim}, n_bins={cfg.n_bins}")
        check(cfg.use_aux_features, "Auxiliary features enabled")
        check(cfg.ss2d_d_state > 0, f"SS2D state size: {cfg.ss2d_d_state}")
        check(cfg.lefn_expansion > 0, f"LEFN expansion: {cfg.lefn_expansion}")
        d_inner = cfg.d_model * cfg.expand
        check(d_inner % cfg.headdim == 0,
              f"d_inner={d_inner} divisible by headdim={cfg.headdim} ✓")
        return cfg
    except Exception as e:
        check(False, f"Config failed: {e}")
        traceback.print_exc()
        return None


# ── Test 3: dataset ──────────────────────────────────────────────

def test_dataset(cfg, n_samples=8) -> Optional[object]:
    banner("3 / 9  SyntheticHiCDataset")
    from dataset import SyntheticHiCDataset, build_dataloaders
    try:
        ds = SyntheticHiCDataset(
            n_samples=n_samples, n_bins=cfg.n_bins,
            n_conditions=cfg.n_conditions, bin_size=cfg.test_bin_size
        )
        check(len(ds) == n_samples, f"len={len(ds)}")
        s = ds[0]
        check(s['dna_seq'].shape == (cfg.n_bins * cfg.test_bin_size,),
              f"dna_seq: {s['dna_seq'].shape}")
        check(s['rna_signal'].shape == (cfg.n_conditions, cfg.n_bins),
              f"rna_signal: {s['rna_signal'].shape}")
        check(s['target'].shape == (cfg.n_conditions, cfg.n_bins, cfg.n_bins),
              f"target: {s['target'].shape}")
        check(s['tad_priors'].shape == (4, cfg.n_bins, cfg.n_bins),
              f"tad_priors: {s['tad_priors'].shape}")
        dl_tr, _ = build_dataloaders(ds, None, batch_size=cfg.batch_size,
                                      num_workers=0, pin_memory=False, val_split=0.25)
        batch = next(iter(dl_tr))
        check(batch['rna_signal'].shape[1] == cfg.n_conditions, "Batch rna ok")
        check(batch['tad_priors'].shape[1] == 4, "Batch TAD priors channels=4")
        return batch
    except Exception as e:
        check(False, f"Dataset failed: {e}")
        traceback.print_exc()
        return None


# ── Test 4: model forward ────────────────────────────────────────

def test_model_forward(cfg, batch) -> Optional[dict]:
    banner("4 / 9  M2++ Model forward pass")
    from model_m2pp import HiCVerseModelM2PP
    device = torch.device(cfg.device)
    try:
        model = HiCVerseModelM2PP(cfg).to(device)
        print(model.summary())
        dna = batch['dna_seq'].to(device)
        rna = batch['rna_signal'].to(device)
        tad = batch['tad_priors'].to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model(dna, rna, tad_priors=tad)
        B, N, L = dna.shape[0], cfg.n_conditions, cfg.n_bins
        check(out['contact_maps'].shape == (B, N, L, L),
              f"contact_maps: {out['contact_maps'].shape}")
        check(not out['contact_maps'].isnan().any(), "No NaN in contact_maps")
        if cfg.enable_loop_head:
            check(out['loop_logits'].shape == (B, L, L),
                  f"loop_logits: {out['loop_logits'].shape}")
            check(not out['loop_logits'].isnan().any(), "No NaN in loop_logits")
        else:
            check(out['loop_logits'] is None, "loop head disabled")
        print(f"  ℹ  Forward took {time.time() - t0:.3f}s")
        return out
    except Exception as e:
        check(False, f"Forward failed: {e}")
        traceback.print_exc()
        return None


# ── Test 5: loss + backward ──────────────────────────────────────

def test_loss(cfg, batch) -> bool:
    banner("5 / 9  M2++ Loss + backward (L1 + Pearson)")
    from model_m2pp import HiCVerseModelM2PP
    from losses_m2pp import CompositeHiCLossM2PP
    device = torch.device(cfg.device)
    try:
        model = HiCVerseModelM2PP(cfg).to(device)
        dna = batch['dna_seq'].to(device)
        rna = batch['rna_signal'].to(device)
        tad = batch['tad_priors'].to(device)
        preds = model(dna, rna, tad_priors=tad)
        loss, info = CompositeHiCLossM2PP(cfg)(preds, batch)
        check(not torch.isnan(loss), f"Loss={loss.item():.4f}")
        check('l1' in info, "L1 loss component present")
        print(f"  ℹ  Breakdown: {info}")
        loss.backward()
        nan_grads = [n for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None
                     and p.grad.isnan().any()]
        check(len(nan_grads) == 0, f"No NaN gradients ({len(nan_grads)} found)")
        return True
    except Exception as e:
        check(False, f"Loss test failed: {e}")
        traceback.print_exc()
        return False


# ── Test 6: SS2D component ───────────────────────────────────────

def test_ss2d(cfg) -> bool:
    banner("6 / 9  SS2D (2D Selective Scan) component")
    from model_m2pp import SS2D
    try:
        d_model = cfg.d_outer * 2
        ss2d = SS2D(d_model=d_model, d_state=cfg.ss2d_d_state)
        B, C, H, W = 2, d_model, 50, 50
        x = torch.randn(B, C, H, W)
        out = ss2d(x)
        check(out.shape == (B, C, H, W), f"SS2D output shape: {out.shape}")
        check(not out.isnan().any(), "No NaN in SS2D output")
        return True
    except Exception as e:
        check(False, f"SS2D test failed: {e}")
        traceback.print_exc()
        return False


# ── Test 7: LEFN component ───────────────────────────────────────

def test_lefn(cfg) -> bool:
    banner("7 / 9  LEFN (Locally Enhanced FFN) component")
    from model_m2pp import LEFN
    try:
        d_model = cfg.d_outer * 2
        lefn = LEFN(d_model=d_model, expansion=cfg.lefn_expansion)
        B, C, H, W = 2, d_model, 50, 50
        x = torch.randn(B, C, H, W)
        out = lefn(x)
        check(out.shape == (B, C, H, W), f"LEFN output shape: {out.shape}")
        check(not out.isnan().any(), "No NaN in LEFN output")
        return True
    except Exception as e:
        check(False, f"LEFN test failed: {e}")
        traceback.print_exc()
        return False


# ── Test 8: Auxiliary features ───────────────────────────────────

def test_aux_features(cfg) -> bool:
    banner("8 / 9  Auxiliary biological features")
    from model_m2pp import AuxiliaryBiologicalFeatures
    try:
        B, L = 2, cfg.n_bins
        device = torch.device(cfg.device)
        aux = AuxiliaryBiologicalFeatures(cfg).to(device)
        tad = torch.zeros(B, 4, L, L, device=device)
        out = aux(B, L, device, tad_priors=tad)
        check(out.shape[0] == B, f"Batch dimension: {out.shape[0]}")
        check(out.shape[2] == L and out.shape[3] == L,
              f"Spatial dims: {out.shape[2]}×{out.shape[3]}")
        check(not out.isnan().any(), "No NaN in aux features")
        return True
    except Exception as e:
        check(False, f"Aux features test failed: {e}")
        traceback.print_exc()
        return False


# ── Test 9: mini epoch ───────────────────────────────────────────

def test_mini_epoch(cfg) -> bool:
    banner("9 / 9  Mini training epoch (2 steps)")
    from train_m2pp import TrainerM2PP
    try:
        cfg.max_epochs = 1
        cfg.mixed_precision = False
        cfg.save_every_n_epochs = 999
        cfg.val_every_n_epochs = 999
        trainer = TrainerM2PP(
            cfg=cfg, mode='synthetic',
            n_synthetic_train=max(cfg.batch_size * 2, 4),
            n_synthetic_val=max(cfg.batch_size, 2),
        )
        it = iter(trainer.dl_train)
        for step in range(2):
            info = trainer._train_step(next(it))
            check(not np.isnan(info['total']),
                  f"Step {step + 1}: loss={info['total']:.4f}")
        return True
    except Exception as e:
        check(False, f"Mini epoch failed: {e}")
        traceback.print_exc()
        return False


# ── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_conditions', type=int, default=6)
    ap.add_argument('--n_bins', type=int, default=50)
    ap.add_argument('--full_model', action='store_true')
    args = ap.parse_args()

    print("\n" + "═" * 60)
    print("  HiC-Verse M2++ · Validation Test Run")
    print("  Enhancements: SS2D + LEFN + Aux Features + L1 Loss")
    print("═" * 60)

    results = {}
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n✗ Fix imports first.")
        sys.exit(1)

    cfg = test_config(args.n_conditions, args.n_bins, args.full_model)
    results['config'] = cfg is not None
    if cfg is None:
        sys.exit(1)

    batch = test_dataset(cfg, n_samples=max(cfg.batch_size * 2, 4))
    results['dataset'] = batch is not None

    if batch is not None:
        out = test_model_forward(cfg, batch)
        results['forward'] = out is not None
        results['loss'] = test_loss(cfg, batch)
    else:
        results['forward'] = results['loss'] = False

    results['ss2d'] = test_ss2d(cfg)
    results['lefn'] = test_lefn(cfg)
    results['aux_features'] = test_aux_features(cfg)
    results['epoch'] = test_mini_epoch(cfg)

    banner("Test Summary")
    all_pass = True
    for name, ok in results.items():
        print(f"{'  ✓' if ok else '  ✗'}  {name}")
        all_pass &= ok

    print()
    if all_pass:
        print("  🎉  All M2++ tests passed!")
        print()
        print("  Next steps:")
        print("    # Sanity test on real data:")
        print("    python train_m2pp.py --mode real --config test")
        print()
        print("    # Full M2++ training:")
        print("    python train_m2pp.py --mode real --config full")
        print()
        print("    # Test saved checkpoint:")
        print("    python test_m2pp.py checkpoints_m2pp/hicverse_m2pp_best.pt")
    else:
        n = sum(1 for v in results.values() if not v)
        print(f"  ⚠  {n} test(s) failed — check logs above.")
    print()


if __name__ == '__main__':
    main()
