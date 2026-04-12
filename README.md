# HiC-Verse M2++

Enhanced Mamba + Transformer model for predicting **multi-condition Hi-C contact maps** from DNA sequence and RNA-seq signals.

This repository contains a full training + evaluation pipeline with:
- hybrid Mamba/attention 1D encoders,
- 1D-to-2D outer-product projection,
- SS2D + LEFN Hi-C refinement blocks,
- optional biological priors (CTCF + TAD),
- synthetic and real-data workflows.

## Table of Contents
- [What The Model Does](#what-the-model-does)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Data Layout](#data-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

## What The Model Does
Given a genomic window, the model predicts one Hi-C map per condition.

- Input `dna_seq`: `(B, L_dna)` where `L_dna = n_bins * bin_size`.
- Input `rna_signal`: `(B, N_cond, n_bins)`.
- Optional priors:
  - `ctcf_prior_2d`: `(B, n_bins, n_bins)`
  - `tad_priors`: `(B, 4, n_bins, n_bins)`
- Output `contact_maps`: `(B, N_cond, n_bins, n_bins)`
- Optional output `loop_logits`: `(B, n_bins, n_bins)`

Default config uses `n_bins=200`, `bin_size=10,000` (about 2 Mb windows), and `n_conditions=6`.

## Architecture

### High-Level Flow

```mermaid
flowchart LR
    A[DNA sequence] --> B[DNAEncoder\nCNN + HybridMamba3AttentionStack]
    C[RNA signal per condition] --> D[DilatedRNAEncoder]
    B --> E[FeatureGating\nDNA * sigmoid(W*RNA)]
    D --> E
    E --> F[OuterProductProjection\n1D -> 2D]
    F --> G[Optional auxiliary priors\nDistance + TAD + CTCF]
    G --> H[HolisticScanBlock\n(SS2D + LEFN) x n_hic_blocks]
    H --> I[Symmetric Hi-C maps]
    F --> J[Optional LoopDetectionHead]
```

### Component Breakdown

1. `DNAEncoder`
- Token embedding + CNN front-end.
- Hybrid sequence stack mixes Mamba blocks with periodic self-attention.

2. `DilatedRNAEncoder`
- Per-condition 1D dilated convolution stack.
- Produces condition-aware sequence features aligned to genomic bins.

3. `FeatureGating`
- Conditions DNA features with RNA features using learned gating.

4. `OuterProductProjection`
- Converts per-bin 1D fused features into pairwise 2D interaction features.

5. `AuxiliaryBiologicalFeatures` (optional)
- Distance-decay channel(s), CTCF-orientation prior, and TAD prior channels.
- Includes training-time aux-feature dropout to avoid over-reliance on priors.

6. `HolisticScanBlock`
- `SS2D`: directional row/column scans for long-range 2D context.
- `LEFN`: local 2D refinement for sharper boundaries and loop-like structures.
- Repeated `n_hic_blocks` times.

7. Output heads
- Main head predicts contact maps.
- Optional loop head predicts loop logits.

## Repository Layout

```text
.
|-- config_m2pp.py         # Model/training/data configuration presets
|-- dataset.py             # Real + synthetic datasets and dataloaders
|-- losses_m2pp.py         # Composite loss (L1 + Pearson + optional focal BCE)
|-- model_m2pp.py          # Core architecture
|-- train_m2pp.py          # Training entry point
|-- test_m2pp.py           # Checkpoint evaluation script
|-- test_run_m2pp.py       # Fast end-to-end validation on synthetic data
|-- QUICKSTART_M2PP.md     # Legacy quick start notes
|-- README_M2PP.md         # Legacy extended notes
```

## Data Layout
For real mode, data is expected under `matrices/`:

```text
matrices/
|-- X/
|   |-- cond01_rep1/
|   |   |-- window_0000_chr1_0_2000000.npy
|   |   `-- ...
|   |-- cond01_rep2/
|   |-- cond02_rep1/
|   `-- ...
`-- Y/
    |-- cond01_rep1/
    |   |-- window_0000_chr1_0_2000000.npy
    |   `-- ...
    `-- ...
```

File naming convention:
- `window_<idx>_<chrom>_<start>_<end>.npy`

Dataset behavior:
- Replicates are grouped by condition prefix (for example, `cond01_*`).
- By default, replicates are averaged per condition.
- DNA sequence is fetched from `genome_fasta_path` using window coordinates.

## Installation

### 1. Create and activate environment
```bash
conda create -n hicverse-m2pp python=3.10 -y
conda activate hicverse-m2pp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Notes:
- `mamba-ssm` is optional. If unavailable, model code falls back to a GRU-based SSM block.
- `matplotlib` and `seaborn` are only needed for plots.
- `tensorboard` is optional but recommended during training.

## Quick Start

### 1. Validate pipeline on synthetic data
```bash
python test_run_m2pp.py --n_conditions 6 --n_bins 50
```

### 2. Run a short synthetic training
```bash
python train_m2pp.py --mode synthetic --config test --epochs 2
```

### 3. Run real-data smoke test
```bash
python train_m2pp.py --mode real --config test --n_conditions 6 --epochs 5
```

## Training

### Full training
```bash
python train_m2pp.py --mode real --config full --n_conditions 6 --epochs 100
```

### Useful overrides
```bash
python train_m2pp.py --mode real --config full \
  --scheduler onecycle \
  --aux_dropout 0.15 \
  --n_hic_blocks 3 \
  --loss_map_weight 10.0 \
  --loss_pearson_weight 1.0
```

### Resume from checkpoint
```bash
python train_m2pp.py --mode real --config full --resume checkpoints_m2pp/hicverse_m2pp_best.pt
```

## Evaluation

Evaluate a saved checkpoint:

```bash
python test_m2pp.py checkpoints_m2pp/hicverse_m2pp_best.pt \
  --mode real \
  --n_samples 50 \
  --out_dir test_results_m2pp
```

Helpful flags:
- `--conditions 0 2 cond03` to limit evaluated conditions.
- `--compare_maps` to save predicted-vs-target panels.
- `--force_cpu` if GPU runtime mismatch appears.

## Configuration
Main settings live in `config_m2pp.py`.

Most commonly edited fields:
- Data paths: `x_dir`, `y_dir`, `genome_fasta_path`
- Model size: `d_model`, `n_hybrid_stacks`, `d_outer`
- Hi-C head: `ss2d_d_state`, `lefn_expansion`, `n_hic_blocks`
- Priors: `use_aux_features`, `use_ctcf_fimo_prior`, `use_tad_priors`
- Training: `learning_rate`, `batch_size`, `max_epochs`, `scheduler_type`
- Loss: `loss_map_weight`, `loss_pearson_weight`, `loss_loop_weight`

Portable path defaults:
- `x_dir=./matrices/X`
- `y_dir=./matrices/Y`
- FASTA can be set by either:
  - `cfg.genome_fasta_path`, or
  - environment variable `HICVERSE_GENOME_FASTA`

## Outputs
Training outputs:
- `./checkpoints_m2pp/` checkpoints
- `./hicverse_m2pp_output/logs/` TensorBoard events
- `./hicverse_m2pp_output/visualizations/` map visualizations
- `./checkpoints_m2pp/training_history.json` metric history

Evaluation outputs (default):
- `./test_results_m2pp/test_results.json`
- summary plots and optional contact-map comparisons

## Troubleshooting

- `Genome FASTA not found`:
  - Set `genome_fasta_path` in config or export `HICVERSE_GENOME_FASTA`.

- `mamba_ssm not installed`:
  - Install `mamba-ssm` if desired, or continue with GRU fallback.

- CUDA/cuDNN mismatch during evaluation:
  - Use `--force_cpu` or `--allow_cpu_fallback` in `test_m2pp.py`.

- Missing real-data files:
  - Verify `matrices/X`, `matrices/Y`, and filename convention.
