# Dataset Spec (run bundle -> processed -> ML export)

This document describes file formats and shapes across the dataset pipeline.

## Run bundle (one simulation)

Each run folder contains:

- `fields.npz`
  - `C_snap`: `[T, H, W]`
  - `D`: `[H, W]`
  - `k`: `[H, W]`
  - `patch_mask`: `[H, W]`
  - `t`: `[T]`
  - `J`: `[T]`
- `meta.json`
  - grid (`H`, `W`, `dx`, `dt`, `T`, `save_every`)
  - boundary settings (`mode`, `C0`, `decay_rate`, patch config, BC flags)
  - `seed`, `regime`, `extras`, stability info
- `metrics.json`
  - scalar summaries such as `P`, `Tlag`, `J_ss`, `AUC_J`, `J_peak`, `t_peak`, `M_delivered_24h`

## Processed dataset arrays

Saved in `data/processed/`:

- split arrays: `v3_train.npz`, `v3_val.npz`, `v3_test.npz`
- index metadata: `index.json`

Each split file stores stacked arrays with shape `[N, ...]`:

- `C_snap`: `[N, T, H, W]`
- `D`: `[N, H, W]`
- `k`: `[N, H, W]`
- `patch_mask`: `[N, H, W]`
- `t`: `[N, T]`
- `J`: `[N, T]`

Lightweight assembly mode:

- `scripts/sim/assemble_dataset.py --lightweight` writes only `t` and `J`.
- This is intended for large ML runs where full fields are not needed during training.

## ML export arrays

`scripts/sim/export_ml_dataset.py` writes split files plus `meta.json` in `data/processed/ml/`.

Each split file contains:

- `X`: feature matrix
- `y_scalar`: selected scalar targets
- `J`: flux curve
- `t`: time grid

Split naming note:

- Trainers consume logical splits (`train`, `val`, `test`).
- Split keys in `meta.json` must be exactly `train`, `val`, and `test`.

Feature columns in `X`:

- sampled inputs: `patch_width`, `patch_offset`, `C0`, `decay_rate`, `k_dermis`, `heterogeneity_sigma`, `heterogeneity_steps`
- derived terms: `dose_proxy_c0_over_decay`, `log_decay_rate`, `width_times_sigma`
- interactions: `c0_times_width`, `decay_times_width`, `sigma_times_width`
- D-field summaries: `D_mean`, `D_std`, `D_p10`, `D_p50`, `D_p90`, `D_top_mean`, `D_bottom_mean`

`meta.json` includes:

- `feature_names`
- `scalar_target_names`
- `scalar_target_source`
- split index rows with `run_dir`, `meta_path`, and `metrics_path`

## Split policy

- fractions: `train=0.7`, `val=0.15`, `test=0.15`
- deterministic split seed recorded in `index.json`
- `scikit-learn` split logic is used for reproducibility

## Placement policy (dataset v1)

- `patch_offset` values: `left`, `center`, `right`
- for `patch_width = 1.0`, offset is fixed to `center`
