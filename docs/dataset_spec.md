# Dataset Spec (run bundle -> processed)

This describes the files produced by dataset runs and the shapes you should
expect when loading them.

## Run bundle (one simulation)

Each run folder contains:

- `fields.npz`
  - `C_snap` shape: `[T, H, W]`
  - `D` shape: `[H, W]`
  - `k` shape: `[H, W]`
  - `patch_mask` shape: `[H, W]`
  - `t` shape: `[T]` (saved times aligned with `C_snap`)
  - `J` shape: `[T]` (bottom flux at saved times)

- `meta.json`
  - grid info: `H`, `W`, `dx`, `dt`, `T`, `save_every`
  - boundary info: mode, `C0`, `decay_rate`, patch params, BC flags
  - `seed`, `regime`, `extras`
  - stability info (dt limits)

- `metrics.json`
  - `P`, `Tlag`, `J_ss`
  - finite-dose scalars: `AUC_J`, `J_peak`, `t_peak`, `M_delivered_24h`
  - `J_min`, `J_max`, `J_mean`, `J_sum`

Notes:
- Runs created with `scripts/sim/run_sim.py` or the validation scripts may
  include extra files such as `diagnostics.json` and `bc.json`.

---

## Processed datasets (ID + optional OOD)

Saved in `data/processed/`:

- ID split folder:
  - `id/v3_train.npz`
  - `id/v3_val.npz`
  - `id/v3_test.npz`
- OOD folder (optional):
  - `ood/v3_ood_primary.npz` (written only when OOD is enabled and non-empty)
- Legacy compatibility copies are also saved at root:
  - `v3_train.npz`, `v3_val.npz`, `v3_test.npz`

Each file stores stacked arrays with shape `[N, ...]`:

- `C_snap` shape: `[N, T, H, W]`
- `D` shape: `[N, H, W]`
- `k` shape: `[N, H, W]`
- `patch_mask` shape: `[N, H, W]`
- `t` shape: `[N, T]`
- `J` shape: `[N, T]`

Lightweight assemble mode:
- If `scripts/sim/assemble_dataset.py` is run with `--lightweight`,
  split files store only `t` and `J`.
- This mode is intended for large-run ML workflows where full fields are
  not needed during assembly/export/training.

`index.json` stores split settings, counts, and index maps.

Important:
- `assemble_dataset` reads runs from the `output_root` inside the spec/config
  you pass in.
- Use the same spec for `make_dataset`, `check_runs`, `fix_runs`, and
  `assemble_dataset` to avoid pointing at the wrong run folder.

---

## ML-ready export

`scripts/sim/export_ml_dataset.py` creates:
- `id_train.npz`, `id_val.npz`, `id_test.npz`
- optional: `ood_primary.npz` (only when OOD exists in processed inputs)
- each file contains:
  - `X`: tabular features
  - `y_scalar`: scalar targets selected from dataset-spec `targets.primary`
    (e.g. `P`, `J_ss`, `AUC_J`, `J_peak`, `t_peak`, `M_delivered_24h`)
  - `J`: flux curve
  - `t`: time grid

Feature columns in `X`:
- `patch_width`
- `patch_offset`
- `C0`
- `decay_rate`
- `k_dermis`
- `heterogeneity_sigma`
- `heterogeneity_steps`
- `dose_proxy_c0_over_decay`
- `log_decay_rate`
- `width_times_sigma`
- `c0_times_width`
- `decay_times_width`
- `sigma_times_width`
- `D_mean`
- `D_std`
- `D_p10`
- `D_p50`
- `D_p90`
- `D_top_mean`
- `D_bottom_mean`

Split behavior during export:
- `export_ml_dataset` reads assembled ID arrays and then rebuilds
  `id_train/id_val/id_test` with input-only stratification
  (`patch_width`, `patch_offset`) while keeping the same split sizes.
- Scalar target policy is read from `data/processed/index.json`:
  - if `dataset_spec.scalar_primary` exists, that list is used
  - otherwise export falls back to `[P, Tlag, J_ss]` for compatibility
- Export writes `meta.json` with:
  - `feature_names`
  - `scalar_target_names`
  - `scalar_target_source` (for example: `dataset_spec.primary`)
  - row-level split index mapping (`run_dir`, `meta_path`, `metrics_path`)

---

## Deterministic splits

The split uses a fixed random seed:

- ID train fraction: 0.7
- ID val fraction: 0.15
- ID test fraction: 0.15
- OOD holdout default: `patch_width = 0.25` when OOD is enabled
- You can disable OOD with `--no_ood`
- You can change OOD settings with `--ood_param` and `--ood_value`

The split seed is stored in `index.json`.
Splits are made with `scikit-learn` so the shuffle is repeatable.

## Placement policy (dataset v1)

- `patch_offset` is sampled from a discrete set: `left`, `center`, `right`.
- If `patch_width = 1.0`, offset is fixed to `center`.
