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
  - `J_min`, `J_max`, `J_mean`, `J_sum`

Notes:
- Runs created with `scripts/sim/run_sim.py` or the validation scripts may
  include extra files such as `diagnostics.json` and `bc.json`.

---

## Processed datasets (ID + OOD)

Saved in `data/processed/`:

- ID split folder:
  - `id/v3_train.npz`
  - `id/v3_val.npz`
  - `id/v3_test.npz`
- OOD folder:
  - `ood/v3_ood_primary.npz`
- Legacy compatibility copies are also saved at root:
  - `v3_train.npz`, `v3_val.npz`, `v3_test.npz`

Each file stores stacked arrays with shape `[N, ...]`:

- `C_snap` shape: `[N, T, H, W]`
- `D` shape: `[N, H, W]`
- `k` shape: `[N, H, W]`
- `patch_mask` shape: `[N, H, W]`
- `t` shape: `[N, T]`
- `J` shape: `[N, T]`

`index.json` stores split settings, counts, and index maps.

---

## Deterministic splits

The split uses a fixed random seed:

- ID train fraction: 0.7
- ID val fraction: 0.15
- ID test fraction: 0.15
- OOD holdout: `patch_width = 0.25`

The split seed is stored in `index.json`.
Splits are made with `scikit-learn` so the shuffle is repeatable.
