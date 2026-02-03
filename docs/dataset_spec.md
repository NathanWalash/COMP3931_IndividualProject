# Dataset Spec (run-level -> processed)

## Run-level bundle (one simulation)

Each run folder contains:

- `fields.npz`
  - `C_snap` shape: `[T, H, W]`
  - `D` shape: `[H, W]`
  - `k` shape: `[H, W]`
  - `patch_mask` shape: `[H, W]`
  - `t` shape: `[T]`
  - `J` shape: `[T]`

- `meta.json`
  - grid info: `H`, `W`, `dx`, `dt`, `T`, `save_every`
  - boundary info: mode, `C0`, `decay_rate`, patch params, BC flags
  - `seed`, `regime`, `extras`
  - stability info (dt limits)

- `metrics.json`
  - `P`, `Tlag`, `J_ss`
  - `J_min`, `J_max`, `J_mean`, `J_sum`

---

## Processed datasets (train/val/test)

Saved in `data/processed/`:

- `v3_train.npz`
- `v3_val.npz`
- `v3_test.npz`

Each file stores stacked arrays with shape `[N, ...]`:

- `C_snap` shape: `[N, T, H, W]`
- `D` shape: `[N, H, W]`
- `k` shape: `[N, H, W]`
- `patch_mask` shape: `[N, H, W]`
- `t` shape: `[N, T]`
- `J` shape: `[N, T]`

`index.json` maps each row to its run metadata path.

---

## Deterministic splits

The split uses a fixed random seed:

- Train fraction: 0.8
- Val fraction: 0.1
- Test fraction: 0.1

The split seed is stored in `index.json`.

Splits are made with `scikit-learn` so the shuffle is repeatable.
