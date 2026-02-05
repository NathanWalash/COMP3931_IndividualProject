# Skin Diffusion + ML Surrogates

A small 2D finite-difference skin diffusion simulator with config-driven regimes (V1/V2/V3), validation/benchmark scripts, and a dataset pipeline for ML surrogates.

## What this repo does

- Simulates diffusion through skin with a top donor patch and bottom sink.
- Supports constant D (V1), layered D(y) with optional clearance k(y) (V2), and 2D patch geometry + heterogeneity + time-decay donor (V3).
- Produces figures, metrics (flux, permeability, lag time), and run bundles for ML datasets.

## Quickstart

1) Install deps:
   - `pip install -r requirements.txt`
2) Run a baseline simulation:
   - `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml`
3) If Python cannot see `src/`, set the path:
   - Windows PowerShell: `$env:PYTHONPATH="src"`

## Repo structure (high level)

- `src/`: core solver and utilities (`skin_diffusion`)
- `scripts/`: runnable entry points (simulation, validation, benchmarks)
- `configs/`: YAML configs for the regimes
- `docs/`: plain-English explanations
- `tests/`: unit tests for core math
- `outputs/`: saved run bundles and reports
- `figures/`: saved validation/benchmark plots
- `data/`: processed datasets for ML

## Core commands

### 1) Run a simulation

`python -m scripts.sim.run_sim --config <path>`

Examples:
- `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml`
- `python -m scripts.sim.run_sim --config configs/sim/v2_layers_clearance.yaml`
- `python -m scripts.sim.run_sim --config configs/sim/v3_hetero_patch_timeDecay.yaml`

Options:
- `--demo_step`: prints a constant-D stencil demo.
- `--demo_bc`: prints a quick BC sanity check.
- `--print_meta`: prints `meta.json` after the run.
- `--no_bc`: runs a no-BC loop (debug only).

Outputs (per run):
- `fields.npz` with `C_snap`, `D`, `k`, `patch_mask`, `t`, `J`
- `meta.json` with grid/boundary/stability info
- `metrics.json` with `P`, `Tlag`, `J_ss` and flux stats

### 2) Validation figures

- V1: `python -m scripts.sim.validate_v1 --config configs/sim/v1_baseline.yaml`
- V2: `python -m scripts.sim.validate_v2 --config configs/sim/v2_layers_clearance.yaml`
- V3: `python -m scripts.sim.validate_v3 --config configs/sim/v3_hetero_patch_timeDecay.yaml`

Outputs:
- `figures/validation/v1`, `figures/validation/v2`, `figures/validation/v3`
- run bundles in `outputs/sim/<regime>/...`

### 3) Benchmarks (verification)

- Grid refinement: `python -m scripts.sim.benchmark_v1 --config configs/sim/v1_baseline.yaml`
  - `figures/validation/v1_convergence.png`
  - `outputs/sim/v1/benchmark/report.json`
- 1D analytic comparison: `python -m scripts.sim.benchmark_v1_1d --config configs/sim/v1_baseline.yaml`
  - `outputs/sim/v1/benchmark/report_1d.json`

### 4) Literature comparison (lidocaine example)

`python -m scripts.sim.compare_literature --config configs/sim/v2_lidocaine_compare.yaml`

### 5) Dataset pipeline (V3)

Generate run folders:
- `python -m scripts.sim.make_dataset --config configs/sim/v3_hetero_patch_timeDecay.yaml --num_runs 5`

Assemble train/val/test:
- `python -m scripts.sim.assemble_dataset --config configs/sim/v3_hetero_patch_timeDecay.yaml`

Outputs:
- `data/processed/v3_train.npz`
- `data/processed/v3_val.npz`
- `data/processed/v3_test.npz`
- `data/processed/index.json`

### 6) Runtime profiling

- `python -m scripts.sim.profile_solver`
  - `outputs/sim/profile/runtime.csv`

### 7) Clean generated folders

- `python -m scripts.sim.clean_outputs --figures`
- `python -m scripts.sim.clean_outputs --outputs`
- `python -m scripts.sim.clean_outputs --data`

Optional subdir:
- `python -m scripts.sim.clean_outputs --outputs --subdir sim\v3\dataset`

### 8) Tests

- `python -m pytest -q`

## Configs

Main simulation configs:
- `configs/sim/v1_baseline.yaml`
- `configs/sim/v2_layers_clearance.yaml`
- `configs/sim/v3_hetero_patch_timeDecay.yaml`
- `configs/sim/v2_lidocaine_compare.yaml` (literature compare)

Key sections:
- `grid`: `H, W, dx, dt, T, save_every`
- `boundary`: donor patch settings and BCs
- `layers`: per-layer D and clearance (V2)
- `heterogeneity`: IID or correlated D noise (V3)
- `literature`: target permeability/lag time (compare script)

## Docs

- `docs/metrics_process_overview.md`: pipeline summary and key metrics
- `docs/validation_overview.md`: validation/benchmark map
- `docs/validation_methods.md`: methods and evidence for the report
- `docs/figures_guide.md`: how to interpret figures
- `docs/dataset_spec.md`: run bundle + processed dataset schema
- `docs/outputs_guide.md`: outputs and where they are written
- `docs/config_guide.md`: config fields and options
- `docs/tests_overview.md`: unit test summary
