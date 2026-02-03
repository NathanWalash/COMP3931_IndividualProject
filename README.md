# Skin Diffusion + ML Surrogates

Finite-difference skin diffusion solver with config-driven regimes (V1/V2/V3) plus ML surrogate stubs.

This README is a note-to-self list of what exists now and how to run it.

## V1 / V2 / V3 regimes

- V1: constant D, basic BCs, used for verification and benchmarks.
- V2: layered D(y) and optional clearance k(y), more physiological.
- V3: stress-test: patch geometry variations + heterogeneous D(x,y) + time-decaying donor.

## Quickstart (basic)

1) Install deps:
   - `pip install -r requirements.txt`
2) Run a script (example):
   - `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml`
3) If Python cannot see `src/`, set the path:
   - Windows PowerShell: `$env:PYTHONPATH="src"`

## Run commands

### 1) Core runner

`python -m scripts.sim.run_sim --config <path>`

Why:
- This is the one place to run a full sim from a config and save the standard outputs.
- used as the “baseline run” and a quick sanity check before I do any validation.

Options:
- `--config <path>`: which YAML config file to load (required).
- `--demo_step`: quick constant‑D single step demo (prints 5‑point stencil values).
- `--demo_bc`: quick boundary condition demo (prints top/bottom checks).
- `--print_meta`: prints the saved `meta.json` after the run.
- `--no_bc`: run a no-BC loop (debug test only).

Notes:
- This uses the shared solver for any config i.e. type of simulation passed (V1/V2/V3).
- Output goes to `outputs/sim/<regime>/meta.json` and `fields.npz`

Example commands:
- `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml`
- `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml --demo_step`
- `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml --demo_bc`
- `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml --print_meta`
- `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml --no_bc`

### 2) Validation scripts (save figures)

V1 validation (time series heatmaps + profiles + patch mask):
- `python -m scripts.sim.validate_v1 --config configs/sim/v1_baseline.yaml`

Why:
- V1 is the simplest case, so it is my first check to make sure simple fundamentals are right.
- The heatmaps and profiles give me a visual sanity check for diffusion and BCs.

V2 validation (D/k maps + time series heatmaps + profiles):
- `python -m scripts.sim.validate_v2 --config configs/sim/v2_layers_clearance.yaml`

Why:
- I want to see the layered D(y) and k(y) maps directly.
- I will use it to check that clearance and layer transitions actually show up.

V3 validation (patch width/offset variants + lateral profile + time series):
- `python -m scripts.sim.validate_v3 --config configs/sim/v3_hetero_patch_timeDecay.yaml`

Why:
- This is my “stress test” to confirm 2D effects and patch geometry changes.
- The lateral profile helps show that small patches spread sideways.

Outputs:
- figures: `figures/validation/v1`, `figures/validation/v2`, `figures/validation/v3`
- run data: `outputs/sim/<regime>/...`

### 3) Benchmarks (verification)

V1 self‑convergence (grid refinement):
- `python -m scripts.sim.benchmark_v1 --config configs/sim/v1_baseline.yaml`
  - outputs: `figures/validation/v1_convergence.png`
  - report: `outputs/sim/v1/benchmark/report.json`

Why:
- This is my quantitative evidence that the solver is behaving as expected.
- I will cite the convergence plot/report in the diss paper - i think.

V1 1D analytic placeholder (needs finishing):
- `python -m scripts.sim.benchmark_v1_1d --config configs/sim/v1_baseline.yaml`
  - report: `outputs/sim/v1/benchmark/report_1d.json`

Why:
- This is here so I can drop in the Crank (or other) series later and compare profiles for validation.

### 4) Dataset pipeline (V3 runs ? train/val/test)

Generate runs (each run is a folder with fields/meta/metrics):
- `python -m scripts.sim.make_dataset --config configs/sim/v3_hetero_patch_timeDecay.yaml --num_runs 5`

Options:
- `--num_runs <int>`: how many run folders to create.
- `--seed_start <int>`: optional start seed; if omitted, uses config seed.

Outputs per run:
- `outputs/sim/v3/dataset/run_###/fields.npz`
- `outputs/sim/v3/dataset/run_###/meta.json`
- `outputs/sim/v3/dataset/run_###/metrics.json`

Assemble processed datasets (split into train/val/test):
- `python -m scripts.sim.assemble_dataset --config configs/sim/v3_hetero_patch_timeDecay.yaml`

Options:
- `--out_dir <path>`: where to write processed datasets (default `data/processed`).
- `--split_seed <int>`: seed for split.
- `--train_frac <float>`: train fraction (default 0.8).
- `--val_frac <float>`: validation fraction (default 0.1).

Outputs:
- `data/processed/v3_train.npz`
- `data/processed/v3_val.npz`
- `data/processed/v3_test.npz`
- `data/processed/index.json`

### 5) Runtime profiling

Baseline runtime table:
- `python -m scripts.sim.profile_solver`

Outputs:
- `outputs/sim/profile/runtime.csv`

### 6) Clean up utility

Delete whole folders (no prompt):
- `python -m scripts.sim.clean_outputs --figures`
- `python -m scripts.sim.clean_outputs --outputs`
- `python -m scripts.sim.clean_outputs --data`

Optional subdir:
- `python -m scripts.sim.clean_outputs --outputs --subdir sim\\v3\\dataset`

### 7) Tests

Run tests:
- `python -m pytest -q`

## Config files

Main simulation configs:
- `configs/sim/v1_baseline.yaml`
- `configs/sim/v2_layers_clearance.yaml`
- `configs/sim/v3_hetero_patch_timeDecay.yaml`

Notes:
- `grid`: controls `H, W, dx, dt, T, save_every`
- `boundary`: donor patch + BCs
- `extras`: used for heterogeneity options

## Repo layout (quick map)

- `configs/`: YAML configs for simulation and ML
- `src/`: core code
- `scripts/`: runnable experiments
- `tests/`: unit tests
- `data/`: raw and processed data
- `outputs/`: simulation and ML outputs
- `figures/`: saved plots
