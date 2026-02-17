# Skin Diffusion + ML Surrogates

A small 2D finite-difference skin diffusion simulator with config-driven regimes (V1/V2/V3), validation/benchmark scripts, and a dataset pipeline for ML surrogates.

## What this repo does

- Simulates diffusion through skin with a top donor patch and bottom sink.
- Supports constant D (V1), layered D(y) with optional clearance k(y) (V2), and 2D patch geometry + heterogeneity + time-decay donor (V3).
- Produces figures, metrics (flux, permeability, lag time), and run bundles for ML datasets.

## Quickstart

1) Create and activate a virtual environment (recommended):
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
2) Install deps and package:
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`
   - `pip install -e .`
3) Run a baseline simulation:
   - `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml`

Notes:
- `pip install -e .` makes `skin_diffusion` importable without setting `PYTHONPATH`.
- `requirements.txt` includes both runtime and test dependencies.

Notebook kernel (VS Code/Jupyter):
- Register this venv as a kernel:
  - `python -m ipykernel install --user --name comp3931-venv --display-name "Python (.venv COMP3931)"`
- In the notebook kernel picker, choose `Python (.venv COMP3931)`.

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
- `python -m scripts.sim.make_dataset --config configs/sim/v3_literature_dataset_spec.yaml --num_runs 5`

Check run integrity (find broken/incomplete runs):
- `python -m scripts.sim.check_runs --config configs/sim/v3_literature_dataset_spec.yaml`
- optional run index slicing:
  - `python -m scripts.sim.check_runs --config configs/sim/v3_literature_dataset_spec.yaml --run_start_index 500 --run_end_index 999 --out_path outputs/qc/run_integrity_500_999.json`
- regenerate missing/corrupt runs in-place:
  - dry run:
    - `python -m scripts.sim.fix_runs --config configs/sim/v3_literature_dataset_spec.yaml --run_start_index 500 --run_end_index 999 --out_path outputs/qc/fix_runs_500_999.json`
  - apply fixes:
    - `python -m scripts.sim.fix_runs --config configs/sim/v3_literature_dataset_spec.yaml --run_start_index 500 --run_end_index 999 --apply --out_path outputs/qc/fix_runs_500_999_apply.json`
  - optional explicit seed mapping:
    - `python -m scripts.sim.fix_runs --config configs/sim/v3_literature_dataset_spec.yaml --run_start_index 500 --run_end_index 999 --seed_offset 1000 --apply`

Assemble ID/OOD splits:
- `python -m scripts.sim.assemble_dataset --config configs/sim/v3_literature_dataset_spec.yaml`
- low-memory option for large datasets:
  - `python -m scripts.sim.assemble_dataset --config configs/sim/v3_literature_dataset_spec.yaml --lightweight`
- optional run index slicing (inclusive):
  - `python -m scripts.sim.assemble_dataset --config configs/sim/v3_literature_dataset_spec.yaml --run_start_index 0 --run_end_index 499`

Outputs:
- `data/processed/id/v3_train.npz`
- `data/processed/id/v3_val.npz`
- `data/processed/id/v3_test.npz`
- `data/processed/ood/v3_ood_primary.npz`
- `data/processed/index.json`

Export ML-ready splits:
- `python -m scripts.sim.export_ml_dataset --processed_dir data/processed --out_dir data/processed/ml`

ML feature vector now includes:
- base sampled inputs (`patch_width`, `patch_offset`, `C0`, `decay_rate`, `heterogeneity_sigma`, `heterogeneity_steps`)
- simple derived terms (`C0/decay_rate`, `log(decay_rate)`, `patch_width*heterogeneity_sigma`)
- D-field summaries from each run (`mean/std/p10/p50/p90/top_mean/bottom_mean`)

Subset evaluation/training by run index (inclusive):
- QC:
  - `python -m scripts.sim.qc_dataset --processed_dir data/processed --out_dir outputs/qc/processed --run_start_index 0 --run_end_index 499`
- Train:
  - `python -m scripts.ml.train_blackbox --ml_dir data/processed/ml --out_dir outputs/ml/blackbox --run_start_index 0 --run_end_index 499`

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

- `pip install -r requirements.txt`
- `python -m pytest -q`


## Notebooks

The `notebooks/` folder provides runnable, narrative walkthroughs:

- `01_quickstart_simulation.ipynb`: baseline run and inspection
- `02_validation_v1_v2_v3.ipynb`: visual validation for V1/V2/V3
- `03_convergence_and_1d_benchmark.ipynb`: grid refinement + analytic check
- `04_literature_compare_lidocaine.ipynb`: compare to literature targets
- `05_dataset_pipeline.ipynb`: build a small dataset and inspect splits

## Configs

Main simulation configs:
- `configs/sim/v1_baseline.yaml`
- `configs/sim/v2_layers_clearance.yaml`
- `configs/sim/v3_hetero_patch_timeDecay.yaml`
- `configs/sim/v2_lidocaine_compare.yaml` (literature compare)
- `configs/sim/v3_layers_literature.yaml`
- `configs/sim/v3_literature_dataset_spec.yaml` (dataset spec)

Key sections:
- `grid`: `H, W, dx, dt, T, save_every`
- `boundary`: donor patch settings and BCs
- `layers`: per-layer D and clearance (V2)
- `heterogeneity`: IID or correlated D noise (V3)
- `literature`: target permeability/lag time (compare script)

`boundary.patch_offset` supports:
- `left`, `center`, `right`

For dataset generation (`v3_literature_dataset_spec.yaml`), patch placement is
sampled from this discrete set to keep the input space simple and consistent.

## Docs

- `docs/metrics_process_overview.md`: pipeline summary and key metrics
- `docs/validation_overview.md`: validation/benchmark map
- `docs/validation_methods.md`: methods and evidence for the report
- `docs/figures_guide.md`: how to interpret figures
- `docs/dataset_spec.md`: run bundle + processed dataset schema
- `docs/outputs_guide.md`: outputs and where they are written
- `docs/config_guide.md`: config fields and options
- `docs/notebooks_guide.md`: what each notebook demonstrates
- `docs/tests_overview.md`: unit test summary
