# Validation Methods

This document explains the validation methods used in the project and what each
method demonstrates. It is written as a short note-to-self for the report.

## 1) V1 baseline validation

Script: `scripts/sim/validate_v1.py`
Config: `configs/sim/v1_baseline.yaml`

Purpose:
- Sanity check for the simplest constant-D case.
- Confirms boundary conditions are applied correctly.
- Shows expected diffusion from the patch into depth.

Evidence:
- `figures/validation/v1/*` (heatmaps + depth profiles)
- `outputs/sim/v1/meta.json` (run config + stability limits)

## 2) V2 layered diffusion validation

Script: `scripts/sim/validate_v2.py`
Config: `configs/sim/v2_layers_clearance.yaml`

Purpose:
- Checks layered D (SC / VE / dermis).
- Shows slower transport through SC.
- Optional clearance (k) reduces deeper concentrations.

Evidence:
- `figures/validation/v2/*` (D map, k map, heatmaps, profiles)
- `outputs/sim/v2/metrics.json`

## 3) V3 2D patch + heterogeneity validation

Script: `scripts/sim/validate_v3.py`
Config: `configs/sim/v3_hetero_patch_timeDecay.yaml`

Purpose:
- Tests full 2D behavior (patch offsets + widths).
- Confirms lateral diffusion and heterogeneity effects.

Evidence:
- `figures/validation/v3/*` (per-case heatmaps + lateral profiles)

## 4) Convergence (grid refinement)

Script: `scripts/sim/benchmark_v1.py`

Method:
- Run multiple grids (e.g., 16/32/64/128).
- Keep physical time and stability scaling consistent.
- Compare coarse vs fine using block-averaged restriction.

What it proves:
- Errors decrease under refinement -> numerical convergence.

Evidence:
- `figures/validation/v1_convergence.png`
- `outputs/sim/v1/benchmark/report.json` (full error curves + summary)

## 5) Analytic 1D comparison (Crank)

Script: `scripts/sim/benchmark_v1_1d.py`

Method:
- Full-width patch so the problem is 1D in depth.
- Compare numeric depth profile to the analytic finite-slab series solution.

What it proves:
- Solver matches a textbook analytic solution.

Evidence:
- `outputs/sim/v1/benchmark/report_1d.json`
- Errors are tiny for the baseline config.

## 6) Literature comparison (lidocaine example)

Script: `scripts/sim/compare_literature.py`
Config: `configs/sim/v2_lidocaine_compare.yaml`

Method:
- Simulate a layered skin case matching the experiment.
- Compare simulated permeability (P) and lag time (Tlag) to the paper.
- Calibration details are recorded in the YAML.

What it proves:
- Model can reproduce reported permeation parameters within ~10-20%.

Evidence:
- Console output from `compare_literature`.
- Calibration notes in `configs/sim/v2_lidocaine_compare.yaml`.
