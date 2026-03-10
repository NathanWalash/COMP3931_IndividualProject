# Validation And Trustworthiness Overview

This note combines the validation map and method details in one place.

## 1) V1 baseline validation

Script: `scripts/sim/validate_v1.py`  
Config: `configs/sim/v1_baseline.yaml`

Purpose:
- Sanity check for the simplest constant-D case.
- Confirm boundary conditions are applied correctly.
- Show expected diffusion from patch into depth.

Evidence:
- `figures/validation/v1/*` (heatmaps + depth profiles)
- `outputs/sim/v1/meta.json` (run config + stability limits)

## 2) V2 layered diffusion validation

Script: `scripts/sim/validate_v2.py`  
Config: `configs/sim/v2_layers_clearance.yaml`

Purpose:
- Check layered D (SC / VE / dermis).
- Show slower transport through SC.
- Show dermal clearance impact when k is enabled.

Evidence:
- `figures/validation/v2/*` (D map, k map, heatmaps, profiles)
- `outputs/sim/v2/metrics.json`

## 3) V3 2D patch + heterogeneity validation

Script: `scripts/sim/validate_v3.py`  
Config: `configs/sim/v3_hetero_patch_timeDecay.yaml`

Purpose:
- Test full 2D behavior (patch widths/offsets).
- Confirm lateral diffusion and heterogeneity effects.

Evidence:
- `figures/validation/v3/*` (per-case heatmaps + lateral profiles)

## 4) Convergence (grid refinement)

Script: `scripts/sim/benchmark_v1.py`

Method:
- Run multiple grids (e.g., 16/32/64/128).
- Keep physical time and stability scaling consistent.
- Compare coarse vs fine using block-averaged restriction.

What it proves:
- Errors decrease under refinement (numerical convergence).

Evidence:
- `figures/validation/v1_convergence.png`
- `outputs/sim/v1/benchmark/report.json`

## 5) Analytic 1D comparison (Crank)

Script: `scripts/sim/benchmark_v1_1d.py`

Method:
- Use full-width patch so the setup is 1D in depth.
- Compare numeric depth profile to finite-slab analytic series solution.

What it proves:
- Solver matches textbook diffusion behavior.

Evidence:
- `outputs/sim/v1/benchmark/report_1d.json`

## 6) Literature validation (lidocaine)

Script: `scripts/sim/compare_literature.py`  
Config: `configs/sim/v2_lidocaine_compare.yaml`

Method:
- Simulate layered case matching experiment assumptions.
- Compare simulated `P` and `Tlag` to paper targets.

What it proves:
- The calibration setup can be benchmarked directly against literature
  permeability and lag-time targets; exact deltas depend on the selected
  parameterization and should be reported from the run output.

Evidence:
- compare script output
- calibration notes in YAML
- `notebooks/04_literature_calibration_lidocaine.ipynb`

## 7) Unit tests

- Run: `python -m pytest -q`
- Setup: `pip install -r requirements.txt`
- Covers BCs, operators, metrics, dataset assembly, and stability logic.
