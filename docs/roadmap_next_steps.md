# Project Specification And Execution Plan

Specification status: locked for implementation phase (can be revised only by explicit decision change).

## Current Status Snapshot

1. Black-box training/evaluation pipeline is in place with scalar and curve plots.
2. PINN training and evaluation scripts are in place, and PINN evaluation now writes the same curve plot types as black-box.
3. Main remaining comparison gap is unified PINN scalar parity/residual outputs for direct side-by-side reporting.

## 1. Purpose

1. Define the final implementation plan for the modelling phase.
2. Deliver two core modelling tracks:
   1. Supervised black-box surrogate.
   2. Physics-Informed Neural Network (PINN).
3. Compare both models on the same primary targets.

## 2. Project Positioning (Important Note)

1. The simulator is the ground-truth generator for ML/PINN training and testing.
2. The lidocaine literature case is used as an external plausibility anchor.
3. Main dataset variability is model-space variability (simulation-domain variability).
4. Dissertation claim boundary:
   1. Strong claim: surrogate/PINN emulate the validated simulator.
   2. Limited realism claim: simulator outputs are pharmacokinetically plausible based on literature alignment, not direct clinical prediction.

## 3. Core Targets And Deliverables

1. Primary outputs for both black-box and PINN:
   1. `J(t)` curve.
   2. Scalar metrics: `P`, `J_ss`, `AUC_J`, `J_peak`, `t_peak`, `M_delivered_24h`.
2. Secondary scalar (reported but not primary for finite-dose runs):
   1. `Tlag`.
3. Full-field output:
   1. One focused demonstration run for side-by-side visual comparison in dissertation (simulator vs black-box vs PINN).
4. Required comparison dimensions:
   1. Accuracy.
   2. Physical consistency.
   3. Compute cost.

## 4. Final Dataset V1 Design (Locked)

1. Primary regime for ML/PINN: V3.
2. Grid and time settings are fixed to current V3 baseline discretization.
3. Heterogeneity mode for v1:
   1. correlated only (primary mode).
4. Noise policy for v1:
   1. input-side variability enabled,
   2. output/measurement noise disabled.
5. All implementation should read these locked settings from a dedicated dataset-v1 config/spec file.

## 5. Parameter Ranges (Chosen For Dataset V1)

1. `patch_width`:
   1. discrete set: `[0.25, 0.50, 1.00]`.
2. `patch_offset`:
   1. allowed values: `left`, `center`, `right`,
   2. for `patch_width = 1.00`, use `center` only.
3. `C0`:
   1. uniform range: `[0.8, 1.2]`.
4. `decay_rate`:
   1. uniform range: `[0.05, 0.30]`.
5. correlated heterogeneity `sigma`:
   1. uniform range: `[0.01, 0.08]`.
6. correlated smoothing `steps`:
   1. integer range: `[3, 9]`.
7. heterogeneity seed:
   1. varied per run (deterministic from run index/seed policy).

## 6. Split Strategy (Chosen)

1. ID split:
   1. `70/15/15` train/val/test,
   2. random by run with fixed split seed.
2. Required OOD split:
   1. hold out `patch_width = 0.25` completely from training,
   2. evaluate both black-box and PINN on this held-out condition.
3. Optional second OOD split:
   1. hold out highest `sigma` band,
   2. include only if time/compute allows.

## 7. Dataset Size Tiers (Progressive Execution)

1. Laptop tier:
   1. pilot: `100` runs,
   2. full: `400` to `800` runs.
2. Lab PC tier (RTX 4070):
   1. pilot: `200` runs,
   2. full: `1,500` to `4,000` runs.
3. Cluster tier:
   1. pilot: `300` runs,
   2. full: `8,000` to `20,000+` runs.
4. Execution rule:
   1. complete pilot first,
   2. scale up only after pilot QC and baseline model sanity checks pass.

## 8. Evaluation Metrics (Chosen)

### 8.1 Curve-Level Metrics (`J(t)`)

1. MAE over time samples.
2. RMSE over time samples.
3. Relative L2 error on full curve.
4. Integrated absolute error (area between curves).
5. Pearson correlation between predicted and reference `J(t)`.

### 8.2 Scalar Metrics (`P`, `J_ss`, `AUC_J`, `J_peak`, `t_peak`, `M_delivered_24h`, optional `Tlag`)

1. MAE.
2. RMSE.
3. Relative error (%), with safe epsilon handling for near-zero denominators.
4. `R^2` for regression quality.

### 8.3 PINN Physics Metrics

1. PDE residual magnitude (mean and distribution).
2. Boundary-condition violation magnitude.
3. Initial-condition violation magnitude.
4. Non-negativity violation rate (if applicable).

### 8.4 Compute Metrics

1. Training wall-clock time.
2. Inference time per sample.
3. Throughput (samples per second).
4. Effective speedup vs simulator generation time.

## 9. Required Figures And Tables

1. `J(t)` overlays: simulator vs black-box vs PINN.
2. `J(t)` error-over-time plots.
3. Scatter plots for primary scalars (and optional `Tlag`) (predicted vs simulator).
4. Residual histograms for scalar metrics.
5. ID vs OOD comparison table.
6. Runtime/compute comparison table.
7. One full-field side-by-side figure (single representative run).

## 10. Execution Phases

### Phase 1: Dataset Pilot And QC

1. Generate pilot dataset on selected tier.
2. Validate run bundles and metric distributions.
3. Validate ID and OOD split correctness.

### Phase 2: Black-Box Baseline

1. Train baseline black-box model(s) on primary targets.
2. Evaluate on ID and OOD.
3. Generate baseline figures and metrics tables.

### Phase 3: PINN Development

1. Train PINN on aligned targets.
2. Evaluate on ID and OOD.
3. Produce physics-consistency metrics and plots.

### Phase 4: Comparative Reporting

1. Produce side-by-side black-box vs PINN comparison outputs.
2. Add single full-field demonstration comparison run.
3. Finalize dissertation-ready narrative and evidence.

## 11. Acceptance Criteria

1. Black-box and PINN both evaluated on same primary targets.
2. ID and required OOD results reported with fixed seeds/splits.
3. Accuracy, physics, and compute comparisons are complete.
4. Full-field demonstration figure is included (single representative run).
5. Conclusions clearly separate:
   1. simulator-emulation claims,
   2. pharmacokinetic plausibility claims.
