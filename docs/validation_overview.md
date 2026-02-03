# Validation And Trustworthiness Overview

## Core validations

- V1 baseline validation (scripts/sim/validate_v1.py)
  - What it does: Runs the simplest constant-D case with a top patch and bottom
    sink; saves heatmaps and depth profiles over time.
  - What it proves: The solver behaves correctly in the simplest setup; the
    patch drives diffusion downward, boundary conditions apply correctly.
  - Evidence: figures/validation/v1 and outputs/sim/v1/meta.json

- V2 layered diffusion validation (scripts/sim/validate_v2.py)
  - What it does: Uses layered D (SC/VE/dermis) and optional clearance k.
  - What it proves: The variable-D solver behaves as expected with layered
    structure; slower penetration through SC; k reduces concentration.
  - Evidence: figures/validation/v2, outputs/sim/v2/diagnostics.json,
    outputs/sim/v2/metrics.json

- V3 heterogeneity + patch geometry validation (scripts/sim/validate_v3.py)
  - What it does: Multiple patch widths/offsets and heterogeneous D fields.
  - What it proves: The solver works in full 2D; lateral diffusion behaves as
    expected; heterogeneity produces spatial variation.
  - Evidence: figures/validation/v3

## Convergence / numerical correctness

- Grid convergence study (scripts/sim/benchmark_v1.py)
  - What it does: Runs multiple grid sizes with consistent physical timing;
    compares coarse vs fine by block-averaging the fine grid to the coarse grid.
  - What it proves: Error decreases under refinement, showing convergence.
  - Evidence: figures/validation/v1_convergence.png and
    outputs/sim/v1/benchmark/report.json

- Analytic 1D comparison (Crank) (scripts/sim/benchmark_v1_1d.py)
  - What it does: Full-width patch to make the problem 1D in depth. Compares
    numeric depth profile to the finite-slab analytic series solution from Crank.
  - What it proves: The solver matches a textbook analytic solution for a
    canonical diffusion case.
  - Evidence: outputs/sim/v1/benchmark/report_1d.json (very small L2 errors)

## Physics sanity checks

- Stability checks (src/skin_diffusion/checks.py)
  - What it does: Computes diffusion and reaction time-step limits and warns
    when dt is too large.
  - What it proves: The run respects explicit stability limits.
  - Evidence: recorded in meta.json under "stability"

- Diagnostics: mass and minimum concentration
  - What it does: Tracks total mass and minimum C over time.
  - What it proves: Detects unphysical mass drift or negative values.
  - Evidence: outputs/sim/*/diagnostics.json

## Literature validation

- Lidocaine Franz-cell comparison (scripts/sim/compare_literature.py and
  configs/sim/v2_lidocaine_compare.yaml)
  - What it does: Calibrates layered D to match published permeability and
    lag time (Adamiak-Giera et al., 2023).
  - What it proves: The model can reproduce reported experimental permeation
    parameters within ~10–20%.
  - Evidence: compare_literature output and calibration notes in the YAML.

## Unit tests

- Run: python -m pytest -q
  - BC tests: patch, bottom sink, side Neumann
  - Operator tests: constant D vs var-D equivalence
  - Metrics tests: flux for zero/linear fields
  - Checks tests: stability and warning logic
  - What it proves: Core numerical building blocks behave as expected and do
    not regress.
