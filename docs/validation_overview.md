# Validation And Trustworthiness Overview

This is a short map of what to run and what each step proves.

## Core validations

- V1 baseline validation (`scripts/sim/validate_v1.py`)
  - Proves: solver behaves correctly in the simplest setup.
  - Evidence: `figures/validation/v1`, `outputs/sim/v1/meta.json`.

- V2 layered diffusion (`scripts/sim/validate_v2.py`)
  - Proves: variable-D solver behaves as expected with layered structure.
  - Evidence: `figures/validation/v2`, `outputs/sim/v2/metrics.json`.

- V3 heterogeneity + patch geometry (`scripts/sim/validate_v3.py`)
  - Proves: 2D behavior and lateral diffusion are handled correctly.
  - Evidence: `figures/validation/v3`.

## Convergence / numerical correctness

- Grid convergence study (`scripts/sim/benchmark_v1.py`)
  - Proves: error decreases under refinement.
  - Evidence: `figures/validation/v1_convergence.png`,
    `outputs/sim/v1/benchmark/report.json`.

- Analytic 1D comparison (`scripts/sim/benchmark_v1_1d.py`)
  - Proves: solver matches a textbook diffusion solution.
  - Evidence: `outputs/sim/v1/benchmark/report_1d.json`.

## Literature validation

- Lidocaine Franz-cell comparison (`scripts/sim/compare_literature.py`)
  - Proves: model can match reported permeability/lag time within ~10-20%.
  - Evidence: console output + calibration notes in the YAML.

## Unit tests

- Run: `python -m pytest -q`
  - setup: `pip install -r requirements.txt`
  - Covers BCs, operators, metrics, and stability logic.
