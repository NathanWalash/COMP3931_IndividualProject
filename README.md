# Skin Diffusion + ML Surrogates

Finite-difference skin diffusion solver with regime-driven configs, plus ML surrogate models.

## Quickstart

1) Create a virtual environment (optional)
2) Install deps:
   - `pip install -r requirements.txt`
3) Run a script (placeholder until Commit 1+):
   - `python scripts/sim/run_sim.py --config configs/sim/v1_baseline.yaml`

## Repo layout

- `configs/`: YAML configs for simulation and ML
- `src/`: core code (source of truth)
- `scripts/`: runnable experiments
- `tests/`: unit tests
- `data/`: raw and processed data
- `outputs/`: simulation and ML outputs
- `figures/`: saved plots
