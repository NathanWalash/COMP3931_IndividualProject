# Skin Diffusion + ML Surrogates

Finite-difference skin diffusion solver with regime-driven configs, plus ML surrogate models.

## Quickstart

1) Create a virtual environment (optional)
2) Install deps:
   - `pip install -r requirements.txt`
3) Run a script:
   - `python -m scripts.sim.run_sim --config configs/sim/v1_baseline.yaml`
   - If import error, set the Python path first:
     - Windows PowerShell: `$env:PYTHONPATH="src"`

## Repo layout

- `configs/`: YAML configs for simulation and ML
- `src/`: core code (source of truth)
- `scripts/`: runnable experiments
- `tests/`: unit tests
- `data/`: raw and processed data
- `outputs/`: simulation and ML outputs
- `figures/`: saved plots
