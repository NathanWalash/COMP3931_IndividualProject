# Report Results Reproducibility (Final ML Runs)

This document defines the reproducible workflow for the dissertation ML comparison using the active physics-corrected surrogate code path.

All report-facing artifacts are read directly from:

- `outputs/ml/final_comparison/blackbox_ridge`
- `outputs/ml/final_comparison/corrective_ridge_seed42`
- `outputs/ml/final_comparison/corrective_ridge_seed7`
- `outputs/ml/final_comparison/corrective_ridge_seed123`

Report artifacts are consumed directly from these `outputs/ml/final_comparison/*` directories.

## 1) Environment

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Notes:

- Python: `>=3.12` (from `pyproject.toml`)
- Final parity runs were executed on `cuda`.

## 2) Locked Dataset/Config Basis

Dataset and split policy used by report-final runs:

- Dataset spec: `configs/sim/v3_literature_dataset_spec_clearance.yaml`
- Base sim config: `configs/sim/v3_layers_literature_clearance.yaml`
- Split seed: `321`
- Split fractions: `train=0.70`, `val=0.15`, `test=0.15`
- Final split counts: `700 / 150 / 150`

Expected input artifacts:

- `data/processed/index.json`
- `data/processed/ml/meta.json`
- `data/processed/ml/train.npz`
- `data/processed/ml/val.npz`
- `data/processed/ml/test.npz`

Optional full regeneration:

```bash
python -m scripts.sim.assemble_dataset \
  --config configs/sim/v3_literature_dataset_spec_clearance.yaml \
  --out_dir data/processed \
  --split_seed 321 \
  --train_frac 0.7 \
  --val_frac 0.15

python -m scripts.sim.export_ml_dataset \
  --processed_dir data/processed \
  --out_dir data/processed/ml
```

## 3) Reproducible Training Commands

Blackbox baseline:

```bash
python -m scripts.ml.train_blackbox \
  --ml_dir data/processed/ml \
  --out_dir outputs/ml/final_comparison/blackbox_ridge \
  --curve_components 20 \
  --alpha_grid 0.001,0.01,0.1,1,10,100
```

Physics-corrected surrogate (seed 42):

```bash
python -m scripts.ml.train_corrective \
  --ml_dir data/processed/ml \
  --out_dir outputs/ml/final_comparison/corrective_ridge_seed42 \
  --device cuda \
  --seed 42 \
  --max_train_rows 700 \
  --max_val_rows 150 \
  --max_test_rows 150 \
  --backbone_curve_components 20 \
  --backbone_alphas 1e-4,1e-3,1e-2,1e-1,1,10 \
  --epochs 800 \
  --eval_every 5 \
  --lr 0.0003 \
  --batch_runs 48 \
  --predict_batch_runs 48 \
  --hidden_dim 128 \
  --depth 4 \
  --correction_scale 0.5 \
  --initial_correction_scale 0.3 \
  --use_learned_gate \
  --gate_init_bias 0.0 \
  --flux_warmup_epochs 1 \
  --flux_ramp_epochs 2 \
  --flux_points 2048 \
  --weight_flux 10.0 \
  --weight_anchor 0.05 \
  --weight_nonneg 0.02 \
  --weight_peak 3.0 \
  --weight_gate_entropy 0.05 \
  --flux_log_scale 1e11 \
  --worst_case_top_n 10
```

Repeat for other report seeds:

- `--seed 7 --out_dir outputs/ml/final_comparison/corrective_ridge_seed7`
- `--seed 123 --out_dir outputs/ml/final_comparison/corrective_ridge_seed123`

`train_corrective` now uses additive correction only (multiplicative mode removed).

## 4) Expected Metrics

Report-reference test metrics (`stageFinal`):

| Run ID | relative_l2 | pearson_r |
|---|---:|---:|
| blackbox_ridge | 0.0788934931 | 0.9967120791 |
| corrective_ridge_seed42 | 0.0487797596 | 0.9991678188 |
| corrective_ridge_seed7 | 0.0506992266 | 0.9990412214 |
| corrective_ridge_seed123 | 0.0500499271 | 0.9991374296 |

Current-code verification (April 16, 2026):

- `seed 42` rerun with trimmed code path produced:
  - `relative_l2 = 0.0487319566`
  - `pearson_r = 0.9991747864`
- This is within normal numerical drift vs report seed42 (`~0.098%` relative_l2 difference).

## 5) Output-Only Sanity Check

Use this to print the report-critical metrics directly from `outputs/ml/final_comparison`:

```bash
python - <<'PY'
import json
from pathlib import Path

runs = [
    "blackbox_ridge",
    "corrective_ridge_seed42",
    "corrective_ridge_seed7",
    "corrective_ridge_seed123",
]
root = Path("outputs/ml/final_comparison")
for run in runs:
    s = json.loads((root / run / "summary.json").read_text())
    rel = s.get("test_relative_l2")
    p = s.get("test_pearson_r")
    print(f"{run}: relative_l2={rel:.10f}, pearson_r={p:.10f}")
PY
```

## 6) Current Naming/Code State

- Active trainer: `python -m scripts.ml.train_corrective`
- Corrective checkpoint name: `corrective_model.pt`
- Corrective stage key in predictions/metrics: `stageCorrected`
- Summary config block: `corrective_config`
