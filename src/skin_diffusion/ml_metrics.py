import numpy as np


def compute_curve_metrics(j_true, j_pred, t, eps=1e-12):
    # Curve metrics across all time points.
    diff = j_pred - j_true

    # Per-run integrated absolute error over each run's own time grid.
    iae_values = []
    for i in range(j_true.shape[0]):
        value = float(np.trapezoid(np.abs(diff[i]), x=t[i]))
        iae_values.append(value)

    # Pearson is computed on flattened matrices as one global agreement score.
    true_flat = j_true.ravel()
    pred_flat = j_pred.ravel()
    pearson_r = float("nan")
    if len(true_flat) > 1:
        pearson_r = float(np.corrcoef(true_flat, pred_flat)[0, 1])

    # Report split-level scalar summaries for parity with existing blackbox outputs.
    metrics = {}
    metrics["mae"] = float(np.mean(np.abs(diff)))
    metrics["rmse"] = float(np.sqrt(np.mean(diff**2)))
    metrics["relative_l2"] = float(np.linalg.norm(diff) / (np.linalg.norm(j_true) + eps))
    metrics["integrated_absolute_error"] = float(np.mean(iae_values))
    metrics["pearson_r"] = pearson_r
    return metrics
