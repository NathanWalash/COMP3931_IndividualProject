from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from skin_diffusion.metrics import (
    compute_finite_dose_scalars,
    compute_permeability,
    estimate_lag_time,
    estimate_steady_state_flux,
)


def rel_percent_error(y_true, y_pred, eps=1e-12):
    # Mean relative error in percent.
    denom = np.abs(y_true) + eps
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / denom))


def safe_r2(y_true, y_pred):
    # Return NaN instead of hard errors for degenerate inputs.
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(np.var(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def compute_scalar_metrics(y_true, y_pred):
    # Scalar regression metrics for one target vector.
    metrics = {}
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["r2"] = safe_r2(y_true, y_pred)
    metrics["relative_error_percent"] = rel_percent_error(y_true, y_pred)
    return metrics


def scalar_report(y_true, y_pred, target_names):
    # Per-target report with explicit accounting for invalid predictions.
    report = {}
    for i in range(len(target_names)):
        name = target_names[i]
        target_mask = np.isfinite(y_true[:, i])
        pred_mask = np.isfinite(y_pred[:, i])
        valid_mask = target_mask & pred_mask

        target_n = int(np.sum(target_mask))
        valid_n = int(np.sum(valid_mask))
        invalid_pred_n = int(np.sum(target_mask & (~pred_mask)))

        if valid_n == 0:
            stats = {}
            stats["mae"] = float("nan")
            stats["rmse"] = float("nan")
            stats["r2"] = float("nan")
            stats["relative_error_percent"] = float("nan")
            stats["n"] = 0
            stats["target_n"] = target_n
            stats["invalid_pred_n"] = invalid_pred_n
            report[name] = stats
        else:
            stats = compute_scalar_metrics(y_true[valid_mask, i], y_pred[valid_mask, i])
            stats["n"] = valid_n
            stats["target_n"] = target_n
            stats["invalid_pred_n"] = invalid_pred_n
            report[name] = stats
    return report


def build_scalar_rmse_summary(metrics_scalar, target_names):
    # Keep only RMSE per scalar target for compact summaries.
    summary = {}
    for name in target_names:
        target_metrics = metrics_scalar.get(name, {})
        summary[name] = target_metrics.get("rmse")
    return summary


def subplot_grid(item_count, max_cols=3):
    # Pick a compact grid so target plots stay readable as target count grows.
    ncols = min(max_cols, max(1, int(item_count)))
    nrows = int(np.ceil(float(item_count) / float(ncols)))
    return nrows, ncols


def should_use_log_axis(true_values, pred_values):
    # Use log scale when values are strictly positive and span a wide range.
    min_true = float(np.min(true_values))
    min_pred = float(np.min(pred_values))
    if min_true <= 0.0 or min_pred <= 0.0:
        return False

    max_true = float(np.max(true_values))
    max_pred = float(np.max(pred_values))
    min_value = min(min_true, min_pred)
    max_value = max(max_true, max_pred)
    if min_value <= 0.0:
        return False
    dynamic_range = max_value / min_value
    return dynamic_range >= 100.0


def has_nearly_zero_span(lo_value, hi_value, rtol=1e-6):
    # Avoid collapsed parity axes only when values are effectively identical.
    span = float(hi_value) - float(lo_value)
    scale = max(abs(float(lo_value)), abs(float(hi_value)), np.finfo(float).eps)
    return span <= rtol * scale


def plot_scalar_parity(y_true, y_pred, target_names, out_path, title):
    # Save predicted-vs-true scatter plots for scalar targets.
    nrows, ncols = subplot_grid(len(target_names), max_cols=3)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for i in range(len(target_names)):
        name = target_names[i]
        ax = axes[i]
        mask = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        if np.sum(mask) == 0:
            ax.set_title(name + " (no valid points)")
            ax.axis("off")
            continue

        true_values = y_true[mask, i]
        pred_values = y_pred[mask, i]
        use_log_axis = should_use_log_axis(true_values, pred_values)
        ax.scatter(true_values, pred_values, s=24, alpha=0.75, edgecolors="none")

        lo_raw = min(float(np.min(true_values)), float(np.min(pred_values)))
        hi_raw = max(float(np.max(true_values)), float(np.max(pred_values)))
        if use_log_axis:
            lo = lo_raw / 1.2
            hi = hi_raw * 1.2
            if lo <= 0.0:
                lo = lo_raw * 0.8
            if hi <= lo:
                hi = lo * 10.0
        else:
            if has_nearly_zero_span(lo_raw, hi_raw):
                center = lo_raw
                half_width = max(abs(center) * 0.1, 1e-12)
                lo = center - half_width
                hi = center + half_width
            else:
                pad = 0.04 * (hi_raw - lo_raw)
                lo = lo_raw - pad
                hi = hi_raw + pad

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if use_log_axis:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("True (log scale)")
            ax.set_ylabel("Pred (log scale)")
        else:
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")

        mae = float(mean_absolute_error(true_values, pred_values))
        rmse = float(np.sqrt(mean_squared_error(true_values, pred_values)))
        r2 = safe_r2(true_values, pred_values)
        n_points = int(np.sum(mask))
        ax.set_title(name)
        text = f"n={n_points}\nMAE={mae:.3g}\nRMSE={rmse:.3g}\nR2={r2:.3f}"
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
        )
        ax.grid(alpha=0.25)

    for j in range(len(target_names), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.savefig(Path(out_path), dpi=180)
    plt.close(fig)


def plot_scalar_residual_hist(y_true, y_pred, target_names, out_path, title):
    # Plot scalar residual distributions for fast bias/variance inspection.
    nrows, ncols = subplot_grid(len(target_names), max_cols=3)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for i in range(len(target_names)):
        ax = axes[i]
        name = target_names[i]
        mask = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        if np.sum(mask) == 0:
            ax.set_title(name + " (no valid points)")
            ax.axis("off")
            continue

        true_values = y_true[mask, i]
        pred_values = y_pred[mask, i]
        if np.min(true_values) > 0.0:
            residual = 100.0 * (pred_values - true_values) / np.maximum(true_values, 1e-12)
            x_label = "Relative error (%)"
        else:
            residual = pred_values - true_values
            x_label = "Residual (pred - true)"

        ax.hist(residual, bins=24, alpha=0.8, color="#4C78A8")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
        mean_value = float(np.mean(residual))
        ax.axvline(mean_value, color="#D62728", linestyle="-", linewidth=1.2, label=f"mean={mean_value:.3g}")
        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(len(target_names), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.savefig(Path(out_path), dpi=180)
    plt.close(fig)


def scalar_targets_from_flux_curves(j_curves, t_curves, c0_values, target_names):
    # Derive scalar targets from predicted J(t) so hybrid can report scalar plots/metrics.
    j_arr = np.asarray(j_curves, dtype=float)
    t_arr = np.asarray(t_curves, dtype=float)
    c0_arr = np.asarray(c0_values, dtype=float)
    if j_arr.ndim != 2 or t_arr.ndim != 2:
        raise ValueError("j_curves and t_curves must be 2D arrays [N,T]")
    if j_arr.shape != t_arr.shape:
        raise ValueError("j_curves and t_curves must have the same shape")
    if c0_arr.ndim != 1 or c0_arr.shape[0] != j_arr.shape[0]:
        raise ValueError("c0_values must be 1D with one value per run")

    out = np.full((j_arr.shape[0], len(target_names)), np.nan, dtype=np.float32)
    for i in range(j_arr.shape[0]):
        flux = j_arr[i]
        t = t_arr[i]
        c0 = float(c0_arr[i])

        j_ss = estimate_steady_state_flux(flux, t)
        tlag = estimate_lag_time(flux, t)
        p = None
        if np.isfinite(c0):
            p = compute_permeability(j_ss, c0)
        finite = compute_finite_dose_scalars(flux, t, delivered_time_s=86400.0)

        scalars = {
            "P": p,
            "Tlag": tlag,
            "J_ss": j_ss,
            "AUC_J": finite.get("AUC_J"),
            "J_peak": finite.get("J_peak"),
            "t_peak": finite.get("t_peak"),
            "M_delivered_24h": finite.get("M_delivered_24h"),
        }
        for j in range(len(target_names)):
            name = target_names[j]
            value = scalars.get(name)
            if value is None:
                out[i, j] = np.nan
            else:
                out[i, j] = float(value)
    return out
