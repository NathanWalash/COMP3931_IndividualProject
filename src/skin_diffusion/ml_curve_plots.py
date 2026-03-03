from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def should_use_log_axis(true_values, pred_values):
    # Use log scale when values are positive and span a wide range.
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


def plot_curve_examples(t, J_true, J_pred, out_path, title, max_examples=3):
    # Save sample true-vs-predicted J(t) overlays.
    count = min(int(max_examples), int(J_true.shape[0]))
    ncols = 3
    nrows = int(np.ceil(float(count) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    if count == 1:
        sample_idx = np.array([0], dtype=int)
    else:
        sample_idx = np.linspace(0, J_true.shape[0] - 1, count, dtype=int)

    for i in range(count):
        row_idx = int(sample_idx[i])
        ax = axes[i]
        time_values = t[row_idx]
        true_curve = J_true[row_idx]
        pred_curve = J_pred[row_idx]
        ax.plot(time_values, true_curve, label="True", linewidth=2.0)
        ax.plot(time_values, pred_curve, label="Pred", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Flux J")
        if should_use_log_axis(true_curve, pred_curve):
            ax.set_yscale("log")
        rel = np.linalg.norm(pred_curve - true_curve) / (np.linalg.norm(true_curve) + 1e-12)
        mae = float(np.mean(np.abs(true_curve - pred_curve)))
        ax.set_title("Sample " + str(row_idx) + f"\nMAE={mae:.3g}, relL2={rel:.3f}")
        ax.legend()
        ax.grid(alpha=0.25)

    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.savefig(Path(out_path), dpi=180)
    plt.close(fig)


def plot_curve_error_over_time(t, J_true, J_pred, out_path, title):
    # Plot split-wide absolute and relative curve error over time.
    abs_err = np.abs(J_pred - J_true)
    rel_err = abs_err / np.maximum(np.abs(J_true), 1e-12)

    mean_abs = np.mean(abs_err, axis=0)
    p10_abs = np.percentile(abs_err, 10, axis=0)
    p90_abs = np.percentile(abs_err, 90, axis=0)

    mean_rel = 100.0 * np.mean(rel_err, axis=0)
    p10_rel = 100.0 * np.percentile(rel_err, 10, axis=0)
    p90_rel = 100.0 * np.percentile(rel_err, 90, axis=0)

    time_axis = t[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax_abs = axes[0]
    ax_rel = axes[1]

    abs_line_color = "#1D4E89"
    abs_band_color = "#8EC5FF"
    rel_line_color = "#B42318"
    rel_band_color = "#F7B0B0"
    band_edge_alpha = 0.55

    ax_abs.fill_between(time_axis, p10_abs, p90_abs, color=abs_band_color, alpha=0.30, label="10-90 percentile", zorder=1)
    ax_abs.plot(time_axis, p10_abs, color=abs_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_abs.plot(time_axis, p90_abs, color=abs_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_abs.plot(time_axis, mean_abs, color=abs_line_color, linewidth=2.3, label="Mean |pred - true|", zorder=3)
    ax_abs.set_ylabel("Absolute error")
    ax_abs.set_title(title)
    ax_abs.grid(alpha=0.25)
    ax_abs.legend(framealpha=0.9)

    ax_rel.fill_between(time_axis, p10_rel, p90_rel, color=rel_band_color, alpha=0.30, label="10-90 percentile", zorder=1)
    ax_rel.plot(time_axis, p10_rel, color=rel_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_rel.plot(time_axis, p90_rel, color=rel_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_rel.plot(time_axis, mean_rel, color=rel_line_color, linewidth=2.3, label="Mean relative error", zorder=3)
    ax_rel.set_xlabel("Time (s)")
    ax_rel.set_ylabel("Relative error (%)")
    ax_rel.grid(alpha=0.25)
    ax_rel.legend(framealpha=0.9)

    fig.savefig(Path(out_path), dpi=180)
    plt.close(fig)


def plot_training_history(history, out_path, title="Training history"):
    # Plot training loss curves and validation metric from history records.
    # history is a list of dicts with keys: epoch, loss_total, val_rel_l2,
    # and optionally loss_flux, loss_pde, loss_anchor, loss_peak, gate_mean.
    if not history or len(history) < 2:
        return

    epochs = [int(row["epoch"]) for row in history]
    val_rel_l2 = [float(row.get("val_rel_l2", float("nan"))) for row in history]
    loss_total = [float(row.get("loss_total", float("nan"))) for row in history]

    has_components = "loss_flux" in history[0]
    has_gate = any(float(row.get("gate_mean", 0.0)) > 0.0 for row in history)

    panel_count = 2
    if has_components:
        panel_count = 3
    if has_gate:
        panel_count = panel_count + 1

    fig, axes = plt.subplots(panel_count, 1, figsize=(10, 3.2 * panel_count),
                             sharex=True, constrained_layout=True)

    # Panel 1: Validation relative L2.
    ax = axes[0]
    ax.plot(epochs, val_rel_l2, color="#1D4E89", linewidth=2.0, label="val rel_l2")
    ax.set_ylabel("Relative L2")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.9)

    # Panel 2: Total loss.
    ax = axes[1]
    ax.plot(epochs, loss_total, color="#B42318", linewidth=1.8, label="total loss")
    ax.set_ylabel("Loss (total)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.9)

    # Panel 3 (optional): Individual loss components.
    if has_components:
        ax = axes[2]
        component_names = ["loss_flux", "loss_pde", "loss_anchor", "loss_peak"]
        component_colors = ["#2E7D32", "#1565C0", "#E65100", "#6A1B9A"]
        for name, color in zip(component_names, component_colors):
            values = [float(row.get(name, 0.0)) for row in history]
            if any(v > 0.0 for v in values):
                ax.plot(epochs, values, linewidth=1.4, label=name.replace("loss_", ""), color=color)
        ax.set_ylabel("Loss component")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(framealpha=0.9, ncols=2)

    # Panel 4 (optional): Gate mean.
    if has_gate:
        ax = axes[-1]
        gate_vals = [float(row.get("gate_mean", 0.5)) for row in history]
        ax.plot(epochs, gate_vals, color="#00695C", linewidth=1.8, label="gate mean lam")
        ax.set_ylabel("Gate lambda")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.legend(framealpha=0.9)

    axes[-1].set_xlabel("Epoch")
    fig.savefig(Path(out_path), dpi=180)
    plt.close(fig)


def plot_per_run_error_distribution(j_true, j_pred, out_path, title="Per-run error distribution"):
    # Histogram of per-run relative L2 error across the test set.
    run_count = int(j_true.shape[0])
    per_run_rel_l2 = np.empty(run_count, dtype=np.float64)
    for i in range(run_count):
        true_norm = float(np.linalg.norm(j_true[i]))
        diff_norm = float(np.linalg.norm(j_pred[i] - j_true[i]))
        per_run_rel_l2[i] = diff_norm / (true_norm + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # Left: histogram.
    ax = axes[0]
    ax.hist(per_run_rel_l2, bins=min(40, max(10, run_count // 4)),
            color="#1D4E89", edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.axvline(float(np.median(per_run_rel_l2)), color="#B42318", linewidth=2.0,
               linestyle="--", label=f"median={np.median(per_run_rel_l2):.4f}")
    ax.axvline(float(np.mean(per_run_rel_l2)), color="#E65100", linewidth=2.0,
               linestyle=":", label=f"mean={np.mean(per_run_rel_l2):.4f}")
    ax.set_xlabel("Per-run relative L2")
    ax.set_ylabel("Count")
    ax.set_title("Error histogram")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25)

    # Right: sorted error curve (CDF-style).
    ax = axes[1]
    sorted_errors = np.sort(per_run_rel_l2)
    ranks = np.arange(1, run_count + 1) / float(run_count)
    ax.plot(sorted_errors, ranks, color="#1D4E89", linewidth=2.0)
    ax.set_xlabel("Per-run relative L2")
    ax.set_ylabel("Fraction of runs <= threshold")
    ax.set_title("Cumulative error distribution")
    p90 = float(np.percentile(per_run_rel_l2, 90))
    p95 = float(np.percentile(per_run_rel_l2, 95))
    ax.axvline(p90, color="#E65100", linestyle="--", linewidth=1.4, label=f"P90={p90:.4f}")
    ax.axvline(p95, color="#B42318", linestyle="--", linewidth=1.4, label=f"P95={p95:.4f}")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.savefig(Path(out_path), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-model comparison plots
# ---------------------------------------------------------------------------
# These are for notebook or post-hoc comparison across models.
# Each function can optionally save to a file (out_path) or just return the
# figure so the caller can plt.show() in a notebook.


def plot_curve_comparison(t, j_true, model_preds, out_path=None, title="Flux curve comparison",
                          max_examples=6, time_units="hours"):
    # Overlay simulator ground truth with one or more model predictions.
    # model_preds: list of (label, color, j_pred_array) tuples.
    count = min(int(max_examples), int(j_true.shape[0]))
    ncols = 3
    nrows = int(np.ceil(float(count) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    if count == 1:
        sample_idx = np.array([0], dtype=int)
    else:
        sample_idx = np.linspace(0, j_true.shape[0] - 1, count, dtype=int)

    time_divisor = 3600.0 if time_units == "hours" else 1.0
    x_label = "Time (h)" if time_units == "hours" else "Time (s)"

    for i in range(count):
        ri = int(sample_idx[i])
        ax = axes[i]
        t_plot = t[ri] / time_divisor
        ax.plot(t_plot, j_true[ri], "k-", linewidth=1.8, label="Simulator")
        for label, color, j_pred in model_preds:
            rel = np.linalg.norm(j_pred[ri] - j_true[ri]) / (np.linalg.norm(j_true[ri]) + 1e-12)
            ax.plot(t_plot, j_pred[ri], "--", color=color, linewidth=1.4, alpha=0.85,
                    label=label + f" ({rel:.3f})")
        ax.set_xlabel(x_label)
        if i % ncols == 0:
            ax.set_ylabel("Flux J(t)")
        ax.set_title(f"Run {ri}", fontsize=9)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if out_path is not None:
        fig.savefig(Path(out_path), dpi=180)
    return fig


def per_run_rel_l2(j_pred, j_true):
    # Per-run relative L2 as a 1-D array.
    norms_true = np.linalg.norm(j_true, axis=1) + 1e-12
    norms_diff = np.linalg.norm(j_pred - j_true, axis=1)
    return norms_diff / norms_true


def plot_error_distribution_comparison(j_true, model_preds, out_path=None,
                                       title="Per-run error distribution comparison"):
    # Overlaid histogram + CDF for multiple models.
    # model_preds: list of (label, color, j_pred_array) tuples.
    errors = [(label, color, per_run_rel_l2(j_pred, j_true)) for label, color, j_pred in model_preds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    # Left: overlaid histograms (first two models to avoid clutter).
    all_max = max(e.max() for _, _, e in errors)
    bins = np.linspace(0, all_max * 1.05, 40)
    for label, color, err in errors[:2]:
        ax1.hist(err, bins=bins, color=color, alpha=0.45, edgecolor="white", linewidth=0.5,
                 label=f"{label} (med={np.median(err):.4f})")
    ax1.set_xlabel("Per-run relative L2")
    ax1.set_ylabel("Count")
    ax1.set_title("Error histogram")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Right: CDF comparison (all models).
    for label, color, err in errors:
        sorted_err = np.sort(err)
        ranks = np.arange(1, len(sorted_err) + 1) / float(len(sorted_err))
        ax2.plot(sorted_err, ranks, color=color, linewidth=1.8, label=label)
    ax2.set_xlabel("Per-run relative L2")
    ax2.set_ylabel("Fraction of runs <= threshold")
    ax2.set_title("Cumulative error distribution")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if out_path is not None:
        fig.savefig(Path(out_path), dpi=180)
    return fig


def plot_scalar_parity_comparison(y_true, model_preds, target_names, out_path=None,
                                  title="Scalar parity comparison"):
    # Multi-model parity scatter: true on x-axis, each model a different colour.
    # model_preds: list of (label, color, y_pred_array) tuples.
    ncols = min(3, len(target_names))
    nrows = int(np.ceil(float(len(target_names)) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for i in range(len(target_names)):
        name = target_names[i]
        ax = axes[i]
        tv = y_true[:, i]
        mask = np.isfinite(tv)
        if np.sum(mask) == 0:
            ax.set_title(name + " (no valid points)")
            ax.axis("off")
            continue
        tv_valid = tv[mask]
        lo, hi = tv_valid.min() * 0.9, tv_valid.max() * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")
        for label, color, y_pred in model_preds:
            pv = y_pred[mask, i]
            ax.scatter(tv_valid, pv, s=18, alpha=0.5, color=color, edgecolors="none", label=label)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)

    for j in range(len(target_names), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if out_path is not None:
        fig.savefig(Path(out_path), dpi=180)
    return fig
