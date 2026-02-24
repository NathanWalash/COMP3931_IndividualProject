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
