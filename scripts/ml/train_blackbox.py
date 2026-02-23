import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


def load_json(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def run_index_from_path(run_dir):
    # Parse run index from run directory path.
    name = Path(run_dir).name
    if not name.startswith("run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def in_index_range(run_idx, start_idx, end_idx):
    # Inclusive range filter; None means unbounded.
    if run_idx is None:
        return False
    if start_idx is not None and run_idx < start_idx:
        return False
    if end_idx is not None and run_idx > end_idx:
        return False
    return True


def load_target_names(ml_dir):
    # Load scalar target names written by export_ml_dataset.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing ml/meta.json: " + str(meta_path))

    meta = load_json(meta_path)
    names = meta.get("scalar_target_names")
    if not isinstance(names, list):
        raise ValueError("meta.json is missing scalar_target_names list")
    if len(names) == 0:
        raise ValueError("scalar_target_names list is empty in meta.json")
    return [str(name) for name in names]


def load_feature_names(ml_dir):
    # Load feature column names written by export_ml_dataset.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing ml/meta.json: " + str(meta_path))

    meta = load_json(meta_path)
    names = meta.get("feature_names")
    if not isinstance(names, list):
        raise ValueError("meta.json is missing feature_names list")
    if len(names) == 0:
        raise ValueError("feature_names list is empty in meta.json")
    return [str(name) for name in names]


def load_npz(path):
    # Load one dataset split from NPZ.
    data = np.load(path)
    result = {}
    result["X"] = data["X"]
    result["y_scalar"] = data["y_scalar"]
    result["J"] = data["J"]
    result["t"] = data["t"]
    return result


def filter_split_by_run_index(split_data, split_index_entries, start_idx, end_idx):
    # Filter one split by run index range using index entries from ml/meta.json.
    if start_idx is None and end_idx is None:
        return split_data, split_index_entries

    keep_rows = []
    for i in range(len(split_index_entries)):
        entry = split_index_entries[i]
        run_idx = run_index_from_path(entry["run_dir"])
        if in_index_range(run_idx, start_idx, end_idx):
            keep_rows.append(i)

    keep_rows = np.asarray(keep_rows, dtype=int)
    filtered = {}
    filtered["X"] = split_data["X"][keep_rows]
    filtered["y_scalar"] = split_data["y_scalar"][keep_rows]
    filtered["J"] = split_data["J"][keep_rows]
    filtered["t"] = split_data["t"][keep_rows]

    filtered_index = []
    for idx in keep_rows:
        filtered_index.append(split_index_entries[int(idx)])

    return filtered, filtered_index


def rel_percent_error(y_true, y_pred, eps=1e-12):
    # Mean relative error in percent.
    denom = np.abs(y_true) + eps
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def safe_r2(y_true, y_pred):
    # R^2 needs at least two points.
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def compute_scalar_metrics(y_true, y_pred):
    # Scalar metrics for one target.
    metrics = {}
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["r2"] = safe_r2(y_true, y_pred)
    metrics["relative_error_percent"] = rel_percent_error(y_true, y_pred)
    return metrics


def compute_curve_metrics(J_true, J_pred, t, eps=1e-12):
    # Curve metrics across all time points.
    diff = J_pred - J_true

    iae_values = []
    for i in range(J_true.shape[0]):
        value = float(np.trapezoid(np.abs(diff[i]), x=t[i]))
        iae_values.append(value)

    true_flat = J_true.ravel()
    pred_flat = J_pred.ravel()
    pearson_r = float("nan")
    if len(true_flat) > 1:
        pearson_r = float(np.corrcoef(true_flat, pred_flat)[0, 1])

    metrics = {}
    metrics["mae"] = float(np.mean(np.abs(diff)))
    metrics["rmse"] = float(np.sqrt(np.mean(diff**2)))
    metrics["relative_l2"] = float(np.linalg.norm(diff) / (np.linalg.norm(J_true) + eps))
    metrics["integrated_absolute_error"] = float(np.mean(iae_values))
    metrics["pearson_r"] = pearson_r
    return metrics


def choose_transform(values):
    # Use log10 when positive values span a wide range.
    # Zero values are allowed; they are clipped by apply_transform.
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {"kind": "identity"}

    positive = finite[finite > 0.0]
    if positive.size < 2:
        return {"kind": "identity"}

    min_value = max(float(np.min(positive)), 1e-30)
    max_value = float(np.max(positive))
    spread_ratio = max_value / min_value

    if spread_ratio < 20.0:
        return {"kind": "identity"}

    eps = max(min_value * 1e-3, 1e-30)
    transform = {}
    transform["kind"] = "log10"
    transform["eps"] = eps
    return transform


def choose_scalar_log_transform(values):
    # Scalars are trained in log-space to handle wide dynamic ranges.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        transform = {}
        transform["kind"] = "identity"
        return transform

    positive = finite[finite > 0.0]
    if positive.size == 0:
        # Fallback for unexpected non-positive data.
        transform = {}
        transform["kind"] = "identity"
        return transform

    min_value = float(np.min(positive))
    eps = max(min_value * 1e-3, 1e-30)
    transform = {}
    transform["kind"] = "log10"
    transform["eps"] = eps
    return transform


def apply_transform(values, transform):
    # Apply target transform before fitting.
    if transform["kind"] == "identity":
        return values
    eps = transform["eps"]
    clipped = np.maximum(values, eps)
    return np.log10(clipped)


def invert_transform(values, transform):
    # Convert predictions back to original units.
    if transform["kind"] == "identity":
        return values
    # Clamp log-space predictions before exponentiation to avoid inf values.
    # 10^300 is already far beyond any expected scale in this project.
    clipped = np.clip(values, -300.0, 300.0)
    return np.power(10.0, clipped)


def build_scalar_candidates(use_xgboost, alpha_grid):
    # Build candidate models for scalar targets.
    candidates = []

    ridge_params = []
    for alpha in alpha_grid:
        entry = {}
        entry["alpha"] = alpha
        ridge_params.append(entry)
    candidates.append({"family": "ridge", "params": ridge_params})

    if use_xgboost:
        xgb_params = []
        xgb_params.append({"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05})
        xgb_params.append({"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05})
        candidates.append({"family": "xgboost", "params": xgb_params})
    return candidates


def build_curve_candidates(use_xgboost, alpha_grid):
    # Build candidate models for PCA curve coordinates.
    candidates = []

    ridge_params = []
    for alpha in alpha_grid:
        entry = {}
        entry["alpha"] = alpha
        ridge_params.append(entry)
    candidates.append({"family": "ridge", "params": ridge_params})

    if use_xgboost:
        xgb_params = []
        xgb_params.append({"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05})
        xgb_params.append({"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05})
        candidates.append({"family": "xgboost_multioutput", "params": xgb_params})
    return candidates


def make_scalar_estimator(family, params):
    # Construct a scalar estimator from family + params.
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if family == "xgboost":
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=1,
        )
    raise ValueError("Unknown scalar model family: " + str(family))


def make_curve_estimator(family, params):
    # Construct a curve estimator in PCA space.
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if family == "xgboost_multioutput":
        base = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=1,
        )
        return MultiOutputRegressor(base)
    raise ValueError("Unknown curve model family: " + str(family))


def fit_scalar_models(
    X_train,
    y_train,
    X_val,
    y_val,
    target_names,
    use_xgboost,
    alpha_grid,
    min_rows_for_xgboost,
):
    # Fit one model per scalar target and pick the best on validation RMSE.
    # We train each scalar target separately because scales/availability differ.
    models = []
    for target_idx in range(y_train.shape[1]):
        target_name = target_names[target_idx]
        ytr = y_train[:, target_idx]
        yva = y_val[:, target_idx]
        mask_train = np.isfinite(ytr)
        mask_val = np.isfinite(yva)
        train_rows = int(np.sum(mask_train))
        val_rows = int(np.sum(mask_val))

        # Avoid complex models when a target has limited supervision.
        use_xgboost_for_target = bool(use_xgboost) and (train_rows >= min_rows_for_xgboost)
        candidates = build_scalar_candidates(use_xgboost_for_target, alpha_grid)

        # If a target has no valid training values, keep a placeholder model.
        if train_rows == 0:
            model_info = {}
            model_info["target_name"] = target_name
            model_info["model_family"] = None
            model_info["transform"] = {"kind": "identity"}
            model_info["hyperparams"] = {}
            model_info["estimator"] = None
            model_info["train_rows"] = train_rows
            model_info["val_rows"] = val_rows
            models.append(model_info)
        else:
            # Keep scalar targets in log-space for stable fitting.
            transform = choose_scalar_log_transform(ytr[mask_train])
            ytr_transformed = apply_transform(ytr[mask_train], transform)

            best_rmse = float("inf")
            best_family = None
            best_params = {}
            best_estimator = None

            # Try each candidate setting and keep the lowest validation RMSE.
            for candidate in candidates:
                family = candidate["family"]
                for params in candidate["params"]:
                    estimator = make_scalar_estimator(family, params)
                    estimator.fit(X_train[mask_train], ytr_transformed)

                    if val_rows == 0:
                        # No validation rows: rank by train RMSE as fallback.
                        pred_transformed = estimator.predict(X_train[mask_train])
                        pred = invert_transform(pred_transformed, transform)
                        rmse = float(np.sqrt(mean_squared_error(ytr[mask_train], pred)))
                    else:
                        pred_transformed = estimator.predict(X_val[mask_val])
                        pred = invert_transform(pred_transformed, transform)
                        rmse = float(np.sqrt(mean_squared_error(yva[mask_val], pred)))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_family = family
                        best_params = dict(params)
                        best_estimator = estimator

            model_info = {}
            model_info["target_name"] = target_name
            model_info["model_family"] = best_family
            model_info["transform"] = transform
            model_info["hyperparams"] = best_params
            model_info["estimator"] = best_estimator
            model_info["train_rows"] = train_rows
            model_info["val_rows"] = val_rows
            models.append(model_info)

    return models


def predict_scalar_models(models, X):
    # Predict all scalar targets for one feature matrix.
    target_count = len(models)
    y_pred = np.full((X.shape[0], target_count), np.nan, dtype=float)

    for i in range(target_count):
        model_info = models[i]
        estimator = model_info["estimator"]
        transform = model_info["transform"]
        if estimator is not None:
            pred_transformed = estimator.predict(X)
            y_pred[:, i] = invert_transform(pred_transformed, transform)
    return y_pred


def fit_curve_model(X_train, J_train, X_val, J_val, curve_components, use_xgboost, alpha_grid):
    # Train curve model: transform J, compress with PCA, regress PCA scores.
    # This reduces curve dimensionality so the regressor has a manageable target.
    transform = choose_transform(J_train)
    J_train_transformed = apply_transform(J_train, transform)

    max_components = min(curve_components, J_train_transformed.shape[0], J_train_transformed.shape[1])
    pca = PCA(n_components=max_components)
    Z_train = pca.fit_transform(J_train_transformed)

    candidates = build_curve_candidates(use_xgboost, alpha_grid)
    best_rmse = float("inf")
    best_family = None
    best_params = {}
    best_estimator = None

    for candidate in candidates:
        family = candidate["family"]
        for params in candidate["params"]:
            estimator = make_curve_estimator(family, params)
            estimator.fit(X_train, Z_train)

            Z_val_pred = estimator.predict(X_val)
            J_val_pred_transformed = pca.inverse_transform(Z_val_pred)
            J_val_pred = invert_transform(J_val_pred_transformed, transform)
            rmse = float(np.sqrt(mean_squared_error(J_val, J_val_pred)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_family = family
                best_params = dict(params)
                best_estimator = estimator

    curve_config = {}
    curve_config["model_family"] = best_family
    for key, value in best_params.items():
        curve_config[key] = value
    return pca, transform, best_estimator, curve_config


def predict_curve_model(pca, curve_transform, estimator, X):
    # Predict J(t) by decoding predicted PCA coordinates.
    Z_pred = estimator.predict(X)
    J_pred_transformed = pca.inverse_transform(Z_pred)
    J_pred = invert_transform(J_pred_transformed, curve_transform)
    return np.maximum(J_pred, 0.0)


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
        else:
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
            # Keep equal axis scaling so distance from diagonal is visually meaningful.
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_curve_examples(t, J_true, J_pred, out_path, title, max_examples=3):
    # Save example true-vs-predicted curve overlays.
    # We use a 3x3 layout by default so the plot is more informative.
    count = min(max_examples, J_true.shape[0])
    ncols = 3
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    # Spread examples through the split instead of plotting only first few rows.
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
        mae = float(mean_absolute_error(true_curve, pred_curve))
        ax.set_title("Sample " + str(row_idx) + f"\nMAE={mae:.3g}, relL2={rel:.3f}")
        ax.legend()
        ax.grid(alpha=0.25)

    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
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
        else:
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_curve_error_over_time(t, J_true, J_pred, out_path, title):
    # Plot average absolute and relative curve error over time across a split.
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

    # Keep bands subtle but readable by adding faint percentile boundaries.
    abs_line_color = "#1D4E89"
    abs_band_color = "#8EC5FF"
    rel_line_color = "#B42318"
    rel_band_color = "#F7B0B0"
    band_edge_alpha = 0.55

    ax_abs.fill_between(
        time_axis,
        p10_abs,
        p90_abs,
        color=abs_band_color,
        alpha=0.30,
        label="10-90 percentile",
        zorder=1,
    )
    ax_abs.plot(time_axis, p10_abs, color=abs_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_abs.plot(time_axis, p90_abs, color=abs_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_abs.plot(time_axis, mean_abs, color=abs_line_color, linewidth=2.3, label="Mean |pred - true|", zorder=3)
    ax_abs.set_ylabel("Absolute error")
    ax_abs.set_title(title)
    ax_abs.grid(alpha=0.25)
    ax_abs.legend(framealpha=0.9)

    ax_rel.fill_between(
        time_axis,
        p10_rel,
        p90_rel,
        color=rel_band_color,
        alpha=0.30,
        label="10-90 percentile",
        zorder=1,
    )
    ax_rel.plot(time_axis, p10_rel, color=rel_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_rel.plot(time_axis, p90_rel, color=rel_line_color, linewidth=0.9, linestyle=":", alpha=band_edge_alpha, zorder=2)
    ax_rel.plot(time_axis, mean_rel, color=rel_line_color, linewidth=2.3, label="Mean relative error", zorder=3)
    ax_rel.set_xlabel("Time (s)")
    ax_rel.set_ylabel("Relative error (%)")
    ax_rel.grid(alpha=0.25)
    ax_rel.legend(framealpha=0.9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scalar_report(y_true, y_pred, target_names):
    # Per-target metric report with explicit accounting for invalid predictions.
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
    # Keep only RMSE per scalar target for quick comparisons.
    summary = {}
    for name in target_names:
        summary[name] = metrics_scalar[name]["rmse"]
    return summary


def run_training(
    ml_dir,
    out_dir,
    curve_components,
    use_xgboost,
    alpha_grid,
    min_rows_for_xgboost,
    run_start_index=None,
    run_end_index=None,
):
    # Run one full train/evaluate pass and save outputs.
    # Outputs: per-split metrics, runtime info, compact summary, and plots.
    ml_dir = Path(ml_dir)
    out_dir = Path(out_dir)
    out_plot = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot.mkdir(parents=True, exist_ok=True)

    target_names = load_target_names(ml_dir)
    feature_names = load_feature_names(ml_dir)
    ml_meta = load_json(ml_dir / "meta.json")
    split_index_map = ml_meta["splits"]

    id_train = load_npz(ml_dir / "id_train.npz")
    id_val = load_npz(ml_dir / "id_val.npz")
    id_test = load_npz(ml_dir / "id_test.npz")
    ood = None
    has_ood = False
    ood_path = ml_dir / "ood_primary.npz"
    if ood_path.exists() and "ood_primary" in split_index_map:
        ood = load_npz(ood_path)
        has_ood = True

    id_train, _ = filter_split_by_run_index(
        id_train,
        split_index_map["id_train"]["index"],
        run_start_index,
        run_end_index,
    )
    id_val, _ = filter_split_by_run_index(
        id_val,
        split_index_map["id_val"]["index"],
        run_start_index,
        run_end_index,
    )
    id_test, _ = filter_split_by_run_index(
        id_test,
        split_index_map["id_test"]["index"],
        run_start_index,
        run_end_index,
    )
    if has_ood:
        ood, _ = filter_split_by_run_index(
            ood,
            split_index_map["ood_primary"]["index"],
            run_start_index,
            run_end_index,
        )

    if id_train["X"].shape[0] == 0:
        raise ValueError("No id_train rows left after run-index filtering")
    if id_val["X"].shape[0] == 0:
        raise ValueError("No id_val rows left after run-index filtering")
    if id_test["X"].shape[0] == 0:
        raise ValueError("No id_test rows left after run-index filtering")
    if has_ood and ood["X"].shape[0] == 0:
        raise ValueError("No ood_primary rows left after run-index filtering")

    # Ensure feature schema and arrays agree before fitting.
    if id_train["X"].shape[1] != len(feature_names):
        raise ValueError("Feature count mismatch: X has " + str(id_train["X"].shape[1]) + " columns but meta feature_names has " + str(len(feature_names)))
    # Fit scaler on training data only.
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(id_train["X"])
    X_val = x_scaler.transform(id_val["X"])
    X_test = x_scaler.transform(id_test["X"])
    if has_ood:
        X_ood = x_scaler.transform(ood["X"])
    else:
        X_ood = None

    t0 = time.time()
    # Scalar stage: select model/hyperparameters per target on validation split.
    scalar_models = fit_scalar_models(
        X_train,
        id_train["y_scalar"],
        X_val,
        id_val["y_scalar"],
        target_names,
        use_xgboost=use_xgboost,
        alpha_grid=alpha_grid,
        min_rows_for_xgboost=min_rows_for_xgboost,
    )
    scalar_train_seconds = float(time.time() - t0)

    y_id_pred = predict_scalar_models(scalar_models, X_test)
    if has_ood:
        y_ood_pred = predict_scalar_models(scalar_models, X_ood)
    else:
        y_ood_pred = None

    t1 = time.time()
    # Curve stage: fit model in PCA space, then decode back to J(t).
    pca, curve_transform, curve_model, curve_config = fit_curve_model(
        X_train,
        id_train["J"],
        X_val,
        id_val["J"],
        curve_components=curve_components,
        use_xgboost=use_xgboost,
        alpha_grid=alpha_grid,
    )
    curve_train_seconds = float(time.time() - t1)

    J_id_pred = predict_curve_model(pca, curve_transform, curve_model, X_test)
    if has_ood:
        J_ood_pred = predict_curve_model(pca, curve_transform, curve_model, X_ood)
    else:
        J_ood_pred = None

    metrics_id = {}
    metrics_id["rows"] = int(id_test["X"].shape[0])
    metrics_id["scalar"] = scalar_report(id_test["y_scalar"], y_id_pred, target_names)
    metrics_id["curve"] = compute_curve_metrics(id_test["J"], J_id_pred, id_test["t"])

    metrics_ood = {}
    if has_ood:
        metrics_ood["available"] = True
        metrics_ood["rows"] = int(ood["X"].shape[0])
        metrics_ood["scalar"] = scalar_report(ood["y_scalar"], y_ood_pred, target_names)
        metrics_ood["curve"] = compute_curve_metrics(ood["J"], J_ood_pred, ood["t"])
    else:
        metrics_ood["available"] = False
        metrics_ood["rows"] = 0
        metrics_ood["scalar"] = {}
        metrics_ood["curve"] = {}

    runtime = {}
    runtime["scalar_train_seconds"] = scalar_train_seconds
    runtime["curve_train_seconds"] = curve_train_seconds
    runtime["use_xgboost"] = bool(use_xgboost)
    runtime["target_names"] = target_names
    runtime["feature_names"] = feature_names
    runtime["feature_count"] = int(len(feature_names))
    runtime["has_ood"] = bool(has_ood)

    runtime_models = []
    for model_info in scalar_models:
        item = {}
        item["target"] = model_info["target_name"]
        item["model_family"] = model_info["model_family"]
        item["transform"] = model_info["transform"]
        item["hyperparams"] = model_info["hyperparams"]
        item["train_rows"] = int(model_info["train_rows"])
        item["val_rows"] = int(model_info["val_rows"])
        runtime_models.append(item)
    runtime["scalar_models"] = runtime_models

    runtime["curve_model"] = curve_config
    runtime["curve_transform"] = curve_transform
    runtime["curve_components"] = int(pca.n_components_)
    runtime["split_rows"] = {
        "id_train": int(id_train["X"].shape[0]),
        "id_val": int(id_val["X"].shape[0]),
        "id_test": int(id_test["X"].shape[0]),
        "ood_primary": int(ood["X"].shape[0]) if has_ood else 0,
    }
    runtime["run_index_filter"] = {
        "start": run_start_index,
        "end": run_end_index,
    }

    # Keep a short summary for quick checks
    summary = {}
    summary["id_scalar_rmse"] = build_scalar_rmse_summary(metrics_id["scalar"], target_names)
    if has_ood:
        summary["ood_scalar_rmse"] = build_scalar_rmse_summary(metrics_ood["scalar"], target_names)
    else:
        summary["ood_scalar_rmse"] = {}
    summary["id_curve_relative_l2"] = metrics_id["curve"]["relative_l2"]
    summary["ood_curve_relative_l2"] = metrics_ood["curve"].get("relative_l2")
    summary["id_curve_pearson_r"] = metrics_id["curve"]["pearson_r"]
    summary["ood_curve_pearson_r"] = metrics_ood["curve"].get("pearson_r")

    (out_dir / "metrics_id.json").write_text(json.dumps(metrics_id, indent=2), encoding="utf-8")
    (out_dir / "metrics_ood.json").write_text(json.dumps(metrics_ood, indent=2), encoding="utf-8")
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_scalar_parity(
        id_test["y_scalar"],
        y_id_pred,
        target_names,
        out_plot / "scalar_parity_id.png",
        "Scalar parity (ID)",
    )
    if has_ood:
        plot_scalar_parity(
            ood["y_scalar"],
            y_ood_pred,
            target_names,
            out_plot / "scalar_parity_ood.png",
            "Scalar parity (OOD)",
        )
    plot_scalar_residual_hist(
        id_test["y_scalar"],
        y_id_pred,
        target_names,
        out_plot / "scalar_residuals_id.png",
        "Scalar residuals (ID)",
    )
    if has_ood:
        plot_scalar_residual_hist(
            ood["y_scalar"],
            y_ood_pred,
            target_names,
            out_plot / "scalar_residuals_ood.png",
            "Scalar residuals (OOD)",
        )
    plot_curve_examples(
        id_test["t"],
        id_test["J"],
        J_id_pred,
        out_plot / "curve_examples_id.png",
        "J(t) examples (ID)",
        max_examples=9,
    )
    if has_ood:
        plot_curve_examples(
            ood["t"],
            ood["J"],
            J_ood_pred,
            out_plot / "curve_examples_ood.png",
            "J(t) examples (OOD)",
            max_examples=9,
        )
    plot_curve_error_over_time(
        id_test["t"],
        id_test["J"],
        J_id_pred,
        out_plot / "curve_error_over_time_id.png",
        "Curve error over time (ID)",
    )
    if has_ood:
        plot_curve_error_over_time(
            ood["t"],
            ood["J"],
            J_ood_pred,
            out_plot / "curve_error_over_time_ood.png",
            "Curve error over time (OOD)",
        )
    return out_dir


def main():
    # Parse CLI arguments and run one training job.
    # This script is intentionally single-run
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", default="data/processed/ml")
    parser.add_argument("--out_dir", default="outputs/ml/blackbox")
    parser.add_argument("--curve_components", type=int, default=20)
    parser.add_argument("--use_xgboost", action="store_true")
    parser.add_argument("--min_rows_for_xgboost", type=int, default=150)
    parser.add_argument("--alpha_grid", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    args = parser.parse_args()

    alpha_grid = []
    for part in args.alpha_grid.split(","):
        text = part.strip()
        if text:
            alpha_grid.append(float(text))
    if len(alpha_grid) == 0:
        raise ValueError("--alpha_grid is empty")

    out_dir = run_training(
        ml_dir=args.ml_dir,
        out_dir=args.out_dir,
        curve_components=args.curve_components,
        use_xgboost=bool(args.use_xgboost),
        alpha_grid=alpha_grid,
        min_rows_for_xgboost=int(args.min_rows_for_xgboost),
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
    )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
