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


def load_npz(path):
    # Load one dataset split from NPZ.
    data = np.load(path)
    result = {}
    result["X"] = data["X"]
    result["y_scalar"] = data["y_scalar"]
    result["J"] = data["J"]
    result["t"] = data["t"]
    return result


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
    # Use log10 only when values are positive and span a wide range.
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {"kind": "identity"}
    if np.min(finite) <= 0.0:
        return {"kind": "identity"}

    min_value = max(float(np.min(finite)), 1e-30)
    max_value = float(np.max(finite))
    spread_ratio = max_value / min_value

    if spread_ratio < 20.0:
        return {"kind": "identity"}

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
    return np.power(10.0, values)


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


def fit_scalar_models(X_train, y_train, X_val, y_val, target_names, use_xgboost, alpha_grid):
    # Fit one model per scalar target and pick the best on validation RMSE.
    # We train each scalar target separately because scales/availability differ.
    models = []
    candidates = build_scalar_candidates(use_xgboost, alpha_grid)

    for target_idx in range(y_train.shape[1]):
        target_name = target_names[target_idx]
        ytr = y_train[:, target_idx]
        yva = y_val[:, target_idx]
        mask_train = np.isfinite(ytr)
        mask_val = np.isfinite(yva)

        # If a target has no valid training values, keep a placeholder model.
        if np.sum(mask_train) == 0:
            model_info = {}
            model_info["target_name"] = target_name
            model_info["model_family"] = None
            model_info["transform"] = {"kind": "identity"}
            model_info["hyperparams"] = {}
            model_info["estimator"] = None
            models.append(model_info)
        else:
            transform = choose_transform(ytr[mask_train])
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

                    if np.sum(mask_val) == 0:
                        rmse = 0.0
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


def enforce_curve_jss_consistency(J_pred, jss_pred, tail_points=100, eps=1e-30):
    # Scale each predicted curve so its tail matches predicted J_ss.
    # This is optional and only used when want scalar/curve outputs aligned.
    J_adjusted = np.array(J_pred, copy=True)
    n_tail = max(1, min(tail_points, J_adjusted.shape[1]))

    for i in range(J_adjusted.shape[0]):
        target = jss_pred[i]
        if np.isfinite(target):
            tail_mean = float(np.mean(J_adjusted[i, -n_tail:]))
            if abs(tail_mean) > eps:
                scale = target / tail_mean
                scale = float(np.clip(scale, 0.0, 50.0))
                J_adjusted[i] = np.maximum(J_adjusted[i] * scale, 0.0)

    return J_adjusted


def plot_scalar_parity(y_true, y_pred, target_names, out_path, title):
    # Save predicted-vs-true scatter plots for scalar targets.
    fig, axes = plt.subplots(1, len(target_names), figsize=(12, 3.5), constrained_layout=True)
    if len(target_names) == 1:
        axes = [axes]

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
            ax.scatter(true_values, pred_values, s=18)
            lo = min(float(np.min(true_values)), float(np.min(pred_values)))
            hi = max(float(np.max(true_values)), float(np.max(pred_values)))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel("true")
            ax.set_ylabel("pred")
            ax.set_title(name)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curve_examples(t, J_true, J_pred, out_path, title, max_examples=3):
    # Save a few true-vs-predicted curve overlays.
    count = min(max_examples, J_true.shape[0])
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.8 * count), constrained_layout=True)
    if count == 1:
        axes = [axes]

    for i in range(count):
        ax = axes[i]
        ax.plot(t[i], J_true[i], label="true")
        ax.plot(t[i], J_pred[i], label="pred")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("J")
        ax.set_title("sample " + str(i))
        ax.legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def scalar_report(y_true, y_pred, target_names):
    # Per-target metric report with NaN-safe masking.
    report = {}
    for i in range(len(target_names)):
        name = target_names[i]
        mask = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        if np.sum(mask) == 0:
            stats = {}
            stats["mae"] = float("nan")
            stats["rmse"] = float("nan")
            stats["r2"] = float("nan")
            stats["relative_error_percent"] = float("nan")
            stats["n"] = 0
            report[name] = stats
        else:
            stats = compute_scalar_metrics(y_true[mask, i], y_pred[mask, i])
            stats["n"] = int(np.sum(mask))
            report[name] = stats
    return report


def build_scalar_rmse_summary(metrics_scalar, target_names):
    # Keep only RMSE per scalar target for quick comparisons.
    summary = {}
    for name in target_names:
        summary[name] = metrics_scalar[name]["rmse"]
    return summary


def run_training(ml_dir, out_dir, curve_components, use_xgboost, curve_consistency, alpha_grid):
    # Run one full train/evaluate pass and save outputs.
    # Outputs: per-split metrics, runtime info, compact summary, and plots.
    ml_dir = Path(ml_dir)
    out_dir = Path(out_dir)
    out_plot = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot.mkdir(parents=True, exist_ok=True)

    target_names = load_target_names(ml_dir)

    id_train = load_npz(ml_dir / "id_train.npz")
    id_val = load_npz(ml_dir / "id_val.npz")
    id_test = load_npz(ml_dir / "id_test.npz")
    ood = load_npz(ml_dir / "ood_primary.npz")

    # Fit scaler on training data only.
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(id_train["X"])
    X_val = x_scaler.transform(id_val["X"])
    X_test = x_scaler.transform(id_test["X"])
    X_ood = x_scaler.transform(ood["X"])

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
    )
    scalar_train_seconds = float(time.time() - t0)

    y_id_pred = predict_scalar_models(scalar_models, X_test)
    y_ood_pred = predict_scalar_models(scalar_models, X_ood)

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
    J_ood_pred = predict_curve_model(pca, curve_transform, curve_model, X_ood)

    if curve_consistency:
        # Optional: force J(t) tail to be consistent with predicted J_ss.
        J_id_pred = enforce_curve_jss_consistency(J_id_pred, y_id_pred[:, 2])
        J_ood_pred = enforce_curve_jss_consistency(J_ood_pred, y_ood_pred[:, 2])

    metrics_id = {}
    metrics_id["rows"] = int(id_test["X"].shape[0])
    metrics_id["scalar"] = scalar_report(id_test["y_scalar"], y_id_pred, target_names)
    metrics_id["curve"] = compute_curve_metrics(id_test["J"], J_id_pred, id_test["t"])

    metrics_ood = {}
    metrics_ood["rows"] = int(ood["X"].shape[0])
    metrics_ood["scalar"] = scalar_report(ood["y_scalar"], y_ood_pred, target_names)
    metrics_ood["curve"] = compute_curve_metrics(ood["J"], J_ood_pred, ood["t"])

    runtime = {}
    runtime["scalar_train_seconds"] = scalar_train_seconds
    runtime["curve_train_seconds"] = curve_train_seconds
    runtime["use_xgboost"] = bool(use_xgboost)
    runtime["curve_consistency"] = bool(curve_consistency)
    runtime["target_names"] = target_names

    runtime_models = []
    for model_info in scalar_models:
        item = {}
        item["target"] = model_info["target_name"]
        item["model_family"] = model_info["model_family"]
        item["transform"] = model_info["transform"]
        item["hyperparams"] = model_info["hyperparams"]
        runtime_models.append(item)
    runtime["scalar_models"] = runtime_models

    runtime["curve_model"] = curve_config
    runtime["curve_transform"] = curve_transform
    runtime["curve_components"] = int(pca.n_components_)
    runtime["split_rows"] = {
        "id_train": int(id_train["X"].shape[0]),
        "id_val": int(id_val["X"].shape[0]),
        "id_test": int(id_test["X"].shape[0]),
        "ood_primary": int(ood["X"].shape[0]),
    }

    # Keep a short summary for quick checks
    summary = {}
    summary["id_scalar_rmse"] = build_scalar_rmse_summary(metrics_id["scalar"], target_names)
    summary["ood_scalar_rmse"] = build_scalar_rmse_summary(metrics_ood["scalar"], target_names)
    summary["id_curve_relative_l2"] = metrics_id["curve"]["relative_l2"]
    summary["ood_curve_relative_l2"] = metrics_ood["curve"]["relative_l2"]
    summary["id_curve_pearson_r"] = metrics_id["curve"]["pearson_r"]
    summary["ood_curve_pearson_r"] = metrics_ood["curve"]["pearson_r"]

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
    plot_scalar_parity(
        ood["y_scalar"],
        y_ood_pred,
        target_names,
        out_plot / "scalar_parity_ood.png",
        "Scalar parity (OOD)",
    )
    plot_curve_examples(id_test["t"], id_test["J"], J_id_pred, out_plot / "curve_examples_id.png", "J(t) examples (ID)")
    plot_curve_examples(ood["t"], ood["J"], J_ood_pred, out_plot / "curve_examples_ood.png", "J(t) examples (OOD)")
    return out_dir


def main():
    # Parse CLI arguments and run one training job.
    # This script is intentionally single-run
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", default="data/processed/ml")
    parser.add_argument("--out_dir", default="outputs/ml/blackbox")
    parser.add_argument("--curve_components", type=int, default=20)
    parser.add_argument("--use_xgboost", action="store_true")
    parser.add_argument("--no_curve_consistency", action="store_true")
    parser.add_argument("--alpha_grid", default="0.001,0.01,0.1,1,10,100")
    args = parser.parse_args()

    alpha_grid = []
    for part in args.alpha_grid.split(","):
        text = part.strip()
        if text:
            alpha_grid.append(float(text))
    if len(alpha_grid) == 0:
        raise ValueError("--alpha_grid is empty")

    # Train once with selected options and write outputs.
    out_dir = run_training(
        ml_dir=args.ml_dir,
        out_dir=args.out_dir,
        curve_components=args.curve_components,
        use_xgboost=bool(args.use_xgboost),
        curve_consistency=not bool(args.no_curve_consistency),
        alpha_grid=alpha_grid,
    )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
