import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from skin_diffusion.ml_curve_plots import plot_curve_error_over_time, plot_curve_examples
from skin_diffusion.ml_metrics import compute_curve_metrics
from skin_diffusion.ml_physics_diagnostics import (
    build_worst_case_report,
    compute_split_physics_diagnostics,
    write_rows_csv,
)
from skin_diffusion.ml_run_dataset import remap_run_dir
from skin_diffusion.ml_scalar_diagnostics import (
    build_scalar_rmse_summary,
    plot_scalar_parity,
    plot_scalar_residual_hist,
    scalar_report,
    scalar_targets_from_flux_curves,
)
from skin_diffusion.run_index import in_index_range, run_index_from_path
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
    target_names = []
    for name in names:
        target_names.append(str(name))
    return target_names


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
    feature_names = []
    for name in names:
        feature_names.append(str(name))
    return feature_names


def load_npz(path):
    # Load one dataset split from NPZ.
    data = np.load(path)
    result = {}
    result["X"] = data["X"]
    result["y_scalar"] = data["y_scalar"]
    result["J"] = data["J"]
    result["t"] = data["t"]
    return result


def resolve_split_key(split_index_map, logical_name):
    # Use canonical split keys only.
    split_text = str(logical_name)
    if split_text in split_index_map:
        return split_text
    raise ValueError(
        "Could not find split key "
        + split_text
        + ". Expected one of: train, val, test"
    )


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


def remap_entry_run_dirs(entries, run_root_override):
    # Keep split index rows but rewrite run_dir to staged-local location when requested.
    remapped = []
    for entry in entries:
        row = dict(entry)
        row["run_dir"] = remap_run_dir(entry["run_dir"], run_root_override)
        remapped.append(row)
    return remapped


def extract_c0_feature(x_raw, feature_names):
    # C0 is needed for fair scalar comparison from predicted curves.
    if "C0" not in feature_names:
        return np.full((int(x_raw.shape[0]),), np.nan, dtype=np.float32)
    c0_col = int(feature_names.index("C0"))
    return np.asarray(x_raw[:, c0_col], dtype=np.float32)


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


def run_training(
    ml_dir,
    out_dir,
    curve_components,
    use_xgboost,
    alpha_grid,
    min_rows_for_xgboost,
    run_start_index=None,
    run_end_index=None,
    run_root_override=None,
    worst_case_top_n=10,
):
    # Run one full train/evaluate pass and save outputs for train/val/test only.
    ml_dir = Path(ml_dir)
    out_dir = Path(out_dir)
    out_plot = out_dir / "plots"
    out_plot_test = out_plot / "test"
    out_diag = out_dir / "diagnostics"
    out_diag_physics = out_diag / "physics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot_test.mkdir(parents=True, exist_ok=True)
    out_diag_physics.mkdir(parents=True, exist_ok=True)

    target_names = load_target_names(ml_dir)
    feature_names = load_feature_names(ml_dir)
    ml_meta = load_json(ml_dir / "meta.json")
    split_index_map = ml_meta["splits"]

    train_split_key = resolve_split_key(split_index_map, "train")
    val_split_key = resolve_split_key(split_index_map, "val")
    test_split_key = resolve_split_key(split_index_map, "test")

    train = load_npz(ml_dir / (train_split_key + ".npz"))
    val = load_npz(ml_dir / (val_split_key + ".npz"))
    test = load_npz(ml_dir / (test_split_key + ".npz"))

    train, train_index_after_filter = filter_split_by_run_index(
        train,
        split_index_map[train_split_key]["index"],
        run_start_index,
        run_end_index,
    )
    val, val_index = filter_split_by_run_index(
        val,
        split_index_map[val_split_key]["index"],
        run_start_index,
        run_end_index,
    )
    test, test_index = filter_split_by_run_index(
        test,
        split_index_map[test_split_key]["index"],
        run_start_index,
        run_end_index,
    )
    # Keep this for row-count traceability in runtime/debug tooling.
    train_filtered_row_count = int(len(train_index_after_filter))

    val_entries = remap_entry_run_dirs(val_index, run_root_override)
    test_entries = remap_entry_run_dirs(test_index, run_root_override)

    if train["X"].shape[0] == 0:
        raise ValueError("No train rows left after run-index filtering")
    if val["X"].shape[0] == 0:
        raise ValueError("No val rows left after run-index filtering")
    if test["X"].shape[0] == 0:
        raise ValueError("No test rows left after run-index filtering")

    if train["X"].shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch: X has "
            + str(train["X"].shape[1])
            + " columns but meta feature_names has "
            + str(len(feature_names))
        )

    # Fit tabular scaler on train only.
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(train["X"])
    x_val = x_scaler.transform(val["X"])
    x_test = x_scaler.transform(test["X"])

    t0 = time.time()
    scalar_models = fit_scalar_models(
        x_train,
        train["y_scalar"],
        x_val,
        val["y_scalar"],
        target_names,
        use_xgboost=use_xgboost,
        alpha_grid=alpha_grid,
        min_rows_for_xgboost=min_rows_for_xgboost,
    )
    scalar_train_seconds = float(time.time() - t0)

    y_test_pred = predict_scalar_models(scalar_models, x_test)
    y_val_pred = predict_scalar_models(scalar_models, x_val)

    t1 = time.time()
    pca, curve_transform, curve_model, curve_config = fit_curve_model(
        x_train,
        train["J"],
        x_val,
        val["J"],
        curve_components=curve_components,
        use_xgboost=use_xgboost,
        alpha_grid=alpha_grid,
    )
    curve_train_seconds = float(time.time() - t1)

    j_val_pred = predict_curve_model(pca, curve_transform, curve_model, x_val)
    j_test_pred = predict_curve_model(pca, curve_transform, curve_model, x_test)

    c0_val = extract_c0_feature(val["X"], feature_names)
    c0_test = extract_c0_feature(test["X"], feature_names)
    y_val_curve = scalar_targets_from_flux_curves(j_val_pred, val["t"], c0_val, target_names)
    y_test_curve = scalar_targets_from_flux_curves(j_test_pred, test["t"], c0_test, target_names)

    # Keep a common staged metric structure across all models.
    metrics_val = {
        "available": True,
        "stageBase": {
            "curve": compute_curve_metrics(val["J"], j_val_pred, val["t"]),
            "scalar": scalar_report(val["y_scalar"], y_val_curve, target_names),
        },
        "stageFinal": {
            "curve": compute_curve_metrics(val["J"], j_val_pred, val["t"]),
            "scalar": scalar_report(val["y_scalar"], y_val_curve, target_names),
        },
    }
    metrics_test = {
        "available": True,
        "stageBase": {
            "curve": compute_curve_metrics(test["J"], j_test_pred, test["t"]),
            "scalar": scalar_report(test["y_scalar"], y_test_curve, target_names),
        },
        "stageFinal": {
            "curve": compute_curve_metrics(test["J"], j_test_pred, test["t"]),
            "scalar": scalar_report(test["y_scalar"], y_test_curve, target_names),
        },
    }

    # Save prediction bundles with aligned key names.
    np.savez(
        out_dir / "pred_val.npz",
        j_true=val["J"],
        j_stageBase=j_val_pred,
        j_stageFinal=j_val_pred,
        t=val["t"],
    )
    np.savez(
        out_dir / "pred_test.npz",
        j_true=test["J"],
        j_stageBase=j_test_pred,
        j_stageFinal=j_test_pred,
        t=test["t"],
    )

    print("computing physics diagnostics for val ...")
    physics_val_rows, physics_val_summary = compute_split_physics_diagnostics(
        entries=val_entries,
        split_name="val",
        prediction_map={
            "truth": val["J"],
            "stageBase": j_val_pred,
            "stageFinal": j_val_pred,
        },
        progress_label="physics_val",
        truth_key="truth",
    )
    print("computing physics diagnostics for test ...")
    physics_test_rows, physics_test_summary = compute_split_physics_diagnostics(
        entries=test_entries,
        split_name="test",
        prediction_map={
            "truth": test["J"],
            "stageBase": j_test_pred,
            "stageFinal": j_test_pred,
        },
        progress_label="physics_test",
        truth_key="truth",
    )
    write_rows_csv(out_diag_physics / "physics_diag_val.csv", physics_val_rows)
    write_rows_csv(out_diag_physics / "physics_diag_test.csv", physics_test_rows)
    (out_diag_physics / "physics_diag_val_summary.json").write_text(
        json.dumps(physics_val_summary, indent=2),
        encoding="utf-8",
    )
    (out_diag_physics / "physics_diag_test_summary.json").write_text(
        json.dumps(physics_test_summary, indent=2),
        encoding="utf-8",
    )
    worst_val = build_worst_case_report(
        physics_val_rows,
        split_name="val",
        stage_name="stageFinal",
        top_n=worst_case_top_n,
    )
    worst_test = build_worst_case_report(
        physics_test_rows,
        split_name="test",
        stage_name="stageFinal",
        top_n=worst_case_top_n,
    )
    (out_diag_physics / "physics_diag_val_worst.json").write_text(
        json.dumps(worst_val, indent=2),
        encoding="utf-8",
    )
    (out_diag_physics / "physics_diag_test_worst.json").write_text(
        json.dumps(worst_test, indent=2),
        encoding="utf-8",
    )

    runtime = {}
    runtime["scalar_train_seconds"] = scalar_train_seconds
    runtime["curve_train_seconds"] = curve_train_seconds
    runtime["use_xgboost"] = bool(use_xgboost)
    runtime["target_names"] = target_names
    runtime["feature_names"] = feature_names
    runtime["feature_count"] = int(len(feature_names))

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
        "train": int(train["X"].shape[0]),
        "val": int(val["X"].shape[0]),
        "test": int(test["X"].shape[0]),
    }
    runtime["train_filtered_row_count"] = int(train_filtered_row_count)
    runtime["run_index_filter"] = {
        "start": run_start_index,
        "end": run_end_index,
    }
    runtime["run_root_override"] = run_root_override
    runtime["physics_diagnostics"] = {
        "val_summary_file": "diagnostics/physics/physics_diag_val_summary.json",
        "test_summary_file": "diagnostics/physics/physics_diag_test_summary.json",
        "val_worst_file": "diagnostics/physics/physics_diag_val_worst.json",
        "test_worst_file": "diagnostics/physics/physics_diag_test_worst.json",
    }

    summary = {}
    summary["test_scalar_rmse"] = build_scalar_rmse_summary(metrics_test["stageFinal"]["scalar"], target_names)
    summary["val_curve_relative_l2"] = metrics_val["stageFinal"]["curve"]["relative_l2"]
    summary["test_curve_relative_l2"] = metrics_test["stageFinal"]["curve"]["relative_l2"]
    summary["val_curve_pearson_r"] = metrics_val["stageFinal"]["curve"]["pearson_r"]
    summary["test_curve_pearson_r"] = metrics_test["stageFinal"]["curve"]["pearson_r"]

    (out_dir / "metrics_val.json").write_text(json.dumps(metrics_val, indent=2), encoding="utf-8")
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Final-stage visual diagnostics mirror the other trainers.
    plot_scalar_parity(
        test["y_scalar"],
        y_test_curve,
        target_names,
        out_plot_test / "scalar_parity.png",
        "Scalar parity (test)",
    )
    plot_scalar_residual_hist(
        test["y_scalar"],
        y_test_curve,
        target_names,
        out_plot_test / "scalar_residuals.png",
        "Scalar residuals (test)",
    )
    plot_curve_examples(
        test["t"],
        test["J"],
        j_test_pred,
        out_plot_test / "curve_examples.png",
        "J(t) examples (test)",
        max_examples=9,
    )
    plot_curve_error_over_time(
        test["t"],
        test["J"],
        j_test_pred,
        out_plot_test / "curve_error_over_time.png",
        "Curve error over time (test)",
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
    parser.add_argument("--run_root_override", default=None)
    parser.add_argument("--worst_case_top_n", type=int, default=10)
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
        run_root_override=args.run_root_override,
        worst_case_top_n=max(1, int(args.worst_case_top_n)),
    )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
