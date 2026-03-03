import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.ml.blackbox_data import filter_split_by_run_index, load_feature_names, load_npz, load_target_names, remap_entry_run_dirs
from scripts.ml.common import extract_c0_feature, load_json, resolve_split_key
from scripts.ml.blackbox_model import fit_curve_model, fit_scalar_models, predict_curve_model, predict_scalar_models
from skin_diffusion.ml_curve_plots import plot_curve_error_over_time, plot_curve_examples
from skin_diffusion.ml_metrics import compute_curve_metrics
from skin_diffusion.ml_physics_diagnostics import build_worst_case_report, compute_split_physics_diagnostics, write_rows_csv
from skin_diffusion.ml_scalar_diagnostics import build_scalar_rmse_summary, plot_scalar_parity, plot_scalar_residual_hist, scalar_report, scalar_targets_from_flux_curves


def run_training(ml_dir, out_dir, curve_components, use_xgboost, alpha_grid, min_rows_for_xgboost, run_start_index=None, run_end_index=None, run_root_override=None, worst_case_top_n=10):
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

    train, train_index_after_filter = filter_split_by_run_index(train, split_index_map[train_split_key]["index"], run_start_index, run_end_index)
    val, val_index = filter_split_by_run_index(val, split_index_map[val_split_key]["index"], run_start_index, run_end_index)
    test, test_index = filter_split_by_run_index(test, split_index_map[test_split_key]["index"], run_start_index, run_end_index)
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
        raise ValueError("Feature count mismatch: X has " + str(train["X"].shape[1]) + " columns but meta feature_names has " + str(len(feature_names)))

    # Fit tabular scaler on train only.
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(train["X"])
    x_val = x_scaler.transform(val["X"])
    x_test = x_scaler.transform(test["X"])

    t0 = time.time()
    scalar_models = fit_scalar_models(x_train, train["y_scalar"], x_val, val["y_scalar"], target_names, use_xgboost=use_xgboost, alpha_grid=alpha_grid, min_rows_for_xgboost=min_rows_for_xgboost)
    scalar_train_seconds = float(time.time() - t0)

    y_test_pred = predict_scalar_models(scalar_models, x_test)
    y_val_pred = predict_scalar_models(scalar_models, x_val)

    t1 = time.time()
    pca, curve_transform, curve_model, curve_config = fit_curve_model(x_train, train["J"], x_val, val["J"], curve_components=curve_components, use_xgboost=use_xgboost, alpha_grid=alpha_grid)
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
    np.savez(out_dir / "pred_val.npz", j_true=val["J"], j_stageBase=j_val_pred, j_stageFinal=j_val_pred, t=val["t"])
    np.savez(out_dir / "pred_test.npz", j_true=test["J"], j_stageBase=j_test_pred, j_stageFinal=j_test_pred, t=test["t"])

    print("computing physics diagnostics for val ...")
    physics_val_rows, physics_val_summary = compute_split_physics_diagnostics(entries=val_entries, split_name="val", prediction_map={"truth": val["J"], "stageBase": j_val_pred, "stageFinal": j_val_pred}, progress_label="physics_val", truth_key="truth")
    print("computing physics diagnostics for test ...")
    physics_test_rows, physics_test_summary = compute_split_physics_diagnostics(entries=test_entries, split_name="test", prediction_map={"truth": test["J"], "stageBase": j_test_pred, "stageFinal": j_test_pred}, progress_label="physics_test", truth_key="truth")
    write_rows_csv(out_diag_physics / "physics_diag_val.csv", physics_val_rows)
    write_rows_csv(out_diag_physics / "physics_diag_test.csv", physics_test_rows)
    (out_diag_physics / "physics_diag_val_summary.json").write_text(json.dumps(physics_val_summary, indent=2), encoding="utf-8")
    (out_diag_physics / "physics_diag_test_summary.json").write_text(json.dumps(physics_test_summary, indent=2), encoding="utf-8")
    worst_val = build_worst_case_report(physics_val_rows, split_name="val", stage_name="stageFinal", top_n=worst_case_top_n)
    worst_test = build_worst_case_report(physics_test_rows, split_name="test", stage_name="stageFinal", top_n=worst_case_top_n)
    (out_diag_physics / "physics_diag_val_worst.json").write_text(json.dumps(worst_val, indent=2), encoding="utf-8")
    (out_diag_physics / "physics_diag_test_worst.json").write_text(json.dumps(worst_test, indent=2), encoding="utf-8")

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
    plot_scalar_parity(test["y_scalar"], y_test_curve, target_names, out_plot_test / "scalar_parity.png", "Scalar parity (test)")
    plot_scalar_residual_hist(test["y_scalar"], y_test_curve, target_names, out_plot_test / "scalar_residuals.png", "Scalar residuals (test)")
    plot_curve_examples(test["t"], test["J"], j_test_pred, out_plot_test / "curve_examples.png", "J(t) examples (test)", max_examples=9)
    plot_curve_error_over_time(test["t"], test["J"], j_test_pred, out_plot_test / "curve_error_over_time.png", "Curve error over time (test)")
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

    out_dir = run_training(ml_dir=args.ml_dir, out_dir=args.out_dir, curve_components=args.curve_components, use_xgboost=bool(args.use_xgboost), alpha_grid=alpha_grid, min_rows_for_xgboost=int(args.min_rows_for_xgboost), run_start_index=args.run_start_index, run_end_index=args.run_end_index, run_root_override=args.run_root_override, worst_case_top_n=max(1, int(args.worst_case_top_n)))
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
