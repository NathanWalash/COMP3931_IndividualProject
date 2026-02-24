import argparse
import time
from pathlib import Path

import numpy as np

from scripts.ml.train_pinn import (
    build_pinn_model,
    choose_device,
    detect_torch,
    feature_stats_from_json,
    load_pinn_config,
    normalize_feature_matrix,
)
from skin_diffusion.pinn_dataset import (
    load_run_fields,
    load_run_grid,
    load_run_meta,
    load_split_entries,
    load_split_feature_matrix,
)
from skin_diffusion.utils import ensure_dir, read_json, write_json


def choose_checkpoint_path(run_dir, checkpoint_arg):
    # Resolve which checkpoint file to load for evaluation.
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        return checkpoint_path

    best_path = run_dir / "checkpoints" / "best.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"
    if best_path.exists():
        return best_path
    if latest_path.exists():
        return latest_path
    raise ValueError(f"No checkpoint found in: {run_dir / 'checkpoints'}")


def choose_config_path(run_dir, config_arg):
    # Resolve config path. If not provided, read it from runtime.json.
    if config_arg is not None:
        config_path = Path(config_arg)
        if not config_path.exists():
            raise ValueError(f"Config file does not exist: {config_path}")
        return config_path

    runtime_path = run_dir / "runtime.json"
    if not runtime_path.exists():
        raise ValueError(f"Missing runtime.json in run_dir: {runtime_path}")
    runtime = read_json(runtime_path)
    config_text = runtime.get("config")
    if not isinstance(config_text, str):
        raise ValueError("runtime.json does not contain config path")
    config_path = Path(config_text)
    if not config_path.exists():
        raise ValueError(f"Config path in runtime.json does not exist: {config_path}")
    return config_path


def split_exists(ml_dir, split_name):
    # Check split presence in ml/meta.json.
    meta = read_json(Path(ml_dir) / "meta.json")
    splits = meta.get("splits", {})
    return split_name in splits


def predict_concentration(model, torch, device, x_unit, y_unit, t_unit, run_features, chunk_size):
    # Run model inference on flattened coordinates in chunks.
    count = int(x_unit.shape[0])
    pred = np.zeros(count, dtype=float)

    start = 0
    while start < count:
        end = min(count, start + int(chunk_size))
        x_tensor = torch.as_tensor(x_unit[start:end], dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y_unit[start:end], dtype=torch.float32, device=device)
        t_tensor = torch.as_tensor(t_unit[start:end], dtype=torch.float32, device=device)
        inputs = torch.stack([x_tensor, y_tensor, t_tensor], dim=1)
        if run_features is not None:
            # Concatenate normalized run features so eval matches conditional training.
            f_tensor = torch.as_tensor(run_features[start:end], dtype=torch.float32, device=device)
            inputs = torch.cat([inputs, f_tensor], dim=1)
        with torch.no_grad():
            c_pred = model(inputs).squeeze(-1)
        pred[start:end] = c_pred.detach().cpu().numpy().astype(float)
        start = end

    return pred


def predict_flux_curve_for_run(model, torch, device, fields, grid, run_feature, chunk_size):
    # Reconstruct J(t) from predicted concentrations at bottom boundary.
    # This uses the same discrete flux formula as the simulator metrics.
    d_field = fields["D"]
    t_curve = fields["t"]

    height = int(d_field.shape[0])
    width = int(d_field.shape[1])
    if height < 2:
        raise ValueError("Grid height must be at least 2 to compute bottom gradient")

    dx = float(grid["dx"])
    time_count = int(t_curve.shape[0])
    if time_count < 1:
        raise ValueError("Time curve is empty")

    # Build one flattened coordinate table for every x at every t.
    x_idx = np.tile(np.arange(width, dtype=float), time_count)
    t_idx = np.repeat(np.arange(time_count, dtype=int), width)

    x_den = float(max(1, width - 1))
    y_den = float(max(1, height - 1))
    t_den = float(max(1.0, float(t_curve[-1])))

    # Convert physical coordinates to normalized model inputs.
    x_unit = x_idx / x_den
    t_unit = t_curve[t_idx].astype(float) / t_den
    y_bottom_unit = np.full_like(x_unit, float(height - 1) / y_den, dtype=float)
    y_inner_unit = np.full_like(x_unit, float(height - 2) / y_den, dtype=float)
    run_features = None
    if run_feature is not None:
        # Repeat one feature row across all sampled points in this run.
        run_features = np.repeat(run_feature[None, :], x_unit.shape[0], axis=0)

    c_bottom = predict_concentration(model, torch, device, x_unit, y_bottom_unit, t_unit, run_features, chunk_size)
    c_inner = predict_concentration(model, torch, device, x_unit, y_inner_unit, t_unit, run_features, chunk_size)

    c_bottom = c_bottom.reshape(time_count, width)
    c_inner = c_inner.reshape(time_count, width)

    # Flux uses the finite-difference bottom gradient and bottom-adjacent diffusivity row.
    d_cdy = (c_bottom - c_inner) / dx
    d_bottom_row = d_field[-2, :]
    flux_profile = -d_bottom_row[None, :] * d_cdy
    return np.mean(flux_profile, axis=1)


def evaluate_split(model, torch, device, split_name, entries, split_features, chunk_size, max_rows_per_split):
    # Evaluate one split and return stacked true/predicted curves.
    use_entries = entries
    use_features = split_features
    if max_rows_per_split is not None:
        max_rows = int(max_rows_per_split)
        use_entries = entries[:max_rows]
        use_features = split_features[:max_rows]

    j_true_rows = []
    j_pred_rows = []
    t_rows = []
    run_dir_rows = []

    row_index = 0
    for entry in use_entries:
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)
        meta = load_run_meta(run_dir)
        grid = load_run_grid(meta)
        run_feature = use_features[row_index]

        j_true = fields["J"].astype(float)
        t_curve = fields["t"].astype(float)
        j_pred = predict_flux_curve_for_run(model, torch, device, fields, grid, run_feature, chunk_size)

        if j_pred.shape != j_true.shape:
            raise ValueError(f"J shape mismatch in {run_dir}: pred {j_pred.shape} vs true {j_true.shape}")

        j_true_rows.append(j_true)
        j_pred_rows.append(j_pred)
        t_rows.append(t_curve)
        run_dir_rows.append(str(run_dir))

        row_index += 1
        if row_index % 10 == 0:
            print("split", split_name, "rows_done", row_index, "of", len(use_entries))

    result = {}
    result["rows"] = int(len(use_entries))
    result["J_true"] = np.asarray(j_true_rows, dtype=float)
    result["J_pred"] = np.asarray(j_pred_rows, dtype=float)
    result["t"] = np.asarray(t_rows, dtype=float)
    result["run_dir"] = np.asarray(run_dir_rows, dtype=str)
    return result


def save_split_prediction(out_dir, split_name, split_result):
    # Save one split prediction bundle as NPZ.
    out_path = out_dir / f"pred_{split_name}.npz"
    np.savez(
        out_path,
        J_true=split_result["J_true"],
        J_pred=split_result["J_pred"],
        t=split_result["t"],
        run_dir=split_result["run_dir"],
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=131072)
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    ml_dir = Path(args.ml_dir)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not ml_dir.exists():
        raise ValueError(f"ml_dir does not exist: {ml_dir}")
    if not run_dir.exists():
        raise ValueError(f"run_dir does not exist: {run_dir}")

    config_path = choose_config_path(run_dir, args.config)
    checkpoint_path = choose_checkpoint_path(run_dir, args.checkpoint)

    torch_info = detect_torch()
    if not torch_info["available"]:
        raise ValueError("torch is required for PINN evaluation: " + str(torch_info["import_error"]))
    import torch

    # Rebuild the exact model architecture from config/checkpoint.
    cfg = load_pinn_config(config_path)

    chosen_device_text = choose_device(args.device, cfg["runtime"].get("device", "auto"), torch_info)
    if chosen_device_text == "cuda" and not torch_info["cuda_available"]:
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    device = torch.device(chosen_device_text)

    # Load checkpoint state and model weights.
    state = torch.load(checkpoint_path, map_location=device)
    model_cfg = cfg["model"]
    if isinstance(state.get("model_cfg_used"), dict):
        model_cfg = dict(state["model_cfg_used"])
    model = build_pinn_model(torch, model_cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Feature normalization must match training.
    feature_stats = None
    if isinstance(state.get("feature_stats"), dict):
        feature_stats = feature_stats_from_json(state["feature_stats"])
    condition_dim = int(model_cfg.get("condition_dim", 0))
    if condition_dim > 0 and feature_stats is None:
        raise ValueError("Checkpoint is missing feature_stats required for conditional PINN evaluation")

    print("PINN eval device:", chosen_device_text)
    print("PINN eval checkpoint:", checkpoint_path)

    # Evaluate ID test and optionally OOD if that split exists.
    split_map = {}
    split_map["id_test"] = load_split_entries(ml_dir=ml_dir, split_name="id_test", run_start_index=args.run_start_index, run_end_index=args.run_end_index)
    if split_exists(ml_dir, "ood_primary"):
        split_map["ood_primary"] = load_split_entries(ml_dir=ml_dir, split_name="ood_primary", run_start_index=args.run_start_index, run_end_index=args.run_end_index)
    else:
        split_map["ood_primary"] = []
    split_feature_map = {}

    for split_name in split_map:
        split_entries = split_map[split_name]
        if len(split_entries) == 0:
            split_feature_map[split_name] = np.zeros((0, condition_dim), dtype=float)
            continue
        if condition_dim == 0:
            split_feature_map[split_name] = np.zeros((len(split_entries), 0), dtype=float)
            continue
        # Pull raw split features, then normalize with train stats before inference.
        split_features_raw = load_split_feature_matrix(ml_dir, split_name, split_entries)
        split_features_norm = normalize_feature_matrix(split_features_raw, feature_stats)
        if int(split_features_norm.shape[1]) != condition_dim:
            raise ValueError(
                f"Condition dim mismatch in split {split_name}: "
                f"features {split_features_norm.shape[1]} vs model {condition_dim}"
            )
        split_feature_map[split_name] = split_features_norm

    saved_files = {}
    split_rows = {}

    # Run one full pass per split and write one prediction bundle per split.
    for split_name in split_map:
        entries = split_map[split_name]
        if len(entries) == 0:
            split_rows[split_name] = 0
            continue

        split_result = evaluate_split(model=model, torch=torch, device=device, split_name=split_name, entries=entries, split_features=split_feature_map[split_name], chunk_size=int(args.chunk_size), max_rows_per_split=args.max_rows_per_split)
        saved_path = save_split_prediction(out_dir, split_name, split_result)
        saved_files[split_name] = str(saved_path.resolve())
        split_rows[split_name] = int(split_result["rows"])

    runtime = {}
    runtime["status"] = "evaluated"
    runtime["ml_dir"] = str(ml_dir.resolve())
    runtime["run_dir"] = str(run_dir.resolve())
    runtime["config"] = str(config_path.resolve())
    runtime["checkpoint"] = str(checkpoint_path.resolve())
    runtime["device"] = chosen_device_text
    runtime["torch"] = torch_info
    runtime["condition_dim"] = int(condition_dim)
    runtime["run_index_filter"] = {"start": args.run_start_index, "end": args.run_end_index}
    runtime["chunk_size"] = int(args.chunk_size)
    runtime["max_rows_per_split"] = args.max_rows_per_split
    runtime["split_rows"] = split_rows
    runtime["prediction_files"] = saved_files
    runtime["seconds"] = float(time.time() - t0)

    write_json(out_dir / "runtime.json", runtime)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
