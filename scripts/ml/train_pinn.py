import argparse
import time
from pathlib import Path

import numpy as np

from skin_diffusion.dataset_spec import load_yaml_file
from skin_diffusion.pinn_dataset import (
    choose_run_rows,
    load_run_fields,
    load_run_grid,
    load_run_meta,
    load_split_feature_matrix,
    load_split_entries,
    sample_boundary_batch,
    sample_collocation_batch,
    sample_initial_batch,
)
from skin_diffusion.utils import ensure_dir, read_json, write_json


"""
PINN training script.

This file trains a concentration model C(x, y, t) with data, PDE, boundary,
initial-condition, and nonnegativity losses. It reads ML split metadata plus
run-level bundles, then writes checkpoints and training reports.
"""


def load_split_shapes(path):
    # Read NPZ array shapes only (memory-map) so validation stays lightweight.
    with np.load(path, mmap_mode="r") as data:
        shapes = {}
        shapes["rows"] = int(data["X"].shape[0])
        shapes["feature_count"] = int(data["X"].shape[1])
        shapes["scalar_target_count"] = int(data["y_scalar"].shape[1])
        shapes["curve_points"] = int(data["J"].shape[1])
    return shapes


def validate_ml_dir(ml_dir):
    # Check expected files and ensure split schema matches meta.json.
    # This catches wrong directories early before expensive training starts.
    required_files = []
    required_files.append("meta.json")
    required_files.append("id_train.npz")
    required_files.append("id_val.npz")
    required_files.append("id_test.npz")

    for name in required_files:
        path = ml_dir / name
        if not path.exists():
            raise ValueError(f"Missing required file: {path}")

    meta = read_json(ml_dir / "meta.json")

    scalar_target_names = meta.get("scalar_target_names", [])
    feature_names = meta.get("feature_names", [])
    if not isinstance(scalar_target_names, list) or len(scalar_target_names) == 0:
        raise ValueError("meta.json is missing scalar_target_names")
    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("meta.json is missing feature_names")

    # Build split map explicitly for readability.
    splits = {}
    splits["id_train"] = load_split_shapes(ml_dir / "id_train.npz")
    splits["id_val"] = load_split_shapes(ml_dir / "id_val.npz")
    splits["id_test"] = load_split_shapes(ml_dir / "id_test.npz")

    # OOD split can exist in the same folder but is not required for training.
    ood_path = ml_dir / "ood_primary.npz"
    if ood_path.exists():
        splits["ood_primary"] = load_split_shapes(ood_path)

    expected_feature_count = len(feature_names)
    expected_target_count = len(scalar_target_names)

    # Keep split-level shape checks explicit for easier debugging.
    for split_name in splits:
        split_info = splits[split_name]
        if split_info["feature_count"] != expected_feature_count:
            raise ValueError(f"Feature count mismatch in {split_name}: npz has {split_info['feature_count']} but meta has {expected_feature_count}")
        if split_info["scalar_target_count"] != expected_target_count:
            raise ValueError(f"Target count mismatch in {split_name}: npz has {split_info['scalar_target_count']} but meta has {expected_target_count}")

    summary = {}
    summary["feature_names"] = feature_names
    summary["scalar_target_names"] = scalar_target_names
    summary["splits"] = splits
    return summary


def load_pinn_config(config_path):
    # Load training config and enforce required top-level blocks.
    cfg = load_yaml_file(str(config_path))
    if not isinstance(cfg, dict):
        raise ValueError(f"PINN config is not a dictionary: {config_path}")

    required_blocks = ["model", "training", "optimizer", "loss_weights", "physics", "runtime"]

    for name in required_blocks:
        if name not in cfg:
            raise ValueError(f"Missing block in PINN config: {name}")

    return cfg


def compute_feature_stats(feature_matrix):
    # Fit normalization stats on one feature matrix (normally id_train features).
    if feature_matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2D")
    mean = np.mean(feature_matrix, axis=0)
    std = np.std(feature_matrix, axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    stats = {}
    stats["mean"] = mean.astype(float)
    stats["std"] = std_safe.astype(float)
    return stats


def normalize_feature_matrix(feature_matrix, feature_stats):
    # Apply z-score normalization with precomputed stats.
    mean = np.asarray(feature_stats["mean"], dtype=float)
    std = np.asarray(feature_stats["std"], dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2D")
    if feature_matrix.shape[1] != mean.shape[0]:
        raise ValueError("Feature matrix width does not match feature stats")
    return (feature_matrix - mean[None, :]) / std[None, :]


def feature_stats_to_json(feature_stats):
    # Convert numpy feature stats to plain Python lists for JSON/checkpoints.
    data = {}
    data["mean"] = np.asarray(feature_stats["mean"], dtype=float).tolist()
    data["std"] = np.asarray(feature_stats["std"], dtype=float).tolist()
    return data


def feature_stats_from_json(data):
    # Read feature stats dict saved in checkpoint/runtime metadata.
    stats = {}
    stats["mean"] = np.asarray(data["mean"], dtype=float)
    stats["std"] = np.asarray(data["std"], dtype=float)
    return stats


def load_split_curve_matrix(ml_dir, split_name, entries):
    # Load split J rows that correspond to the provided entry list.
    # This keeps train/val/test curves aligned with the same split rows used for features.
    split_path = Path(ml_dir) / f"{split_name}.npz"
    if not split_path.exists():
        raise ValueError(f"Missing split array file: {split_path}")

    with np.load(split_path, mmap_mode="r") as data:
        j_all = np.asarray(data["J"], dtype=float)

    row_ids = []
    for entry in entries:
        # split_row is attached by load_split_entries and points into split NPZ rows.
        if "split_row" not in entry:
            raise ValueError(f"Entry is missing split_row for split {split_name}")
        row_ids.append(int(entry["split_row"]))

    if len(row_ids) == 0:
        return np.zeros((0, int(j_all.shape[1])), dtype=float)

    row_array = np.asarray(row_ids, dtype=int)
    return np.asarray(j_all[row_array], dtype=float)


def compute_flux_normalization(j_curves, physics_cfg):
    # Pick a stable scale for relative flux loss from train split curves.
    # Using a quantile avoids tiny denominators from near-zero tail values.
    quantile = float(physics_cfg.get("flux_rel_eps_quantile", 90.0))
    positive = np.asarray(j_curves, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return 1e-12
    eps = float(np.percentile(positive, quantile))
    return max(eps, 1e-12)


def sample_flux_time_indices(rng, j_curve, point_count, weighted_fraction, weight_power):
    # Sample time indices with a weighted+uniform mix to focus on peak regions.
    count = int(point_count)
    if count <= 0:
        return np.array([], dtype=int)

    time_count = int(len(j_curve))
    if time_count <= 0:
        return np.array([], dtype=int)

    weighted_fraction = float(np.clip(weighted_fraction, 0.0, 1.0))
    n_weighted = int(round(weighted_fraction * count))
    n_uniform = int(count - n_weighted)

    picks = []
    # Always include the peak index so high-flux behavior is never skipped.
    peak_idx = int(np.argmax(j_curve))
    picks.append(np.array([peak_idx], dtype=int))

    remaining = count - 1
    if remaining <= 0:
        return picks[0]

    if n_weighted > remaining:
        n_weighted = remaining
        n_uniform = 0
    else:
        n_uniform = remaining - n_weighted

    if n_weighted > 0:
        weights = np.asarray(j_curve, dtype=float)
        weights = np.maximum(weights, 0.0)
        weights = np.power(weights, float(weight_power))
        total = float(np.sum(weights))
        if total > 0.0:
            # Draw weighted time points when the curve has usable positive mass.
            probs = weights / total
            weighted_idx = rng.choice(time_count, size=n_weighted, replace=True, p=probs)
        else:
            # Fallback to uniform when all weights collapse to zero.
            weighted_idx = rng.integers(0, time_count, size=n_weighted)
        picks.append(np.asarray(weighted_idx, dtype=int))

    if n_uniform > 0:
        uniform_idx = rng.integers(0, time_count, size=n_uniform)
        picks.append(np.asarray(uniform_idx, dtype=int))

    return np.concatenate(picks, axis=0)


def detect_torch():
    # Collect torch and CUDA availability once for reporting and device selection.
    info = {}
    info["available"] = False
    info["version"] = None
    info["cuda_available"] = False
    info["cuda_device_count"] = 0
    info["import_error"] = None

    try:
        import torch

        info["available"] = True
        info["version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        info["import_error"] = str(exc)

    return info


def choose_device(device_arg, config_device, torch_info):
    # Device selection uses CLI value first, then config value.
    # The "auto" setting resolves to cuda when available, otherwise cpu.
    if device_arg is not None:
        requested = str(device_arg)
    else:
        requested = str(config_device)

    if requested == "auto":
        if torch_info["available"] and torch_info["cuda_available"]:
            return "cuda"
        return "cpu"

    return requested


def build_activation(torch, name):
    # Map simple config text to torch activation module.
    text = str(name).strip().lower()
    if text == "tanh":
        return torch.nn.Tanh()
    if text == "relu":
        return torch.nn.ReLU()
    if text == "softplus":
        return torch.nn.Softplus()
    if text == "identity":
        return None
    if text == "none":
        return None
    raise ValueError("Unknown activation: " + str(name))


def build_pinn_model(torch, model_cfg):
    # Build a plain MLP that maps [x_unit, y_unit, t_unit, run_features...] to concentration.
    # Coordinates are normalized here and converted to physical derivatives in the loss.
    hidden_layers = int(model_cfg.get("hidden_layers", 4))
    hidden_width = int(model_cfg.get("hidden_width", 128))
    condition_dim = int(model_cfg.get("condition_dim", 0))
    activation_name = model_cfg.get("activation", "tanh")
    output_activation_name = model_cfg.get("output_activation", "softplus")

    activation = build_activation(torch, activation_name)
    output_activation = build_activation(torch, output_activation_name)

    layers = []
    in_width = 3 + int(condition_dim)
    layer_count = 0
    while layer_count < hidden_layers:
        layers.append(torch.nn.Linear(in_width, hidden_width))
        if activation is not None:
            layers.append(build_activation(torch, activation_name))
        in_width = hidden_width
        layer_count += 1
    layers.append(torch.nn.Linear(in_width, 1))
    if output_activation is not None:
        layers.append(output_activation)

    return torch.nn.Sequential(*layers)


def to_tensor(torch, values, device):
    # Convert numpy -> float32 torch tensor on selected device.
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def build_input_tensor(torch, x_vals, y_vals, t_vals, device, run_features_vals=None, requires_grad=False):
    # Stack coordinate arrays and optional run features into one input tensor.
    # Set requires_grad only when derivative terms are needed in the loss.
    x_t = to_tensor(torch, x_vals, device)
    y_t = to_tensor(torch, y_vals, device)
    t_t = to_tensor(torch, t_vals, device)
    inputs = torch.stack([x_t, y_t, t_t], dim=1)
    if run_features_vals is not None:
        run_features_t = to_tensor(torch, run_features_vals, device)
        inputs = torch.cat([inputs, run_features_t], dim=1)
    if requires_grad:
        inputs.requires_grad_(True)
    return inputs


def zero_loss(torch, device):
    # Return scalar zero tensor on correct device.
    return torch.zeros(1, dtype=torch.float32, device=device).mean()


def mse_loss_safe(torch, pred, target, device):
    # Standard MSE with safe empty-input handling.
    if pred.numel() == 0:
        return zero_loss(torch, device)
    return torch.mean((pred - target) ** 2)


def compute_collocation_losses(torch, model, batch, device):
    # Interior loss terms for data fit, PDE residual, and nonnegativity.
    inputs = build_input_tensor(
        torch,
        batch["x_unit"],
        batch["y_unit"],
        batch["t_unit"],
        device,
        run_features_vals=batch["run_features"],
        requires_grad=True,
    )
    c_now_pred = model(inputs).squeeze(-1)

    # Data term keeps the PINN tied to simulator concentrations.
    c_now_true = to_tensor(torch, batch["C_now_true"], device)
    loss_data = mse_loss_safe(torch, c_now_pred, c_now_true, device)

    # First derivatives are taken in normalized coordinates.
    # The graph is kept because second derivatives are needed for the PDE term.
    grad_out = torch.ones_like(c_now_pred)
    grad_now = torch.autograd.grad(
        c_now_pred,
        inputs,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
    )[0]
    dc_dx_unit = grad_now[:, 0]
    dc_dy_unit = grad_now[:, 1]
    dc_dt_unit = grad_now[:, 2]

    # Second derivatives for the Laplacian are also in normalized coordinates.
    # These terms form the diffusion operator d2C/dx2 + d2C/dy2.
    grad_x = torch.autograd.grad(
        dc_dx_unit,
        inputs,
        grad_outputs=torch.ones_like(dc_dx_unit),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_y = torch.autograd.grad(
        dc_dy_unit,
        inputs,
        grad_outputs=torch.ones_like(dc_dy_unit),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2c_dx2_unit = grad_x[:, 0]
    d2c_dy2_unit = grad_y[:, 1]

    # Convert derivatives back to physical units with the chain rule.
    x_scale = to_tensor(torch, batch["x_scale"], device)
    y_scale = to_tensor(torch, batch["y_scale"], device)
    t_scale = to_tensor(torch, batch["t_scale"], device)
    dc_dt = dc_dt_unit / t_scale
    d2c_dx2 = d2c_dx2_unit / (x_scale * x_scale)
    d2c_dy2 = d2c_dy2_unit / (y_scale * y_scale)

    d_vals = to_tensor(torch, batch["D"], device)
    k_vals = to_tensor(torch, batch["k"], device)

    # PDE: dC/dt = D * Laplacian(C) - k * C.
    laplacian = d2c_dx2 + d2c_dy2
    residual = dc_dt - (d_vals * laplacian - k_vals * c_now_pred)
    loss_pde = torch.mean(residual ** 2)

    # Penalize negative concentrations directly in prediction space.
    loss_nonneg = torch.mean(torch.relu(-c_now_pred) ** 2)

    losses = {}
    losses["data_curve"] = loss_data
    losses["pde"] = loss_pde
    losses["nonnegativity"] = loss_nonneg
    return losses


def compute_initial_loss(torch, model, batch, device):
    # Initial condition loss at t=0 against simulator field snapshot.
    inputs = build_input_tensor(
        torch,
        batch["x_unit"],
        batch["y_unit"],
        batch["t_unit"],
        device,
        run_features_vals=batch["run_features"],
        requires_grad=False,
    )
    pred = model(inputs).squeeze(-1)
    target = to_tensor(torch, batch["C_true"], device)
    return mse_loss_safe(torch, pred, target, device)


def compute_boundary_loss(torch, model, batch, device):
    # Average loss over active BC groups: top patch, top off-patch, bottom sink, sides.
    loss_total = zero_loss(torch, device)
    count = 0

    top_patch = batch["top_patch"]
    if len(top_patch["x_unit"]) > 0:
        # Enforce concentration target on patch-covered top boundary points.
        top_patch_inputs = build_input_tensor(
            torch,
            top_patch["x_unit"],
            top_patch["y_unit"],
            top_patch["t_unit"],
            device,
            run_features_vals=top_patch["run_features"],
            requires_grad=False,
        )
        top_patch_pred = model(top_patch_inputs).squeeze(-1)
        top_patch_target = to_tensor(torch, top_patch["C_target"], device)
        loss_total = loss_total + mse_loss_safe(torch, top_patch_pred, top_patch_target, device)
        count += 1

    top_offpatch = batch["top_offpatch"]
    if len(top_offpatch["x_unit"]) > 0:
        # Use boundary-minus-inner difference as a finite-difference no-flux term.
        top_offpatch_boundary_inputs = build_input_tensor(
            torch,
            top_offpatch["x_unit"],
            top_offpatch["y_boundary_unit"],
            top_offpatch["t_unit"],
            device,
            run_features_vals=top_offpatch["run_features"],
            requires_grad=False,
        )
        top_offpatch_inner_inputs = build_input_tensor(
            torch,
            top_offpatch["x_unit"],
            top_offpatch["y_inner_unit"],
            top_offpatch["t_unit"],
            device,
            run_features_vals=top_offpatch["run_features"],
            requires_grad=False,
        )
        top_offpatch_boundary_pred = model(top_offpatch_boundary_inputs).squeeze(-1)
        top_offpatch_inner_pred = model(top_offpatch_inner_inputs).squeeze(-1)
        top_offpatch_target = torch.zeros_like(top_offpatch_boundary_pred)
        top_offpatch_diff = top_offpatch_boundary_pred - top_offpatch_inner_pred
        loss_total = loss_total + mse_loss_safe(torch, top_offpatch_diff, top_offpatch_target, device)
        count += 1

    bottom_sink = batch["bottom_sink"]
    if len(bottom_sink["x_unit"]) > 0:
        # Apply sink condition at the bottom boundary.
        bottom_sink_inputs = build_input_tensor(
            torch,
            bottom_sink["x_unit"],
            bottom_sink["y_unit"],
            bottom_sink["t_unit"],
            device,
            run_features_vals=bottom_sink["run_features"],
            requires_grad=False,
        )
        bottom_sink_pred = model(bottom_sink_inputs).squeeze(-1)
        bottom_sink_target = to_tensor(torch, bottom_sink["C_target"], device)
        loss_total = loss_total + mse_loss_safe(torch, bottom_sink_pred, bottom_sink_target, device)
        count += 1

    side_neumann = batch["side_neumann"]
    if len(side_neumann["x_boundary_unit"]) > 0:
        # Same no-flux form on left/right sides.
        side_boundary_inputs = build_input_tensor(
            torch,
            side_neumann["x_boundary_unit"],
            side_neumann["y_unit"],
            side_neumann["t_unit"],
            device,
            run_features_vals=side_neumann["run_features"],
            requires_grad=False,
        )
        side_inner_inputs = build_input_tensor(
            torch,
            side_neumann["x_inner_unit"],
            side_neumann["y_unit"],
            side_neumann["t_unit"],
            device,
            run_features_vals=side_neumann["run_features"],
            requires_grad=False,
        )
        side_boundary_pred = model(side_boundary_inputs).squeeze(-1)
        side_inner_pred = model(side_inner_inputs).squeeze(-1)
        side_target = torch.zeros_like(side_boundary_pred)
        side_diff = side_boundary_pred - side_inner_pred
        loss_total = loss_total + mse_loss_safe(torch, side_diff, side_target, device)
        count += 1

    if count == 0:
        # Should not normally happen, but keep output tensor valid.
        return zero_loss(torch, device)

    return loss_total / float(count)


def compute_flux_curve_loss(torch, model, entries, feature_matrix, physics_cfg, device, seed_base, flux_eps):
    # Match predicted bottom flux to simulator J(t) on sampled runs/time points.
    flux_runs_per_batch = int(physics_cfg.get("flux_runs_per_batch", 2))
    flux_time_points_per_run = int(physics_cfg.get("flux_time_points_per_run", 32))
    weighted_fraction = float(physics_cfg.get("flux_time_weighted_fraction", 0.7))
    weight_power = float(physics_cfg.get("flux_time_weight_power", 1.0))
    residual_clip = float(physics_cfg.get("flux_rel_residual_clip", 20.0))
    if flux_runs_per_batch <= 0 or flux_time_points_per_run <= 0:
        return zero_loss(torch, device)

    rng = np.random.default_rng(int(seed_base) + 71)
    # Sample a small run subset each step to keep this term affordable.
    run_rows = choose_run_rows(rng, len(entries), flux_runs_per_batch)
    loss_terms = []

    for run_row in run_rows:
        entry = entries[int(run_row)]
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)
        meta = load_run_meta(run_dir)
        grid = load_run_grid(meta)

        d_field = np.asarray(fields["D"], dtype=float)
        t_curve = np.asarray(fields["t"], dtype=float)
        j_true = np.asarray(fields["J"], dtype=float)
        dx = float(grid["dx"])

        height = int(d_field.shape[0])
        width = int(d_field.shape[1])
        if height < 2:
            raise ValueError(f"Grid height must be >= 2 for flux loss: {run_dir}")
        if width < 1:
            raise ValueError(f"Grid width must be >= 1 for flux loss: {run_dir}")
        if len(t_curve) < 1:
            raise ValueError(f"Time curve is empty for flux loss: {run_dir}")

        x_den = float(max(1, width - 1))
        y_den = float(max(1, height - 1))
        t_den = float(max(1.0, float(t_curve[-1])))
        d_bottom = d_field[-2, :].astype(float)
        run_feature = feature_matrix[int(run_row)]
        picked_times = sample_flux_time_indices(
            rng,
            j_true,
            int(flux_time_points_per_run),
            weighted_fraction,
            weight_power,
        )
        for t_idx in picked_times:
            x_idx = np.arange(width, dtype=float)
            x_unit = x_idx / x_den
            y_bottom_unit = np.full(width, float(height - 1) / y_den, dtype=float)
            y_inner_unit = np.full(width, float(height - 2) / y_den, dtype=float)
            t_value = float(t_curve[int(t_idx)])
            t_unit = np.full(width, t_value / t_den, dtype=float)
            run_features = np.repeat(run_feature[None, :], width, axis=0)

            bottom_inputs = build_input_tensor(torch, x_unit, y_bottom_unit, t_unit, device, run_features_vals=run_features, requires_grad=False)
            inner_inputs = build_input_tensor(torch, x_unit, y_inner_unit, t_unit, device, run_features_vals=run_features, requires_grad=False)

            c_bottom = model(bottom_inputs).squeeze(-1)
            c_inner = model(inner_inputs).squeeze(-1)
            d_cdy = (c_bottom - c_inner) / float(dx)
            d_bottom_tensor = to_tensor(torch, d_bottom, device)
            flux_profile = -d_bottom_tensor * d_cdy
            flux_pred = torch.mean(flux_profile)
            flux_true = torch.as_tensor(float(j_true[int(t_idx)]), dtype=torch.float32, device=device)
            flux_scale = max(abs(float(j_true[int(t_idx)])), float(flux_eps))
            # Relative residual keeps low- and high-flux times on a similar scale.
            flux_scale_tensor = torch.as_tensor(flux_scale, dtype=torch.float32, device=device)
            residual = (flux_pred - flux_true) / flux_scale_tensor
            if residual_clip > 0.0:
                residual = torch.clamp(residual, min=-residual_clip, max=residual_clip)
            loss_terms.append(residual ** 2)

    if len(loss_terms) == 0:
        return zero_loss(torch, device)
    return torch.mean(torch.stack(loss_terms))


def compute_curve_probe_metrics(torch, model, entries, feature_matrix, physics_cfg, device, seed_base):
    # Compute quick validation curve metrics on a fixed-size sampled subset.
    # This is a cheap proxy signal for curve quality during early stopping.
    curve_eval_runs = int(physics_cfg.get("curve_eval_runs", 16))
    curve_eval_time_points = int(physics_cfg.get("curve_eval_time_points", 96))
    weighted_fraction = float(physics_cfg.get("flux_time_weighted_fraction", 0.7))
    weight_power = float(physics_cfg.get("flux_time_weight_power", 1.0))
    if curve_eval_runs <= 0 or curve_eval_time_points <= 0:
        result = {}
        result["relative_l2"] = float("nan")
        result["pearson_r"] = float("nan")
        result["point_count"] = 0
        return result

    rng = np.random.default_rng(int(seed_base) + 131)
    run_rows = choose_run_rows(rng, len(entries), curve_eval_runs)

    pred_values = []
    true_values = []
    # No gradients here: this is monitoring only, not part of optimization.
    with torch.no_grad():
        for run_row in run_rows:
            entry = entries[int(run_row)]
            run_dir = Path(entry["run_dir"])
            fields = load_run_fields(run_dir)
            meta = load_run_meta(run_dir)
            grid = load_run_grid(meta)

            d_field = np.asarray(fields["D"], dtype=float)
            t_curve = np.asarray(fields["t"], dtype=float)
            j_true = np.asarray(fields["J"], dtype=float)
            dx = float(grid["dx"])

            height = int(d_field.shape[0])
            width = int(d_field.shape[1])
            if height < 2 or width < 1 or len(t_curve) < 1:
                continue

            x_den = float(max(1, width - 1))
            y_den = float(max(1, height - 1))
            t_den = float(max(1.0, float(t_curve[-1])))
            d_bottom = d_field[-2, :].astype(float)
            run_feature = feature_matrix[int(run_row)]

            x_idx = np.arange(width, dtype=float)
            x_unit = x_idx / x_den
            y_bottom_unit = np.full(width, float(height - 1) / y_den, dtype=float)
            y_inner_unit = np.full(width, float(height - 2) / y_den, dtype=float)
            run_features = np.repeat(run_feature[None, :], width, axis=0)

            picked_times = sample_flux_time_indices(rng, j_true, int(curve_eval_time_points), weighted_fraction, weight_power)
            for t_idx in picked_times:
                t_value = float(t_curve[int(t_idx)])
                t_unit = np.full(width, t_value / t_den, dtype=float)

                bottom_inputs = build_input_tensor(torch, x_unit, y_bottom_unit, t_unit, device, run_features_vals=run_features, requires_grad=False)
                inner_inputs = build_input_tensor(torch, x_unit, y_inner_unit, t_unit, device, run_features_vals=run_features, requires_grad=False)

                c_bottom = model(bottom_inputs).squeeze(-1)
                c_inner = model(inner_inputs).squeeze(-1)
                d_cdy = (c_bottom - c_inner) / float(dx)
                d_bottom_tensor = to_tensor(torch, d_bottom, device)
                flux_profile = -d_bottom_tensor * d_cdy
                flux_pred = float(torch.mean(flux_profile).detach().cpu().item())
                pred_values.append(flux_pred)
                true_values.append(float(j_true[int(t_idx)]))

    pred_arr = np.asarray(pred_values, dtype=float)
    true_arr = np.asarray(true_values, dtype=float)
    out = {}
    out["point_count"] = int(len(pred_arr))
    if len(pred_arr) == 0:
        out["relative_l2"] = float("nan")
        out["pearson_r"] = float("nan")
        return out

    diff = pred_arr - true_arr
    out["relative_l2"] = float(np.linalg.norm(diff) / (np.linalg.norm(true_arr) + 1e-12))
    if len(pred_arr) > 1:
        # Pearson r is useful to track shape agreement independent of magnitude.
        out["pearson_r"] = float(np.corrcoef(true_arr, pred_arr)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    return out


def combine_weighted_losses(losses, weights):
    # Build the weighted objective used for backpropagation.
    # Weights from config control the fit-versus-physics balance.
    total = losses["data_curve"] * float(weights.get("data_curve", 1.0))
    total = total + losses["pde"] * float(weights.get("pde", 1.0))
    total = total + losses["bc"] * float(weights.get("bc", 1.0))
    total = total + losses["ic"] * float(weights.get("ic", 1.0))
    total = total + losses["flux_curve"] * float(weights.get("flux_curve", 0.0))
    total = total + losses["nonnegativity"] * float(weights.get("nonnegativity", 0.0))
    return total


def choose_run_count(batch_size, points_per_run):
    # Convert a point budget into number of sampled runs per step.
    # Sampling is run-based, so each run contributes points_per_run points.
    if points_per_run <= 0:
        raise ValueError("points_per_run must be > 0")
    run_count = int(batch_size // points_per_run)
    if run_count < 1:
        run_count = 1
    return run_count


def compute_loss_map(torch, model, entries, feature_matrix, training_cfg, physics_cfg, loss_weights, device, seed_base, flux_eps):
    # Build stochastic batches for interior, initial, and boundary losses.
    # Seed offsets keep these batch draws decorrelated inside each epoch.
    if len(entries) != int(feature_matrix.shape[0]):
        raise ValueError("entries and feature_matrix row count must match")

    collocation_points_per_run = int(physics_cfg.get("collocation_points_per_run", 256))
    boundary_points_per_run = int(physics_cfg.get("boundary_points_per_run", 128))
    initial_points_per_run = int(physics_cfg.get("initial_points_per_run", 128))
    batch_size_physics = int(training_cfg.get("batch_size_physics", collocation_points_per_run))
    batch_size_data = int(training_cfg.get("batch_size_data", initial_points_per_run))

    collocation_run_count = choose_run_count(batch_size_physics, collocation_points_per_run)
    boundary_run_count = choose_run_count(batch_size_physics, boundary_points_per_run)
    initial_run_count = choose_run_count(batch_size_data, initial_points_per_run)
    # Each sampler reads different point types from the same selected run set.

    collocation_batch = sample_collocation_batch(
        entries,
        run_count=collocation_run_count,
        points_per_run=collocation_points_per_run,
        seed=int(seed_base) + 11,
        feature_matrix=feature_matrix,
    )
    initial_batch = sample_initial_batch(
        entries,
        run_count=initial_run_count,
        points_per_run=initial_points_per_run,
        seed=int(seed_base) + 29,
        feature_matrix=feature_matrix,
    )
    boundary_batch = sample_boundary_batch(
        entries,
        run_count=boundary_run_count,
        points_per_run=boundary_points_per_run,
        seed=int(seed_base) + 47,
        feature_matrix=feature_matrix,
    )

    collocation_losses = compute_collocation_losses(torch, model, collocation_batch, device)
    loss_ic = compute_initial_loss(torch, model, initial_batch, device)
    loss_bc = compute_boundary_loss(torch, model, boundary_batch, device)
    loss_flux_curve = compute_flux_curve_loss(torch, model, entries, feature_matrix, physics_cfg, device, seed_base, flux_eps)

    # Keep explicit keys so history/summary files stay stable.
    losses = {}
    losses["data_curve"] = collocation_losses["data_curve"]
    losses["pde"] = collocation_losses["pde"]
    losses["bc"] = loss_bc
    losses["ic"] = loss_ic
    losses["flux_curve"] = loss_flux_curve
    losses["nonnegativity"] = collocation_losses["nonnegativity"]
    # The final optimization objective is just the weighted sum of all terms.
    losses["total"] = combine_weighted_losses(losses, loss_weights)
    return losses


def losses_to_float_dict(losses):
    # Convert torch scalar tensors to plain floats for JSON serialization.
    values = {}
    for name in losses:
        values[name] = float(losses[name].detach().cpu().item())
    return values


def save_checkpoint(torch, path, model, optimizer, epoch, best_val_total, model_cfg_used, feature_stats_json):
    # Save train state required to resume or evaluate later.
    state = {}
    state["epoch"] = int(epoch)
    state["best_val_total"] = float(best_val_total)
    state["model_state_dict"] = model.state_dict()
    state["optimizer_state_dict"] = optimizer.state_dict()
    state["model_cfg_used"] = model_cfg_used
    state["feature_stats"] = feature_stats_json
    torch.save(state, path)


def main():
    # This is the full training entrypoint.
    # It validates inputs, trains with early stopping, evaluates, and writes artifacts.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default="configs/ml/pinn_v1.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    # Resolve and validate paths first.
    ml_dir = Path(args.ml_dir)
    if not ml_dir.exists():
        raise ValueError(f"ml_dir does not exist: {ml_dir}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise ValueError(f"config does not exist: {config_path}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load config and ML schema metadata.
    pinn_cfg = load_pinn_config(config_path)
    data_summary = validate_ml_dir(ml_dir)

    torch_info = detect_torch()
    if not torch_info["available"]:
        raise ValueError("torch is required for PINN training: " + str(torch_info["import_error"]))

    import torch

    chosen_device_text = choose_device(
        args.device,
        pinn_cfg["runtime"].get("device", "auto"),
        torch_info,
    )
    if chosen_device_text == "cuda" and not torch_info["cuda_available"]:
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    device = torch.device(chosen_device_text)
    if chosen_device_text == "cuda":
        cuda_index = int(torch.cuda.current_device())
        cuda_name = str(torch.cuda.get_device_name(cuda_index))
        print("PINN device:", "cuda", "(index", cuda_index, "-", cuda_name + ")")
    else:
        print("PINN device:", chosen_device_text)

    # Keep block extraction explicit for readability.
    training_cfg = pinn_cfg["training"]
    model_cfg = pinn_cfg["model"]
    optimizer_cfg = pinn_cfg["optimizer"]
    loss_weights = pinn_cfg["loss_weights"]
    physics_cfg = pinn_cfg["physics"]

    # Seed numpy and torch for reproducible sampling and initialization.
    seed = int(training_cfg.get("seed", 321))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_info["cuda_available"]:
        torch.cuda.manual_seed_all(seed)

    # Load run-entry lists for each split.
    train_entries = load_split_entries(
        ml_dir=ml_dir,
        split_name="id_train",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
    )
    val_entries = load_split_entries(
        ml_dir=ml_dir,
        split_name="id_val",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
    )
    test_entries = load_split_entries(
        ml_dir=ml_dir,
        split_name="id_test",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
    )

    # Load split features and normalize them using train split statistics.
    train_features_raw = load_split_feature_matrix(ml_dir, "id_train", train_entries)
    val_features_raw = load_split_feature_matrix(ml_dir, "id_val", val_entries)
    test_features_raw = load_split_feature_matrix(ml_dir, "id_test", test_entries)
    feature_stats = compute_feature_stats(train_features_raw)
    train_features = normalize_feature_matrix(train_features_raw, feature_stats)
    val_features = normalize_feature_matrix(val_features_raw, feature_stats)
    test_features = normalize_feature_matrix(test_features_raw, feature_stats)
    feature_stats_json = feature_stats_to_json(feature_stats)
    train_curves = load_split_curve_matrix(ml_dir, "id_train", train_entries)
    flux_eps = compute_flux_normalization(train_curves, physics_cfg)

    # Model conditioning width is tied to ML feature count.
    condition_dim = int(train_features.shape[1])
    model_cfg_used = dict(model_cfg)
    if "condition_dim" in model_cfg_used:
        cfg_condition_dim = int(model_cfg_used["condition_dim"])
        if cfg_condition_dim != condition_dim:
            raise ValueError(f"model.condition_dim ({cfg_condition_dim}) does not match ML feature count ({condition_dim})")
    model_cfg_used["condition_dim"] = int(condition_dim)

    # Build model and optimizer.
    model = build_pinn_model(torch, model_cfg_used).to(device)

    optimizer_name = str(optimizer_cfg.get("name", "adam")).strip().lower()
    if optimizer_name != "adam":
        raise ValueError("Only adam optimizer is supported in this training script")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )

    # Checkpoint locations.
    checkpoint_dir = out_dir / "checkpoints"
    ensure_dir(checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"

    epochs = int(training_cfg.get("epochs", 1))
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_delta = float(training_cfg.get("early_stop_min_delta", 0.0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
    early_stop_metric = str(training_cfg.get("early_stop_metric", "val_total")).strip().lower()
    curve_eval_every = int(training_cfg.get("curve_eval_every", 5))
    if curve_eval_every < 1:
        curve_eval_every = 1
    supported_metrics = ["val_total", "curve_rel_l2", "curve_pearson"]
    if early_stop_metric not in supported_metrics:
        raise ValueError("training.early_stop_metric must be one of: " + ", ".join(supported_metrics))

    # Early-stopping and logging state.
    best_val_total = None
    best_monitor_value = None
    best_curve_relative_l2 = None
    best_curve_pearson_r = None
    best_epoch = 0
    wait_count = 0
    epochs_ran = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Run one training step.
        model.train()
        optimizer.zero_grad()

        train_seed = int(seed + epoch * 101)
        train_losses = compute_loss_map(torch, model, train_entries, train_features, training_cfg, physics_cfg, loss_weights, device, train_seed, flux_eps)
        train_total = train_losses["total"]
        train_total.backward()

        # Optional gradient clipping can improve stability on early epochs.
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # Run one validation step. Gradients stay enabled because PDE loss needs derivatives.
        model.eval()
        with torch.enable_grad():
            # Keep validation sampling fixed across epochs so early stopping compares like-for-like.
            val_seed = int(seed + 50000)
            val_losses = compute_loss_map(
                torch,
                model,
                val_entries,
                val_features,
                training_cfg,
                physics_cfg,
                loss_weights,
                device,
                val_seed,
                flux_eps,
            )

        train_values = losses_to_float_dict(train_losses)
        val_values = losses_to_float_dict(val_losses)
        train_total_value = train_values["total"]
        val_total_value = val_values["total"]
        if best_val_total is None or val_total_value < best_val_total:
            best_val_total = val_total_value

        val_curve_metrics = None
        need_curve_metric = early_stop_metric != "val_total"
        if need_curve_metric and (epoch == 1 or epoch % curve_eval_every == 0):
            val_curve_metrics = compute_curve_probe_metrics(torch, model, val_entries, val_features, physics_cfg, device, seed + 700000)

        history_row = {"epoch": int(epoch), "train": train_values, "val": val_values}
        if val_curve_metrics is not None:
            history_row["val_curve_relative_l2"] = float(val_curve_metrics["relative_l2"])
            history_row["val_curve_pearson_r"] = float(val_curve_metrics["pearson_r"])
            history_row["val_curve_points"] = int(val_curve_metrics["point_count"])
        history.append(history_row)

        # Save latest checkpoint every epoch.
        save_checkpoint(
            torch,
            latest_checkpoint_path,
            model,
            optimizer,
            epoch,
            val_total_value,
            model_cfg_used,
            feature_stats_json,
        )

        # Track best validation checkpoint for final reporting.
        improved = False
        monitor_value = val_total_value
        if early_stop_metric == "curve_rel_l2":
            if val_curve_metrics is not None:
                monitor_value = float(val_curve_metrics["relative_l2"])
            else:
                monitor_value = None
        if early_stop_metric == "curve_pearson":
            if val_curve_metrics is not None:
                monitor_value = float(val_curve_metrics["pearson_r"])
            else:
                monitor_value = None
        if monitor_value is not None and not np.isfinite(monitor_value):
            monitor_value = None

        if monitor_value is None:
            improved = False
        else:
            if best_monitor_value is None:
                improved = True
            elif early_stop_metric == "curve_pearson":
                if monitor_value > (best_monitor_value + min_delta):
                    improved = True
            else:
                if monitor_value < (best_monitor_value - min_delta):
                    improved = True

        if improved and monitor_value is not None:
            best_monitor_value = monitor_value
            if val_curve_metrics is not None:
                best_curve_relative_l2 = float(val_curve_metrics["relative_l2"])
                best_curve_pearson_r = float(val_curve_metrics["pearson_r"])
            best_epoch = int(epoch)
            wait_count = 0
            save_checkpoint(torch, best_checkpoint_path, model, optimizer, epoch, float(best_monitor_value), model_cfg_used, feature_stats_json)
        else:
            if early_stop_metric == "val_total":
                wait_count += 1
            elif val_curve_metrics is not None:
                wait_count += int(curve_eval_every)

        epochs_ran = int(epoch)
        if val_curve_metrics is None:
            print("epoch", epoch, "train_total=", f"{train_total_value:.6e}", "val_total=", f"{val_total_value:.6e}")
        else:
            print(
                "epoch",
                epoch,
                "train_total=",
                f"{train_total_value:.6e}",
                "val_total=",
                f"{val_total_value:.6e}",
                "val_curve_rel_l2=",
                f"{float(val_curve_metrics['relative_l2']):.6e}",
                "val_curve_r=",
                f"{float(val_curve_metrics['pearson_r']):.4f}",
            )

        # Stop early when no validation improvement for "patience" epochs.
        if patience > 0 and wait_count >= patience:
            print("early_stop at epoch", epoch)
            break

    # Always evaluate using the best validation checkpoint when available.
    if best_checkpoint_path.exists():
        best_state = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

    # Test split loss report (same loss map used in training/validation).
    test_values = None
    if len(test_entries) > 0:
        model.eval()
        with torch.enable_grad():
            test_seed = int(seed + 900000)
            test_losses = compute_loss_map(torch, model, test_entries, test_features, training_cfg, physics_cfg, loss_weights, device, test_seed, flux_eps)
        test_values = losses_to_float_dict(test_losses)

    # Runtime metadata helps reproduce and audit each run.
    runtime = {}
    runtime["status"] = "trained"
    runtime["ml_dir"] = str(ml_dir.resolve())
    runtime["config"] = str(config_path.resolve())
    runtime["device"] = chosen_device_text
    runtime["torch"] = torch_info
    runtime["run_index_filter"] = {"start": args.run_start_index, "end": args.run_end_index}
    runtime["condition_dim"] = int(condition_dim)
    runtime["early_stop_metric"] = early_stop_metric
    runtime["curve_eval_every"] = int(curve_eval_every)
    runtime["flux_eps"] = float(flux_eps)
    runtime["seconds"] = float(time.time() - t0)

    # Summary is a concise training report for quick checks.
    summary = {}
    summary["status"] = "trained"
    summary["description"] = "PINN training completed with PDE/BC/IC losses."
    summary["feature_count"] = int(len(data_summary["feature_names"]))
    summary["condition_dim"] = int(condition_dim)
    summary["feature_stats"] = feature_stats_json
    summary["scalar_target_count"] = int(len(data_summary["scalar_target_names"]))
    summary["splits"] = data_summary["splits"]
    summary["train_rows"] = int(len(train_entries))
    summary["val_rows"] = int(len(val_entries))
    summary["test_rows"] = int(len(test_entries))
    summary["val_source"] = "id_val"
    summary["early_stop_metric"] = early_stop_metric
    summary["curve_eval_every"] = int(curve_eval_every)
    summary["flux_eps"] = float(flux_eps)
    summary["epochs_requested"] = int(epochs)
    summary["epochs_ran"] = int(epochs_ran)
    summary["best_epoch"] = int(best_epoch)
    summary["best_monitor_value"] = None if best_monitor_value is None else float(best_monitor_value)
    summary["best_val_total"] = None if best_val_total is None else float(best_val_total)
    summary["best_val_curve_relative_l2"] = None if best_curve_relative_l2 is None else float(best_curve_relative_l2)
    summary["best_val_curve_pearson_r"] = None if best_curve_pearson_r is None else float(best_curve_pearson_r)
    summary["checkpoint_best"] = str(best_checkpoint_path.resolve())
    summary["checkpoint_latest"] = str(latest_checkpoint_path.resolve())
    if len(history) > 0:
        summary["final_train_total"] = float(history[-1]["train"]["total"])
        summary["final_val_total"] = float(history[-1]["val"]["total"])
    summary["test"] = test_values

    # Persist run artifacts.
    write_json(out_dir / "runtime.json", runtime)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "history.json", history)

    print("saved:", out_dir)


if __name__ == "__main__":
    main()
