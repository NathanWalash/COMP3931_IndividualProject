import argparse
import time
from pathlib import Path

import numpy as np

from skin_diffusion.dataset_spec import load_yaml_file
from skin_diffusion.pinn_dataset import (
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
    # Keeping this strict prevents silent fallback behavior.
    cfg = load_yaml_file(str(config_path))
    if not isinstance(cfg, dict):
        raise ValueError(f"PINN config is not a dictionary: {config_path}")

    required_blocks = []
    required_blocks.append("model")
    required_blocks.append("training")
    required_blocks.append("optimizer")
    required_blocks.append("loss_weights")
    required_blocks.append("physics")
    required_blocks.append("runtime")

    for name in required_blocks:
        if name not in cfg:
            raise ValueError(f"Missing block in PINN config: {name}")

    return cfg


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
    # Build a plain MLP that maps [x_unit, y_unit, t_unit] to concentration.
    # Coordinates are normalized here and converted to physical derivatives in the loss.
    hidden_layers = int(model_cfg.get("hidden_layers", 4))
    hidden_width = int(model_cfg.get("hidden_width", 128))
    activation_name = model_cfg.get("activation", "tanh")
    output_activation_name = model_cfg.get("output_activation", "softplus")

    activation = build_activation(torch, activation_name)
    output_activation = build_activation(torch, output_activation_name)

    layers = []
    in_width = 3
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


def build_input_tensor(torch, x_vals, y_vals, t_vals, device, requires_grad=False):
    # Stack coordinate arrays into an [N, 3] input tensor for the network.
    # Set requires_grad only when derivative terms are needed in the loss.
    x_t = to_tensor(torch, x_vals, device)
    y_t = to_tensor(torch, y_vals, device)
    t_t = to_tensor(torch, t_vals, device)
    inputs = torch.stack([x_t, y_t, t_t], dim=1)
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
    # Compute interior losses for data fit, PDE residual, and nonnegativity.
    # The model predicts C(x, y, t) at sampled interior collocation points.
    inputs = build_input_tensor(
        torch,
        batch["x_unit"],
        batch["y_unit"],
        batch["t_unit"],
        device,
        requires_grad=True,
    )
    c_now_pred = model(inputs).squeeze(-1)

    # Data term anchors the PINN to simulator concentrations at sampled points.
    # Without this term the network could satisfy PDE constraints but drift from data.
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

    # Convert normalized-coordinate derivatives back to physical units with the chain rule.
    # Here x = x_unit * x_scale, y = y_unit * y_scale, t = t_unit * t_scale.
    # Therefore dC/dt = (dC/dt_unit)/t_scale and d2C/dx2 = (d2C/dx_unit2)/x_scale^2.
    x_scale = to_tensor(torch, batch["x_scale"], device)
    y_scale = to_tensor(torch, batch["y_scale"], device)
    t_scale = to_tensor(torch, batch["t_scale"], device)
    dc_dt = dc_dt_unit / t_scale
    d2c_dx2 = d2c_dx2_unit / (x_scale * x_scale)
    d2c_dy2 = d2c_dy2_unit / (y_scale * y_scale)

    d_vals = to_tensor(torch, batch["D"], device)
    k_vals = to_tensor(torch, batch["k"], device)

    # PDE used here is dC/dt = D * Laplacian(C) - k * C.
    # We move everything to one side to build a residual that should be zero.
    # The PDE loss is the mean squared residual over sampled collocation points.
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
        requires_grad=False,
    )
    pred = model(inputs).squeeze(-1)
    target = to_tensor(torch, batch["C_true"], device)
    return mse_loss_safe(torch, pred, target, device)


def compute_boundary_loss(torch, model, batch, device):
    # Boundary loss is the average over active boundary components.
    # It enforces top patch concentration, top off-patch no-flux,
    # bottom sink concentration, and side no-flux behavior.
    loss_total = zero_loss(torch, device)
    count = 0

    top_patch = batch["top_patch"]
    if len(top_patch["x_unit"]) > 0:
        # Apply concentration target on the patch-covered top boundary.
        top_patch_inputs = build_input_tensor(
            torch,
            top_patch["x_unit"],
            top_patch["y_unit"],
            top_patch["t_unit"],
            device,
            requires_grad=False,
        )
        top_patch_pred = model(top_patch_inputs).squeeze(-1)
        top_patch_target = to_tensor(torch, top_patch["C_target"], device)
        loss_total = loss_total + mse_loss_safe(torch, top_patch_pred, top_patch_target, device)
        count += 1

    top_offpatch = batch["top_offpatch"]
    if len(top_offpatch["x_unit"]) > 0:
        # This is a one-step finite-difference form of no-flux at the top off-patch region.
        # Enforcing boundary_pred - inner_pred = 0 is proportional to enforcing dC/dn = 0.
        top_offpatch_boundary_inputs = build_input_tensor(
            torch,
            top_offpatch["x_unit"],
            top_offpatch["y_boundary_unit"],
            top_offpatch["t_unit"],
            device,
            requires_grad=False,
        )
        top_offpatch_inner_inputs = build_input_tensor(
            torch,
            top_offpatch["x_unit"],
            top_offpatch["y_inner_unit"],
            top_offpatch["t_unit"],
            device,
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
            requires_grad=False,
        )
        bottom_sink_pred = model(bottom_sink_inputs).squeeze(-1)
        bottom_sink_target = to_tensor(torch, bottom_sink["C_target"], device)
        loss_total = loss_total + mse_loss_safe(torch, bottom_sink_pred, bottom_sink_target, device)
        count += 1

    side_neumann = batch["side_neumann"]
    if len(side_neumann["x_boundary_unit"]) > 0:
        # Same no-flux idea on left/right sides using boundary-minus-inner difference.
        side_boundary_inputs = build_input_tensor(
            torch,
            side_neumann["x_boundary_unit"],
            side_neumann["y_unit"],
            side_neumann["t_unit"],
            device,
            requires_grad=False,
        )
        side_inner_inputs = build_input_tensor(
            torch,
            side_neumann["x_inner_unit"],
            side_neumann["y_unit"],
            side_neumann["t_unit"],
            device,
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


def combine_weighted_losses(losses, weights):
    # Build the weighted objective used for backpropagation.
    # Weights from config control the fit-versus-physics balance.
    total = losses["data_curve"] * float(weights.get("data_curve", 1.0))
    total = total + losses["pde"] * float(weights.get("pde", 1.0))
    total = total + losses["bc"] * float(weights.get("bc", 1.0))
    total = total + losses["ic"] * float(weights.get("ic", 1.0))
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


def compute_loss_map(
    torch,
    model,
    entries,
    training_cfg,
    physics_cfg,
    loss_weights,
    device,
    seed_base,
):
    # Build stochastic batches for interior, initial, and boundary losses.
    # Seed offsets keep these batch draws decorrelated inside each epoch.
    collocation_points_per_run = int(physics_cfg.get("collocation_points_per_run", 256))
    boundary_points_per_run = int(physics_cfg.get("boundary_points_per_run", 128))
    initial_points_per_run = int(physics_cfg.get("initial_points_per_run", 128))
    batch_size_physics = int(training_cfg.get("batch_size_physics", collocation_points_per_run))
    batch_size_data = int(training_cfg.get("batch_size_data", initial_points_per_run))

    collocation_run_count = choose_run_count(batch_size_physics, collocation_points_per_run)
    boundary_run_count = choose_run_count(batch_size_physics, boundary_points_per_run)
    initial_run_count = choose_run_count(batch_size_data, initial_points_per_run)

    collocation_batch = sample_collocation_batch(
        entries,
        run_count=collocation_run_count,
        points_per_run=collocation_points_per_run,
        seed=int(seed_base) + 11,
    )
    initial_batch = sample_initial_batch(
        entries,
        run_count=initial_run_count,
        points_per_run=initial_points_per_run,
        seed=int(seed_base) + 29,
    )
    boundary_batch = sample_boundary_batch(
        entries,
        run_count=boundary_run_count,
        points_per_run=boundary_points_per_run,
        seed=int(seed_base) + 47,
    )

    collocation_losses = compute_collocation_losses(torch, model, collocation_batch, device)
    loss_ic = compute_initial_loss(torch, model, initial_batch, device)
    loss_bc = compute_boundary_loss(torch, model, boundary_batch, device)

    # Keep explicit keys so history/summary files stay stable.
    losses = {}
    losses["data_curve"] = collocation_losses["data_curve"]
    losses["pde"] = collocation_losses["pde"]
    losses["bc"] = loss_bc
    losses["ic"] = loss_ic
    losses["nonnegativity"] = collocation_losses["nonnegativity"]
    losses["total"] = combine_weighted_losses(losses, loss_weights)
    return losses


def losses_to_float_dict(losses):
    # Convert torch scalar tensors to plain floats for JSON serialization.
    values = {}
    for name in losses:
        values[name] = float(losses[name].detach().cpu().item())
    return values


def save_checkpoint(torch, path, model, optimizer, epoch, best_val_total):
    # Save train state required to resume or evaluate later.
    state = {}
    state["epoch"] = int(epoch)
    state["best_val_total"] = float(best_val_total)
    state["model_state_dict"] = model.state_dict()
    state["optimizer_state_dict"] = optimizer.state_dict()
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

    chosen_device_text = choose_device(args.device, pinn_cfg["runtime"].get("device", "auto"), torch_info)
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
    train_entries = load_split_entries(ml_dir=ml_dir, split_name="id_train", run_start_index=args.run_start_index, run_end_index=args.run_end_index)
    val_entries = load_split_entries(ml_dir=ml_dir, split_name="id_val", run_start_index=args.run_start_index, run_end_index=args.run_end_index)
    test_entries = load_split_entries(ml_dir=ml_dir, split_name="id_test", run_start_index=args.run_start_index, run_end_index=args.run_end_index)

    # Build model and optimizer.
    model = build_pinn_model(torch, model_cfg).to(device)

    optimizer_name = str(optimizer_cfg.get("name", "adam")).strip().lower()
    if optimizer_name != "adam":
        raise ValueError("Only adam optimizer is supported in this training script")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(optimizer_cfg.get("lr", 1e-3)), weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)))

    # Checkpoint locations.
    checkpoint_dir = out_dir / "checkpoints"
    ensure_dir(checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"

    epochs = int(training_cfg.get("epochs", 1))
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_delta = float(training_cfg.get("early_stop_min_delta", 0.0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))

    # Early-stopping and logging state.
    best_val_total = None
    best_epoch = 0
    wait_count = 0
    epochs_ran = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Run one training step.
        model.train()
        optimizer.zero_grad()

        train_seed = int(seed + epoch * 101)
        train_losses = compute_loss_map(torch, model, train_entries, training_cfg, physics_cfg, loss_weights, device, train_seed)
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
            val_losses = compute_loss_map(torch, model, val_entries, training_cfg, physics_cfg, loss_weights, device, val_seed)

        train_values = losses_to_float_dict(train_losses)
        val_values = losses_to_float_dict(val_losses)
        train_total_value = train_values["total"]
        val_total_value = val_values["total"]

        history_row = {"epoch": int(epoch), "train": train_values, "val": val_values}
        history.append(history_row)

        # Save latest checkpoint every epoch.
        save_checkpoint(torch, latest_checkpoint_path, model, optimizer, epoch, val_total_value)

        # Track best validation checkpoint for final reporting.
        improved = False
        if best_val_total is None:
            improved = True
        else:
            if val_total_value < (best_val_total - min_delta):
                improved = True

        if improved:
            best_val_total = val_total_value
            best_epoch = int(epoch)
            wait_count = 0
            save_checkpoint(torch, best_checkpoint_path, model, optimizer, epoch, best_val_total)
        else:
            wait_count += 1

        epochs_ran = int(epoch)
        print("epoch", epoch, "train_total=", f"{train_total_value:.6e}", "val_total=", f"{val_total_value:.6e}")

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
            test_losses = compute_loss_map(torch, model, test_entries, training_cfg, physics_cfg, loss_weights, device, test_seed)
        test_values = losses_to_float_dict(test_losses)

    # Runtime metadata helps reproduce and audit each run.
    runtime = {}
    runtime["status"] = "trained"
    runtime["ml_dir"] = str(ml_dir.resolve())
    runtime["config"] = str(config_path.resolve())
    runtime["device"] = chosen_device_text
    runtime["torch"] = torch_info
    runtime["run_index_filter"] = {"start": args.run_start_index, "end": args.run_end_index}
    runtime["seconds"] = float(time.time() - t0)

    # Summary is a concise training report for quick checks.
    summary = {}
    summary["status"] = "trained"
    summary["description"] = "PINN training completed with PDE/BC/IC losses."
    summary["feature_count"] = int(len(data_summary["feature_names"]))
    summary["scalar_target_count"] = int(len(data_summary["scalar_target_names"]))
    summary["splits"] = data_summary["splits"]
    summary["train_rows"] = int(len(train_entries))
    summary["val_rows"] = int(len(val_entries))
    summary["test_rows"] = int(len(test_entries))
    summary["val_source"] = "id_val"
    summary["epochs_requested"] = int(epochs)
    summary["epochs_ran"] = int(epochs_ran)
    summary["best_epoch"] = int(best_epoch)
    summary["best_val_total"] = float(best_val_total)
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
