import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from skin_diffusion.ml_curve_plots import plot_curve_error_over_time, plot_curve_examples
from skin_diffusion.ml_curve_backbone import (
    fit_curve_backbone,
    parse_float_csv,
    predict_curve_backbone,
)
from skin_diffusion.ml_metrics import compute_curve_metrics
from skin_diffusion.ml_physics_diagnostics import (
    build_worst_case_report,
    compute_split_physics_diagnostics,
    write_rows_csv,
)
from skin_diffusion.ml_run_dataset import (
    load_split_2d_array,
    load_split_entries,
    remap_run_dir,
)
from skin_diffusion.ml_scalar_diagnostics import (
    build_scalar_rmse_summary,
    plot_scalar_parity,
    plot_scalar_residual_hist,
    scalar_report,
    scalar_targets_from_flux_curves,
)
from skin_diffusion.utils import ensure_dir


def choose_device(device_arg):
    # Keep device selection explicit so CUDA runs fail fast when requested.
    text = str(device_arg).strip().lower()
    if text in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise ValueError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def set_global_seed(seed_value):
    # Use one seed path for numpy + torch so repeated runs are comparable.
    seed_int = int(seed_value)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def build_entries(
    ml_dir,
    split_name,
    run_start_index=None,
    run_end_index=None,
    max_rows=None,
    run_root_override=None,
):
    # Load split rows from ml/meta and remap run paths when using staged storage.
    entries = load_split_entries(
        ml_dir=ml_dir,
        split_name=split_name,
        run_start_index=run_start_index,
        run_end_index=run_end_index,
    )
    out = []
    for entry in entries:
        row = dict(entry)
        row["run_dir"] = remap_run_dir(entry["run_dir"], run_root_override)
        out.append(row)
    if max_rows is not None:
        out = out[: int(max_rows)]
    if len(out) == 0:
        raise ValueError("No rows selected for split " + str(split_name))
    return out


def load_split_matrix(ml_dir, split_name, entries, key):
    # Read one array block (X/J/t/...) for the selected split rows.
    return load_split_2d_array(
        ml_dir=ml_dir,
        split_name=split_name,
        entries=entries,
        key=key,
    ).astype(np.float32)


def resolve_split_key(split_map, logical_name):
    # Use canonical split keys only.
    split_text = str(logical_name)
    if split_text in split_map:
        return split_text
    raise ValueError(
        "Could not find split key "
        + split_text
        + ". Expected one of: train, val, test"
    )


def load_split_name_map(ml_dir):
    # Build a stable train/val/test -> file/index key map from ml/meta.json.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing meta.json in " + str(ml_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_map = meta.get("splits", {})
    if not isinstance(split_map, dict):
        raise ValueError("meta.json is missing splits")

    result = {}
    result["train"] = resolve_split_key(split_map, "train")
    result["val"] = resolve_split_key(split_map, "val")
    result["test"] = resolve_split_key(split_map, "test")
    return result


def load_ml_meta_names(ml_dir):
    # Read feature and scalar target names used by scalar diagnostics.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing meta.json in " + str(ml_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_names = meta.get("feature_names", [])
    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("meta.json is missing feature_names")

    scalar_target_names = meta.get("scalar_target_names", [])
    if not isinstance(scalar_target_names, list) or len(scalar_target_names) == 0:
        raise ValueError("meta.json is missing scalar_target_names")

    feature_out = []
    feature_index = 0
    while feature_index < len(feature_names):
        feature_out.append(str(feature_names[feature_index]))
        feature_index += 1

    scalar_out = []
    scalar_index = 0
    while scalar_index < len(scalar_target_names):
        scalar_out.append(str(scalar_target_names[scalar_index]))
        scalar_index += 1
    return feature_out, scalar_out


def extract_c0_feature(x_raw, feature_names):
    # C0 is needed to derive permeability-style scalar diagnostics from J(t).
    if "C0" not in feature_names:
        return np.full((int(x_raw.shape[0]),), np.nan, dtype=np.float32)
    c0_column = int(feature_names.index("C0"))
    return np.asarray(x_raw[:, c0_column], dtype=np.float32)


def load_run_physics_1d(entries):
    # Load 1D x-averaged concentration/material profiles from the exact simulator outputs.
    d1d_rows = []
    d1d_dy_rows = []
    k1d_rows = []
    c_star_rows = []
    t_norm_rows = []
    t_phys_rows = []
    patch_width_rows = []
    c0_rows = []
    decay_rows = []
    mode_rows = []
    depth_rows = []
    t_end_rows = []

    t_count_ref = None
    h_count_ref = None
    row_count = len(entries)
    row_index = 0
    while row_index < row_count:
        run_dir = Path(entries[row_index]["run_dir"])
        fields_path = run_dir / "fields.npz"
        meta_path = run_dir / "meta.json"
        if not fields_path.exists():
            raise ValueError("Missing fields.npz: " + str(fields_path))
        if not meta_path.exists():
            raise ValueError("Missing meta.json: " + str(meta_path))

        with np.load(fields_path) as data:
            c_snap = np.asarray(data["C_snap"], dtype=np.float32)
            d_field = np.asarray(data["D"], dtype=np.float32)
            k_field = np.asarray(data["k"], dtype=np.float32)
            t_curve = np.asarray(data["t"], dtype=np.float32)

        if c_snap.ndim != 3 or d_field.ndim != 2 or k_field.ndim != 2 or t_curve.ndim != 1:
            raise ValueError("Invalid array shapes for run: " + str(run_dir))

        t_count = int(c_snap.shape[0])
        h_count = int(c_snap.shape[1])
        if t_count_ref is None:
            t_count_ref = t_count
            h_count_ref = h_count
        if t_count != int(t_count_ref) or h_count != int(h_count_ref):
            raise ValueError("All runs must share C_snap shape [T,H,W] in selected split")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        grid = meta.get("grid", {})
        boundary = meta.get("boundary", {})
        dx_value = float(grid.get("dx", 0.0))
        if dx_value <= 0.0:
            raise ValueError("Invalid dx in run metadata: " + str(run_dir))
        depth_value = float((h_count - 1) * dx_value)
        t_end_value = float(t_curve[-1])
        if t_end_value <= 0.0:
            raise ValueError("Invalid t_end in run fields: " + str(run_dir))

        d1d = np.mean(d_field, axis=1).astype(np.float32)
        y_star = np.linspace(0.0, 1.0, h_count, dtype=np.float32)
        d1d_dy = np.gradient(d1d.astype(np.float64), y_star.astype(np.float64), edge_order=1).astype(np.float32)
        k1d = np.mean(k_field, axis=1).astype(np.float32)
        c1d = np.mean(c_snap, axis=2).astype(np.float32)

        # Convert simulator concentration into dimensionless c*=C/C0 for PINN training.
        c0_value = float(boundary.get("C0", 0.0))
        c0_safe = max(c0_value, 1e-12)
        c_star = np.clip(c1d / c0_safe, a_min=0.0, a_max=None).astype(np.float32)
        # Normalize time to [0,1] so one model can cover different t_end values.
        t_norm = np.asarray(t_curve / t_end_value, dtype=np.float32)

        mode_text = str(boundary.get("mode", "infinite_dose"))
        mode_rows.append(1.0 if mode_text == "time_decay" else 0.0)
        patch_width_rows.append(float(boundary.get("patch_width", 0.5)))
        c0_rows.append(c0_value)
        decay_rows.append(float(boundary.get("decay_rate", 0.0)))
        depth_rows.append(depth_value)
        t_end_rows.append(t_end_value)
        d1d_rows.append(d1d)
        d1d_dy_rows.append(d1d_dy)
        k1d_rows.append(k1d)
        c_star_rows.append(c_star)
        t_norm_rows.append(t_norm)
        t_phys_rows.append(np.asarray(t_curve, dtype=np.float32))

        row_index += 1
        if row_index == row_count or row_index % 50 == 0:
            print("loaded run physics", row_index, "/", row_count)

    return {
        "d1d": np.asarray(d1d_rows, dtype=np.float32),
        "d1d_dy_star": np.asarray(d1d_dy_rows, dtype=np.float32),
        "k1d": np.asarray(k1d_rows, dtype=np.float32),
        "c_star": np.asarray(c_star_rows, dtype=np.float32),
        "t_norm": np.asarray(t_norm_rows, dtype=np.float32),
        "t_phys": np.asarray(t_phys_rows, dtype=np.float32),
        "patch_width": np.asarray(patch_width_rows, dtype=np.float32),
        "c0": np.asarray(c0_rows, dtype=np.float32),
        "decay": np.asarray(decay_rows, dtype=np.float32),
        "mode_flag": np.asarray(mode_rows, dtype=np.float32),
        "depth": np.asarray(depth_rows, dtype=np.float32),
        "t_end": np.asarray(t_end_rows, dtype=np.float32),
    }


def pack_for_training(features, j_true, j_base, phys):
    # Keep all per-run arrays in one structure before moving to torch tensors.
    return {
        "x_feat": np.asarray(features, dtype=np.float32),
        "j_true": np.asarray(j_true, dtype=np.float32),
        "j_base": np.asarray(j_base, dtype=np.float32),
        "d1d": np.asarray(phys["d1d"], dtype=np.float32),
        "d1d_dy_star": np.asarray(phys["d1d_dy_star"], dtype=np.float32),
        "k1d": np.asarray(phys["k1d"], dtype=np.float32),
        "c_star": np.asarray(phys["c_star"], dtype=np.float32),
        "t_norm": np.asarray(phys["t_norm"], dtype=np.float32),
        "t_phys": np.asarray(phys["t_phys"], dtype=np.float32),
        "patch_width": np.asarray(phys["patch_width"], dtype=np.float32),
        "c0": np.asarray(phys["c0"], dtype=np.float32),
        "decay": np.asarray(phys["decay"], dtype=np.float32),
        "mode_flag": np.asarray(phys["mode_flag"], dtype=np.float32),
        "depth": np.asarray(phys["depth"], dtype=np.float32),
        "t_end": np.asarray(phys["t_end"], dtype=np.float32),
    }


def tensorize_pack(np_pack, device):
    # Move numpy arrays to the selected device with one predictable dtype.
    out = {}
    for key in np_pack:
        value = np_pack[key]
        out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return out


class RunConditionedPINN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, depth=4, fourier_frequencies=6):
        super().__init__()
        # Concentration head takes run features + (y,t) coordinates.
        self.fourier_frequencies = int(max(0, fourier_frequencies))
        if self.fourier_frequencies > 0:
            freq_tensor = torch.arange(self.fourier_frequencies, dtype=torch.float32)
            self.register_buffer("freq_values", torch.pow(2.0, freq_tensor))

        layers = []
        extra_dim = 0
        if self.fourier_frequencies > 0:
            # sin/cos for y and t.
            extra_dim = 4 * self.fourier_frequencies
        in_dim = int(feature_dim) + 2 + int(extra_dim)
        layer_index = 0
        while layer_index < int(depth):
            layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
            layers.append(torch.nn.Tanh())
            in_dim = int(hidden_dim)
            layer_index += 1
        self.c_hidden = torch.nn.Sequential(*layers)
        self.c_head = torch.nn.Linear(in_dim, 1)
        torch.nn.init.zeros_(self.c_head.weight)
        torch.nn.init.zeros_(self.c_head.bias)

        # Flux head predicts a smooth multiplicative correction over a time basis.
        flux_basis_count = 24
        flux_layers = []
        flux_in_dim = int(feature_dim)
        flux_index = 0
        while flux_index < int(depth):
            flux_layers.append(torch.nn.Linear(flux_in_dim, int(hidden_dim)))
            flux_layers.append(torch.nn.Tanh())
            flux_in_dim = int(hidden_dim)
            flux_index += 1
        self.j_hidden = torch.nn.Sequential(*flux_layers)
        self.j_coeff_head = torch.nn.Linear(flux_in_dim, flux_basis_count)
        torch.nn.init.zeros_(self.j_coeff_head.weight)
        torch.nn.init.zeros_(self.j_coeff_head.bias)
        self.j_bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("flux_basis_centers", torch.linspace(0.0, 1.0, flux_basis_count))
        self.flux_basis_sigma = 1.5 / float(max(1, flux_basis_count - 1))

    def forward_c(self, x_feat, y_star, t_star):
        feature_parts = [x_feat, y_star, t_star]
        if self.fourier_frequencies > 0:
            freq = self.freq_values.to(dtype=x_feat.dtype, device=x_feat.device).reshape(1, -1)
            y_phase = (2.0 * math.pi) * (y_star * freq)
            t_phase = (2.0 * math.pi) * (t_star * freq)
            feature_parts.extend(
                [
                    torch.sin(y_phase),
                    torch.cos(y_phase),
                    torch.sin(t_phase),
                    torch.cos(t_phase),
                ]
            )
        inp = torch.cat(feature_parts, dim=1)
        # Output is dimensionless concentration c*=C/C0; nonnegativity is handled by loss terms.
        return self.c_head(self.c_hidden(inp))

    def forward_j(self, x_feat, t_star, j_base=None):
        hidden = self.j_hidden(x_feat)
        coeff = self.j_coeff_head(hidden)
        t_clamped = torch.clamp(t_star, min=0.0, max=1.0)
        centers = self.flux_basis_centers.to(dtype=t_clamped.dtype, device=t_clamped.device).reshape(1, -1)
        z = (t_clamped - centers) / float(self.flux_basis_sigma)
        basis = torch.exp(-0.5 * (z * z))
        raw = torch.sum(coeff * basis, dim=1, keepdim=True) + self.j_bias
        if j_base is None:
            # Standalone flux branch is only used for debugging or ablations.
            return torch.nn.functional.softplus(raw) * 1e-9
        # Default path: apply a bounded multiplicative correction to the backbone flux.
        corr = torch.exp(0.05 * torch.tanh(raw))
        return torch.clamp(j_base * corr, min=0.0)

    def forward(self, x_feat, y_star, t_star):
        return self.forward_c(x_feat, y_star, t_star)


def curriculum_scale(epoch_index, warmup_epochs, ramp_epochs):
    # Increase physics-loss influence gradually after warmup.
    if int(epoch_index) <= int(warmup_epochs):
        return 0.0
    ramp = max(1, int(ramp_epochs))
    progress = float(int(epoch_index) - int(warmup_epochs)) / float(ramp)
    return float(min(1.0, max(0.0, progress)))


def sample_time_ids(signal_per_time, sample_count):
    # Bias sampling toward informative times where concentration/flux is larger.
    weights = torch.clamp(signal_per_time, min=0.0) + 1e-8
    weights = weights / torch.sum(weights)
    return torch.multinomial(weights, num_samples=int(sample_count), replacement=True)


def sample_depth_ids(signal_per_depth, sample_count):
    # Reuse weighted sampling on depth so collocation focuses on active regions.
    weights = torch.clamp(signal_per_depth, min=0.0) + 1e-8
    weights = weights / torch.sum(weights)
    return torch.multinomial(weights, num_samples=int(sample_count), replacement=True)


def evaluate_cstar_rmse(model, split_torch, max_points=16384):
    # Fast concentration sanity metric sampled across run/time/depth points.
    model.eval()
    n_rows = int(split_torch["x_feat"].shape[0])
    t_count = int(split_torch["c_star"].shape[1])
    h_count = int(split_torch["c_star"].shape[2])
    total_points = n_rows * t_count * h_count
    sample_points = int(min(int(max_points), int(total_points)))

    run_ids = torch.randint(low=0, high=n_rows, size=(sample_points,), device=split_torch["x_feat"].device)
    time_ids = torch.randint(low=0, high=t_count, size=(sample_points,), device=split_torch["x_feat"].device)
    y_ids = torch.randint(low=0, high=h_count, size=(sample_points,), device=split_torch["x_feat"].device)

    x_sample = split_torch["x_feat"][run_ids]
    t_sample = split_torch["t_norm"][run_ids, time_ids].reshape(-1, 1)
    y_sample = (y_ids.to(torch.float32) / float(max(1, h_count - 1))).reshape(-1, 1)
    target = split_torch["c_star"][run_ids, time_ids, y_ids].reshape(-1, 1)

    with torch.no_grad():
        pred = model(x_sample, y_sample, t_sample)

    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return float(rmse.detach().cpu().item())


def predict_flux_curves(model, split_torch, batch_runs=16):
    # Predict full J(t) curves in batches to keep GPU memory bounded.
    model.eval()
    n_rows = int(split_torch["x_feat"].shape[0])
    t_count = int(split_torch["t_norm"].shape[1])
    out_rows = []

    start = 0
    while start < n_rows:
        end = min(start + int(batch_runs), n_rows)
        row_ids = torch.arange(start, end, device=split_torch["x_feat"].device, dtype=torch.long)
        x_batch = split_torch["x_feat"][row_ids]
        j_base_batch = split_torch["j_base"][row_ids]
        t_batch = split_torch["t_norm"][row_ids]
        local_count = int(end - start)

        run_ids = torch.arange(local_count, device=split_torch["x_feat"].device, dtype=torch.long)
        run_ids = run_ids.repeat_interleave(t_count)
        x_points = x_batch[run_ids]
        j_base_points = j_base_batch.reshape(-1, 1)
        t_points = t_batch.reshape(-1, 1)
        j_points = model.forward_j(x_points, t_points, j_base=j_base_points)
        out_rows.append(j_points.reshape(local_count, t_count).detach().cpu().numpy().astype(np.float32))

        start = end

    return np.concatenate(out_rows, axis=0)


def train_pinn(model, train_torch, val_torch, args):
    # Main training loop: data fit + physics losses with optional phase-2 tightening.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    n_train = int(train_torch["x_feat"].shape[0])
    t_count = int(train_torch["t_norm"].shape[1])
    h_count = int(train_torch["d1d"].shape[1])
    phase1_epochs = int(args.epochs)
    phase2_epochs = int(args.phase2_epochs)
    total_epochs = int(phase1_epochs + phase2_epochs)

    best_state = None
    best_val_rel_l2 = float("inf")
    best_val_cstar_rmse = float("inf")
    best_phase1_val_rel_l2 = float("inf")
    phase2_stopped_early = False
    final_epoch_ran = 0
    history = []

    epoch_index = 1
    while epoch_index <= int(total_epochs):
        model.train()
        is_phase2 = bool(epoch_index > int(phase1_epochs))

        # Sample a fresh run mini-batch each epoch.
        run_pick = np.random.permutation(n_train)[: int(min(args.batch_runs, n_train))]
        run_pick_t = torch.as_tensor(run_pick, dtype=torch.long, device=train_torch["x_feat"].device)
        x_batch = train_torch["x_feat"][run_pick_t]
        j_true_batch = train_torch["j_true"][run_pick_t]
        j_base_batch = train_torch["j_base"][run_pick_t]
        d_batch = train_torch["d1d"][run_pick_t]
        d_dy_batch = train_torch["d1d_dy_star"][run_pick_t]
        k_batch = train_torch["k1d"][run_pick_t]
        c_star_batch = train_torch["c_star"][run_pick_t]
        t_norm_batch = train_torch["t_norm"][run_pick_t]
        t_phys_batch = train_torch["t_phys"][run_pick_t]
        patch_width_batch = train_torch["patch_width"][run_pick_t]
        c0_batch = train_torch["c0"][run_pick_t]
        decay_batch = train_torch["decay"][run_pick_t]
        mode_batch = train_torch["mode_flag"][run_pick_t]
        depth_batch = train_torch["depth"][run_pick_t]
        t_end_batch = train_torch["t_end"][run_pick_t]
        batch_count = int(x_batch.shape[0])
        c_signal_time = torch.mean(c_star_batch.detach(), dim=(0, 2))
        c_signal_depth = torch.mean(c_star_batch.detach(), dim=(0, 1))
        j_signal_time = torch.mean(torch.abs(j_true_batch.detach()), dim=0)

        # Concentration supervision from simulator snapshots (stage-A sanity target).
        conc_count = int(args.concentration_points)
        run_ids_conc = torch.randint(low=0, high=batch_count, size=(conc_count,), device=x_batch.device)
        time_ids_conc = sample_time_ids(c_signal_time, conc_count).to(x_batch.device)
        y_ids_conc = sample_depth_ids(c_signal_depth, conc_count).to(x_batch.device)
        x_conc = x_batch[run_ids_conc]
        t_conc = t_norm_batch[run_ids_conc, time_ids_conc].reshape(-1, 1)
        y_conc = (y_ids_conc.to(torch.float32) / float(max(1, h_count - 1))).reshape(-1, 1)
        c_conc_true = c_star_batch[run_ids_conc, time_ids_conc, y_ids_conc].reshape(-1, 1)
        c_conc_pred = model(x_conc, y_conc, t_conc)
        conc_scale = float(args.conc_log_scale)
        c_pred_asinh = torch.asinh(c_conc_pred * conc_scale)
        c_true_asinh = torch.asinh(c_conc_true * conc_scale)
        loss_c_data = torch.mean((c_pred_asinh - c_true_asinh) ** 2)
        loss_c_nonneg = torch.mean(torch.relu(-c_conc_pred) ** 2)

        # Near-bottom concentration supervision to shape the discrete gradient used for flux.
        inner_count = int(args.inner_profile_points)
        run_ids_inner = torch.randint(low=0, high=batch_count, size=(inner_count,), device=x_batch.device)
        time_ids_inner = sample_time_ids(j_signal_time, inner_count).to(x_batch.device)
        x_inner = x_batch[run_ids_inner]
        t_inner = t_norm_batch[run_ids_inner, time_ids_inner].reshape(-1, 1)
        y_inner = torch.full_like(
            t_inner,
            float(h_count - 2) / float(max(1, h_count - 1)),
        )
        c_inner_true = c_star_batch[run_ids_inner, time_ids_inner, h_count - 2].reshape(-1, 1)
        c_inner_pred = model(x_inner, y_inner, t_inner)
        c_inner_pred_asinh = torch.asinh(c_inner_pred * conc_scale)
        c_inner_true_asinh = torch.asinh(c_inner_true * conc_scale)
        loss_inner_profile = torch.mean((c_inner_pred_asinh - c_inner_true_asinh) ** 2)

        # PDE residual on interior collocation points in dimensionless form.
        pde_count = int(args.collocation_points)
        run_ids_pde = torch.randint(low=0, high=batch_count, size=(pde_count,), device=x_batch.device)
        time_ids_pde = sample_time_ids(c_signal_time[1:], pde_count).to(x_batch.device) + 1
        y_ids_pde = sample_depth_ids(c_signal_depth[1 : h_count - 1], pde_count).to(x_batch.device) + 1
        x_pde = x_batch[run_ids_pde]
        t_pde = t_norm_batch[run_ids_pde, time_ids_pde].reshape(-1, 1)
        y_pde = (y_ids_pde.to(torch.float32) / float(max(1, h_count - 1))).reshape(-1, 1)
        y_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        c_pde = model(x_pde, y_pde, t_pde)
        c_y = torch.autograd.grad(
            c_pde,
            y_pde,
            grad_outputs=torch.ones_like(c_pde),
            create_graph=True,
            retain_graph=True,
        )[0]
        c_t = torch.autograd.grad(
            c_pde,
            t_pde,
            grad_outputs=torch.ones_like(c_pde),
            create_graph=True,
            retain_graph=True,
        )[0]
        c_yy = torch.autograd.grad(
            c_y,
            y_pde,
            grad_outputs=torch.ones_like(c_y),
            create_graph=True,
            retain_graph=True,
        )[0]

        d_points = d_batch[run_ids_pde, y_ids_pde].reshape(-1, 1)
        d_dy_points = d_dy_batch[run_ids_pde, y_ids_pde].reshape(-1, 1)
        k_points = k_batch[run_ids_pde, y_ids_pde].reshape(-1, 1)
        depth_points = depth_batch[run_ids_pde].reshape(-1, 1)
        t_end_points = t_end_batch[run_ids_pde].reshape(-1, 1)
        alpha_points = t_end_points / torch.clamp(depth_points * depth_points, min=1e-12)
        beta_points = t_end_points * k_points
        pde_residual = c_t - (alpha_points * ((d_points * c_yy) + (d_dy_points * c_y))) + (beta_points * c_pde)
        loss_pde = torch.mean(pde_residual * pde_residual)

        # Top/bottom BC losses.
        bc_count = int(args.bc_points)
        run_ids_bc = torch.randint(low=0, high=batch_count, size=(bc_count,), device=x_batch.device)
        time_ids_bc = sample_time_ids(c_signal_time, bc_count).to(x_batch.device)
        x_bc = x_batch[run_ids_bc]
        t_bc = t_norm_batch[run_ids_bc, time_ids_bc].reshape(-1, 1)
        t_phys_bc = t_phys_batch[run_ids_bc, time_ids_bc].reshape(-1, 1)
        y_top = torch.zeros_like(t_bc)
        y_bottom = torch.ones_like(t_bc)
        c_top_pred = model(x_bc, y_top, t_bc)
        c_bottom_pred = model(x_bc, y_bottom, t_bc)
        c_patch_star = (
            mode_batch[run_ids_bc].reshape(-1, 1) * torch.exp(-decay_batch[run_ids_bc].reshape(-1, 1) * t_phys_bc)
        ) + (1.0 - mode_batch[run_ids_bc].reshape(-1, 1))
        c_top_target = patch_width_batch[run_ids_bc].reshape(-1, 1) * c_patch_star
        loss_bc_top = torch.mean((c_top_pred - c_top_target) ** 2)
        loss_bc_bottom = torch.mean(c_bottom_pred**2)

        # Initial condition loss c*(y,0)=0.
        ic_count = int(args.ic_points)
        run_ids_ic = torch.randint(low=0, high=batch_count, size=(ic_count,), device=x_batch.device)
        y_ids_ic = torch.randint(low=0, high=h_count, size=(ic_count,), device=x_batch.device)
        x_ic = x_batch[run_ids_ic]
        t_ic = torch.zeros((ic_count, 1), device=x_batch.device, dtype=torch.float32)
        y_ic = (y_ids_ic.to(torch.float32) / float(max(1, h_count - 1))).reshape(-1, 1)
        c_ic_pred = model(x_ic, y_ic, t_ic)
        loss_ic = torch.mean(c_ic_pred**2)

        # Flux data loss plus anchor-to-backbone loss.
        flux_count = int(args.flux_points)
        run_ids_flux = torch.randint(low=0, high=batch_count, size=(flux_count,), device=x_batch.device)
        time_ids_flux = sample_time_ids(j_signal_time, flux_count).to(x_batch.device)
        x_flux = x_batch[run_ids_flux]
        t_flux = t_norm_batch[run_ids_flux, time_ids_flux].reshape(-1, 1)
        j_flux_base = j_base_batch[run_ids_flux, time_ids_flux].reshape(-1, 1)
        j_flux_pred = model.forward_j(x_flux, t_flux, j_base=j_flux_base)

        y_flux_bottom = torch.ones_like(t_flux)
        y_flux_inner = torch.full_like(t_flux, float(h_count - 2) / float(max(1, h_count - 1)))
        c_flux_bottom = model.forward_c(x_flux, y_flux_bottom, t_flux)
        c_flux_inner = model.forward_c(x_flux, y_flux_inner, t_flux)
        d_bottom_flux = d_batch[run_ids_flux, h_count - 2].reshape(-1, 1)
        c0_flux = c0_batch[run_ids_flux].reshape(-1, 1)
        depth_flux = depth_batch[run_ids_flux].reshape(-1, 1)
        dy_star_inv = float(max(1, h_count - 1))
        grad_star_flux = (c_flux_bottom - c_flux_inner) * dy_star_inv
        j_flux_from_c = -d_bottom_flux * ((c0_flux / torch.clamp(depth_flux, min=1e-12)) * grad_star_flux)

        j_flux_true = j_true_batch[run_ids_flux, time_ids_flux].reshape(-1, 1)
        flux_scale_log = float(args.flux_log_scale)
        j_pred_asinh = torch.asinh(j_flux_pred * flux_scale_log)
        j_true_asinh = torch.asinh(j_flux_true * flux_scale_log)
        j_base_asinh = torch.asinh(j_flux_base * flux_scale_log)
        j_from_c_asinh = torch.asinh(j_flux_from_c * flux_scale_log)
        loss_flux = torch.mean((j_pred_asinh - j_true_asinh) ** 2)
        loss_anchor = torch.mean((j_pred_asinh - j_base_asinh) ** 2)
        loss_flux_consistency = torch.mean((j_pred_asinh - j_from_c_asinh) ** 2)
        loss_nonneg = torch.mean(torch.relu(-j_flux_pred) ** 2)

        if is_phase2:
            # In phase-2 we fully enable flux and PDE terms.
            flux_scale = 1.0
            pde_scale = 1.0
        else:
            flux_scale = curriculum_scale(
                epoch_index=epoch_index,
                warmup_epochs=int(args.flux_warmup_epochs),
                ramp_epochs=int(args.flux_ramp_epochs),
            )
            pde_scale = curriculum_scale(
                epoch_index=epoch_index,
                warmup_epochs=int(args.pde_warmup_epochs),
                ramp_epochs=int(args.pde_ramp_epochs),
            )

        weight_flux_consistency = float(args.weight_flux_consistency)
        weight_pde = float(args.weight_pde)
        weight_bc_top = float(args.weight_bc_top)
        weight_bc_bottom = float(args.weight_bc_bottom)
        weight_ic = float(args.weight_ic)
        if is_phase2:
            # Tighten selected physics terms during phase-2 fine-tuning.
            weight_flux_consistency = weight_flux_consistency * float(args.phase2_weight_flux_consistency_mult)
            weight_pde = weight_pde * float(args.phase2_weight_pde_mult)
            weight_bc_top = weight_bc_top * float(args.phase2_weight_bc_top_mult)
            weight_bc_bottom = weight_bc_bottom * float(args.phase2_weight_bc_bottom_mult)
            weight_ic = weight_ic * float(args.phase2_weight_ic_mult)

        total_loss = (
            (float(args.weight_c_data) * loss_c_data)
            + (float(args.weight_c_nonneg) * loss_c_nonneg)
            + (float(args.weight_inner_profile) * loss_inner_profile)
            + (float(args.weight_flux) * float(flux_scale) * loss_flux)
            + (float(args.weight_anchor) * float(flux_scale) * loss_anchor)
            + (float(weight_flux_consistency) * float(flux_scale) * loss_flux_consistency)
            + (float(weight_pde) * float(pde_scale) * loss_pde)
            + (float(weight_bc_top) * loss_bc_top)
            + (float(weight_bc_bottom) * loss_bc_bottom)
            + (float(weight_ic) * loss_ic)
            + (float(args.weight_nonneg) * loss_nonneg)
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch_index == 1 or epoch_index == int(total_epochs) or epoch_index % int(args.eval_every) == 0:
            # Validation is based on curve rel-L2 because that is the final deployment target.
            j_val_pinn = predict_flux_curves(
                model=model,
                split_torch=val_torch,
                batch_runs=int(args.predict_batch_runs),
            )
            val_metrics = compute_curve_metrics(
                val_torch["j_true"].detach().cpu().numpy(),
                j_val_pinn,
                val_torch["t_phys"].detach().cpu().numpy(),
                eps=1e-12,
            )
            val_rel_l2 = float(val_metrics["relative_l2"])
            val_cstar_rmse = evaluate_cstar_rmse(
                model=model,
                split_torch=val_torch,
                max_points=int(args.eval_concentration_points),
            )

            if val_rel_l2 < best_val_rel_l2:
                # Keep the best validation checkpoint to avoid end-of-training regressions.
                best_val_rel_l2 = val_rel_l2
                best_val_cstar_rmse = val_cstar_rmse
                state = {}
                for param_name in model.state_dict():
                    state[param_name] = model.state_dict()[param_name].detach().cpu().clone()
                best_state = state

            if not is_phase2 and val_rel_l2 < best_phase1_val_rel_l2:
                best_phase1_val_rel_l2 = val_rel_l2

            row = {
                "epoch": int(epoch_index),
                "phase": "phase2_physics_tighten" if is_phase2 else "phase1_main",
                "loss_total": float(total_loss.detach().cpu().item()),
                "val_rel_l2": float(val_rel_l2),
            }
            history.append(row)
            print(
                "epoch",
                int(epoch_index),
                "loss_total=",
                f"{row['loss_total']:.6e}",
                "val_rel_l2=",
                f"{row['val_rel_l2']:.6e}",
                "val_cstar_rmse=",
                f"{float(val_cstar_rmse):.6e}",
            )

            if is_phase2 and phase2_epochs > 0 and np.isfinite(best_phase1_val_rel_l2):
                # Guard: stop phase-2 if validation rel-L2 drifts above the allowed margin.
                guard_limit = float(best_phase1_val_rel_l2) * (1.0 + float(args.phase2_max_rel_l2_regression))
                if float(val_rel_l2) > float(guard_limit):
                    phase2_stopped_early = True
                    print(
                        "phase2_guard_stop",
                        "epoch",
                        int(epoch_index),
                        "val_rel_l2=",
                        f"{float(val_rel_l2):.6e}",
                        "guard_limit=",
                        f"{float(guard_limit):.6e}",
                    )
                    final_epoch_ran = int(epoch_index)
                    break

        final_epoch_ran = int(epoch_index)
        epoch_index += 1

    if best_state is None:
        raise ValueError("PINN training did not record any validation checkpoint")

    if not np.isfinite(best_phase1_val_rel_l2):
        best_phase1_val_rel_l2 = float(best_val_rel_l2)

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_val_rel_l2": float(best_val_rel_l2),
        "best_val_cstar_rmse": float(best_val_cstar_rmse),
        "best_phase1_val_rel_l2": float(best_phase1_val_rel_l2),
        "phase1_epochs": int(phase1_epochs),
        "phase2_epochs": int(phase2_epochs),
        "phase2_stopped_early": bool(phase2_stopped_early),
        "final_epoch_ran": int(final_epoch_ran),
    }


def pick_blend_lambda(j_val_true, t_val, j_val_base, j_val_pinn):
    # No-harm selection on validation split; lambda=0 means fallback to backbone.
    best_lambda = 0.0
    best_metric = None
    lambda_values = np.linspace(0.0, 1.0, 11).astype(np.float32)
    value_index = 0
    while value_index < int(lambda_values.shape[0]):
        lam = float(lambda_values[value_index])
        j_mix = j_val_base + (lam * (j_val_pinn - j_val_base))
        metrics = compute_curve_metrics(j_val_true, j_mix, t_val, eps=1e-12)
        if best_metric is None or float(metrics["relative_l2"]) < float(best_metric["relative_l2"]):
            best_metric = metrics
            best_lambda = lam
        value_index += 1
    return float(best_lambda), best_metric


def build_metrics_bundle(j_true, t_split, j_base, j_pinn, j_final, y_true, c0_split, target_names):
    # Keep output shape aligned with existing blackbox/hybrid metric conventions.
    t_matrix = np.asarray(t_split, dtype=np.float32)
    y_stage_base = scalar_targets_from_flux_curves(j_base, t_matrix, c0_split, target_names)
    y_stage_pinn = scalar_targets_from_flux_curves(j_pinn, t_matrix, c0_split, target_names)
    y_stage_final = scalar_targets_from_flux_curves(j_final, t_matrix, c0_split, target_names)
    return {
        "available": True,
        "stageBase": {
            "curve": compute_curve_metrics(j_true, j_base, t_matrix, eps=1e-12),
            "scalar": scalar_report(y_true, y_stage_base, target_names),
        },
        "stagePINN": {
            "curve": compute_curve_metrics(j_true, j_pinn, t_matrix, eps=1e-12),
            "scalar": scalar_report(y_true, y_stage_pinn, target_names),
        },
        "stageFinal": {
            "curve": compute_curve_metrics(j_true, j_final, t_matrix, eps=1e-12),
            "scalar": scalar_report(y_true, y_stage_final, target_names),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    # Dataset and output paths.
    parser.add_argument("--ml_dir", default="data/processed/ml")
    parser.add_argument("--out_dir", default="outputs/ml/pinn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_root_override", default=None)
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    parser.add_argument("--max_train_rows", type=int, default=256)
    parser.add_argument("--max_val_rows", type=int, default=64)
    parser.add_argument("--max_test_rows", type=int, default=64)

    # Stage-0 curve backbone.
    parser.add_argument("--backbone_curve_components", type=int, default=20)
    parser.add_argument("--backbone_alphas", default="1e-4,1e-3,1e-2,1e-1,1,10")

    # Neural model and optimizer settings.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--phase2_epochs", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_runs", type=int, default=24)
    parser.add_argument("--predict_batch_runs", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fourier_frequencies", type=int, default=6)

    # Curriculum schedule.
    parser.add_argument("--flux_warmup_epochs", type=int, default=6)
    parser.add_argument("--flux_ramp_epochs", type=int, default=6)
    parser.add_argument("--pde_warmup_epochs", type=int, default=10)
    parser.add_argument("--pde_ramp_epochs", type=int, default=10)

    # Sampling counts for each loss component.
    parser.add_argument("--concentration_points", type=int, default=4096)
    parser.add_argument("--collocation_points", type=int, default=2048)
    parser.add_argument("--bc_points", type=int, default=512)
    parser.add_argument("--ic_points", type=int, default=512)
    parser.add_argument("--flux_points", type=int, default=2048)
    parser.add_argument("--inner_profile_points", type=int, default=1024)
    parser.add_argument("--eval_concentration_points", type=int, default=16384)

    # Loss weights.
    parser.add_argument("--weight_c_data", type=float, default=1.0)
    parser.add_argument("--weight_c_nonneg", type=float, default=0.1)
    parser.add_argument("--weight_inner_profile", type=float, default=2.0)
    parser.add_argument("--weight_flux", type=float, default=2.0)
    parser.add_argument("--weight_anchor", type=float, default=1.0)
    parser.add_argument("--weight_flux_consistency", type=float, default=0.2)
    parser.add_argument("--weight_pde", type=float, default=0.01)
    parser.add_argument("--weight_bc_top", type=float, default=0.2)
    parser.add_argument("--weight_bc_bottom", type=float, default=0.2)
    parser.add_argument("--weight_ic", type=float, default=0.1)
    parser.add_argument("--weight_nonneg", type=float, default=0.05)
    parser.add_argument("--conc_log_scale", type=float, default=5000.0)
    parser.add_argument("--flux_log_scale", type=float, default=1e11)

    # Phase-2 physics tightening multipliers.
    parser.add_argument("--phase2_weight_flux_consistency_mult", type=float, default=2.0)
    parser.add_argument("--phase2_weight_pde_mult", type=float, default=4.0)
    parser.add_argument("--phase2_weight_bc_top_mult", type=float, default=2.0)
    parser.add_argument("--phase2_weight_bc_bottom_mult", type=float, default=2.0)
    parser.add_argument("--phase2_weight_ic_mult", type=float, default=2.0)
    parser.add_argument("--phase2_max_rel_l2_regression", type=float, default=0.01)
    parser.add_argument("--worst_case_top_n", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    ml_dir = Path(args.ml_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    set_global_seed(int(args.seed))

    device = choose_device(args.device)
    print("device:", str(device))
    if str(device) == "cuda":
        print("cuda_name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    backbone_alphas = parse_float_csv(
        args.backbone_alphas,
        fallback_values=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    )

    split_name_map = load_split_name_map(ml_dir)

    # Build split row selections from the pre-generated ML dataset index.
    train_entries = build_entries(
        ml_dir=ml_dir,
        split_name=split_name_map["train"],
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
        max_rows=args.max_train_rows,
        run_root_override=args.run_root_override,
    )
    val_entries = build_entries(
        ml_dir=ml_dir,
        split_name=split_name_map["val"],
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
        max_rows=args.max_val_rows,
        run_root_override=args.run_root_override,
    )
    test_entries = build_entries(
        ml_dir=ml_dir,
        split_name=split_name_map["test"],
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
        max_rows=args.max_test_rows,
        run_root_override=args.run_root_override,
    )
    feature_names, target_names = load_ml_meta_names(ml_dir)

    # Load tabular features, target curves, and curve time grids.
    x_train_raw = load_split_matrix(ml_dir, split_name_map["train"], train_entries, key="X")
    x_val_raw = load_split_matrix(ml_dir, split_name_map["val"], val_entries, key="X")
    x_test_raw = load_split_matrix(ml_dir, split_name_map["test"], test_entries, key="X")
    y_val_true = load_split_matrix(ml_dir, split_name_map["val"], val_entries, key="y_scalar")
    y_test_true = load_split_matrix(ml_dir, split_name_map["test"], test_entries, key="y_scalar")
    c0_val = extract_c0_feature(x_val_raw, feature_names)
    c0_test = extract_c0_feature(x_test_raw, feature_names)

    j_train_true = load_split_matrix(ml_dir, split_name_map["train"], train_entries, key="J")
    j_val_true = load_split_matrix(ml_dir, split_name_map["val"], val_entries, key="J")
    j_test_true = load_split_matrix(ml_dir, split_name_map["test"], test_entries, key="J")
    t_val = load_split_matrix(ml_dir, split_name_map["val"], val_entries, key="t")
    t_test = load_split_matrix(ml_dir, split_name_map["test"], test_entries, key="t")

    # Fit and predict the blackbox-style base curve model.
    print("fitting blackbox-style backbone ...")
    backbone = fit_curve_backbone(
        x_train_raw=x_train_raw,
        j_train_true=j_train_true,
        x_val_raw=x_val_raw,
        j_val_true=j_val_true,
        curve_components=int(args.backbone_curve_components),
        alpha_grid=backbone_alphas,
    )
    j_train_base, x_train_scaled = predict_curve_backbone(backbone, x_train_raw)
    j_val_base, x_val_scaled = predict_curve_backbone(backbone, x_val_raw)
    j_test_base, x_test_scaled = predict_curve_backbone(backbone, x_test_raw)

    # Load simulator-side fields needed by the PDE and BC losses.
    print("loading run physics for train split ...")
    train_phys = load_run_physics_1d(train_entries)
    print("loading run physics for val split ...")
    val_phys = load_run_physics_1d(val_entries)
    print("loading run physics for test split ...")
    test_phys = load_run_physics_1d(test_entries)

    # Move all split packs onto the selected device once.
    train_pack = tensorize_pack(
        pack_for_training(x_train_scaled, j_train_true, j_train_base, train_phys),
        device=device,
    )
    val_pack = tensorize_pack(
        pack_for_training(x_val_scaled, j_val_true, j_val_base, val_phys),
        device=device,
    )
    test_pack = tensorize_pack(
        pack_for_training(x_test_scaled, j_test_true, j_test_base, test_phys),
        device=device,
    )

    model = RunConditionedPINN(
        feature_dim=int(x_train_scaled.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        fourier_frequencies=int(args.fourier_frequencies),
    ).to(device)

    # Train with checkpointing and optional phase-2 physics tightening.
    train_result = train_pinn(
        model=model,
        train_torch=train_pack,
        val_torch=val_pack,
        args=args,
    )
    model = train_result["model"]

    # Build full validation/test curve predictions.
    print("predicting full curves for val/test ...")
    j_val_pinn = predict_flux_curves(model, val_pack, batch_runs=int(args.predict_batch_runs))
    j_test_pinn = predict_flux_curves(model, test_pack, batch_runs=int(args.predict_batch_runs))

    # Blend is selected on validation to prevent PINN from harming final deployment curves.
    lambda_best, lambda_metric = pick_blend_lambda(
        j_val_true=j_val_true,
        t_val=t_val,
        j_val_base=j_val_base,
        j_val_pinn=j_val_pinn,
    )
    j_val_final = j_val_base + (lambda_best * (j_val_pinn - j_val_base))
    j_test_final = j_test_base + (lambda_best * (j_test_pinn - j_test_base))

    # Save metrics bundles in train/val/test terminology only.
    metrics_val = build_metrics_bundle(
        j_val_true,
        t_val,
        j_val_base,
        j_val_pinn,
        j_val_final,
        y_val_true,
        c0_val,
        target_names,
    )
    metrics_test = build_metrics_bundle(
        j_test_true,
        t_test,
        j_test_base,
        j_test_pinn,
        j_test_final,
        y_test_true,
        c0_test,
        target_names,
    )
    (out_dir / "metrics_val.json").write_text(json.dumps(metrics_val, indent=2), encoding="utf-8")
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")

    np.savez(
        out_dir / "pred_val.npz",
        j_true=j_val_true.astype(np.float32),
        j_stageBase=j_val_base.astype(np.float32),
        j_stagePINN=j_val_pinn.astype(np.float32),
        j_stageFinal=j_val_final.astype(np.float32),
        t=t_val.astype(np.float32),
    )
    np.savez(
        out_dir / "pred_test.npz",
        j_true=j_test_true.astype(np.float32),
        j_stageBase=j_test_base.astype(np.float32),
        j_stagePINN=j_test_pinn.astype(np.float32),
        j_stageFinal=j_test_final.astype(np.float32),
        t=t_test.astype(np.float32),
    )

    out_diag = out_dir / "diagnostics"
    out_diag_physics = out_diag / "physics"
    ensure_dir(out_diag)
    ensure_dir(out_diag_physics)

    # Physics diagnostics are computed from run bundles and predicted curves.
    print("computing physics diagnostics for val ...")
    physics_val_rows, physics_val_summary = compute_split_physics_diagnostics(
        entries=val_entries,
        split_name="val",
        prediction_map={
            "truth": j_val_true,
            "stageBase": j_val_base,
            "stagePINN": j_val_pinn,
            "stageFinal": j_val_final,
        },
        progress_label="physics_val",
        truth_key="truth",
        pde_anchor_key="stageBase",
    )
    print("computing physics diagnostics for test ...")
    physics_test_rows, physics_test_summary = compute_split_physics_diagnostics(
        entries=test_entries,
        split_name="test",
        prediction_map={
            "truth": j_test_true,
            "stageBase": j_test_base,
            "stagePINN": j_test_pinn,
            "stageFinal": j_test_final,
        },
        progress_label="physics_test",
        truth_key="truth",
        pde_anchor_key="stageBase",
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
    # Keep explicit worst-case rows for manual inspection after each run.
    worst_val = build_worst_case_report(
        physics_val_rows,
        split_name="val",
        stage_name="stageFinal",
        top_n=int(args.worst_case_top_n),
    )
    worst_test = build_worst_case_report(
        physics_test_rows,
        split_name="test",
        stage_name="stageFinal",
        top_n=int(args.worst_case_top_n),
    )
    (out_diag_physics / "physics_diag_val_worst.json").write_text(
        json.dumps(worst_val, indent=2),
        encoding="utf-8",
    )
    (out_diag_physics / "physics_diag_test_worst.json").write_text(
        json.dumps(worst_test, indent=2),
        encoding="utf-8",
    )

    # Curve and scalar plots for quick visual comparison with blackbox.
    plot_dir = out_dir / "plots"
    plot_test_dir = plot_dir / "test"
    ensure_dir(plot_dir)
    ensure_dir(plot_test_dir)
    plot_curve_examples(
        t_test,
        j_test_true,
        j_test_final,
        plot_test_dir / "curve_examples.png",
        "J(t) examples (test)",
        max_examples=9,
    )
    plot_curve_error_over_time(
        t_test,
        j_test_true,
        j_test_final,
        plot_test_dir / "curve_error_over_time.png",
        "Curve error over time (test)",
    )
    y_test_stage_final = scalar_targets_from_flux_curves(j_test_final, t_test, c0_test, target_names)
    plot_scalar_parity(
        y_test_true,
        y_test_stage_final,
        target_names,
        plot_test_dir / "scalar_parity.png",
        "Scalar parity (test)",
    )
    plot_scalar_residual_hist(
        y_test_true,
        y_test_stage_final,
        target_names,
        plot_test_dir / "scalar_residuals.png",
        "Scalar residuals (test)",
    )

    # Write one summary artifact with config, metrics, diagnostics, and runtime.
    summary = {
        "status": "ok",
        "description": "PINN: dimensionless C*(y,t) with c-data warmup, PDE curriculum, and no-harm blend",
        "seed": int(args.seed),
        "device": str(device),
        "run_root_override": args.run_root_override,
        "split_rows": {
            "train": int(len(train_entries)),
            "val": int(len(val_entries)),
            "test": int(len(test_entries)),
        },
        "backbone": {
            "curve_components": int(args.backbone_curve_components),
            "alpha_grid": [float(value) for value in backbone_alphas],
            "selected_ridge_alpha": float(backbone["ridge_alpha"]),
            "val_rmse_for_selection": float(backbone["val_rmse"]),
        },
        "pinn_config": {
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "fourier_frequencies": int(args.fourier_frequencies),
            "epochs": int(args.epochs),
            "phase2_epochs": int(args.phase2_epochs),
            "flux_warmup_epochs": int(args.flux_warmup_epochs),
            "flux_ramp_epochs": int(args.flux_ramp_epochs),
            "pde_warmup_epochs": int(args.pde_warmup_epochs),
            "pde_ramp_epochs": int(args.pde_ramp_epochs),
            "lr": float(args.lr),
            "batch_runs": int(args.batch_runs),
            "concentration_points": int(args.concentration_points),
            "collocation_points": int(args.collocation_points),
            "bc_points": int(args.bc_points),
            "ic_points": int(args.ic_points),
            "flux_points": int(args.flux_points),
            "inner_profile_points": int(args.inner_profile_points),
            "conc_log_scale": float(args.conc_log_scale),
            "flux_log_scale": float(args.flux_log_scale),
            "phase2_weight_multipliers": {
                "flux_consistency": float(args.phase2_weight_flux_consistency_mult),
                "pde": float(args.phase2_weight_pde_mult),
                "bc_top": float(args.phase2_weight_bc_top_mult),
                "bc_bottom": float(args.phase2_weight_bc_bottom_mult),
                "ic": float(args.phase2_weight_ic_mult),
            },
            "phase2_max_rel_l2_regression": float(args.phase2_max_rel_l2_regression),
            "weights": {
                "c_data": float(args.weight_c_data),
                "c_nonneg": float(args.weight_c_nonneg),
                "inner_profile": float(args.weight_inner_profile),
                "flux": float(args.weight_flux),
                "anchor": float(args.weight_anchor),
                "flux_consistency": float(args.weight_flux_consistency),
                "pde": float(args.weight_pde),
                "bc_top": float(args.weight_bc_top),
                "bc_bottom": float(args.weight_bc_bottom),
                "ic": float(args.weight_ic),
                "nonneg": float(args.weight_nonneg),
            },
        },
        "best_val_rel_l2_stagePINN": float(train_result["best_val_rel_l2"]),
        "best_val_cstar_rmse_stagePINN": float(train_result["best_val_cstar_rmse"]),
        "best_phase1_val_rel_l2_stagePINN": float(train_result["best_phase1_val_rel_l2"]),
        "phase2_status": {
            "phase1_epochs": int(train_result["phase1_epochs"]),
            "phase2_epochs": int(train_result["phase2_epochs"]),
            "phase2_stopped_early": bool(train_result["phase2_stopped_early"]),
            "final_epoch_ran": int(train_result["final_epoch_ran"]),
        },
        "blend_lambda": float(lambda_best),
        "blend_lambda_val_metric": lambda_metric,
        "metrics": {
            "val": metrics_val,
            "test": metrics_test,
        },
        "test_scalar_rmse": build_scalar_rmse_summary(metrics_test["stageFinal"]["scalar"], target_names),
        "physics_diagnostics": {
            "val_summary": physics_val_summary,
            "test_summary": physics_test_summary,
        },
        "runtime_seconds": float(time.time() - t0),
    }
    # Persist a compact machine-readable run record for comparisons and plotting.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(train_result["history"], indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "pinn_model.pt")

    print("saved:", str(out_dir))
    print("stageBase test rel_l2:", metrics_test["stageBase"]["curve"]["relative_l2"])
    print("stagePINN test rel_l2:", metrics_test["stagePINN"]["curve"]["relative_l2"])
    print("stageFinal test rel_l2:", metrics_test["stageFinal"]["curve"]["relative_l2"])


if __name__ == "__main__":
    main()
