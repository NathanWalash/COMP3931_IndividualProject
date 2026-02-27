import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from skin_diffusion.ml_curve_plots import plot_curve_error_over_time, plot_curve_examples
from skin_diffusion.ml_metrics import compute_curve_metrics
from skin_diffusion.ml_physics_diagnostics import (
    build_worst_case_report,
    compute_split_physics_diagnostics,
    write_rows_csv,
)
from skin_diffusion.ml_run_dataset import (
    load_run_bundle,
    load_split_entries,
    load_split_feature_matrix,
    load_split_scalar_matrix,
    remap_run_dir,
)
from skin_diffusion.ml_scalar_diagnostics import (
    build_scalar_rmse_summary,
    plot_scalar_parity,
    plot_scalar_residual_hist,
    scalar_report,
    scalar_targets_from_flux_curves,
)
from skin_diffusion.run_index import in_index_range, run_index_from_path
from skin_diffusion.utils import ensure_dir


# Hybrid PINN path:
# Stage A: coarse discrete PDE baseline
# Stage B: learned amplitude/time-warp correction
# Stage C: low-rank residual correction


def choose_device(device_arg):
    # Keep device selection simple: auto->cuda if available, else cpu.
    text = str(device_arg).strip().lower()
    if text in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_entries(
    ml_dir,
    split_name,
    run_start_index=None,
    run_end_index=None,
    max_rows=None,
    run_root_override=None,
    allow_empty=False,
):
    # Read split index rows from ml/meta, then apply optional row and path filters.
    entries = load_split_entries(
        ml_dir=ml_dir,
        split_name=split_name,
        run_start_index=run_start_index,
        run_end_index=run_end_index,
        allow_empty=bool(allow_empty),
    )

    out = []
    for entry in entries:
        run_idx = run_index_from_path(entry["run_dir"])
        if in_index_range(run_idx, run_start_index, run_end_index):
            row = dict(entry)
            # Allows running from fast staged storage while keeping split definitions unchanged.
            row["run_dir"] = remap_run_dir(entry["run_dir"], run_root_override)
            out.append(row)

    if max_rows is not None:
        out = out[: int(max_rows)]

    if len(out) == 0 and not bool(allow_empty):
        raise ValueError("No rows selected for split " + split_name)
    return out


def load_ml_meta_names(ml_dir):
    # Hybrid scalar diagnostics need both feature and scalar target names.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing meta.json in " + str(ml_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_names = meta.get("feature_names", [])
    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("meta.json is missing feature_names")

    # Scalar targets are dataset-configurable; keep legacy defaults as fallback.
    target_names = meta.get("scalar_target_names", [])
    if not isinstance(target_names, list) or len(target_names) == 0:
        target_names = ["P", "Tlag", "J_ss"]

    out_feature_names = []
    for name in feature_names:
        out_feature_names.append(str(name))

    out_target_names = []
    for name in target_names:
        out_target_names.append(str(name))

    return out_feature_names, out_target_names


def extract_c0_feature(x_raw, feature_names):
    # Use C0 from raw feature matrix to derive permeability from predicted J(t).
    if "C0" not in feature_names:
        return np.full((int(x_raw.shape[0]),), np.nan, dtype=np.float32)
    c0_col = int(feature_names.index("C0"))
    return np.asarray(x_raw[:, c0_col], dtype=np.float32)


def patch_concentration(t_value, mode, c0, decay_rate):
    # Match simulator top-patch BC modes used during dataset generation.
    if str(mode) == "time_decay":
        return float(c0) * math.exp(-float(decay_rate) * float(t_value))
    return float(c0)


def downsample_mean_2d(field_2d, factor):
    # Coarsen high-res fields onto Stage A grid by block averaging.
    h, w = field_2d.shape
    if (h % factor) != 0 or (w % factor) != 0:
        raise ValueError("Grid shape is not divisible by coarsen_factor")
    h2 = h // factor
    w2 = w // factor
    reshaped = field_2d.reshape(h2, factor, w2, factor)
    return reshaped.mean(axis=(1, 3))


def downsample_patch_mask_top(mask_2d, factor):
    # Stage A assumes top-row patch only; downsample mask on the top edge.
    h, w = mask_2d.shape
    if (h % factor) != 0 or (w % factor) != 0:
        raise ValueError("Patch mask shape is not divisible by coarsen_factor")
    w2 = w // factor
    top = mask_2d[0].reshape(w2, factor)
    top_coarse = np.any(top, axis=1)
    h2 = h // factor
    mask_coarse = np.zeros((h2, w2), dtype=bool)
    mask_coarse[0, :] = top_coarse
    return mask_coarse


def downsample_mean_3d(fields_3d, factor):
    # fields_3d: [B, H, W]
    b, h, w = fields_3d.shape
    if (h % factor) != 0 or (w % factor) != 0:
        raise ValueError("Grid shape is not divisible by coarsen_factor")
    h2 = h // factor
    w2 = w // factor
    reshaped = fields_3d.reshape(b, h2, factor, w2, factor)
    return reshaped.mean(axis=(2, 4))


def downsample_patch_mask_top_3d(mask_3d, factor):
    # mask_3d: [B, H, W] -> [B, H2, W2] with only top row active.
    b, h, w = mask_3d.shape
    if (h % factor) != 0 or (w % factor) != 0:
        raise ValueError("Patch mask shape is not divisible by coarsen_factor")
    w2 = w // factor
    top = mask_3d[:, 0, :].reshape(b, w2, factor)
    top_coarse = np.any(top, axis=2)
    h2 = h // factor
    out = np.zeros((b, h2, w2), dtype=bool)
    out[:, 0, :] = top_coarse
    return out


def harmonic_mean_torch(a, b):
    # Harmonic averaging gives conservative face diffusivity in heterogeneous media.
    return (2.0 * a * b) / (a + b + 1e-30)


def step_var_d_conservative_torch(c_field, d_field, dt, dx):
    # One explicit conservative diffusion step with spatially varying D.
    dx_plus = harmonic_mean_torch(d_field[:, 1:], d_field[:, :-1])
    dy_plus = harmonic_mean_torch(d_field[1:, :], d_field[:-1, :])

    jx = -dx_plus * (c_field[:, 1:] - c_field[:, :-1]) / dx
    jy = -dy_plus * (c_field[1:, :] - c_field[:-1, :]) / dx

    div = torch.zeros_like(c_field)
    div[:, 1:-1] += (jx[:, :-1] - jx[:, 1:]) / dx
    div[1:-1, :] += (jy[:-1, :] - jy[1:, :]) / dx
    return c_field + dt * div


def step_var_d_conservative_torch_batch(c_field, d_field, dt, dx):
    # Batched conservative variable-D step: [B, H, W].
    dx_plus = harmonic_mean_torch(d_field[:, :, 1:], d_field[:, :, :-1])
    dy_plus = harmonic_mean_torch(d_field[:, 1:, :], d_field[:, :-1, :])

    jx = -dx_plus * (c_field[:, :, 1:] - c_field[:, :, :-1]) / dx
    jy = -dy_plus * (c_field[:, 1:, :] - c_field[:, :-1, :]) / dx

    div = torch.zeros_like(c_field)
    div[:, :, 1:-1] += (jx[:, :, :-1] - jx[:, :, 1:]) / dx
    div[:, 1:-1, :] += (jy[:, :-1, :] - jy[:, 1:, :]) / dx
    return c_field + dt * div


def apply_bc_torch(c_field, patch_top_mask, c_patch):
    # Top boundary: patch Dirichlet + off-patch Neumann.
    c_next = c_field.clone()
    c_next[0, patch_top_mask] = c_patch
    c_next[0, ~patch_top_mask] = c_next[1, ~patch_top_mask]

    # Bottom sink.
    c_next[-1, :] = 0.0

    # Side Neumann.
    c_next[:, 0] = c_next[:, 1]
    c_next[:, -1] = c_next[:, -2]

    # Re-enforce patch after side updates.
    c_next[0, patch_top_mask] = c_patch
    return c_next


def apply_bc_batch_torch(c_field, patch_top_mask, c_patch):
    # Batched BC application in-place where possible.
    # c_field: [B, H, W], patch_top_mask: [B, W], c_patch: [B].
    c_field[:, 0, :] = torch.where(
        patch_top_mask,
        c_patch[:, None],
        c_field[:, 1, :],
    )
    c_field[:, -1, :] = 0.0
    c_field[:, :, 0] = c_field[:, :, 1]
    c_field[:, :, -1] = c_field[:, :, -2]
    c_field[:, 0, :] = torch.where(
        patch_top_mask,
        c_patch[:, None],
        c_field[:, 0, :],
    )
    return c_field


def simulate_coarse_flux(
    d_field,
    k_field,
    patch_mask,
    t_curve,
    boundary,
    dx,
    device,
    coarsen_factor=8,
    dt_stability_safety=0.9,
):
    # Stage A single-run simulator on a coarsened grid.
    d_coarse = downsample_mean_2d(np.asarray(d_field, dtype=np.float32), int(coarsen_factor))
    k_coarse = downsample_mean_2d(np.asarray(k_field, dtype=np.float32), int(coarsen_factor))
    patch_coarse = downsample_patch_mask_top(np.asarray(patch_mask, dtype=bool), int(coarsen_factor))

    dx_coarse = float(dx) * float(coarsen_factor)
    t_arr = np.asarray(t_curve, dtype=np.float32)
    if t_arr.ndim != 1 or t_arr.shape[0] < 2:
        raise ValueError("t curve must be 1D with at least 2 points")
    dt_save = float(t_arr[1] - t_arr[0])
    if dt_save <= 0.0:
        raise ValueError("Saved-time step must be positive")

    dmax = float(np.max(d_coarse))
    # CFL-like explicit stability rule for diffusion; split each saved interval if needed.
    dt_limit = (dx_coarse * dx_coarse) / max(4.0 * dmax, 1e-30)
    stable_dt = max(1e-9, float(dt_stability_safety) * dt_limit)
    n_substeps = max(1, int(math.ceil(dt_save / stable_dt)))
    dt_inner = dt_save / float(n_substeps)

    d_t = torch.as_tensor(d_coarse, dtype=torch.float32, device=device)
    k_t = torch.as_tensor(k_coarse, dtype=torch.float32, device=device)
    patch_top_t = torch.as_tensor(patch_coarse[0], dtype=torch.bool, device=device)

    h2, w2 = d_coarse.shape
    c_t = torch.zeros((h2, w2), dtype=torch.float32, device=device)
    j_pred = torch.zeros((int(t_arr.shape[0]),), dtype=torch.float32, device=device)
    d_bottom = d_t[-2, :]

    mode = str(boundary.get("mode", "infinite_dose"))
    c0 = float(boundary.get("C0", 0.0))
    decay = float(boundary.get("decay_rate", 0.0))

    for i in range(int(t_arr.shape[0])):
        # Evaluate BC and bottom flux at saved times to build J(t).
        t_now = float(t_arr[i])
        c_patch = patch_concentration(t_now, mode, c0, decay)
        c_t = apply_bc_torch(c_t, patch_top_t, c_patch)

        d_cdy = (c_t[-1, :] - c_t[-2, :]) / dx_coarse
        flux_profile = -d_bottom * d_cdy
        j_pred[i] = torch.mean(flux_profile)

        if i < int(t_arr.shape[0]) - 1:
            for s in range(n_substeps):
                # Inner explicit solver steps between saved samples.
                t_sub = t_now + float(s) * dt_inner
                c_pre = patch_concentration(t_sub, mode, c0, decay)
                c_t = apply_bc_torch(c_t, patch_top_t, c_pre)
                c_t = step_var_d_conservative_torch(c_t, d_t, dt_inner, dx_coarse)
                c_t = c_t - (dt_inner * k_t * c_t)
                c_post = patch_concentration(t_sub + dt_inner, mode, c0, decay)
                c_t = apply_bc_torch(c_t, patch_top_t, c_post)

    return np.asarray(j_pred.detach().cpu().numpy(), dtype=np.float32)


def simulate_coarse_flux_batch(
    d_fields,
    k_fields,
    patch_masks,
    t_curve,
    boundary_modes,
    c0_values,
    decay_values,
    dx,
    device,
    coarsen_factor=8,
    dt_stability_safety=0.9,
):
    # Run many coarse simulations in one tensor batch to cut Python overhead.
    d_np = np.asarray(d_fields, dtype=np.float32)
    k_np = np.asarray(k_fields, dtype=np.float32)
    m_np = np.asarray(patch_masks, dtype=bool)
    if d_np.ndim != 3 or k_np.ndim != 3 or m_np.ndim != 3:
        raise ValueError("d_fields/k_fields/patch_masks must be [B,H,W]")

    b = int(d_np.shape[0])
    if b < 1:
        return np.zeros((0, int(len(t_curve))), dtype=np.float32)

    d_coarse = downsample_mean_3d(d_np, int(coarsen_factor))
    k_coarse = downsample_mean_3d(k_np, int(coarsen_factor))
    patch_coarse = downsample_patch_mask_top_3d(m_np, int(coarsen_factor))

    dx_coarse = float(dx) * float(coarsen_factor)
    t_arr = np.asarray(t_curve, dtype=np.float32)
    if t_arr.ndim != 1 or t_arr.shape[0] < 2:
        raise ValueError("t curve must be 1D with at least 2 points")
    dt_save = float(t_arr[1] - t_arr[0])
    if dt_save <= 0.0:
        raise ValueError("Saved-time step must be positive")

    # One stable inner dt is shared across the whole mini-batch.
    dmax = float(np.max(d_coarse))
    dt_limit = (dx_coarse * dx_coarse) / max(4.0 * dmax, 1e-30)
    stable_dt = max(1e-9, float(dt_stability_safety) * dt_limit)
    n_substeps = max(1, int(math.ceil(dt_save / stable_dt)))
    dt_inner = dt_save / float(n_substeps)

    d_t = torch.as_tensor(d_coarse, dtype=torch.float32, device=device)
    k_t = torch.as_tensor(k_coarse, dtype=torch.float32, device=device)
    patch_top_t = torch.as_tensor(patch_coarse[:, 0, :], dtype=torch.bool, device=device)
    c_t = torch.zeros_like(d_t)
    j_pred = torch.zeros((b, int(t_arr.shape[0])), dtype=torch.float32, device=device)
    d_bottom = d_t[:, -2, :]

    # Encode boundary mode once so per-step patch values stay vectorized.
    mode_flag = np.zeros((b,), dtype=np.float32)
    for i in range(len(boundary_modes)):
        mode = boundary_modes[i]
        # Encode boundary mode as 0/1 for vectorized patch concentration updates.
        mode_flag[i] = 1.0 if str(mode) == "time_decay" else 0.0
    mode_t = torch.as_tensor(mode_flag, dtype=torch.float32, device=device)
    c0_t = torch.as_tensor(np.asarray(c0_values, dtype=np.float32), dtype=torch.float32, device=device)
    decay_t = torch.as_tensor(np.asarray(decay_values, dtype=np.float32), dtype=torch.float32, device=device)

    def patch_value_at(time_value):
        # Per-run patch concentration on current time for all runs in the batch.
        t_val = torch.as_tensor(float(time_value), dtype=torch.float32, device=device)
        decay_curve = c0_t * torch.exp(-decay_t * t_val)
        return (mode_t * decay_curve) + ((1.0 - mode_t) * c0_t)

    # March through saved times, then optionally substep between them.
    for ti in range(int(t_arr.shape[0])):
        t_now = float(t_arr[ti])
        c_patch = patch_value_at(t_now)
        c_t = apply_bc_batch_torch(c_t, patch_top_t, c_patch)

        d_cdy = (c_t[:, -1, :] - c_t[:, -2, :]) / dx_coarse
        flux_profile = -d_bottom * d_cdy
        j_pred[:, ti] = torch.mean(flux_profile, dim=1)

        if ti < int(t_arr.shape[0]) - 1:
            for s in range(n_substeps):
                # Substeps improve explicit-stability while preserving saved-time outputs.
                t_sub = t_now + (float(s) * dt_inner)
                c_pre = patch_value_at(t_sub)
                c_t = apply_bc_batch_torch(c_t, patch_top_t, c_pre)
                c_t = step_var_d_conservative_torch_batch(c_t, d_t, dt_inner, dx_coarse)
                c_t = c_t - (dt_inner * k_t * c_t)
                c_post = patch_value_at(t_sub + dt_inner)
                c_t = apply_bc_batch_torch(c_t, patch_top_t, c_post)

    return np.asarray(j_pred.detach().cpu().numpy(), dtype=np.float32)


def build_correction_head(feature_dim, hidden_dim=64):
    # Minimal functional style: plain Sequential network.
    return torch.nn.Sequential(
        torch.nn.Linear(int(feature_dim), int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), 2),
    )


def raw_to_amp_tau(raw):
    # Map unconstrained outputs to positive amplitude/time-warp factors.
    amp = torch.exp(raw[:, 0])
    tau = torch.exp(raw[:, 1])
    return amp, tau


def warp_curve_batch(j_coarse, amp, tau):
    # j_coarse: [B, T]
    # amp/tau: [B]
    batch_size, time_count = j_coarse.shape
    idx = torch.arange(time_count, device=j_coarse.device, dtype=torch.float32).reshape(1, -1)
    q = idx / tau.reshape(-1, 1)
    q = torch.clamp(q, 0.0, float(time_count - 1))

    lo = torch.floor(q).long()
    hi = torch.clamp(lo + 1, max=time_count - 1)
    w = q - lo.to(torch.float32)

    # Linear interpolation in time index space implements differentiable time warping.
    j_lo = torch.gather(j_coarse, dim=1, index=lo)
    j_hi = torch.gather(j_coarse, dim=1, index=hi)
    j_warp = ((1.0 - w) * j_lo) + (w * j_hi)
    return amp.reshape(batch_size, 1) * j_warp


def relative_curve_mse(j_true, j_pred, eps):
    # Unweighted relative MSE used as a baseline loss form.
    scale = torch.clamp(torch.abs(j_true), min=float(eps))
    rel = (j_pred - j_true) / scale
    return torch.mean(rel * rel)


def relative_curve_mse_weighted(j_true, j_pred, eps, time_weights=None):
    # Optional time-weighted relative curve MSE for Stage B training.
    scale = torch.clamp(torch.abs(j_true), min=float(eps))
    rel_sq = ((j_pred - j_true) / scale) ** 2
    if time_weights is None:
        return torch.mean(rel_sq)
    weight_t = time_weights.reshape(1, -1)
    return torch.mean(rel_sq * weight_t) / torch.mean(weight_t)


def to_numpy_float32(arr):
    # Keep all numpy arrays in one dtype for stable serialization/training interop.
    return np.asarray(arr, dtype=np.float32)


def compute_split_curves(entries, device, coarsen_factor, progress_label, sim_batch_runs=32):
    # Build truth + Stage A coarse curves for one split.
    true_curves = []
    coarse_curves = []
    time_curves = []

    total = len(entries)
    batch_size = max(1, int(sim_batch_runs))
    done = 0
    for start in range(0, total, batch_size):
        # Load one run chunk, then simulate them together on the selected device.
        chunk = entries[start : start + batch_size]

        d_batch = []
        k_batch = []
        mask_batch = []
        j_true_batch = []
        boundary_modes = []
        c0_values = []
        decay_values = []
        t_curve_ref = None
        dx_ref = None

        for entry in chunk:
            bundle = load_run_bundle(entry["run_dir"])
            fields = bundle["fields"]
            meta = bundle["meta"]
            grid = meta["grid"]
            boundary = meta["boundary"]

            j_true = to_numpy_float32(fields["J"])
            t_curve = to_numpy_float32(fields["t"])
            if t_curve_ref is None:
                t_curve_ref = t_curve
                dx_ref = float(grid["dx"])
            else:
                # A split batch is expected to share solver save grid and spacing.
                if t_curve.shape != t_curve_ref.shape:
                    raise ValueError("All runs in a split batch must share t shape")
                if abs(float(grid["dx"]) - float(dx_ref)) > 1e-12:
                    raise ValueError("All runs in a split batch must share dx")

            d_batch.append(np.asarray(fields["D"], dtype=np.float32))
            k_batch.append(np.asarray(fields["k"], dtype=np.float32))
            mask_batch.append(np.asarray(fields["patch_mask"], dtype=bool))
            j_true_batch.append(j_true)
            boundary_modes.append(str(boundary.get("mode", "infinite_dose")))
            c0_values.append(float(boundary.get("C0", 0.0)))
            decay_values.append(float(boundary.get("decay_rate", 0.0)))

        j_coarse_batch = simulate_coarse_flux_batch(
            d_fields=np.stack(d_batch, axis=0),
            k_fields=np.stack(k_batch, axis=0),
            patch_masks=np.stack(mask_batch, axis=0),
            t_curve=t_curve_ref,
            boundary_modes=boundary_modes,
            c0_values=np.asarray(c0_values, dtype=np.float32),
            decay_values=np.asarray(decay_values, dtype=np.float32),
            dx=float(dx_ref),
            device=device,
            coarsen_factor=int(coarsen_factor),
        )

        # Unpack batched arrays back into per-run rows expected downstream.
        for i in range(len(chunk)):
            true_curves.append(j_true_batch[i])
            coarse_curves.append(j_coarse_batch[i])
            time_curves.append(t_curve_ref)

        done += len(chunk)
        if done == len(chunk) or done == total or (done % 20) == 0:
            print(progress_label, done, "/", total)

    return (
        np.stack(true_curves, axis=0),
        np.stack(coarse_curves, axis=0),
        np.stack(time_curves, axis=0),
    )


def normalize_features(x_train, x_other_list):
    # Standardize using train statistics only to avoid split leakage.
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    out = []
    out.append((x_train - mean[None, :]) / std[None, :])
    for x_arr in x_other_list:
        out.append((x_arr - mean[None, :]) / std[None, :])
    return out, mean, std


def evaluate_curve_metrics(j_true, j_pred, t_curve):
    # Shared curve metrics helper (mirrors blackbox reporting).
    return compute_curve_metrics(j_true, j_pred, t_curve, eps=1e-12)


def stable_softplus(x):
    # Numerically stable softplus for numpy arrays.
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def apply_flux_nonnegativity(j_curve, mode="none", softplus_beta=20.0):
    # Optional non-negativity transform for flux predictions.
    arr = np.asarray(j_curve, dtype=np.float32)
    text = str(mode).strip().lower()
    if text in ("", "none", "off"):
        return arr.astype(np.float32)
    if text == "clamp":
        return np.maximum(arr, 0.0).astype(np.float32)
    if text == "softplus":
        beta = max(float(softplus_beta), 1e-6)
        scaled = beta * np.asarray(arr, dtype=np.float64)
        mapped = stable_softplus(scaled) / beta
        return np.asarray(mapped, dtype=np.float32)
    raise ValueError("Unsupported --stagec_nonneg mode: " + str(mode))


def compute_flux_bc_rel_rmse(j_true, j_pred):
    # Relative RMSE against true bottom flux curve (BC consistency proxy).
    true_arr = np.asarray(j_true, dtype=np.float64)
    pred_arr = np.asarray(j_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    scale = float(np.sqrt(np.mean(true_arr * true_arr)) + 1e-12)
    return float(rmse / scale)


def compute_negative_flux_fraction(j_pred):
    # Fraction of predicted flux samples that violate non-negativity.
    pred_arr = np.asarray(j_pred, dtype=np.float64)
    return float(np.mean(pred_arr < -1e-14))


def stage_c_selection_score(j_true, j_pred, curve_metrics, score_w_neg=0.2, score_w_bc=0.05):
    # Composite score keeps accuracy primary but penalizes non-physical outputs.
    rel_l2 = float(curve_metrics["relative_l2"])
    neg_frac = compute_negative_flux_fraction(j_pred)
    bc_rel = compute_flux_bc_rel_rmse(j_true, j_pred)
    score = rel_l2 + (float(score_w_neg) * neg_frac) + (float(score_w_bc) * bc_rel)
    return {
        "score": float(score),
        "val_rel_l2": float(rel_l2),
        "val_neg_flux_fraction": float(neg_frac),
        "val_bc_bottom_flux_rel_rmse": float(bc_rel),
    }


def generate_stage_curve_plots(plot_dir, split_label, t_curve, j_true, stage_map, max_examples=9):
    # Save per-stage plots inside one split folder.
    for stage_name, j_pred in stage_map.items():
        title_suffix = str(split_label).upper() + ", " + str(stage_name)
        plot_curve_examples(
            t_curve,
            j_true,
            j_pred,
            plot_dir / ("curve_examples_" + str(stage_name) + ".png"),
            "J(t) examples (" + title_suffix + ")",
            max_examples=int(max_examples),
        )
        plot_curve_error_over_time(
            t_curve,
            j_true,
            j_pred,
            plot_dir / ("curve_error_over_time_" + str(stage_name) + ".png"),
            "Curve error over time (" + title_suffix + ")",
        )


def build_comparison_report(
    stage_a_val,
    stage_a_test,
    stage_b_val,
    stage_b_test,
    stage_c_val,
    stage_c_test,
    physics_val_summary,
    physics_test_summary,
    blackbox_dir=None,
):
    # Build a single report with hybrid metrics and optional blackbox side-by-side deltas.
    def read_json_if_exists(path):
        # Missing/invalid blackbox files should not fail hybrid training.
        p = Path(path)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def compare_curve_metric(stage_c_curve, bb_curve):
        # Compare final split-level curve metrics between Stage C and blackbox.
        if not isinstance(stage_c_curve, dict) or not isinstance(bb_curve, dict):
            return None
        if "relative_l2" not in stage_c_curve or "relative_l2" not in bb_curve:
            return None
        bb_rel_l2 = float(bb_curve["relative_l2"])
        stagec_rel_l2 = float(stage_c_curve["relative_l2"])
        return {
            "relative_l2_ratio": float(stagec_rel_l2 / max(bb_rel_l2, 1e-12)),
            "relative_l2_gap": float(stagec_rel_l2 - bb_rel_l2),
            "stageC_relative_l2": float(stagec_rel_l2),
            "blackbox_relative_l2": float(bb_rel_l2),
        }

    def compare_stage_metric(stage_c_stage, bb_stage, metric_name):
        # Compare aggregated physics stats (mean/p90) for one metric.
        if not isinstance(stage_c_stage, dict) or not isinstance(bb_stage, dict):
            return None
        if metric_name not in stage_c_stage or metric_name not in bb_stage:
            return None

        stage_c_metric = stage_c_stage[metric_name]
        bb_metric = bb_stage[metric_name]
        if not isinstance(stage_c_metric, dict) or not isinstance(bb_metric, dict):
            return None

        out = {}
        for stat_name in ("mean", "p90"):
            if stat_name in stage_c_metric and stat_name in bb_metric:
                stage_c_value = float(stage_c_metric[stat_name])
                bb_value = float(bb_metric[stat_name])
                out[stat_name + "_ratio"] = float(stage_c_value / max(bb_value, 1e-12))
                out[stat_name + "_gap"] = float(stage_c_value - bb_value)
                out["stageC_" + stat_name] = stage_c_value
                out["blackbox_" + stat_name] = bb_value
        if len(out) == 0:
            return None
        return out

    stage_c_physics_val = physics_val_summary.get("stages", {}).get("stageC", {})
    stage_c_physics_test = physics_test_summary.get("stages", {}).get("stageC", {})

    report = {
        "hybrid": {
            "stageA": {
                "id_val_curve": stage_a_val,
                "id_test_curve": stage_a_test,
            },
            "stageB": {
                "id_val_curve": stage_b_val,
                "id_test_curve": stage_b_test,
            },
            "stageC": {
                "id_val_curve": stage_c_val,
                "id_test_curve": stage_c_test,
                "id_val_physics": stage_c_physics_val,
                "id_test_physics": stage_c_physics_test,
            },
        },
        "blackbox_available": False,
        "blackbox": None,
        "stageC_vs_blackbox": None,
    }

    if blackbox_dir is None:
        return report

    blackbox_root = Path(blackbox_dir)
    # Use files already produced by train_blackbox; support both legacy and subfolder layouts.
    metrics_id_test = read_json_if_exists(blackbox_root / "metrics_id.json")
    metrics_id_val = read_json_if_exists(blackbox_root / "metrics_id_val.json")
    physics_id_test = read_json_if_exists(
        blackbox_root / "diagnostics" / "physics" / "physics_diag_id_test_summary.json"
    )
    if physics_id_test is None:
        physics_id_test = read_json_if_exists(blackbox_root / "physics_diag_id_test_summary.json")
    physics_id_val = read_json_if_exists(
        blackbox_root / "diagnostics" / "physics" / "physics_diag_id_val_summary.json"
    )
    if physics_id_val is None:
        physics_id_val = read_json_if_exists(blackbox_root / "physics_diag_id_val_summary.json")
    if metrics_id_test is None:
        return report

    bb_curve_test = metrics_id_test.get("curve", {})
    bb_curve_val = metrics_id_val.get("curve", {}) if isinstance(metrics_id_val, dict) else {}
    bb_physics_test = {}
    bb_physics_val = {}
    if isinstance(physics_id_test, dict):
        bb_physics_test = physics_id_test.get("stages", {}).get("blackbox", {})
    if isinstance(physics_id_val, dict):
        bb_physics_val = physics_id_val.get("stages", {}).get("blackbox", {})

    # Only compare metrics that are common across both model outputs.
    comparisons = {
        "id_val_curve": compare_curve_metric(stage_c_val, bb_curve_val),
        "id_test_curve": compare_curve_metric(stage_c_test, bb_curve_test),
        "id_val_physics": {},
        "id_test_physics": {},
    }
    shared_physics_metrics = [
        "curve_relative_l2",
        "bc_bottom_flux_rel_rmse",
        "negative_flux_fraction",
        "pde_mass_balance_rel_rmse",
    ]
    for metric_name in shared_physics_metrics:
        val_comp = compare_stage_metric(stage_c_physics_val, bb_physics_val, metric_name)
        if val_comp is not None:
            comparisons["id_val_physics"][metric_name] = val_comp

        test_comp = compare_stage_metric(stage_c_physics_test, bb_physics_test, metric_name)
        if test_comp is not None:
            comparisons["id_test_physics"][metric_name] = test_comp

    report["blackbox_available"] = True
    report["blackbox"] = {
        "dir": str(blackbox_root),
        "id_val_curve": bb_curve_val,
        "id_test_curve": bb_curve_test,
        "id_val_physics": bb_physics_val,
        "id_test_physics": bb_physics_test,
    }
    report["stageC_vs_blackbox"] = comparisons
    return report


def build_stageb_time_weights(j_train_true, mode="none", early_boost=0.0, peak_boost=0.0):
    # Build deterministic per-time weights from train curves.
    arr = np.asarray(j_train_true, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("j_train_true must be [N,T] for time weighting")
    t_count = int(arr.shape[1])
    mode_text = str(mode).strip().lower()
    if mode_text in ("", "none", "off"):
        return np.ones((t_count,), dtype=np.float32)

    if mode_text != "early_peak":
        raise ValueError("Unsupported --stageb_time_weighting mode: " + str(mode))

    # early_profile emphasizes early-time fit; peak_profile emphasizes high-flux windows.
    u = np.linspace(0.0, 1.0, t_count, dtype=np.float32)
    early_profile = 1.0 - u
    mean_curve = np.mean(np.abs(arr), axis=0)
    peak_profile = mean_curve / (float(np.max(mean_curve)) + 1e-12)

    w = (
        1.0
        + (float(early_boost) * early_profile)
        + (float(peak_boost) * peak_profile)
    )
    w = np.maximum(w, 1e-6)
    return np.asarray(w, dtype=np.float32)


def build_stagec_feature_matrix(x_arr, feature_map="linear"):
    # Optional nonlinear feature expansion for Stage C ridge model.
    x = np.asarray(x_arr, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Stage C features must be 2D")

    mode = str(feature_map).strip().lower()
    if mode in ("", "linear"):
        return x.astype(np.float32)
    if mode == "poly2":
        # Add quadratic interaction terms x_i * x_j (including squares).
        b, d = x.shape
        terms = [x]
        quad_terms = []
        for i in range(d):
            for j in range(i, d):
                quad_terms.append((x[:, i] * x[:, j]).reshape(b, 1))
        if len(quad_terms) > 0:
            terms.append(np.concatenate(quad_terms, axis=1).astype(np.float32))
        return np.concatenate(terms, axis=1).astype(np.float32)
    raise ValueError("Unsupported --stagec_feature_map: " + str(feature_map))


def train_correction_head(
    x_train,
    j_train_true,
    j_train_coarse,
    x_val,
    j_val_true,
    j_val_coarse,
    t_val,
    device,
    epochs=120,
    batch_size=64,
    lr=1e-3,
    reg_weight=1e-3,
    time_weighting_mode="none",
    time_weight_early_boost=0.0,
    time_weight_peak_boost=0.0,
):
    # Stage B optimizer loop for the small warp head.
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    j_train_true_t = torch.as_tensor(j_train_true, dtype=torch.float32, device=device)
    j_train_coarse_t = torch.as_tensor(j_train_coarse, dtype=torch.float32, device=device)

    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    j_val_true_t = torch.as_tensor(j_val_true, dtype=torch.float32, device=device)
    j_val_coarse_t = torch.as_tensor(j_val_coarse, dtype=torch.float32, device=device)

    # Relative-loss epsilon from train targets.
    flux_eps = float(np.percentile(np.abs(j_train_true), 90.0))
    if flux_eps <= 0.0:
        flux_eps = 1e-12
    time_weights = build_stageb_time_weights(
        j_train_true,
        mode=time_weighting_mode,
        early_boost=time_weight_early_boost,
        peak_boost=time_weight_peak_boost,
    )
    time_weights_t = torch.as_tensor(time_weights, dtype=torch.float32, device=device)

    model = build_correction_head(feature_dim=x_train.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    rng = np.random.default_rng(321)
    best_state = None
    best_val_rel_l2 = float("inf")
    history = []

    n_train = int(x_train.shape[0])
    for epoch in range(1, int(epochs) + 1):
        model.train()
        # Shuffle rows each epoch so mini-batches see varied run combinations.
        perm = np.asarray(rng.permutation(n_train), dtype=int)
        step_losses = []

        for start in range(0, n_train, int(batch_size)):
            rows = perm[start : start + int(batch_size)]
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)

            xb = x_train_t[rows_t]
            jb_true = j_train_true_t[rows_t]
            jb_coarse = j_train_coarse_t[rows_t]

            amp, tau = raw_to_amp_tau(model(xb))
            jb_pred = warp_curve_batch(jb_coarse, amp, tau)

            loss_data = relative_curve_mse_weighted(
                jb_true,
                jb_pred,
                eps=flux_eps,
                time_weights=time_weights_t,
            )
            reg = torch.mean((amp - 1.0) ** 2) + torch.mean((tau - 1.0) ** 2)
            loss = loss_data + (float(reg_weight) * reg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            amp_val, tau_val = raw_to_amp_tau(model(x_val_t))
            j_val_pred_t = warp_curve_batch(j_val_coarse_t, amp_val, tau_val)
            j_val_pred = np.asarray(j_val_pred_t.detach().cpu().numpy(), dtype=np.float32)
            val_metrics = evaluate_curve_metrics(j_val_true, j_val_pred, t_val)

        val_rel_l2 = float(val_metrics["relative_l2"])
        # Keep best validation checkpoint for final Stage B predictions.
        if val_rel_l2 < best_val_rel_l2:
            best_val_rel_l2 = val_rel_l2
            model_state = {}
            for param_name, param_value in model.state_dict().items():
                model_state[param_name] = param_value.detach().cpu().clone()
            best_state = {
                "model": model_state,
                "epoch": int(epoch),
                "val_rel_l2": float(val_rel_l2),
            }

        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(step_losses)),
            "val_rel_l2": float(val_rel_l2),
            "val_pearson_r": float(val_metrics["pearson_r"]),
        }
        history.append(row)

        if epoch == 1 or epoch == epochs or (epoch % 10) == 0:
            print(
                "epoch",
                epoch,
                "train_loss=",
                f"{row['train_loss']:.6e}",
                "val_rel_l2=",
                f"{row['val_rel_l2']:.6e}",
            )

    if best_state is None:
        raise ValueError("Training produced no best checkpoint state")

    model.load_state_dict(best_state["model"])
    result = {
        "model": model,
        "best_epoch": int(best_state["epoch"]),
        "best_val_rel_l2": float(best_state["val_rel_l2"]),
        "history": history,
        "flux_eps": float(flux_eps),
        "time_weighting": {
            "mode": str(time_weighting_mode),
            "early_boost": float(time_weight_early_boost),
            "peak_boost": float(time_weight_peak_boost),
            "weights_mean": float(np.mean(time_weights)),
            "weights_min": float(np.min(time_weights)),
            "weights_max": float(np.max(time_weights)),
        },
    }
    return result


def predict_with_head(model, x_feat, j_coarse, device):
    # Inference helper for Stage B correction head.
    model.eval()
    x_t = torch.as_tensor(x_feat, dtype=torch.float32, device=device)
    j_t = torch.as_tensor(j_coarse, dtype=torch.float32, device=device)
    with torch.no_grad():
        amp, tau = raw_to_amp_tau(model(x_t))
        j_pred_t = warp_curve_batch(j_t, amp, tau)
    return np.asarray(j_pred_t.detach().cpu().numpy(), dtype=np.float32)


def add_bias_column(x_arr):
    # Bias term for closed-form ridge without relying on estimator intercept options.
    ones = np.ones((int(x_arr.shape[0]), 1), dtype=np.float32)
    return np.concatenate([x_arr.astype(np.float32), ones], axis=1)


def fit_ridge_closed_form(x_aug, y_target, alpha):
    # x_aug: [N, D], y_target: [N, K]
    d = int(x_aug.shape[1])
    eye = np.eye(d, dtype=np.float32)
    lhs = (x_aug.T @ x_aug) + (float(alpha) * eye)
    rhs = x_aug.T @ y_target
    beta = np.linalg.solve(lhs, rhs)
    return beta.astype(np.float32)


def fit_stage_c_residual_model(
    x_train,
    j_train_true,
    j_train_base,
    x_val,
    j_val_true,
    j_val_base,
    t_val,
    component_options=None,
    alpha_grid=None,
    nonneg_mode="clamp",
    softplus_beta=20.0,
    score_w_neg=0.2,
    score_w_bc=0.05,
    feature_map="linear",
):
    # Stage C: low-rank residual model in curve space with grid sweep on val split.
    residual_train = j_train_true - j_train_base
    mean_residual = np.mean(residual_train, axis=0, keepdims=True).astype(np.float32)
    residual_centered = residual_train - mean_residual

    # SVD basis defines low-rank residual coordinates to regress.
    # SVD gives an orthogonal residual basis; we regress only first-k coordinates.
    u_matrix, singular_values, vt = np.linalg.svd(residual_centered, full_matrices=False)
    max_rank = int(vt.shape[0])
    if component_options is None:
        component_options = [8, 16, 24, 32]
    if alpha_grid is None:
        alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    x_train_map = build_stagec_feature_matrix(x_train, feature_map=feature_map)
    x_val_map = build_stagec_feature_matrix(x_val, feature_map=feature_map)
    x_train_aug = add_bias_column(x_train_map)
    x_val_aug = add_bias_column(x_val_map)

    best = None
    sweep_rows = []
    # Grid-search rank and ridge alpha on validation score.
    for k in component_options:
        if int(k) > max_rank:
            continue
        components = vt[: int(k), :].astype(np.float32)
        z_train = residual_centered @ components.T

        for alpha in alpha_grid:
            # Closed-form ridge fit in latent residual space.
            beta = fit_ridge_closed_form(x_train_aug, z_train, alpha=alpha)
            z_val_pred = x_val_aug @ beta
            residual_val_pred = (z_val_pred @ components) + mean_residual
            j_val_pred = j_val_base + residual_val_pred
            j_val_pred = apply_flux_nonnegativity(
                j_val_pred,
                mode=nonneg_mode,
                softplus_beta=softplus_beta,
            )
            metrics = evaluate_curve_metrics(j_val_true, j_val_pred, t_val)
            score_parts = stage_c_selection_score(
                j_true=j_val_true,
                j_pred=j_val_pred,
                curve_metrics=metrics,
                score_w_neg=score_w_neg,
                score_w_bc=score_w_bc,
            )
            score = float(score_parts["score"])

            # Select best model by composite score (accuracy + physics penalties).
            if (best is None) or (score < float(best["val_score"])):
                best = {
                    "k": int(k),
                    "alpha": float(alpha),
                    "components": components,
                    "beta": beta,
                    "mean_residual": mean_residual,
                    "val_score": score,
                    "val_rel_l2": float(score_parts["val_rel_l2"]),
                    "val_neg_flux_fraction": float(score_parts["val_neg_flux_fraction"]),
                    "val_bc_bottom_flux_rel_rmse": float(score_parts["val_bc_bottom_flux_rel_rmse"]),
                    "val_metrics": metrics,
                    "nonneg_mode": str(nonneg_mode),
                    "softplus_beta": float(softplus_beta),
                    "score_w_neg": float(score_w_neg),
                    "score_w_bc": float(score_w_bc),
                    "feature_map": str(feature_map),
                }
            sweep_rows.append(
                {
                    "components": int(k),
                    "alpha": float(alpha),
                    "val_score": float(score),
                    "val_rel_l2": float(score_parts["val_rel_l2"]),
                    "val_neg_flux_fraction": float(score_parts["val_neg_flux_fraction"]),
                    "val_bc_bottom_flux_rel_rmse": float(score_parts["val_bc_bottom_flux_rel_rmse"]),
                    "val_pearson_r": float(metrics["pearson_r"]),
                }
            )

    if best is None:
        raise ValueError("Stage C could not fit a residual model")
    # Keep a small leaderboard for traceability in reports.
    sweep_rows = sorted(sweep_rows, key=lambda row: float(row["val_score"]))
    best["sweep_top10"] = sweep_rows[:10]
    return best


def predict_stage_c(stage_c_model, x_feat, j_base):
    # Decode latent residual correction and add it to Stage B baseline curve.
    x_map = build_stagec_feature_matrix(
        x_feat,
        feature_map=stage_c_model.get("feature_map", "linear"),
    )
    x_aug = add_bias_column(x_map)
    z_pred = x_aug @ stage_c_model["beta"]
    residual_pred = (z_pred @ stage_c_model["components"]) + stage_c_model["mean_residual"]
    j_pred = (j_base + residual_pred).astype(np.float32)
    return apply_flux_nonnegativity(
        j_pred,
        mode=stage_c_model.get("nonneg_mode", "none"),
        softplus_beta=stage_c_model.get("softplus_beta", 20.0),
    )


def main():
    # CLI controls all runtime knobs for the hybrid pipeline.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", default="data/processed/ml")
    parser.add_argument("--out_dir", default="outputs/ml/pinn_discrete_stageA")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run_root_override", default=None)
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)

    parser.add_argument("--max_train_rows", type=int, default=192)
    parser.add_argument("--max_val_rows", type=int, default=64)
    parser.add_argument("--max_test_rows", type=int, default=64)
    parser.add_argument("--max_ood_rows", type=int, default=None)

    parser.add_argument("--coarsen_factor", type=int, default=8)
    parser.add_argument("--sim_batch_runs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--stagec_components", default="8,16,24,32")
    parser.add_argument("--stagec_alphas", default="1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--stagec_nonneg", default="clamp")
    parser.add_argument("--stagec_softplus_beta", type=float, default=20.0)
    parser.add_argument("--stagec_score_w_neg", type=float, default=0.2)
    parser.add_argument("--stagec_score_w_bc", type=float, default=0.05)
    parser.add_argument("--stagec_feature_map", default="linear")
    parser.add_argument("--stageb_time_weighting", default="none")
    parser.add_argument("--stageb_time_weight_early_boost", type=float, default=0.0)
    parser.add_argument("--stageb_time_weight_peak_boost", type=float, default=0.0)
    parser.add_argument("--worst_case_top_n", type=int, default=10)
    parser.add_argument("--plot_max_examples", type=int, default=9)
    parser.add_argument("--blackbox_dir", default="")
    args = parser.parse_args()

    # Setup.
    t0 = time.time()
    ml_dir = Path(args.ml_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = choose_device(args.device)
    print("device:", str(device))
    if str(device) == "cuda":
        print("cuda_name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Clamp to practical minimums instead of hard-failing.
    coarsen_factor = max(1, int(args.coarsen_factor))
    sim_batch_runs = max(1, int(args.sim_batch_runs))

    # Parse optional sweep grids; fallback defaults keep CLI forgiving.
    stagec_components = []
    stagec_component_tokens = str(args.stagec_components).split(",")
    for part in stagec_component_tokens:
        token = part.strip()
        if token != "":
            stagec_components.append(int(token))
    if len(stagec_components) == 0:
        stagec_components = [8, 16, 24, 32]

    stagec_alphas = []
    stagec_alpha_tokens = str(args.stagec_alphas).split(",")
    for part in stagec_alpha_tokens:
        token = part.strip()
        if token != "":
            stagec_alphas.append(float(token))
    if len(stagec_alphas) == 0:
        stagec_alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    stagec_nonneg = str(args.stagec_nonneg).strip().lower()
    if stagec_nonneg not in ("none", "off", "clamp", "softplus"):
        raise ValueError("--stagec_nonneg must be one of: none, clamp, softplus")
    stagec_softplus_beta = max(1e-6, float(args.stagec_softplus_beta))
    stagec_score_w_neg = max(0.0, float(args.stagec_score_w_neg))
    stagec_score_w_bc = max(0.0, float(args.stagec_score_w_bc))
    stagec_feature_map = str(args.stagec_feature_map).strip().lower()
    if stagec_feature_map not in ("linear", "poly2"):
        raise ValueError("--stagec_feature_map must be one of: linear, poly2")

    stageb_time_weighting = str(args.stageb_time_weighting).strip().lower()
    if stageb_time_weighting not in ("none", "off", "early_peak"):
        raise ValueError("--stageb_time_weighting must be one of: none, early_peak")
    stageb_time_weight_early_boost = max(0.0, float(args.stageb_time_weight_early_boost))
    stageb_time_weight_peak_boost = max(0.0, float(args.stageb_time_weight_peak_boost))
    worst_case_top_n = max(1, int(args.worst_case_top_n))
    plot_max_examples = max(1, int(args.plot_max_examples))
    blackbox_dir = str(args.blackbox_dir).strip()
    if blackbox_dir == "":
        blackbox_dir = None
    # Names from ml/meta drive feature extraction and scalar diagnostics output.
    feature_names, target_names = load_ml_meta_names(ml_dir)

    # Resolve split entries with optional run-index slicing and staged-path remap.
    train_entries = build_entries(
        ml_dir=ml_dir,
        split_name="id_train",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
        max_rows=args.max_train_rows,
        run_root_override=args.run_root_override,
    )
    val_entries = build_entries(
        ml_dir=ml_dir,
        split_name="id_val",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
        max_rows=args.max_val_rows,
        run_root_override=args.run_root_override,
    )
    test_entries = build_entries(
        ml_dir=ml_dir,
        split_name="id_test",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
        max_rows=args.max_test_rows,
        run_root_override=args.run_root_override,
    )
    split_meta = json.loads((ml_dir / "meta.json").read_text(encoding="utf-8")).get("splits", {})
    # OOD is optional: only active when split exists and rows remain after filtering.
    has_ood_split = isinstance(split_meta, dict) and ("ood_primary" in split_meta)
    if has_ood_split:
        ood_entries = build_entries(
            ml_dir=ml_dir,
            split_name="ood_primary",
            run_start_index=args.run_start_index,
            run_end_index=args.run_end_index,
            max_rows=args.max_ood_rows,
            run_root_override=args.run_root_override,
            allow_empty=True,
        )
    else:
        ood_entries = []
    has_ood = len(ood_entries) > 0

    # Load tabular inputs and true scalar targets used for scalar diagnostics.
    x_train_raw = load_split_feature_matrix(ml_dir, "id_train", train_entries).astype(np.float32)
    x_val_raw = load_split_feature_matrix(ml_dir, "id_val", val_entries).astype(np.float32)
    x_test_raw = load_split_feature_matrix(ml_dir, "id_test", test_entries).astype(np.float32)
    y_val_true = load_split_scalar_matrix(ml_dir, "id_val", val_entries).astype(np.float32)
    y_test_true = load_split_scalar_matrix(ml_dir, "id_test", test_entries).astype(np.float32)
    c0_val = extract_c0_feature(x_val_raw, feature_names)
    c0_test = extract_c0_feature(x_test_raw, feature_names)
    if has_ood:
        x_ood_raw = load_split_feature_matrix(ml_dir, "ood_primary", ood_entries).astype(np.float32)
        y_ood_true = load_split_scalar_matrix(ml_dir, "ood_primary", ood_entries).astype(np.float32)
        c0_ood = extract_c0_feature(x_ood_raw, feature_names)
    else:
        x_ood_raw = None
        y_ood_true = None
        c0_ood = None

    # Shared feature normalization used by Stage B and Stage C.
    x_other_raw = [x_val_raw, x_test_raw]
    if has_ood:
        x_other_raw.append(x_ood_raw)
    x_norm_all, x_mean, x_std = normalize_features(
        x_train_raw,
        x_other_raw,
    )
    x_train = x_norm_all[0]
    x_val = x_norm_all[1]
    x_test = x_norm_all[2]
    if has_ood:
        x_ood = x_norm_all[3]
    else:
        x_ood = None

    # Stage A: build coarse-physics curves.
    print("computing coarse curves for id_train ...")
    j_train_true, j_train_coarse, t_train = compute_split_curves(
        train_entries,
        device=device,
        coarsen_factor=coarsen_factor,
        progress_label="id_train",
        sim_batch_runs=sim_batch_runs,
    )
    print("computing coarse curves for id_val ...")
    j_val_true, j_val_coarse, t_val = compute_split_curves(
        val_entries,
        device=device,
        coarsen_factor=coarsen_factor,
        progress_label="id_val",
        sim_batch_runs=sim_batch_runs,
    )
    print("computing coarse curves for id_test ...")
    j_test_true, j_test_coarse, t_test = compute_split_curves(
        test_entries,
        device=device,
        coarsen_factor=coarsen_factor,
        progress_label="id_test",
        sim_batch_runs=sim_batch_runs,
    )
    if has_ood:
        print("computing coarse curves for ood_primary ...")
        j_ood_true, j_ood_coarse, t_ood = compute_split_curves(
            ood_entries,
            device=device,
            coarsen_factor=coarsen_factor,
            progress_label="ood_primary",
            sim_batch_runs=sim_batch_runs,
        )
    else:
        j_ood_true = None
        j_ood_coarse = None
        t_ood = None

    # Stage A metrics.
    stage_a_val = evaluate_curve_metrics(j_val_true, j_val_coarse, t_val)
    stage_a_test = evaluate_curve_metrics(j_test_true, j_test_coarse, t_test)
    if has_ood:
        stage_a_ood = evaluate_curve_metrics(j_ood_true, j_ood_coarse, t_ood)
    else:
        stage_a_ood = None

    # Stage B: learn amplitude/time-warp head.
    train_result = train_correction_head(
        x_train=x_train,
        j_train_true=j_train_true,
        j_train_coarse=j_train_coarse,
        x_val=x_val,
        j_val_true=j_val_true,
        j_val_coarse=j_val_coarse,
        t_val=t_val,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_weight=args.reg_weight,
        time_weighting_mode=stageb_time_weighting,
        time_weight_early_boost=stageb_time_weight_early_boost,
        time_weight_peak_boost=stageb_time_weight_peak_boost,
    )
    model = train_result["model"]

    j_val_pred = predict_with_head(model, x_val, j_val_coarse, device)
    j_test_pred = predict_with_head(model, x_test, j_test_coarse, device)
    j_train_pred = predict_with_head(model, x_train, j_train_coarse, device)
    if has_ood:
        j_ood_pred = predict_with_head(model, x_ood, j_ood_coarse, device)
    else:
        j_ood_pred = None

    # Stage B metrics on each available split.
    stage_b_val = evaluate_curve_metrics(j_val_true, j_val_pred, t_val)
    stage_b_test = evaluate_curve_metrics(j_test_true, j_test_pred, t_test)
    if has_ood:
        stage_b_ood = evaluate_curve_metrics(j_ood_true, j_ood_pred, t_ood)
    else:
        stage_b_ood = None

    # Stage C: fit low-rank residual corrector on top of Stage B.
    stage_c_model = fit_stage_c_residual_model(
        x_train=x_train,
        j_train_true=j_train_true,
        j_train_base=j_train_pred,
        x_val=x_val,
        j_val_true=j_val_true,
        j_val_base=j_val_pred,
        t_val=t_val,
        component_options=stagec_components,
        alpha_grid=stagec_alphas,
        nonneg_mode=stagec_nonneg,
        softplus_beta=stagec_softplus_beta,
        score_w_neg=stagec_score_w_neg,
        score_w_bc=stagec_score_w_bc,
        feature_map=stagec_feature_map,
    )
    j_val_stage_c = predict_stage_c(stage_c_model, x_val, j_val_pred)
    j_test_stage_c = predict_stage_c(stage_c_model, x_test, j_test_pred)
    if has_ood:
        j_ood_stage_c = predict_stage_c(stage_c_model, x_ood, j_ood_pred)
    else:
        j_ood_stage_c = None
    # Stage C metrics after residual correction + non-negativity mapping.
    stage_c_val = evaluate_curve_metrics(j_val_true, j_val_stage_c, t_val)
    stage_c_test = evaluate_curve_metrics(j_test_true, j_test_stage_c, t_test)
    if has_ood:
        stage_c_ood = evaluate_curve_metrics(j_ood_true, j_ood_stage_c, t_ood)
    else:
        stage_c_ood = None

    # Derive scalar targets from each stage's predicted J(t) for blackbox-style diagnostics.
    y_val_stage_a = scalar_targets_from_flux_curves(j_val_coarse, t_val, c0_val, target_names)
    y_val_stage_b = scalar_targets_from_flux_curves(j_val_pred, t_val, c0_val, target_names)
    y_val_stage_c = scalar_targets_from_flux_curves(j_val_stage_c, t_val, c0_val, target_names)
    y_test_stage_a = scalar_targets_from_flux_curves(j_test_coarse, t_test, c0_test, target_names)
    y_test_stage_b = scalar_targets_from_flux_curves(j_test_pred, t_test, c0_test, target_names)
    y_test_stage_c = scalar_targets_from_flux_curves(j_test_stage_c, t_test, c0_test, target_names)
    if has_ood:
        y_ood_stage_a = scalar_targets_from_flux_curves(j_ood_coarse, t_ood, c0_ood, target_names)
        y_ood_stage_b = scalar_targets_from_flux_curves(j_ood_pred, t_ood, c0_ood, target_names)
        y_ood_stage_c = scalar_targets_from_flux_curves(j_ood_stage_c, t_ood, c0_ood, target_names)
    else:
        y_ood_stage_a = None
        y_ood_stage_b = None
        y_ood_stage_c = None

    # Keep split metric files aligned with blackbox output naming.
    # Each stage has both curve and scalar sections for side-by-side diagnostics.
    metrics_id_val = {
        "available": True,
        "stageA": {
            "curve": stage_a_val,
            "scalar": scalar_report(y_val_true, y_val_stage_a, target_names),
        },
        "stageB": {
            "curve": stage_b_val,
            "scalar": scalar_report(y_val_true, y_val_stage_b, target_names),
        },
        "stageC": {
            "curve": stage_c_val,
            "scalar": scalar_report(y_val_true, y_val_stage_c, target_names),
        },
    }
    metrics_id = {
        "available": True,
        "stageA": {
            "curve": stage_a_test,
            "scalar": scalar_report(y_test_true, y_test_stage_a, target_names),
        },
        "stageB": {
            "curve": stage_b_test,
            "scalar": scalar_report(y_test_true, y_test_stage_b, target_names),
        },
        "stageC": {
            "curve": stage_c_test,
            "scalar": scalar_report(y_test_true, y_test_stage_c, target_names),
        },
    }
    metrics_ood = {
        "available": bool(has_ood),
        "stageA": {
            "curve": stage_a_ood,
            "scalar": scalar_report(y_ood_true, y_ood_stage_a, target_names),
        }
        if has_ood
        else {},
        "stageB": {
            "curve": stage_b_ood,
            "scalar": scalar_report(y_ood_true, y_ood_stage_b, target_names),
        }
        if has_ood
        else {},
        "stageC": {
            "curve": stage_c_ood,
            "scalar": scalar_report(y_ood_true, y_ood_stage_c, target_names),
        }
        if has_ood
        else {},
    }

    # Persist model and prediction bundles for later inspection.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "coarsen_factor": int(coarsen_factor),
        },
        out_dir / "correction_head.pt",
    )

    np.savez(
        out_dir / "pred_id_val.npz",
        j_true=j_val_true,
        j_stageA=j_val_coarse,
        j_stageB=j_val_pred,
        j_stageC=j_val_stage_c,
        t=t_val,
    )
    np.savez(
        out_dir / "pred_id_test.npz",
        j_true=j_test_true,
        j_stageA=j_test_coarse,
        j_stageB=j_test_pred,
        j_stageC=j_test_stage_c,
        t=t_test,
    )
    if has_ood:
        np.savez(
            out_dir / "pred_ood_primary.npz",
            j_true=j_ood_true,
            j_stageA=j_ood_coarse,
            j_stageB=j_ood_pred,
            j_stageC=j_ood_stage_c,
            t=t_ood,
        )

    # Metric bundles mirror blackbox filenames for easier tooling reuse.
    (out_dir / "metrics_id_val.json").write_text(
        json.dumps(metrics_id_val, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics_id.json").write_text(
        json.dumps(metrics_id, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics_ood.json").write_text(
        json.dumps(metrics_ood, indent=2),
        encoding="utf-8",
    )

    # Diagnostics outputs are grouped by type under diagnostics/.
    out_diag = out_dir / "diagnostics"
    out_diag_physics = out_diag / "physics"
    ensure_dir(out_diag)
    ensure_dir(out_diag_physics)

    # Physics diagnostics quantify BC/PDE proxies per stage, per run.
    print("computing physics diagnostics for id_val ...")
    physics_val_rows, physics_val_summary = compute_split_physics_diagnostics(
        entries=val_entries,
        split_name="id_val",
        prediction_map={
            "truth": j_val_true,
            "stageA": j_val_coarse,
            "stageB": j_val_pred,
            "stageC": j_val_stage_c,
        },
        progress_label="physics_id_val",
        truth_key="truth",
        pde_anchor_key="stageA",
    )
    print("computing physics diagnostics for id_test ...")
    physics_test_rows, physics_test_summary = compute_split_physics_diagnostics(
        entries=test_entries,
        split_name="id_test",
        prediction_map={
            "truth": j_test_true,
            "stageA": j_test_coarse,
            "stageB": j_test_pred,
            "stageC": j_test_stage_c,
        },
        progress_label="physics_id_test",
        truth_key="truth",
        pde_anchor_key="stageA",
    )
    write_rows_csv(out_diag_physics / "physics_diag_id_val.csv", physics_val_rows)
    write_rows_csv(out_diag_physics / "physics_diag_id_test.csv", physics_test_rows)
    (out_diag_physics / "physics_diag_id_val_summary.json").write_text(
        json.dumps(physics_val_summary, indent=2),
        encoding="utf-8",
    )
    (out_diag_physics / "physics_diag_id_test_summary.json").write_text(
        json.dumps(physics_test_summary, indent=2),
        encoding="utf-8",
    )
    worst_val = build_worst_case_report(physics_val_rows, split_name="id_val", top_n=worst_case_top_n)
    worst_test = build_worst_case_report(physics_test_rows, split_name="id_test", top_n=worst_case_top_n)
    (out_diag_physics / "physics_diag_id_val_worst.json").write_text(
        json.dumps(worst_val, indent=2),
        encoding="utf-8",
    )
    (out_diag_physics / "physics_diag_id_test_worst.json").write_text(
        json.dumps(worst_test, indent=2),
        encoding="utf-8",
    )

    # Plot outputs are intentionally parallel to blackbox naming for easier comparison.
    plot_dir = out_dir / "plots"
    plot_id_dir = plot_dir / "id"
    plot_val_dir = plot_dir / "id_val"
    plot_ood_dir = plot_dir / "ood_primary"
    ensure_dir(plot_id_dir)
    ensure_dir(plot_val_dir)
    if has_ood:
        ensure_dir(plot_ood_dir)
    # Blackbox-style curve plots for all hybrid stages.
    generate_stage_curve_plots(
        plot_dir=plot_id_dir,
        split_label="id",
        t_curve=t_test,
        j_true=j_test_true,
        stage_map={
            "stageA": j_test_coarse,
            "stageB": j_test_pred,
            "stageC": j_test_stage_c,
        },
        max_examples=plot_max_examples,
    )
    generate_stage_curve_plots(
        plot_dir=plot_val_dir,
        split_label="id_val",
        t_curve=t_val,
        j_true=j_val_true,
        stage_map={
            "stageA": j_val_coarse,
            "stageB": j_val_pred,
            "stageC": j_val_stage_c,
        },
        max_examples=plot_max_examples,
    )
    if has_ood:
        generate_stage_curve_plots(
            plot_dir=plot_ood_dir,
            split_label="ood_primary",
            t_curve=t_ood,
            j_true=j_ood_true,
            stage_map={
                "stageA": j_ood_coarse,
                "stageB": j_ood_pred,
                "stageC": j_ood_stage_c,
            },
            max_examples=plot_max_examples,
        )
    # Keep final-stage aliases aligned with blackbox naming.
    plot_curve_examples(
        t_test,
        j_test_true,
        j_test_stage_c,
        plot_id_dir / "curve_examples.png",
        "J(t) examples (ID)",
        max_examples=plot_max_examples,
    )
    plot_curve_error_over_time(
        t_test,
        j_test_true,
        j_test_stage_c,
        plot_id_dir / "curve_error_over_time.png",
        "Curve error over time (ID)",
    )
    if has_ood:
        plot_curve_examples(
            t_ood,
            j_ood_true,
            j_ood_stage_c,
            plot_ood_dir / "curve_examples.png",
            "J(t) examples (OOD)",
            max_examples=plot_max_examples,
        )
        plot_curve_error_over_time(
            t_ood,
            j_ood_true,
            j_ood_stage_c,
            plot_ood_dir / "curve_error_over_time.png",
            "Curve error over time (OOD)",
        )
    # Final-stage scalar diagnostics mirror blackbox plot naming.
    plot_scalar_parity(
        y_test_true,
        y_test_stage_c,
        target_names,
        plot_id_dir / "scalar_parity.png",
        "Scalar parity (ID)",
    )
    plot_scalar_residual_hist(
        y_test_true,
        y_test_stage_c,
        target_names,
        plot_id_dir / "scalar_residuals.png",
        "Scalar residuals (ID)",
    )
    if has_ood:
        plot_scalar_parity(
            y_ood_true,
            y_ood_stage_c,
            target_names,
            plot_ood_dir / "scalar_parity.png",
            "Scalar parity (OOD)",
        )
        plot_scalar_residual_hist(
            y_ood_true,
            y_ood_stage_c,
            target_names,
            plot_ood_dir / "scalar_residuals.png",
            "Scalar residuals (OOD)",
        )

    comparison_report = build_comparison_report(
        stage_a_val=stage_a_val,
        stage_a_test=stage_a_test,
        stage_b_val=stage_b_val,
        stage_b_test=stage_b_test,
        stage_c_val=stage_c_val,
        stage_c_test=stage_c_test,
        physics_val_summary=physics_val_summary,
        physics_test_summary=physics_test_summary,
        blackbox_dir=blackbox_dir,
    )
    comparison_report_path = out_diag / "comparison_report.json"
    comparison_report_path.write_text(
        json.dumps(comparison_report, indent=2),
        encoding="utf-8",
    )

    # Pre-materialize sweep values as plain JSON scalars.
    stagec_component_values = []
    for value in stagec_components:
        stagec_component_values.append(int(value))
    stagec_alpha_values = []
    for value in stagec_alphas:
        stagec_alpha_values.append(float(value))

    # Summary bundles model choices, metrics, and file pointers for downstream reports.
    summary = {
        "status": "ok",
        "description": "Discrete-physics PINN starter: coarse PDE baseline + learned amp/time correction",
        "device": str(device),
        "run_root_override": args.run_root_override,
        "ood_available": bool(has_ood),
        "split_rows": {
            "id_train": int(len(train_entries)),
            "id_val": int(len(val_entries)),
            "id_test": int(len(test_entries)),
            "ood_primary": int(len(ood_entries)) if has_ood else 0,
        },
        "coarsen_factor": int(coarsen_factor),
        "sim_batch_runs": int(sim_batch_runs),
        "stagec_sweep_grid": {
            "components": stagec_component_values,
            "alphas": stagec_alpha_values,
        },
        "stagec_nonneg": {
            "mode": stagec_nonneg,
            "softplus_beta": float(stagec_softplus_beta),
        },
        "stagec_score_weights": {
            "negative_flux_fraction": float(stagec_score_w_neg),
            "bc_bottom_flux_rel_rmse": float(stagec_score_w_bc),
        },
        "stagec_feature_map": str(stagec_feature_map),
        "stageb_time_weighting": train_result["time_weighting"],
        "scalar_target_names": target_names,
        "id_scalar_rmse": build_scalar_rmse_summary(metrics_id["stageC"]["scalar"], target_names),
        "ood_scalar_rmse": (
            build_scalar_rmse_summary(metrics_ood["stageC"]["scalar"], target_names)
            if has_ood
            else {}
        ),
        "stageA": {
            "id_val_curve": stage_a_val,
            "id_test_curve": stage_a_test,
            "ood_primary_curve": stage_a_ood,
        },
        "stageB": {
            "id_val_curve": stage_b_val,
            "id_test_curve": stage_b_test,
            "ood_primary_curve": stage_b_ood,
            "best_epoch": int(train_result["best_epoch"]),
            "best_val_rel_l2": float(train_result["best_val_rel_l2"]),
            "flux_eps": float(train_result["flux_eps"]),
        },
        "stageC": {
            "id_val_curve": stage_c_val,
            "id_test_curve": stage_c_test,
            "ood_primary_curve": stage_c_ood,
            "pca_components": int(stage_c_model["k"]),
            "ridge_alpha": float(stage_c_model["alpha"]),
            "selection_score": float(stage_c_model["val_score"]),
            "val_rel_l2_for_selection": float(stage_c_model["val_rel_l2"]),
            "val_neg_flux_fraction_for_selection": float(stage_c_model["val_neg_flux_fraction"]),
            "val_bc_bottom_flux_rel_rmse_for_selection": float(stage_c_model["val_bc_bottom_flux_rel_rmse"]),
            "feature_map": str(stage_c_model.get("feature_map", "linear")),
            "sweep_top10": stage_c_model.get("sweep_top10", []),
        },
        "physics_diagnostics": {
            "id_val_summary_file": "diagnostics/physics/physics_diag_id_val_summary.json",
            "id_test_summary_file": "diagnostics/physics/physics_diag_id_test_summary.json",
            "id_val_worst_file": "diagnostics/physics/physics_diag_id_val_worst.json",
            "id_test_worst_file": "diagnostics/physics/physics_diag_id_test_worst.json",
            "id_val": physics_val_summary,
            "id_test": physics_test_summary,
        },
        "metrics_files": {
            "id_val": "metrics_id_val.json",
            "id_test": "metrics_id.json",
            "ood_primary": "metrics_ood.json",
        },
        "comparison_report_file": "diagnostics/comparison_report.json",
        "runtime_seconds": float(time.time() - t0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(train_result["history"], indent=2), encoding="utf-8")

    print("saved:", out_dir)
    print("stageA id_test rel_l2:", stage_a_test["relative_l2"])
    print("stageB id_test rel_l2:", stage_b_test["relative_l2"])
    print("stageC id_test rel_l2:", stage_c_test["relative_l2"])


if __name__ == "__main__":
    main()
