import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ml.pinn_data import build_entries, extract_c0_feature, load_ml_meta_names, load_run_physics_1d, load_split_matrix, load_split_name_map, pack_for_training, tensorize_pack
from scripts.ml.pinn_model import RunConditionedPINN, curriculum_scale, evaluate_cstar_rmse, predict_flux_curves, sample_depth_ids, sample_time_ids
from skin_diffusion.ml_curve_plots import plot_curve_error_over_time, plot_curve_examples, plot_per_run_error_distribution, plot_training_history
from skin_diffusion.ml_curve_backbone import fit_curve_backbone, parse_float_csv, predict_curve_backbone
from skin_diffusion.ml_metrics import compute_curve_metrics
from skin_diffusion.ml_physics_diagnostics import build_worst_case_report, compute_split_physics_diagnostics, write_rows_csv
from skin_diffusion.ml_scalar_diagnostics import build_scalar_rmse_summary, plot_scalar_parity, plot_scalar_residual_hist, scalar_report, scalar_targets_from_flux_curves
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


def train_pinn(model, train_torch, val_torch, args):
    # Main training loop: data fit + physics losses with optional phase-2 tightening.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    n_train = int(train_torch["x_feat"].shape[0])
    t_count = int(train_torch["t_norm"].shape[1])
    h_count = int(train_torch["d1d"].shape[1])
    phase1_epochs = int(args.epochs)
    phase2_epochs = int(args.phase2_epochs)
    total_epochs = int(phase1_epochs + phase2_epochs)

    # Cosine annealing scheduler for smoother convergence over longer runs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=float(args.lr) * 0.01)

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

        # Progressive correction scale: ramp from initial_correction_scale
        # to correction_scale over the first half of training.
        # Computed early because it is needed by forward_j calls below.
        if hasattr(args, 'initial_correction_scale') and float(args.initial_correction_scale) < float(args.correction_scale):
            ramp_frac = min(1.0, float(epoch_index) / float(max(1, phase1_epochs) * 0.5))
            current_correction_scale = float(args.initial_correction_scale) + ramp_frac * (float(args.correction_scale) - float(args.initial_correction_scale))
        else:
            current_correction_scale = float(args.correction_scale)

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
        j_flux_pred = model.forward_j(x_flux, t_flux, j_base=j_flux_base, correction_scale_override=current_correction_scale)

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

        # Peak-focused loss: differentiable J_peak and t_peak supervision.
        # Uses soft-argmax to extract peak location and value from full curves.
        loss_peak = torch.tensor(0.0, device=x_batch.device, dtype=torch.float32)
        if float(args.weight_peak) > 0.0:
            # Predict full curves for this batch for peak extraction.
            peak_run_count = min(int(batch_count), 16)
            peak_run_ids = torch.arange(peak_run_count, device=x_batch.device)
            peak_x = x_batch[:peak_run_count]
            peak_j_base = j_base_batch[:peak_run_count]
            peak_t_norm = t_norm_batch[:peak_run_count]
            peak_j_true_full = j_true_batch[:peak_run_count]
            peak_t_phys = t_phys_batch[:peak_run_count]
            peak_t_steps = int(peak_t_norm.shape[1])
            # Expand for vectorized forward_j.
            peak_x_exp = peak_x.unsqueeze(1).expand(-1, peak_t_steps, -1).reshape(-1, peak_x.shape[1])
            peak_base_exp = peak_j_base.reshape(-1, 1)
            peak_t_exp = peak_t_norm.reshape(-1, 1)
            peak_j_pred_exp = model.forward_j(peak_x_exp, peak_t_exp, j_base=peak_base_exp, correction_scale_override=current_correction_scale)
            peak_j_pred_curves = peak_j_pred_exp.reshape(peak_run_count, peak_t_steps)
            # Soft-argmax with temperature for differentiable peak extraction.
            peak_temperature = 50.0
            peak_weights_pred = torch.softmax(peak_j_pred_curves * peak_temperature, dim=1)
            peak_weights_true = torch.softmax(peak_j_true_full * peak_temperature, dim=1)
            # Predicted peak value and time.
            j_peak_pred = torch.sum(peak_j_pred_curves * peak_weights_pred, dim=1)
            j_peak_true = torch.sum(peak_j_true_full * peak_weights_true, dim=1)
            t_peak_pred = torch.sum(peak_t_phys * peak_weights_pred, dim=1)
            t_peak_true = torch.sum(peak_t_phys * peak_weights_true, dim=1)
            # Relative error on peak flux value.
            j_peak_safe = torch.clamp(torch.abs(j_peak_true.detach()), min=1e-15)
            loss_j_peak = torch.mean(((j_peak_pred - j_peak_true) / j_peak_safe) ** 2)
            # Relative error on peak time.
            t_peak_safe = torch.clamp(torch.abs(t_peak_true.detach()), min=1.0)
            loss_t_peak = torch.mean(((t_peak_pred - t_peak_true) / t_peak_safe) ** 2)
            loss_peak = loss_j_peak + loss_t_peak

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

        # Exponentially decay anchor weight so the PINN can freely deviate from
        # the backbone later in training once physics losses are dominant.
        # Reaches ~0.05 by epoch phase1_epochs/2 and ~0.002 by end.
        anchor_half_life = float(max(1, phase1_epochs)) / 6.0
        anchor_decay = float(math.exp(-0.693 * float(epoch_index) / anchor_half_life))

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

        # Gate regularisation: gently encourage the gate to stay near 0.5
        # (neither fully backbone nor fully PINN) at the start, then let it
        # specialise as training progresses.  Only active with --use_learned_gate.
        loss_gate_reg = torch.tensor(0.0, device=x_batch.device, dtype=torch.float32)
        if model.use_learned_gate and float(args.weight_gate_entropy) > 0.0:
            # Sample gate values for the current batch.
            gate_vals = model.forward_gate(x_batch)
            # Entropy-style regularisation: maximise entropy of gate distribution.
            # H(gate) = -gate*log(gate) - (1-gate)*log(1-gate), maximised at 0.5.
            gate_clamped = torch.clamp(gate_vals, min=1e-6, max=1.0 - 1e-6)
            gate_entropy = -(gate_clamped * torch.log(gate_clamped) + (1.0 - gate_clamped) * torch.log(1.0 - gate_clamped))
            # Negate entropy (we want to maximise it, so minimise -entropy).
            # Decay this regularisation so the gate can specialise later.
            gate_reg_decay = float(math.exp(-0.693 * float(epoch_index) / float(max(1, phase1_epochs) * 0.3)))
            loss_gate_reg = -torch.mean(gate_entropy) * float(gate_reg_decay)

        total_loss = (
            (float(args.weight_c_data) * loss_c_data)
            + (float(args.weight_c_nonneg) * loss_c_nonneg)
            + (float(args.weight_inner_profile) * loss_inner_profile)
            + (float(args.weight_flux) * float(flux_scale) * loss_flux)
            + (float(args.weight_anchor) * float(flux_scale) * float(anchor_decay) * loss_anchor)
            + (float(weight_flux_consistency) * float(flux_scale) * loss_flux_consistency)
            + (float(weight_pde) * float(pde_scale) * loss_pde)
            + (float(weight_bc_top) * loss_bc_top)
            + (float(weight_bc_bottom) * loss_bc_bottom)
            + (float(weight_ic) * loss_ic)
            + (float(args.weight_nonneg) * loss_nonneg)
            + (float(args.weight_peak) * float(flux_scale) * loss_peak)
            + (float(args.weight_gate_entropy) * loss_gate_reg)
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

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

            # Compute gate mean for diagnostics (if gate is active).
            gate_mean_val = 0.0
            if model.use_learned_gate:
                with torch.no_grad():
                    gate_diag = model.forward_gate(x_batch)
                    gate_mean_val = float(gate_diag.mean().item())

            row = {
                "epoch": int(epoch_index),
                "phase": "phase2_physics_tighten" if is_phase2 else "phase1_main",
                "loss_total": float(total_loss.detach().cpu().item()),
                "loss_flux": float(loss_flux.detach().cpu().item()),
                "loss_pde": float(loss_pde.detach().cpu().item()),
                "loss_anchor": float(loss_anchor.detach().cpu().item()),
                "loss_peak": float(loss_peak.detach().cpu().item()) if isinstance(loss_peak, torch.Tensor) else 0.0,
                "gate_mean": gate_mean_val,
                "val_rel_l2": float(val_rel_l2),
            }
            history.append(row)
            gate_str = f"  gate={gate_mean_val:.3f}" if model.use_learned_gate else ""
            print(
                "epoch",
                int(epoch_index),
                "loss_total=",
                f"{row['loss_total']:.6e}",
                "val_rel_l2=",
                f"{row['val_rel_l2']:.6e}",
                f"flux={row['loss_flux']:.4e}",
                f"pde={row['loss_pde']:.4e}",
                f"anch={row['loss_anchor']:.4e}",
                f"peak={row['loss_peak']:.4e}",
                f"{gate_str}",
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
    parser.add_argument("--max_train_rows", type=int, default=700)
    parser.add_argument("--max_val_rows", type=int, default=150)
    parser.add_argument("--max_test_rows", type=int, default=150)

    # Stage-0 curve backbone.
    parser.add_argument("--backbone_curve_components", type=int, default=20)
    parser.add_argument("--backbone_alphas", default="1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--xgb_backbone", action="store_true")

    # Neural model and optimizer settings.
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--phase2_epochs", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_runs", type=int, default=48)
    parser.add_argument("--predict_batch_runs", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fourier_frequencies", type=int, default=6)
    parser.add_argument("--correction_scale", type=float, default=0.4)
    parser.add_argument("--additive_correction", action="store_true")

    # Curriculum schedule.
    parser.add_argument("--flux_warmup_epochs", type=int, default=3)
    parser.add_argument("--flux_ramp_epochs", type=int, default=5)
    parser.add_argument("--pde_warmup_epochs", type=int, default=10)
    parser.add_argument("--pde_ramp_epochs", type=int, default=30)

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
    parser.add_argument("--weight_anchor", type=float, default=0.1)
    parser.add_argument("--weight_flux_consistency", type=float, default=0.5)
    parser.add_argument("--weight_pde", type=float, default=0.5)
    parser.add_argument("--weight_bc_top", type=float, default=0.2)
    parser.add_argument("--weight_bc_bottom", type=float, default=0.2)
    parser.add_argument("--weight_ic", type=float, default=0.1)
    parser.add_argument("--weight_nonneg", type=float, default=0.05)
    parser.add_argument("--weight_peak", type=float, default=1.0)
    parser.add_argument("--weight_gate_entropy", type=float, default=0.1)
    parser.add_argument("--conc_log_scale", type=float, default=5000.0)
    parser.add_argument("--flux_log_scale", type=float, default=1e11)
    parser.add_argument("--use_learned_gate", action="store_true")
    parser.add_argument("--gate_init_bias", type=float, default=-2.0)
    parser.add_argument("--initial_correction_scale", type=float, default=0.1)

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
    max_rows_map = {"train": args.max_train_rows, "val": args.max_val_rows, "test": args.max_test_rows}
    split_entries = {}
    for split in ("train", "val", "test"):
        split_entries[split] = build_entries(
            ml_dir, split_name_map[split], args.run_start_index, args.run_end_index,
            max_rows_map[split], args.run_root_override)
    train_entries, val_entries, test_entries = split_entries["train"], split_entries["val"], split_entries["test"]
    feature_names, target_names = load_ml_meta_names(ml_dir)

    # Load tabular features, target curves, and curve time grids.
    def load_mat(split, key): return load_split_matrix(ml_dir, split_name_map[split], split_entries[split], key)
    x_train_raw, x_val_raw, x_test_raw = load_mat("train", "X"), load_mat("val", "X"), load_mat("test", "X")
    y_val_true = load_mat("val", "y_scalar")
    y_test_true = load_mat("test", "y_scalar")
    c0_val, c0_test = extract_c0_feature(x_val_raw, feature_names), extract_c0_feature(x_test_raw, feature_names)
    j_train_true, j_val_true, j_test_true = load_mat("train", "J"), load_mat("val", "J"), load_mat("test", "J")
    t_val, t_test = load_mat("val", "t"), load_mat("test", "t")

    # Fit and predict the base curve model (Ridge or XGBoost backbone).
    print("fitting backbone (xgb=" + str(bool(args.xgb_backbone)) + ") ...")
    backbone = fit_curve_backbone(x_train_raw, j_train_true, x_val_raw, j_val_true,
                                  int(args.backbone_curve_components), backbone_alphas, bool(args.xgb_backbone))
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
    train_pack = tensorize_pack(pack_for_training(x_train_scaled, j_train_true, j_train_base, train_phys), device)
    val_pack = tensorize_pack(pack_for_training(x_val_scaled, j_val_true, j_val_base, val_phys), device)
    test_pack = tensorize_pack(pack_for_training(x_test_scaled, j_test_true, j_test_base, test_phys), device)

    model = RunConditionedPINN(
        int(x_train_scaled.shape[1]), int(args.hidden_dim), int(args.depth),
        int(args.fourier_frequencies), float(args.correction_scale),
        bool(args.additive_correction), bool(args.use_learned_gate), float(args.gate_init_bias),
    ).to(device)

    # Train with checkpointing and optional phase-2 physics tightening.
    train_result = train_pinn(model, train_pack, val_pack, args)
    model = train_result["model"]

    # Build full validation/test curve predictions.
    print("predicting full curves for val/test ...")
    j_val_pinn = predict_flux_curves(model, val_pack, batch_runs=int(args.predict_batch_runs))
    j_test_pinn = predict_flux_curves(model, test_pack, batch_runs=int(args.predict_batch_runs))

    # Blend is selected on validation to prevent PINN from harming final deployment curves.
    lambda_best, lambda_metric = pick_blend_lambda(j_val_true, t_val, j_val_base, j_val_pinn)
    j_val_final = j_val_base + (lambda_best * (j_val_pinn - j_val_base))
    j_test_final = j_test_base + (lambda_best * (j_test_pinn - j_test_base))

    # Save metrics bundles in train/val/test terminology only.
    metrics_val = build_metrics_bundle(j_val_true, t_val, j_val_base, j_val_pinn, j_val_final,
                                       y_val_true, c0_val, target_names)
    metrics_test = build_metrics_bundle(j_test_true, t_test, j_test_base, j_test_pinn, j_test_final,
                                        y_test_true, c0_test, target_names)
    (out_dir / "metrics_val.json").write_text(json.dumps(metrics_val, indent=2), encoding="utf-8")
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")

    def save_preds(path, j_true, j_base, j_pinn, j_final, t):
        np.savez(path, j_true=j_true.astype(np.float32), j_stageBase=j_base.astype(np.float32),
                 j_stagePINN=j_pinn.astype(np.float32), j_stageFinal=j_final.astype(np.float32),
                 t=t.astype(np.float32))
    save_preds(out_dir / "pred_val.npz", j_val_true, j_val_base, j_val_pinn, j_val_final, t_val)
    save_preds(out_dir / "pred_test.npz", j_test_true, j_test_base, j_test_pinn, j_test_final, t_test)

    out_diag = out_dir / "diagnostics"
    out_diag_physics = out_diag / "physics"
    ensure_dir(out_diag)
    ensure_dir(out_diag_physics)

    # Physics diagnostics are computed from run bundles and predicted curves.
    def pred_map(j_true, j_base, j_pinn, j_final):
        return {"truth": j_true, "stageBase": j_base, "stagePINN": j_pinn, "stageFinal": j_final}
    print("computing physics diagnostics for val ...")
    physics_val_rows, physics_val_summary = compute_split_physics_diagnostics(
        val_entries, "val", pred_map(j_val_true, j_val_base, j_val_pinn, j_val_final),
        progress_label="physics_val", truth_key="truth", pde_anchor_key="stageBase")
    print("computing physics diagnostics for test ...")
    physics_test_rows, physics_test_summary = compute_split_physics_diagnostics(
        test_entries, "test", pred_map(j_test_true, j_test_base, j_test_pinn, j_test_final),
        progress_label="physics_test", truth_key="truth", pde_anchor_key="stageBase")
    write_rows_csv(out_diag_physics / "physics_diag_val.csv", physics_val_rows)
    write_rows_csv(out_diag_physics / "physics_diag_test.csv", physics_test_rows)
    def write_json(path, obj): path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    write_json(out_diag_physics / "physics_diag_val_summary.json", physics_val_summary)
    write_json(out_diag_physics / "physics_diag_test_summary.json", physics_test_summary)
    # Keep explicit worst-case rows for manual inspection after each run.
    worst_val = build_worst_case_report(physics_val_rows, "val", "stageFinal", int(args.worst_case_top_n))
    worst_test = build_worst_case_report(physics_test_rows, "test", "stageFinal", int(args.worst_case_top_n))
    write_json(out_diag_physics / "physics_diag_val_worst.json", worst_val)
    write_json(out_diag_physics / "physics_diag_test_worst.json", worst_test)

    # Curve and scalar plots for quick visual comparison with blackbox.
    plot_dir = out_dir / "plots"
    plot_test_dir = plot_dir / "test"
    ensure_dir(plot_dir)
    ensure_dir(plot_test_dir)
    plot_curve_examples(t_test, j_test_true, j_test_final,
                        plot_test_dir / "curve_examples.png", "J(t) examples (test)", max_examples=9)
    plot_curve_error_over_time(t_test, j_test_true, j_test_final,
                               plot_test_dir / "curve_error_over_time.png", "Curve error over time (test)")
    y_test_stage_final = scalar_targets_from_flux_curves(j_test_final, t_test, c0_test, target_names)
    plot_scalar_parity(y_test_true, y_test_stage_final, target_names,
                       plot_test_dir / "scalar_parity.png", "Scalar parity (test)")
    plot_scalar_residual_hist(y_test_true, y_test_stage_final, target_names,
                              plot_test_dir / "scalar_residuals.png", "Scalar residuals (test)")
    plot_training_history(train_result["history"], plot_dir / "training_history.png",
                          "Training loss & validation (seed=" + str(args.seed) + ")")
    plot_per_run_error_distribution(j_test_true, j_test_final,
                                    plot_test_dir / "per_run_error_distribution.png",
                                    "Per-run error distribution (test, stageFinal)")

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
            "xgb_backbone": bool(args.xgb_backbone),
            "selected_family": str(backbone.get("backbone_family", "ridge")),
            "selected_hyperparams": backbone.get("backbone_hyperparams", {}),
            "selected_ridge_alpha": float(backbone["ridge_alpha"]),
            "val_rmse_for_selection": float(backbone["val_rmse"]),
        },
        "pinn_config": {
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "fourier_frequencies": int(args.fourier_frequencies),
            "correction_scale": float(args.correction_scale),
            "initial_correction_scale": float(args.initial_correction_scale),
            "additive_correction": bool(args.additive_correction),
            "use_learned_gate": bool(args.use_learned_gate),
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
                "peak": float(args.weight_peak),
                "gate_entropy": float(args.weight_gate_entropy),
            },
            "gate_init_bias": float(args.gate_init_bias),
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
