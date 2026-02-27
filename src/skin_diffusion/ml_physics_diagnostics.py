import csv
from pathlib import Path

import numpy as np

from skin_diffusion.bc import patch_concentration
from skin_diffusion.ml_run_dataset import load_run_bundle


def harmonic_mean_numpy(a, b):
    return (2.0 * a * b) / (a + b + 1e-30)


def build_run_physics_reference(bundle):
    # Build per-run physics terms from saved concentration snapshots.
    fields = bundle["fields"]
    meta = bundle["meta"]
    grid = meta["grid"]
    boundary = meta["boundary"]

    c_snap = np.asarray(fields["C_snap"], dtype=np.float64)
    d_field = np.asarray(fields["D"], dtype=np.float64)
    k_field = np.asarray(fields["k"], dtype=np.float64)
    patch_mask = np.asarray(fields["patch_mask"], dtype=bool)
    t_curve = np.asarray(fields["t"], dtype=np.float64)
    if c_snap.ndim != 3:
        raise ValueError("C_snap must be [T,H,W]")
    if t_curve.shape[0] != c_snap.shape[0]:
        raise ValueError("t and C_snap length mismatch")
    if c_snap.shape[1:] != d_field.shape or d_field.shape != k_field.shape:
        raise ValueError("C/D/k shape mismatch in run bundle")
    if patch_mask.shape != d_field.shape:
        raise ValueError("patch_mask shape mismatch in run bundle")
    if c_snap.shape[1] < 2 or c_snap.shape[2] < 2:
        raise ValueError("Grid must be at least 2x2 for physics diagnostics")

    dt = np.diff(t_curve)
    if np.any(dt <= 0.0):
        raise ValueError("Saved time curve must be strictly increasing")

    dx = float(grid["dx"])
    h = int(c_snap.shape[1])

    # Top and bottom mean fluxes from the saved C snapshots.
    top_face_d = harmonic_mean_numpy(d_field[1, :], d_field[0, :])
    bottom_face_d = harmonic_mean_numpy(d_field[-1, :], d_field[-2, :])
    top_grad = (c_snap[:, 1, :] - c_snap[:, 0, :]) / dx
    bottom_grad = (c_snap[:, -1, :] - c_snap[:, -2, :]) / dx
    j_top_in = np.mean(-top_face_d[None, :] * top_grad, axis=1)
    j_bottom_true = np.mean(-bottom_face_d[None, :] * bottom_grad, axis=1)

    # Domain-mean mass balance terms on saved intervals.
    mass_mean = np.mean(c_snap, axis=(1, 2))
    mass_dt = np.diff(mass_mean) / dt
    reaction_mean = np.mean(k_field[None, :, :] * c_snap[:-1, :, :], axis=(1, 2))
    rhs_true = ((j_top_in[:-1] - j_bottom_true[:-1]) / (float(h) * dx)) - reaction_mean
    residual_true = mass_dt - rhs_true

    # Solver BC/IC adherence from saved concentration fields.
    mode = str(boundary.get("mode", "infinite_dose"))
    c0 = float(boundary.get("C0", 0.0))
    decay = float(boundary.get("decay_rate", 0.0))
    cpatch_curve = np.asarray(
        [patch_concentration(float(ti), mode, c0, decay) for ti in t_curve],
        dtype=np.float64,
    )
    top_patch_mask = patch_mask[0]
    top_offpatch_mask = ~top_patch_mask

    if np.any(top_patch_mask):
        top_patch_vals = c_snap[:, 0, :][:, top_patch_mask]
        top_patch_bc_mae = float(np.mean(np.abs(top_patch_vals - cpatch_curve[:, None])))
    else:
        top_patch_bc_mae = 0.0

    if np.any(top_offpatch_mask):
        top_off = c_snap[:, 0, :][:, top_offpatch_mask]
        top_inner = c_snap[:, 1, :][:, top_offpatch_mask]
        top_offpatch_neumann_mae = float(np.mean(np.abs(top_off - top_inner)))
    else:
        top_offpatch_neumann_mae = 0.0

    bottom_sink_mae = float(np.mean(np.abs(c_snap[:, -1, :])))
    side_left_mae = float(np.mean(np.abs(c_snap[:, :, 0] - c_snap[:, :, 1])))
    side_right_mae = float(np.mean(np.abs(c_snap[:, :, -1] - c_snap[:, :, -2])))
    side_neumann_mae = 0.5 * (side_left_mae + side_right_mae)
    ic_interior_mae = float(np.mean(np.abs(c_snap[0, 1:, :])))

    return {
        "t": t_curve.astype(np.float32),
        "j_top_in": j_top_in.astype(np.float32),
        "j_bottom_true": j_bottom_true.astype(np.float32),
        "mass_dt": mass_dt.astype(np.float32),
        "reaction_mean": reaction_mean.astype(np.float32),
        "residual_true": residual_true.astype(np.float32),
        "h": int(h),
        "dx": float(dx),
        "boundary_mode": str(boundary.get("mode", "infinite_dose")),
        "bc_top_patch_mae": float(top_patch_bc_mae),
        "bc_top_offpatch_neumann_mae": float(top_offpatch_neumann_mae),
        "bc_bottom_sink_mae": float(bottom_sink_mae),
        "bc_side_neumann_mae": float(side_neumann_mae),
        "ic_interior_mae": float(ic_interior_mae),
    }


def evaluate_prediction_physics_metrics(j_pred, j_truth, ref, pde_anchor=None, pde_anchor_name=None):
    # Stage/model-specific metrics using truth flux and run physics reference.
    j_arr = np.asarray(j_pred, dtype=np.float64)
    j_truth_arr = np.asarray(j_truth, dtype=np.float64)
    if j_arr.shape != j_truth_arr.shape:
        raise ValueError("Predicted flux shape mismatch in physics diagnostics")

    # Flux-curve metrics against truth.
    curve_diff = j_arr - j_truth_arr
    curve_mae = float(np.mean(np.abs(curve_diff)))
    curve_rel_l2 = float(np.linalg.norm(curve_diff) / (np.linalg.norm(j_truth_arr) + 1e-12))

    # Bottom boundary mismatch metrics.
    bc_rmse = float(np.sqrt(np.mean(curve_diff**2)))
    bc_scale = float(np.sqrt(np.mean(j_truth_arr * j_truth_arr)) + 1e-12)

    # Mass-balance residual using true C snapshots as reference terms.
    j_top_in = np.asarray(ref["j_top_in"], dtype=np.float64)
    mass_dt = np.asarray(ref["mass_dt"], dtype=np.float64)
    reaction_mean = np.asarray(ref["reaction_mean"], dtype=np.float64)
    h = float(ref["h"])
    dx = float(ref["dx"])
    rhs = ((j_top_in[:-1] - j_arr[:-1]) / (h * dx)) - reaction_mean
    residual = mass_dt - rhs
    pde_rmse = float(np.sqrt(np.mean(residual * residual)))
    mass_scale = float(np.sqrt(np.mean(mass_dt * mass_dt)) + 1e-12)

    metrics = {
        "curve_mae": curve_mae,
        "curve_relative_l2": curve_rel_l2,
        "ic_flux_abs_error": float(abs(j_arr[0] - j_truth_arr[0])),
        "bc_bottom_flux_rmse": bc_rmse,
        "bc_bottom_flux_rel_rmse": float(bc_rmse / bc_scale),
        "pde_mass_balance_rmse": pde_rmse,
        "pde_mass_balance_rel_rmse": float(pde_rmse / mass_scale),
        "negative_flux_fraction": float(np.mean(j_arr < -1e-14)),
    }

    if pde_anchor is not None:
        anchor_arr = np.asarray(pde_anchor, dtype=np.float64)
        if anchor_arr.shape != j_arr.shape:
            raise ValueError("PDE anchor shape mismatch in physics diagnostics")
        anchor_name = str(pde_anchor_name or "anchor")
        pde_anchor_rmse = float(np.sqrt(np.mean((j_arr - anchor_arr) ** 2)))
        pde_anchor_scale = float(np.sqrt(np.mean(anchor_arr * anchor_arr)) + 1e-12)
        metrics["pde_" + anchor_name + "_flux_rmse"] = pde_anchor_rmse
        metrics["pde_" + anchor_name + "_flux_rel_rmse"] = float(pde_anchor_rmse / pde_anchor_scale)

    return metrics


def summarize_scalar(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 1:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def write_rows_csv(path, rows):
    if len(rows) < 1:
        Path(path).write_text("", encoding="utf-8")
        return

    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compute_split_physics_diagnostics(
    entries,
    split_name,
    prediction_map,
    progress_label,
    truth_key="truth",
    pde_anchor_key=None,
):
    # Normalize all stage/model curves into a consistent [N, T] map.
    stage_curves = {}
    stage_names = []
    n_rows = len(entries)
    for stage_name, curves in prediction_map.items():
        arr = np.asarray(curves, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Diagnostics curves must be [N,T]")
        if int(arr.shape[0]) != int(n_rows):
            raise ValueError("Diagnostics rows mismatch for stage " + str(stage_name))
        name = str(stage_name)
        stage_names.append(name)
        stage_curves[name] = arr

    truth_name = str(truth_key)
    if truth_name not in stage_curves:
        raise ValueError("prediction_map must include truth_key: " + truth_name)

    # Optional anchor lets us report distance to a chosen physics baseline (e.g., stageA).
    anchor_name = None
    if pde_anchor_key is not None:
        anchor_name = str(pde_anchor_key)
        if anchor_name not in stage_curves:
            raise ValueError("prediction_map must include pde_anchor_key: " + anchor_name)

    rows = []
    metric_names = None
    done = 0
    for row_idx, entry in enumerate(entries):
        bundle = load_run_bundle(entry["run_dir"])
        ref = build_run_physics_reference(bundle)

        # Keep run identity fields so worst-case reports can link back to runs directly.
        row = {
            "split": str(split_name),
            "row_index": int(row_idx),
            "run_dir": str(entry["run_dir"]),
            "boundary_mode": str(ref["boundary_mode"]),
            "truth_pde_mass_balance_rmse": float(np.sqrt(np.mean(np.asarray(ref["residual_true"], dtype=np.float64) ** 2))),
            "truth_bc_top_patch_mae": float(ref["bc_top_patch_mae"]),
            "truth_bc_top_offpatch_neumann_mae": float(ref["bc_top_offpatch_neumann_mae"]),
            "truth_bc_bottom_sink_mae": float(ref["bc_bottom_sink_mae"]),
            "truth_bc_side_neumann_mae": float(ref["bc_side_neumann_mae"]),
            "truth_ic_interior_mae": float(ref["ic_interior_mae"]),
        }
        truth_mass_scale = float(np.sqrt(np.mean(np.asarray(ref["mass_dt"], dtype=np.float64) ** 2)) + 1e-12)
        row["truth_pde_mass_balance_rel_rmse"] = float(row["truth_pde_mass_balance_rmse"] / truth_mass_scale)

        j_truth = stage_curves[truth_name][row_idx]
        pde_anchor = stage_curves[anchor_name][row_idx] if anchor_name is not None else None

        for stage_name in stage_names:
            metrics = evaluate_prediction_physics_metrics(
                j_pred=stage_curves[stage_name][row_idx],
                j_truth=j_truth,
                ref=ref,
                pde_anchor=pde_anchor,
                pde_anchor_name=anchor_name,
            )
            # All stages emit the same metric set; capture names once from first stage.
            if metric_names is None:
                metric_names = list(metrics.keys())
            for key, value in metrics.items():
                row[stage_name + "_" + key] = float(value)
        rows.append(row)

        done += 1
        if done == 1 or done == n_rows or (done % 20) == 0:
            print(progress_label, done, "/", n_rows)

    if metric_names is None:
        metric_names = []

    summary = {
        "split": str(split_name),
        "rows": int(len(rows)),
        "stages": {},
        "truth_reference": {
            "pde_mass_balance_rmse": summarize_scalar([r["truth_pde_mass_balance_rmse"] for r in rows]),
            "pde_mass_balance_rel_rmse": summarize_scalar([r["truth_pde_mass_balance_rel_rmse"] for r in rows]),
            "bc_top_patch_mae": summarize_scalar([r["truth_bc_top_patch_mae"] for r in rows]),
            "bc_top_offpatch_neumann_mae": summarize_scalar([r["truth_bc_top_offpatch_neumann_mae"] for r in rows]),
            "bc_bottom_sink_mae": summarize_scalar([r["truth_bc_bottom_sink_mae"] for r in rows]),
            "bc_side_neumann_mae": summarize_scalar([r["truth_bc_side_neumann_mae"] for r in rows]),
            "ic_interior_mae": summarize_scalar([r["truth_ic_interior_mae"] for r in rows]),
        },
    }
    # Summarize each stage metric as mean/p50/p90/max for compact reporting.
    for stage_name in stage_names:
        stage_summary = {}
        for metric_name in metric_names:
            metric_key = stage_name + "_" + metric_name
            stage_summary[metric_name] = summarize_scalar([r[metric_key] for r in rows])
        summary["stages"][stage_name] = stage_summary

    return rows, summary


def build_worst_case_report(rows, split_name, stage_name="stageC", top_n=10):
    if len(rows) == 0:
        return {
            "split": str(split_name),
            "top_n": int(max(1, int(top_n))),
            "stage_name": str(stage_name),
            "rankings": {},
        }

    stage = str(stage_name)
    suffixes = [
        "curve_relative_l2",
        "bc_bottom_flux_rel_rmse",
        "negative_flux_fraction",
    ]
    for optional in ("pde_stageA_flux_rel_rmse", "pde_mass_balance_rel_rmse"):
        if (stage + "_" + optional) in rows[0]:
            suffixes.append(optional)

    limit = max(1, int(top_n))

    def score_value(row, key):
        value = row.get(key, float("nan"))
        value = float(value)
        if np.isfinite(value):
            return value
        return float("-inf")

    report = {
        "split": str(split_name),
        "top_n": int(limit),
        "stage_name": stage,
        "rankings": {},
    }

    for suffix in suffixes:
        metric_key = stage + "_" + suffix
        # Higher values are worse for these diagnostics, so sort descending.
        ranked = sorted(rows, key=lambda row: score_value(row, metric_key), reverse=True)
        compact_rows = []
        for row in ranked[:limit]:
            compact = {
                "row_index": int(row["row_index"]),
                "run_dir": str(row["run_dir"]),
                "boundary_mode": str(row["boundary_mode"]),
            }
            for candidate_suffix in suffixes:
                key = stage + "_" + candidate_suffix
                compact[key] = float(row.get(key, float("nan")))
            compact_rows.append(compact)
        report["rankings"][metric_key] = compact_rows
    return report
