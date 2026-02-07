import numpy as np
from datetime import datetime
from pathlib import Path

from skin_diffusion.bc import make_patch_mask, patch_concentration
from skin_diffusion.layers import (
    apply_correlated_heterogeneity,
    apply_iid_heterogeneity,
    build_D_field,
    build_k_field,
    build_layer_id,
)
from skin_diffusion.metrics import (
    compute_bottom_flux,
    compute_permeability,
    estimate_lag_time,
    estimate_steady_state_flux,
)
from skin_diffusion.solver import compute_stability_info, init_state, simulate
from skin_diffusion.utils import ensure_dir, write_json


def build_fields(cfg):
    # build patch mask
    patch_mask = make_patch_mask(
        cfg.grid.H,
        cfg.grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # default constant D and zero k
    D_field = np.full((cfg.grid.H, cfg.grid.W), 1.0, dtype=float)
    k_field = None

    # layered D and k (V2 style)
    layers_cfg = cfg.extras.get("layers", {})
    layer_rows = layers_cfg.get("layer_rows", [])
    D_values = layers_cfg.get("D_values", [])
    k_dermis = layers_cfg.get("k_dermis", 0.0)
    if layer_rows and D_values:
        layer_id = build_layer_id(cfg.grid.H, layer_rows)
        D_field = build_D_field(cfg.grid.H, cfg.grid.W, layer_id, D_values)
        k_field = build_k_field(cfg.grid.H, cfg.grid.W, layer_rows[-1], k_dermis)

    # heterogeneity on D (V3 style)
    het_cfg = cfg.extras.get("heterogeneity", {})
    sigma = float(het_cfg.get("sigma", 0.0))
    seed = int(het_cfg.get("seed", cfg.seed))
    D_min = float(het_cfg.get("D_min", 0.001))
    D_max = float(het_cfg.get("D_max", 1.0))
    mode = het_cfg.get("mode", "iid")
    steps = int(het_cfg.get("steps", 5))
    if sigma > 0.0:
        if mode == "correlated":
            D_field = apply_correlated_heterogeneity(
                D_field, sigma, seed, D_min, D_max, steps
            )
        else:
            D_field = apply_iid_heterogeneity(D_field, sigma, seed, D_min, D_max)

    return D_field, k_field, patch_mask


def run_simulation(cfg):
    # build fields from config
    D_field, k_field, patch_mask = build_fields(cfg)

    # run sim
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    C_snap, t_save, diagnostics = simulate(
        C0, 1.0, cfg.grid, cfg.boundary, patch_mask, k=k_field, D_field=D_field
    )

    # metrics
    # bottom flux is the main curve we use for later summaries
    flux_curve = compute_bottom_flux(C_snap, D_field, cfg.grid.dx)
    steady_flux = estimate_steady_state_flux(flux_curve, t_save)
    lag_time = estimate_lag_time(flux_curve, t_save)
    permeability = compute_permeability(steady_flux, cfg.boundary.C0)

    metrics = {}
    # store series in plain lists for json
    metrics["J"] = []
    for val in flux_curve:
        metrics["J"].append(float(val))
    metrics["t"] = []
    for val in t_save:
        metrics["t"].append(float(val))
    metrics["J_ss"] = steady_flux
    metrics["Tlag"] = lag_time
    metrics["P"] = permeability

    # stability info
    stability_info = compute_stability_info(D_field, k_field, cfg.grid)

    return C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info


def save_run_outputs(out_dir, cfg, C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info):
    # make output folder
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # always store a full k field for the npz
    if k_field is None:
        k_save = np.zeros_like(D_field)
    else:
        k_save = k_field

    # make sure we have J and t for a consistent run bundle
    if metrics is None or "J" not in metrics:
        J = compute_bottom_flux(C_snap, D_field, cfg.grid.dx)
        t = t_save
    else:
        J = np.array(metrics["J"], dtype=float)
        t = np.array(metrics.get("t", t_save), dtype=float)

    # save fields (same schema as dataset runs)
    fields_path = out_dir / "fields.npz"
    np.savez(
        fields_path,
        C_snap=C_snap,
        D=D_field,
        k=k_save,
        patch_mask=patch_mask,
        t=t,
        J=J,
    )

    # build metadata
    # write grid and boundary as plain dicts
    grid_dict = {}
    for key, val in cfg.grid.__dict__.items():
        grid_dict[key] = val
    boundary_dict = {}
    for key, val in cfg.boundary.__dict__.items():
        boundary_dict[key] = val

    meta = {}
    meta["timestamp"] = datetime.now().isoformat()
    meta["config"] = {
        "seed": cfg.seed,
        "output_dir": cfg.output_dir,
        "regime_name": cfg.regime_name,
        "grid": grid_dict,
        "boundary": boundary_dict,
        "extras": cfg.extras,
    }
    meta["t_save"] = t_save.tolist()
    meta["stability"] = stability_info

    # save metadata
    meta_path = out_dir / "meta.json"
    write_json(meta_path, meta)

    # save diagnostics if available
    if diagnostics is not None:
        diag_path = out_dir / "diagnostics.json"
        write_json(diag_path, diagnostics)

    # save metrics if available
    if metrics is not None:
        metrics_path = out_dir / "metrics.json"
        write_json(metrics_path, metrics)

    # save patch concentration over saved times
    cpatch_curve = []
    for t in t_save:
        cpatch_curve.append(
            float(
                patch_concentration(
                    float(t),
                    cfg.boundary.mode,
                    cfg.boundary.C0,
                    cfg.boundary.decay_rate,
                )
            )
        )
    bc_path = out_dir / "bc.json"
    write_json(bc_path, {"t": t_save.tolist(), "Cpatch": cpatch_curve})

    return fields_path, meta_path
