import argparse
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.bc import make_patch_mask, patch_concentration
from skin_diffusion.layers import build_D_field, build_k_field, build_layer_id
from skin_diffusion.metrics import (
    compute_bottom_flux,
    compute_permeability,
    estimate_lag_time,
    estimate_steady_state_flux,
)
from skin_diffusion.solver import init_state, simulate_v1
from skin_diffusion.utils import ensure_dir, write_json
from skin_diffusion.viz import plot_depth_profile, plot_heatmap


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # pull layers section from extras
    layers_cfg = cfg.extras.get("layers", {})
    layer_rows = layers_cfg.get("layer_rows", [])
    D_values = layers_cfg.get("D_values", [])
    k_dermis = layers_cfg.get("k_dermis", 0.0)

    # layer id
    layer_id = build_layer_id(cfg.grid.H, layer_rows)

    # build D and k fields
    D_field = build_D_field(cfg.grid.H, cfg.grid.W, layer_id, D_values)
    k_field = build_k_field(cfg.grid.H, cfg.grid.W, layer_rows[-1], k_dermis)

    # patch mask
    patch_mask = make_patch_mask(
        cfg.grid.H,
        cfg.grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # run sim with reaction
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    C_snap, t_save, diagnostics = simulate_v1(
        C0, 1.0, cfg.grid, cfg.boundary, patch_mask, k=k_field
    )

    # output folder
    fig_dir = Path("figures") / "validation" / "v2"
    ensure_dir(fig_dir)

    # save D and k plots
    for label in tqdm(["D_map", "k_map"], desc="saving figures"):
        if label == "D_map":
            plot_heatmap(D_field, fig_dir / "D_map.png", title="D map")
            plot_depth_profile(D_field, fig_dir / "D_depth_profile.png")
        else:
            plot_heatmap(k_field, fig_dir / "k_map.png", title="k map")
            plot_depth_profile(k_field, fig_dir / "k_depth_profile.png")

    # save diagnostics
    if diagnostics is not None:
        diag_path = Path(cfg.output_dir) / "diagnostics.json"
        write_json(diag_path, diagnostics)

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
    bc_path = Path(cfg.output_dir) / "bc.json"
    write_json(bc_path, {"t": t_save.tolist(), "Cpatch": cpatch_curve})

    # compute bottom flux curve and basic metrics
    flux_curve = compute_bottom_flux(C_snap, D_field, cfg.grid.dx)
    steady_flux = estimate_steady_state_flux(flux_curve, t_save)
    lag_time = estimate_lag_time(flux_curve, t_save)
    permeability = compute_permeability(steady_flux, cfg.boundary.C0)

    # store metrics
    metrics = {}
    metrics["J"] = []
    for val in flux_curve:
        metrics["J"].append(float(val))
    metrics["t"] = []
    for val in t_save:
        metrics["t"].append(float(val))
    metrics["J_ss"] = steady_flux
    metrics["Tlag"] = lag_time
    metrics["P"] = permeability

    metrics_path = Path(cfg.output_dir) / "metrics.json"
    write_json(metrics_path, metrics)

    print("saved figures in:", fig_dir)


if __name__ == "__main__":
    main()
