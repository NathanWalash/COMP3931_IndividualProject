import argparse
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.bc import make_patch_mask
from skin_diffusion.layers import build_D_field, build_k_field, build_layer_id
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

    print("saved figures in:", fig_dir)


if __name__ == "__main__":
    main()
