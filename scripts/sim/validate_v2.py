import argparse
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.layers import build_D_field, build_layer_id
from skin_diffusion.utils import ensure_dir
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

    # layer id
    layer_id = build_layer_id(cfg.grid.H, layer_rows)

    # build D field
    D_field = build_D_field(cfg.grid.H, cfg.grid.W, layer_id, D_values)

    # output folder
    fig_dir = Path("figures") / "validation" / "v2"
    ensure_dir(fig_dir)

    # save D plots
    heat_path = fig_dir / "D_map.png"
    prof_path = fig_dir / "D_depth_profile.png"
    for label in tqdm(["heatmap", "profile"], desc="saving figures"):
        if label == "heatmap":
            plot_heatmap(D_field, heat_path, title="D map")
        else:
            plot_depth_profile(D_field, prof_path)

    print("saved:", heat_path)
    print("saved:", prof_path)


if __name__ == "__main__":
    main()
