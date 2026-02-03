import argparse
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.run_utils import run_simulation, save_run_outputs
from skin_diffusion.utils import ensure_dir
from skin_diffusion.viz import plot_depth_profile, plot_heatmap


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # run sim (with layered D and k)
    C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)

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

    # save time series heatmaps and profiles
    idxs = [0, len(t_save) // 2, len(t_save) - 1]
    # shared color scale for fair comparison
    vmax = float(C_snap.max())
    for i in tqdm(idxs, desc="saving time series"):
        C = C_snap[i]
        t = t_save[i]
        heat_path = fig_dir / f"heat_t{i:04d}.png"
        prof_path = fig_dir / f"profile_t{i:04d}.png"
        plot_heatmap(C, heat_path, title=f"t={t:.4f}", vmin=0.0, vmax=vmax)
        plot_depth_profile(C, prof_path)

    # save outputs
    save_run_outputs(
        Path(cfg.output_dir),
        cfg,
        C_snap,
        t_save,
        D_field,
        k_field,
        patch_mask,
        diagnostics,
        metrics,
        stability_info,
    )

    print("saved figures in:", fig_dir)


if __name__ == "__main__":
    main()
