import argparse
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.run_utils import run_simulation, save_run_outputs
from skin_diffusion.utils import ensure_dir
from skin_diffusion.viz import plot_depth_profile, plot_heatmap, plot_mask


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # output folders
    # heatmaps = 2D concentration images at a few times
    # profiles = depth curves (x-averaged) at the same times
    # patch mask = where the donor sits on the top boundary
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / run_id
    fig_dir = Path("figures") / "validation" / "v1"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    # run sim (with BC)
    C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)

    # save run outputs
    save_run_outputs(
        out_dir,
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

    # pick 3 times to plot
    idxs = [0, len(t_save) // 2, len(t_save) - 1]
    # shared color scale for fair comparison
    vmax = float(C_snap.max())
    for i in tqdm(idxs, desc="saving figures"):
        C = C_snap[i]
        t = t_save[i]
        heat_path = fig_dir / f"heat_t{i:04d}.png"
        prof_path = fig_dir / f"profile_t{i:04d}.png"
        plot_heatmap(C, heat_path, title=f"t={t:.4f}", vmin=0.0, vmax=vmax)
        plot_depth_profile(C, prof_path)

    # save mask image for sanity check
    mask_path = fig_dir / "patch_mask.png"
    plot_mask(patch_mask, mask_path)

    print("saved fields:", out_dir / "fields.npz")
    print("saved meta:", out_dir / "meta.json")
    print("figures in:", fig_dir)


if __name__ == "__main__":
    main()
