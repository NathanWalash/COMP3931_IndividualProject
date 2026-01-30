import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.config import load_config
from skin_diffusion.solver import init_state, simulate_v1
from skin_diffusion.utils import ensure_dir, write_json
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

    # make patch mask for the donor patch
    patch_mask = make_patch_mask(
        cfg.grid.H,
        cfg.grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # run sim (with BC)
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    D_scalar = 1.0
    C_snap, t_save = simulate_v1(C0, D_scalar, cfg.grid, cfg.boundary, patch_mask)

    # save fields for later plots/analysis
    fields_path = out_dir / "fields.npz"
    np.savez(fields_path, C_snap=C_snap, t_save=t_save, patch_mask=patch_mask)

    # save meta info for this run
    meta = {
        "run_id": run_id,
        "config": {
            "seed": cfg.seed,
            "output_dir": cfg.output_dir,
            "regime_name": cfg.regime_name,
            "grid": cfg.grid.__dict__,
            "boundary": cfg.boundary.__dict__,
            "extras": cfg.extras,
        },
    }
    meta_path = out_dir / "meta.json"
    write_json(meta_path, meta)

    # pick 3 times to plot
    idxs = [0, len(t_save) // 2, len(t_save) - 1]
    for i in tqdm(idxs, desc="saving figures"):
        C = C_snap[i]
        t = t_save[i]
        heat_path = fig_dir / f"heat_t{i:04d}.png"
        prof_path = fig_dir / f"profile_t{i:04d}.png"
        plot_heatmap(C, heat_path, title=f"t={t:.4f}")
        plot_depth_profile(C, prof_path)

    # save mask image for sanity check
    mask_path = fig_dir / "patch_mask.png"
    plot_mask(patch_mask, mask_path)

    print("saved fields:", fields_path)
    print("saved meta:", meta_path)
    print("figures in:", fig_dir)


if __name__ == "__main__":
    main()
