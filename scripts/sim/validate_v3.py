import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.config import load_config
from skin_diffusion.run_utils import run_simulation, save_run_outputs
from skin_diffusion.utils import ensure_dir
from skin_diffusion.viz import plot_depth_profile, plot_heatmap


def run_case(cfg, patch_width, patch_offset):
    # set patch settings for this run
    cfg.boundary.patch_width = patch_width
    cfg.boundary.patch_offset = patch_offset

    # patch mask
    patch_mask = make_patch_mask(
        cfg.grid.H,
        cfg.grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # run sim
    C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)

    # return full series
    return C_snap, t_save


def save_single_heatmap(fig_path, field, title):
    # single heatmap for full-width patch
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(field, vmin=0.0, vmax=float(field.max()))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_lateral_profile(fig_path, field, depth_idx):
    # lateral profile at a chosen depth
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(field[depth_idx, :])
    ax.set_xlabel("x index")
    ax.set_ylabel("C")
    ax.set_title("lateral profile")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # output folder
    fig_dir = Path("figures") / "validation" / "v3"
    ensure_dir(fig_dir)

    # output data folder
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # patch options
    widths = [1.0, 0.5, 0.25]
    offsets = ["left", "center", "right"]

    # build one full run to get fields and outputs
    C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)

    # save D field view
    save_single_heatmap(fig_dir / "D_field.png", D_field, "D field")

    # save outputs
    out_dir = Path(cfg.output_dir)
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

    # run each width
    for width in widths:
        if width == 1.0:
            # full-width patch does not need left/center/right
            C_snap, t_save = run_case(cfg, width, "center")
            field = C_snap[-1]
            name = "patch_width_1.00_center.png"
            save_single_heatmap(fig_dir / name, field, "patch width 1.0 (center)")
        else:
            # save one image per offset
            for offset in offsets:
                C_snap, t_save = run_case(cfg, width, offset)
                field = C_snap[-1]
                name = f"patch_width_{width:.2f}_{offset}.png"
                title = f"patch width {width} ({offset})"
                save_single_heatmap(fig_dir / name, field, title)

    # lateral diffusion for small patch
    C_snap_small, t_save_small = run_case(cfg, 0.25, "center")
    field_small = C_snap_small[-1]
    depth_idx = cfg.grid.H // 4
    save_lateral_profile(fig_dir / "lateral_profile_small_patch.png", field_small, depth_idx)

    # save time series heatmaps and profiles for one case
    idxs = [0, len(t_save_small) // 2, len(t_save_small) - 1]
    # shared color scale for fair comparison
    vmax = float(C_snap_small.max())
    for i in idxs:
        C = C_snap_small[i]
        t = t_save_small[i]
        heat_path = fig_dir / f"heat_t{i:04d}.png"
        prof_path = fig_dir / f"profile_t{i:04d}.png"
        plot_heatmap(C, heat_path, title=f"t={t:.4f}", vmin=0.0, vmax=vmax)
        plot_depth_profile(C, prof_path)

    print("saved figures in:", fig_dir)


if __name__ == "__main__":
    main()
