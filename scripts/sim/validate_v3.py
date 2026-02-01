import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.config import load_config
from skin_diffusion.solver import init_state, simulate_v1
from skin_diffusion.utils import ensure_dir


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
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    C_snap, t_save, diagnostics = simulate_v1(
        C0, 1.0, cfg.grid, cfg.boundary, patch_mask, k=None
    )

    # return final field
    return C_snap[-1]


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

    # patch options
    widths = [1.0, 0.5, 0.25]
    offsets = ["left", "center", "right"]

    # run each width
    for width in widths:
        if width == 1.0:
            # full-width patch does not need left/center/right
            field = run_case(cfg, width, "center")
            name = "patch_width_1.00_center.png"
            save_single_heatmap(fig_dir / name, field, "patch width 1.0 (center)")
        else:
            # save one image per offset
            for offset in offsets:
                field = run_case(cfg, width, offset)
                name = f"patch_width_{width:.2f}_{offset}.png"
                title = f"patch width {width} ({offset})"
                save_single_heatmap(fig_dir / name, field, title)

    # lateral diffusion for small patch
    field_small = run_case(cfg, 0.25, "center")
    depth_idx = cfg.grid.H // 4
    save_lateral_profile(fig_dir / "lateral_profile_small_patch.png", field_small, depth_idx)

    print("saved figures in:", fig_dir)


if __name__ == "__main__":
    main()
