import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skin_diffusion.bc import make_patch_mask, patch_concentration
from skin_diffusion.config import load_config
from skin_diffusion.grid import create_time
from skin_diffusion.layers import apply_correlated_heterogeneity, apply_iid_heterogeneity
from skin_diffusion.solver import init_state, simulate_v1
from skin_diffusion.utils import ensure_dir, write_json


def run_case(cfg, patch_width, patch_offset, D_field):
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
        C0, 1.0, cfg.grid, cfg.boundary, patch_mask, k=None, D_field=D_field
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

    # output data folder
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # patch options
    widths = [1.0, 0.5, 0.25]
    offsets = ["left", "center", "right"]

    # base D field (start with constant)
    D_field = np.full((cfg.grid.H, cfg.grid.W), 1.0, dtype=float)

    # optional iid heterogeneity
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

    # save D field view
    save_single_heatmap(fig_dir / "D_field.png", D_field, "D field")

    # run each width
    for width in widths:
        if width == 1.0:
            # full-width patch does not need left/center/right
            field = run_case(cfg, width, "center", D_field)
            name = "patch_width_1.00_center.png"
            save_single_heatmap(fig_dir / name, field, "patch width 1.0 (center)")
        else:
            # save one image per offset
            for offset in offsets:
                field = run_case(cfg, width, offset, D_field)
                name = f"patch_width_{width:.2f}_{offset}.png"
                title = f"patch width {width} ({offset})"
                save_single_heatmap(fig_dir / name, field, title)

    # lateral diffusion for small patch
    field_small = run_case(cfg, 0.25, "center", D_field)
    depth_idx = cfg.grid.H // 4
    save_lateral_profile(fig_dir / "lateral_profile_small_patch.png", field_small, depth_idx)

    # save patch concentration over saved times
    t_all, t_save_idx, t_save = create_time(
        cfg.grid.T, cfg.grid.dt, cfg.grid.save_every
    )
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

    print("saved figures in:", fig_dir)


if __name__ == "__main__":
    main()
