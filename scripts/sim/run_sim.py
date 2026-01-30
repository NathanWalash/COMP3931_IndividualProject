import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from skin_diffusion.bc import apply_bc, make_patch_mask, patch_concentration
from skin_diffusion.checks import stability_limit_diffusion
from skin_diffusion.config import load_config
from skin_diffusion.grid import create_coords
from skin_diffusion.operators import step_constant_D
from skin_diffusion.solver import compute_stability_info, init_state, simulate_v1, simulate_v1_no_bc
from skin_diffusion.utils import ensure_dir, set_seed, write_json


def _demo_step():
    # quick demo for constant D step
    H = 9
    W = 9
    dx = 1.0
    dt = 0.1
    D = 1.0
    C = np.zeros((H, W), dtype=float)
    C[H // 2, W // 2] = 1.0
    C2 = step_constant_D(C, D, dt, dx)
    mid = H // 2
    print("demo center:", C2[mid, mid])
    print("demo up/down:", C2[mid - 1, mid], C2[mid + 1, mid])
    print("demo left/right:", C2[mid, mid - 1], C2[mid, mid + 1])


def _demo_bc(cfg, patch_mask):
    # quick demo for BCs
    C = np.zeros((cfg.grid.H, cfg.grid.W), dtype=float)

    # build patch value at t=0
    Cpatch = patch_concentration(
        0.0,
        cfg.boundary.mode,
        cfg.boundary.C0,
        cfg.boundary.decay_rate,
    )

    # apply BCs to empty field
    apply_bc(
        C,
        patch_mask,
        Cpatch,
        bottom_sink=True,
        neumann_sides=True,
        top_offpatch="neumann",
    )

    # check top and bottom rows
    top_row = C[0]
    print("bc top min/max:", float(top_row.min()), float(top_row.max()))
    print("bc bottom max:", float(C[-1].max()))


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--demo_step", action="store_true")
    parser.add_argument("--demo_bc", action="store_true")
    parser.add_argument("--print_meta", action="store_true")
    parser.add_argument("--no_bc", action="store_true")
    args = parser.parse_args()

    if args.demo_step:
        _demo_step()

    # load config
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # make output folder
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # build coords
    x, y = create_coords(cfg.grid.H, cfg.grid.W, cfg.grid.dx)

    # make patch mask
    patch_mask = make_patch_mask(
        cfg.grid.H,
        cfg.grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    if args.demo_bc:
        _demo_bc(cfg, patch_mask)

    # init and run
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    D_scalar = 1.0
    if args.no_bc:
        C_snap, t_save = simulate_v1_no_bc(C0, D_scalar, cfg.grid)
    else:
        C_snap, t_save = simulate_v1(C0, D_scalar, cfg.grid, cfg.boundary, patch_mask)

    # compute stability info
    D_field = np.full((cfg.grid.H, cfg.grid.W), D_scalar, dtype=float)
    stability_info = compute_stability_info(D_field, None, cfg.grid)

    # build simple metadata
    meta = {}
    meta["timestamp"] = datetime.now().isoformat()
    meta["config"] = {
        "seed": cfg.seed,
        "output_dir": cfg.output_dir,
        "regime_name": cfg.regime_name,
        "grid": cfg.grid.__dict__,
        "boundary": cfg.boundary.__dict__,
        "extras": cfg.extras,
    }
    meta["t_save"] = t_save.tolist()
    meta["stability"] = stability_info

    # save metadata
    meta_path = out_dir / "meta.json"
    write_json(meta_path, meta)

    if args.print_meta:
        print(meta)

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("t_save shape:", t_save.shape)
    print("C0 shape:", C0.shape)
    print("C_snap shape:", C_snap.shape)
    print("patch_mask true count:", int(patch_mask.sum()))
    print("loaded config:", cfg.regime_name)
    print("wrote:", meta_path)


if __name__ == "__main__":
    main()
