import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.checks import stability_limit_diffusion
from skin_diffusion.config import load_config
from skin_diffusion.grid import create_coords
from skin_diffusion.operators import step_constant_D
from skin_diffusion.solver import init_state, simulate_v1_no_bc
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


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--demo_step", action="store_true")
    parser.add_argument("--print_meta", action="store_true")
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

    # init and run (no BC)
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    D_scalar = 1.0
    C_snap, t_save = simulate_v1_no_bc(C0, D_scalar, cfg.grid)

    # compute stability info
    dt_max = stability_limit_diffusion(D_scalar, cfg.grid.dx)

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
    meta["stability"] = {"dt": cfg.grid.dt, "dt_max": dt_max}

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
