import time
from pathlib import Path

import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.config import load_config, GridConfig
from skin_diffusion.solver import init_state, simulate
from skin_diffusion.utils import ensure_dir


def run_case(cfg, H, W, dx, dt, save_every):
    # make a grid copy so we do not edit the original config
    grid = GridConfig(
        H=cfg.grid.H,
        W=cfg.grid.W,
        dx=cfg.grid.dx,
        dt=cfg.grid.dt,
        T=cfg.grid.T,
        save_every=cfg.grid.save_every,
    )
    # override the bits we are testing
    grid.H = H
    grid.W = W
    grid.dx = dx
    grid.dt = dt
    grid.save_every = save_every

    # patch mask for the top boundary
    patch_mask = make_patch_mask(
        grid.H,
        grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # run one sim and return the arrays
    C0 = init_state(grid.H, grid.W)
    D_scalar = 1.0
    C_snap, t_save, _ = simulate(
        C0, D_scalar, grid, cfg.boundary, patch_mask
    )

    return C_snap, t_save, grid


def main():
    # load a baseline config for settings
    cfg = load_config("configs/sim/v1_baseline.yaml")

    # test a few grid sizes
    sizes = [32, 64, 128]
    results = []

    for H in sizes:
        W = H
        # keep the physical size the same by scaling dx
        dx = cfg.grid.dx * (cfg.grid.H / H)
        # keep explicit stability by scaling dt with dx^2
        dt = cfg.grid.dt * (dx / cfg.grid.dx) ** 2
        save_every = cfg.grid.save_every

        # time the run
        t0 = time.time()
        C_snap, t_save, grid = run_case(cfg, H, W, dx, dt, save_every)
        t1 = time.time()

        # timing summary
        steps = int(grid.T / grid.dt) + 1
        seconds = t1 - t0
        sec_per_step = seconds / steps

        # keep one row per grid size
        results.append(
            {
                "H": H,
                "W": W,
                "dx": dx,
                "dt": dt,
                "steps": steps,
                "seconds": seconds,
                "sec_per_step": sec_per_step,
            }
        )

    # write csv so we can compare later
    out_dir = Path("outputs") / "sim" / "profile"
    ensure_dir(out_dir)
    csv_path = out_dir / "runtime.csv"

    header = "H,W,dx,dt,steps,seconds,sec_per_step\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for r in results:
            f.write(
                f"{r['H']},{r['W']},{r['dx']},{r['dt']},{r['steps']},"
                f"{r['seconds']},{r['sec_per_step']}\n"
            )

    # print table for quick view
    print("runtime table")
    for r in results:
        print(
            f"H={r['H']} W={r['W']} steps={r['steps']} "
            f"sec={r['seconds']:.3f} sec/step={r['sec_per_step']:.6f}"
        )
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
