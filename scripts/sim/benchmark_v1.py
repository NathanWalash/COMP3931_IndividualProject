import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.checks import l2_error
from skin_diffusion.config import GridConfig, load_config
from skin_diffusion.solver import init_state, simulate_v1
from skin_diffusion.utils import ensure_dir, write_json


def _run_case(cfg, H, W, dx, dt, save_every):
    # run one case at a given grid size
    # copy grid so we do not change the original
    grid = GridConfig(
        H=cfg.grid.H,
        W=cfg.grid.W,
        dx=cfg.grid.dx,
        dt=cfg.grid.dt,
        T=cfg.grid.T,
        save_every=cfg.grid.save_every,
    )
    grid.H = H
    grid.W = W
    grid.dx = dx
    grid.dt = dt
    grid.save_every = save_every

    # patch mask for BCs
    patch_mask = make_patch_mask(
        grid.H,
        grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # run sim
    C0 = init_state(grid.H, grid.W)
    D_scalar = 1.0
    C_snap, t_save, _ = simulate_v1(
        C0, D_scalar, grid, cfg.boundary, patch_mask
    )

    return C_snap, t_save


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # coarse run (baseline)
    H = cfg.grid.H
    W = cfg.grid.W
    dx = cfg.grid.dx
    dt = cfg.grid.dt
    save_every = cfg.grid.save_every

    Cc, t_save = _run_case(cfg, H, W, dx, dt, save_every)

    # fine run (dx/2, dt/4) to compare
    Hf = H * 2
    Wf = W * 2
    dxf = dx / 2.0
    dtf = dt / 4.0
    save_every_f = save_every * 4

    Cf, t_save_f = _run_case(cfg, Hf, Wf, dxf, dtf, save_every_f)

    # downsample fine to coarse grid (simple stride)
    Cf_down = Cf[:, ::2, ::2]

    # compute errors over time (L2)
    errors = []
    for i in range(len(t_save)):
        err = l2_error(Cc[i], Cf_down[i])
        errors.append(err)

    # plot error curve (smaller is better)
    fig_dir = Path("figures") / "validation"
    ensure_dir(fig_dir)
    fig_path = fig_dir / "v1_convergence.png"
    plt.figure()
    plt.plot(t_save, errors)
    plt.xlabel("time")
    plt.ylabel("L2 error")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # build report lists
    t_list = []
    for t in t_save:
        t_list.append(float(t))

    err_list = []
    for e in errors:
        err_list.append(float(e))

    # write report json for the thesis
    report = {
        "timestamp": datetime.now().isoformat(),
        "t_save": t_list,
        "errors": err_list,
        "coarse": {"H": H, "W": W, "dx": dx, "dt": dt},
        "fine": {"H": Hf, "W": Wf, "dx": dxf, "dt": dtf},
    }

    report_dir = Path(cfg.output_dir) / "benchmark"
    ensure_dir(report_dir)
    report_path = report_dir / "report.json"
    write_json(report_path, report)

    print("saved:", fig_path)
    print("saved:", report_path)


if __name__ == "__main__":
    main()
