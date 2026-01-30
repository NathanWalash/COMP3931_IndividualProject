import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.checks import l2_error
from skin_diffusion.config import load_config
from skin_diffusion.solver import init_state, simulate_v1
from skin_diffusion.utils import ensure_dir, write_json


# placeholder for analytic 1D solution
# fill this in later (Crank series)
# should return a profile with length H

def analytic_profile(depth_idx, t, cfg):
    # TODO: implement 1D series solution from Crank
    return None


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # force full-width patch for 1D behavior
    cfg.boundary.patch_width = 1.0
    cfg.boundary.patch_offset = "center"

    # make patch mask
    patch_mask = make_patch_mask(
        cfg.grid.H,
        cfg.grid.W,
        cfg.boundary.patch_width,
        cfg.boundary.patch_offset,
    )

    # run sim
    C0 = init_state(cfg.grid.H, cfg.grid.W)
    D_scalar = 1.0
    C_snap, t_save = simulate_v1(C0, D_scalar, cfg.grid, cfg.boundary, patch_mask)

    # compare profiles at a few times
    # we pick start, middle, end of the run
    # profile = x-averaged concentration vs depth
    # analytic_profile is a placeholder for now
    idxs = [0, len(t_save) // 2, len(t_save) - 1]
    errors = []
    for i in idxs:
        # time for this snapshot
        t = float(t_save[i])
        # get depth profile
        profile = C_snap[i].mean(axis=1)
        # analytic target (to be filled in)
        analytic = analytic_profile(np.arange(cfg.grid.H), t, cfg)
        if analytic is None:
            errors.append(None)
        else:
            # l2 error between numeric and analytic profile
            errors.append(l2_error(profile, analytic))

    # build report lists
    t_list = []
    for i in idxs:
        t_list.append(float(t_save[i]))

    # write report
    report = {
        "timestamp": datetime.now().isoformat(),
        "note": "analytic series placeholder (fill in later)",
        "times": t_list,
        "errors": errors,
        "grid": {
            "H": cfg.grid.H,
            "W": cfg.grid.W,
            "dx": cfg.grid.dx,
            "dt": cfg.grid.dt,
        },
    }

    # write report json
    report_dir = Path(cfg.output_dir) / "benchmark"
    ensure_dir(report_dir)
    report_path = report_dir / "report_1d.json"
    write_json(report_path, report)

    print("saved:", report_path)


if __name__ == "__main__":
    main()
