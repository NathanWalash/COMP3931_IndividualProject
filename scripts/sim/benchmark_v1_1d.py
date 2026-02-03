import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.checks import l2_error
from skin_diffusion.config import load_config
from skin_diffusion.solver import init_state, simulate
from skin_diffusion.utils import ensure_dir, write_json


# analytic 1D solution: finite slab with fixed boundary concentrations (Crank 1975)
def analytic_profile(depth_idx, t, cfg):
    # depth_idx can be an array; return array of same length
    z = depth_idx * cfg.grid.dx
    C0 = cfg.boundary.C0
    D = 1.0  # must match the scalar D used in this benchmark run

    if t <= 0.0:
        # initial condition: zero everywhere except the surface cell
        return np.where(depth_idx == 0, C0, 0.0)

    L = (cfg.grid.H - 1) * cfg.grid.dx
    if L <= 0.0:
        return np.zeros_like(z, dtype=float)

    # series setup
    # n_terms controls accuracy; 200 is plenty for the grid sizes we use.
    n_terms = 200
    n = np.arange(1, n_terms + 1, dtype=float)

    # Each term is sin(n*pi*z/L) * exp(-n^2*pi^2*D*t/L^2) / n
    # Compute all terms at once for speed and clarity.
    nz = (n[:, None] * np.pi * z[None, :]) / L
    decay = np.exp(-(n[:, None] ** 2) * (np.pi ** 2) * D * t / (L ** 2))
    series_sum = (np.sin(nz) * decay / n[:, None]).sum(axis=0)

    # steady state for fixed top/bottom concentrations is linear in depth
    steady = C0 * (1.0 - z / L)
    # transient decays over time; subtract it from the steady profile
    transient = (2.0 * C0 / np.pi) * series_sum
    return steady - transient


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
    C_snap, t_save, _ = simulate(
        C0, D_scalar, cfg.grid, cfg.boundary, patch_mask
    )

    # compare profiles at a few times
    # we pick start, middle, end of the run
    # profile = x-averaged concentration vs depth
    idxs = [0, len(t_save) // 2, len(t_save) - 1]
    errors = []
    for i in idxs:
        # time for this snapshot
        t = float(t_save[i])
        # get depth profile
        profile = C_snap[i].mean(axis=1)
        # analytic target from Crank's erfc solution
        analytic = analytic_profile(np.arange(cfg.grid.H), t, cfg)
        # l2 error between numeric and analytic profile
        errors.append(l2_error(profile, analytic))

    # build report lists
    t_list = []
    for i in idxs:
        t_list.append(float(t_save[i]))

    # write report
    report = {
        "timestamp": datetime.now().isoformat(),
        "note": "finite slab, fixed C0 at top and sink at bottom (Crank 1975)",
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
