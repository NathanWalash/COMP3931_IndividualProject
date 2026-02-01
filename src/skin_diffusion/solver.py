import numpy as np
from tqdm import tqdm

from skin_diffusion.bc import apply_bc, patch_concentration
from skin_diffusion.checks import (
    diagnostics_over_time,
    check_reaction_stability,
    check_stability,
    stability_limit_diffusion,
)
from skin_diffusion.grid import create_time
from skin_diffusion.operators import step_constant_D, step_varD_conservative


def init_state(H, W):
    # start at zero everywhere
    return np.zeros((H, W), dtype=float)


def allocate_snapshots(Tsave, H, W):
    # store saved states
    return np.zeros((Tsave, H, W), dtype=float)


def apply_reaction(C, k, dt):
    # simple explicit reaction step
    # this is the k*C loss term
    return C - dt * k * C


def simulate_v1_no_bc(C0, D_scalar, grid_cfg, k=None):
    # bc means boundary conditions
    # simple explicit loop without BCs
    t_all, t_save_idx, t_save = create_time(
        grid_cfg.T, grid_cfg.dt, grid_cfg.save_every
    )

    # check stability once
    dt_max = stability_limit_diffusion(D_scalar, grid_cfg.dx)
    check_stability(grid_cfg.dt, dt_max, mode="warn")

    # reaction stability
    if k is not None:
        kmax = float(np.max(k))
        check_reaction_stability(grid_cfg.dt, kmax, mode="warn")

    # start from the initial state
    C = C0.copy()
    # allocate saved frames
    C_snap = allocate_snapshots(len(t_save), grid_cfg.H, grid_cfg.W)

    # constant D field for varD step
    D = np.full((grid_cfg.H, grid_cfg.W), D_scalar, dtype=float)

    # save every save_every steps
    save_i = 0
    steps = tqdm(range(len(t_all)), desc="simulate_v1_no_bc")
    for step in steps:
        # snapshot before step
        if step % grid_cfg.save_every == 0:
            # store a snapshot
            C_snap[save_i] = C
            save_i += 1

        if step < len(t_all) - 1:
            # one diffusion step
            C = step_varD_conservative(C, D, grid_cfg.dt, grid_cfg.dx)

            # reaction step (optional)
            if k is not None:
                C = apply_reaction(C, k, grid_cfg.dt)

    diagnostics = diagnostics_over_time(C_snap, grid_cfg.dx)
    return C_snap, t_save, diagnostics


def simulate_v1(C0, D_scalar, grid_cfg, bc_cfg, patch_mask, k=None):
    # full loop with BCs
    t_all, t_save_idx, t_save = create_time(
        grid_cfg.T, grid_cfg.dt, grid_cfg.save_every
    )

    # check stability once
    dt_max = stability_limit_diffusion(D_scalar, grid_cfg.dx)
    check_stability(grid_cfg.dt, dt_max, mode="warn")

    # reaction stability
    if k is not None:
        kmax = float(np.max(k))
        check_reaction_stability(grid_cfg.dt, kmax, mode="warn")

    # start from the initial state
    C = C0.copy()
    # allocate saved frames
    C_snap = allocate_snapshots(len(t_save), grid_cfg.H, grid_cfg.W)

    # constant D field for varD step
    D = np.full((grid_cfg.H, grid_cfg.W), D_scalar, dtype=float)

    # save every save_every steps
    save_i = 0
    steps = tqdm(range(len(t_all)), desc="simulate_v1")
    for step in steps:
        # current time
        t = t_all[step]

        # patch value for this time
        Cpatch = patch_concentration(t, bc_cfg.mode, bc_cfg.C0, bc_cfg.decay_rate)

        # apply BCs before step
        apply_bc(
            C,
            patch_mask,
            Cpatch,
            bottom_sink=(bc_cfg.bottom == "sink"),
            neumann_sides=(bc_cfg.sides == "neumann"),
            top_offpatch=bc_cfg.top_offpatch_mode,
        )

        # snapshot before step
        if step % grid_cfg.save_every == 0:
            # store a snapshot
            C_snap[save_i] = C
            save_i += 1

        if step < len(t_all) - 1:
            # one diffusion step
            C = step_varD_conservative(C, D, grid_cfg.dt, grid_cfg.dx)

            # reaction step (optional)
            if k is not None:
                C = apply_reaction(C, k, grid_cfg.dt)

        # re-apply BCs after step
        apply_bc(
            C,
            patch_mask,
            Cpatch,
            bottom_sink=(bc_cfg.bottom == "sink"),
            neumann_sides=(bc_cfg.sides == "neumann"),
            top_offpatch=bc_cfg.top_offpatch_mode,
        )

    diagnostics = diagnostics_over_time(C_snap, grid_cfg.dx)
    return C_snap, t_save, diagnostics


def compute_stability_info(D, k, grid_cfg):
    # collect stability info for metadata
    info = {}

    # diffusion max and limit
    Dmax = float(np.max(D))
    info["Dmax"] = Dmax
    info["dt_diff_max"] = stability_limit_diffusion(Dmax, grid_cfg.dx)

    # reaction max and limit
    if k is None:
        info["kmax"] = 0.0
        info["dt_react_max"] = None
    else:
        kmax = float(np.max(k))
        info["kmax"] = kmax
        ok, dt_react_max = check_reaction_stability(grid_cfg.dt, kmax, mode="warn")
        info["dt_react_max"] = dt_react_max

    return info
