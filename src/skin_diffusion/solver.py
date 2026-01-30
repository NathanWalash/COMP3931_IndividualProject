import numpy as np
from tqdm import tqdm

from skin_diffusion.bc import apply_bc, patch_concentration
from skin_diffusion.checks import check_stability, stability_limit_diffusion
from skin_diffusion.grid import create_time
from skin_diffusion.operators import step_constant_D, step_varD_conservative


def init_state(H, W):
    # start at zero everywhere
    return np.zeros((H, W), dtype=float)


def allocate_snapshots(Tsave, H, W):
    # store saved states
    return np.zeros((Tsave, H, W), dtype=float)


def simulate_v1_no_bc(C0, D_scalar, grid_cfg):
    # bc means boundary conditions
    # simple explicit loop without BCs
    t_all, t_save_idx, t_save = create_time(
        grid_cfg.T, grid_cfg.dt, grid_cfg.save_every
    )

    # check stability once
    dt_max = stability_limit_diffusion(D_scalar, grid_cfg.dx)
    check_stability(grid_cfg.dt, dt_max, mode="warn")

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
        if step % grid_cfg.save_every == 0:
            # store a snapshot
            C_snap[save_i] = C
            save_i += 1

        if step < len(t_all) - 1:
            # one diffusion step
            C = step_varD_conservative(C, D, grid_cfg.dt, grid_cfg.dx)

    return C_snap, t_save


def simulate_v1(C0, D_scalar, grid_cfg, bc_cfg, patch_mask):
    # full loop with BCs
    t_all, t_save_idx, t_save = create_time(
        grid_cfg.T, grid_cfg.dt, grid_cfg.save_every
    )

    # check stability once
    dt_max = stability_limit_diffusion(D_scalar, grid_cfg.dx)
    check_stability(grid_cfg.dt, dt_max, mode="warn")

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

        if step % grid_cfg.save_every == 0:
            # store a snapshot
            C_snap[save_i] = C
            save_i += 1

        if step < len(t_all) - 1:
            # one diffusion step
            C = step_varD_conservative(C, D, grid_cfg.dt, grid_cfg.dx)

        # re-apply BCs after step
        apply_bc(
            C,
            patch_mask,
            Cpatch,
            bottom_sink=(bc_cfg.bottom == "sink"),
            neumann_sides=(bc_cfg.sides == "neumann"),
            top_offpatch=bc_cfg.top_offpatch_mode,
        )

    return C_snap, t_save
