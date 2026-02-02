import argparse
from pathlib import Path

import numpy as np

from skin_diffusion.config import load_config
from skin_diffusion.grid import create_coords
from skin_diffusion.operators import step_constant_D
from skin_diffusion.run_utils import run_simulation, save_run_outputs
from skin_diffusion.solver import init_state, simulate_v1_no_bc
from skin_diffusion.utils import ensure_dir, set_seed


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

    # local import keeps CLI fast when demo_bc is unused
    from skin_diffusion.bc import apply_bc, patch_concentration

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

    if args.demo_bc:
        # demo uses a patch mask
        from skin_diffusion.bc import make_patch_mask
        patch_mask = make_patch_mask(
            cfg.grid.H,
            cfg.grid.W,
            cfg.boundary.patch_width,
            cfg.boundary.patch_offset,
        )
        _demo_bc(cfg, patch_mask)

    # init state for prints
    C0 = init_state(cfg.grid.H, cfg.grid.W)

    if args.no_bc:
        # init and run without BCs
        D_scalar = 1.0
        C_snap, t_save, diagnostics = simulate_v1_no_bc(
            C0, D_scalar, cfg.grid
        )
    else:
        # full run via centralized runner
        C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)
        save_run_outputs(
            out_dir,
            cfg,
            C_snap,
            t_save,
            D_field,
            k_field,
            patch_mask,
            diagnostics,
            metrics,
            stability_info,
        )

    if args.print_meta:
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            print(meta_path.read_text())

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("t_save shape:", t_save.shape)
    print("C0 shape:", C0.shape)
    print("C_snap shape:", C_snap.shape)
    if not args.no_bc:
        print("patch_mask true count:", int(patch_mask.sum()))
    print("loaded config:", cfg.regime_name)
    if not args.no_bc:
        print("wrote:", out_dir / "meta.json")


if __name__ == "__main__":
    main()
