import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skin_diffusion.bc import make_patch_mask
from skin_diffusion.checks import l2_error
from skin_diffusion.config import GridConfig, load_config
from skin_diffusion.solver import init_state, simulate
from skin_diffusion.utils import ensure_dir, write_json


def run_case(cfg, H, W, dx, dt, save_every):
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
    C_snap, t_save, _ = simulate(
        C0, D_scalar, grid, cfg.boundary, patch_mask
    )

    return C_snap, t_save


def block_average_restrict(C_fine):
    # average 2x2 blocks so fine data matches coarse cells
    # C_fine is [T, Hf, Wf] and Hf/Wf must be even
    T, Hf, Wf = C_fine.shape
    Hc = Hf // 2
    Wc = Wf // 2
    reshaped = C_fine.reshape(T, Hc, 2, Wc, 2)
    return reshaped.mean(axis=(2, 4))


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # keep sizes modest so memory stays reasonable
    # increase later if have more RAM (e.g., add 256 or 512 here) ask timon about super computer
    sizes = [16, 32, 64, 128, 256]
    runs = {}

    # keep the same physical save interval on every grid
    save_interval = cfg.grid.save_every * cfg.grid.dt

    for H in sizes:
        W = H
        # keep the same physical size
        dx = cfg.grid.dx * (cfg.grid.H / H)
        # scale dt with dx^2 for a consistent explicit stability ratio
        dt = cfg.grid.dt * (dx / cfg.grid.dx) ** 2
        # keep physical save times aligned across grids
        save_every = int(round(save_interval / dt))
        if save_every < 1:
            save_every = 1
        C_snap, t_save = run_case(cfg, H, W, dx, dt, save_every)
        runs[H] = {
            "C": C_snap,
            "t": t_save,
            "dx": dx,
            "dt": dt,
        }

    # compare consecutive grid pairs
    pairs = list(zip(sizes[:-1], sizes[1:]))
    all_errors = {}
    all_times = {}

    for coarse, fine in pairs:
        C_coarse = runs[coarse]["C"]
        t_coarse = runs[coarse]["t"]
        C_restricted = block_average_restrict(runs[fine]["C"])

        n = min(len(t_coarse), len(C_restricted))
        errs = []
        for i in range(n):
            errs.append(l2_error(C_coarse[i], C_restricted[i]))

        label = f"{coarse}_vs_{fine}"
        all_errors[label] = errs
        all_times[label] = [float(t) for t in t_coarse[:n]]

    # plot error curves
    fig_dir = Path("figures") / "validation"
    ensure_dir(fig_dir)
    fig_path = fig_dir / "v1_convergence.png"
    plt.figure()
    for label in all_errors:
        plt.plot(all_times[label], all_errors[label], label=label.replace("_", " "))
    plt.xlabel("time")
    plt.ylabel("L2 error")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # summary numbers for quick evidence
    summary = {}
    for label, errs in all_errors.items():
        summary[label] = {
            "mean": float(np.mean(errs)),
            "final": float(errs[-1]),
            "max": float(np.max(errs)),
        }

    # write report json for the reporting
    report = {"timestamp": datetime.now().isoformat()}
    for label in all_errors:
        report[f"t_save_{label.split('_vs_')[0]}"] = all_times[label]
        report[f"errors_{label}"] = [float(e) for e in all_errors[label]]
    report["summary"] = summary
    for s in sizes:
        report[f"grid_{s}"] = {"H": s, "W": s, "dx": runs[s]["dx"], "dt": runs[s]["dt"]}

    report_dir = Path(cfg.output_dir) / "benchmark"
    ensure_dir(report_dir)
    report_path = report_dir / "report.json"
    write_json(report_path, report)

    print("saved:", fig_path)
    print("saved:", report_path)


if __name__ == "__main__":
    main()
