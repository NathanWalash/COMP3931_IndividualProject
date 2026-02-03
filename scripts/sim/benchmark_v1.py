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
    sizes = [16, 32, 64, 128]
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

    # compare 16 vs 32, 32 vs 64, and 64 vs 128
    C16 = runs[16]["C"]
    t16 = runs[16]["t"]
    C32 = runs[32]["C"]
    t32 = runs[32]["t"]
    C64 = runs[64]["C"]
    t64 = runs[64]["t"]
    C128 = runs[128]["C"]
    t128 = runs[128]["t"]

    # restrict fine grids by block averaging
    C16_from_32 = block_average_restrict(C32)
    C64_from_128 = block_average_restrict(C128)
    C32_from_64 = block_average_restrict(C64)

    # compute errors over time (L2)
    # use the shared time length for each pair
    n16_32 = min(len(t16), len(C16_from_32))
    n32_64 = min(len(t32), len(C32_from_64))
    n64_128 = min(len(t64), len(C64_from_128))
    errors_16_32 = []
    for i in range(n16_32):
        err = l2_error(C16[i], C16_from_32[i])
        errors_16_32.append(err)

    errors_32_64 = []
    for i in range(n32_64):
        err = l2_error(C32[i], C32_from_64[i])
        errors_32_64.append(err)

    errors_64_128 = []
    for i in range(n64_128):
        err = l2_error(C64[i], C64_from_128[i])
        errors_64_128.append(err)

    # plot both error curves
    fig_dir = Path("figures") / "validation"
    ensure_dir(fig_dir)
    fig_path = fig_dir / "v1_convergence.png"
    plt.figure()
    plt.plot(t16[:n16_32], errors_16_32, label="16 vs 32")
    plt.plot(t32[:n32_64], errors_32_64, label="32 vs 64")
    plt.plot(t64[:n64_128], errors_64_128, label="64 vs 128")
    plt.xlabel("time")
    plt.ylabel("L2 error")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # build report lists
    t_list_16 = []
    for t in t16[:n16_32]:
        t_list_16.append(float(t))
    t_list_32 = []
    for t in t32[:n32_64]:
        t_list_32.append(float(t))
    t_list_64 = []
    for t in t64[:n64_128]:
        t_list_64.append(float(t))

    err_list_16_32 = []
    for e in errors_16_32:
        err_list_16_32.append(float(e))
    err_list_32_64 = []
    for e in errors_32_64:
        err_list_32_64.append(float(e))
    err_list_64_128 = []
    for e in errors_64_128:
        err_list_64_128.append(float(e))

    # summary numbers for quick evidence
    summary = {
        "16_vs_32": {
            "mean": float(np.mean(errors_16_32)),
            "final": float(errors_16_32[-1]),
            "max": float(np.max(errors_16_32)),
        },
        "32_vs_64": {
            "mean": float(np.mean(errors_32_64)),
            "final": float(errors_32_64[-1]),
            "max": float(np.max(errors_32_64)),
        },
        "64_vs_128": {
            "mean": float(np.mean(errors_64_128)),
            "final": float(errors_64_128[-1]),
            "max": float(np.max(errors_64_128)),
        },
    }

    # write report json for the reporting
    report = {
        "timestamp": datetime.now().isoformat(),
        "t_save_16": t_list_16,
        "t_save_32": t_list_32,
        "t_save_64": t_list_64,
        "errors_16_vs_32": err_list_16_32,
        "errors_32_vs_64": err_list_32_64,
        "errors_64_vs_128": err_list_64_128,
        "summary": summary,
        "grid_16": {"H": 16, "W": 16, "dx": runs[16]["dx"], "dt": runs[16]["dt"]},
        "grid_32": {"H": 32, "W": 32, "dx": runs[32]["dx"], "dt": runs[32]["dt"]},
        "grid_64": {"H": 64, "W": 64, "dx": runs[64]["dx"], "dt": runs[64]["dt"]},
        "grid_128": {"H": 128, "W": 128, "dx": runs[128]["dx"], "dt": runs[128]["dt"]},
    }

    report_dir = Path(cfg.output_dir) / "benchmark"
    ensure_dir(report_dir)
    report_path = report_dir / "report.json"
    write_json(report_path, report)

    print("saved:", fig_path)
    print("saved:", report_path)


if __name__ == "__main__":
    main()
