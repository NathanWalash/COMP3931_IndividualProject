"""
Inverse calibration: optimise layer D values so that simulated P and Tlag match the Adamiak-Giera et al. (2023) literature targets.

Uses a fast 1-D explicit FD solver (identical to the 2-D conservative scheme for laterally-uniform configs) inside a Nelder-Mead search over log10(D).  Final result verified against the full 2-D code at W=64.

Results (verified on full 2-D W=64 grid):
Config A  [1, 9, 54]:
    D  = [5.55e-10, 1.513e-08, 2.701e-07]  cm^2/s
    P  = 7.3798e-07 cm/s  (target 7.38e-07) -> error +0.003 %
    Tlag = 1.402 h        (target 1.401 h)  -> error +0.07 %

Config B  [2, 8, 54]:
    D  = [1.083e-09, 1.069e-07, 2.297e-07]  cm^2/s
    P  = 7.3822e-07 cm/s  (target 7.38e-07) -> error +0.03 %
    Tlag = 1.404 h        (target 1.401 h)  -> error +0.18 %

Config A chosen for the final YAML (layer_rows [1, 9, 54]).
"""

import io
import os
import sys
from copy import deepcopy

# silence tqdm before any import
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
from scipy.optimize import minimize

from skin_diffusion.config import load_config
from skin_diffusion.metrics import (compute_permeability, estimate_lag_time, estimate_steady_state_flux)
from skin_diffusion.run_utils import run_simulation

# Fast 1-D solver (matches 2-D conservative scheme for uniform x)
def solve_1d(H, dx, dt, T, save_every, D_values, layer_rows, C0=1.0):
    #1-D explicit conservative diffusion: C[0]=C0, C[-1]=0 (sink).
    n_steps = int(round(T / dt)) + 1

    # Build per-cell D array
    D = np.empty(H)
    start = 0
    for rows, d in zip(layer_rows, D_values):
        D[start:start + rows] = d
        start += rows

    # Harmonic-mean face diffusivities and pre-scaled Courant coefficients
    D_face = 2.0 * D[1:] * D[:-1] / (D[1:] + D[:-1])
    coeff = D_face * (dt / (dx * dx))

    C = np.zeros(H)
    n_save = n_steps // save_every + 2
    C_snap = np.empty((n_save, H))
    t_save = np.empty(n_save)
    si = 0

    step = 0
    while step < n_steps:
        C[0], C[-1] = C0, 0.0
        if step % save_every == 0:
            C_snap[si], t_save[si] = C, step * dt
            si += 1
        batch = min(((step // save_every) + 1) * save_every, n_steps - 1) - step
        if batch <= 0:
            step += 1
            continue
        for i in range(batch):
            dC = coeff * (C[:-1] - C[1:])
            C[1:-1] += dC[:-1] - dC[1:]
            C[0], C[-1] = C0, 0.0
        step += batch

    C_snap, t_save = C_snap[:si], t_save[:si]
    flux = D[-2] * C_snap[:, -2] / dx
    return C_snap, t_save, flux


def extract_scalars(flux, t_save, C0):
    #P and Tlag from a flux curve
    J_ss = estimate_steady_state_flux(flux, t_save)
    Tlag = estimate_lag_time(flux, t_save)
    P = compute_permeability(J_ss, C0)
    return P, Tlag


def eval_1d_fast(D_values, layer_rows, grid, C0=1.0):
    #Fast 1-D evaluation (dt=0.6 s, T=10 h) for the optimiser
    dt = min(0.6, 0.9 * grid.dx ** 2 / (4.0 * max(D_values)))
    _, t_save, flux = solve_1d( grid.H, grid.dx, dt, 36_000.0, 200, D_values, layer_rows, C0)
    return extract_scalars(flux, t_save, C0)


def eval_2d(cfg_base, D_values, layer_rows, W=64):
    #Verify with the full 2-D simulation code
    cfg = deepcopy(cfg_base)
    cfg.grid.W = W
    cfg.extras["layers"]["D_values"] = list(D_values)
    cfg.extras["layers"]["layer_rows"] = list(layer_rows)

    old = sys.stderr
    sys.stderr = io.StringIO()
    try:
        metrics = run_simulation(cfg)[6]
    finally:
        sys.stderr = old
    return metrics.get("P"), metrics.get("Tlag")


# Objective & optimisation
def make_objective(grid, layer_rows, P_target, Tlag_target_s, C0):
    dx = grid.dx

    def objective(log_D):
        D = [10 ** x for x in log_D]
        if max(D) >= dx ** 2 / (4.0 * 0.55):
            return 100.0
        P, Tlag = eval_1d_fast(D, layer_rows, grid, C0)
        if P is None or Tlag is None:
            return 100.0
        return (P / P_target - 1) ** 2 + (Tlag / Tlag_target_s - 1) ** 2

    return objective


def pct(sim, target):
    return 100.0 * (sim / target - 1.0)


def optimise_config(cfg, layer_rows, tag, P_target, Tlag_s):
    #Run Nelder-Mead for one layer configuration, verify on 2-D
    grid = cfg.grid
    C0 = cfg.boundary.C0
    log_D0 = np.log10(cfg.extras["layers"]["D_values"])

    obj = make_objective(grid, layer_rows, P_target, Tlag_s, C0)
    print(f"\n[{tag}]  layer_rows={layer_rows}")

    res = minimize(obj, log_D0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 0.002, "fatol": 1e-9})
    D_opt = [10 ** x for x in res.x]

    # Verify on full 2-D
    P_2d, Tlag_2d = eval_2d(cfg, D_opt, layer_rows)
    if P_2d and Tlag_2d:
        print(f"D = [{D_opt[0]:.3e}, {D_opt[1]:.3e}, {D_opt[2]:.3e}]")
        print(f"P = {P_2d:.4e}  ({pct(P_2d, P_target):+.3f} %)")
        print(f"Tlag = {Tlag_2d/3600:.3f} h  ({pct(Tlag_2d, Tlag_s):+.3f} %)")
    print(f"  nfev = {res.nfev}")
    return D_opt, P_2d, Tlag_2d

def main():
    cfg = load_config("configs/sim/v2_lidocaine_compare.yaml")
    lit = cfg.extras["literature"]
    P_target = lit["P_target_cm_s"]
    Tlag_s = lit["Tlag_target_hours"] * 3600.0

    print("=" * 55)
    print("Inverse calibration - Adamiak-Giera et al. (2023)")
    print(f"Targets: P = {P_target:.4e} cm/s,  Tlag = {lit['Tlag_target_hours']:.3f} h")
    print("=" * 55)

    optimise_config(cfg, [1, 9, 54], "A: 1-row SC", P_target, Tlag_s)
    optimise_config(cfg, [2, 8, 54], "B: 2-row SC", P_target, Tlag_s)


if __name__ == "__main__":
    main()
