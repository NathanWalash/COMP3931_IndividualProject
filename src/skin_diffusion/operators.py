import numpy as np


# 5-point laplacian on interior only
# boundaries are left as zeros
# this is for constant D tests
# based on standard finite differences
# see Crank or Strikwerda

def laplacian_5pt(C, dx):
    # interior 5-point laplacian
    L = np.zeros_like(C)
    inv_dx2 = 1.0 / (dx * dx)
    up = C[:-2, 1:-1]
    down = C[2:, 1:-1]
    left = C[1:-1, :-2]
    right = C[1:-1, 2:]
    center = C[1:-1, 1:-1]
    L[1:-1, 1:-1] = (up + down + left + right - 4.0 * center) * inv_dx2
    return L


# one explicit step with constant D
# no boundary conditions here

def step_constant_D(C, D_scalar, dt, dx):
    # one explicit step for constant D
    L = laplacian_5pt(C, dx)
    return C + dt * D_scalar * L


# harmonic mean for faces
# keeps flux continuous at jumps

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b)


# conservative variable-D step
# uses face fluxes then divergence
# boundaries are left untouched

def step_varD_conservative(C, D, dt, dx):
    # face diffusion in x
    Dx_plus = harmonic_mean(D[:, 1:], D[:, :-1])
    # face diffusion in y
    Dy_plus = harmonic_mean(D[1:, :], D[:-1, :])

    # fluxes on faces
    Jx = -Dx_plus * (C[:, 1:] - C[:, :-1]) / dx
    Jy = -Dy_plus * (C[1:, :] - C[:-1, :]) / dx

    # divergence (interior only)
    div = np.zeros_like(C)
    div[:, 1:-1] += (Jx[:, :-1] - Jx[:, 1:]) / dx
    div[1:-1, :] += (Jy[:-1, :] - Jy[1:, :]) / dx

    return C + dt * div
