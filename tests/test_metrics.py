import numpy as np

from skin_diffusion.metrics import compute_bottom_flux, compute_permeability


def test_bottom_flux_zero_field():
    # zero field should give zero flux
    C_snap = np.zeros((3, 4, 5), dtype=float)
    J = compute_bottom_flux(C_snap, 1.0, 0.1)
    assert np.allclose(J, 0.0)


def test_bottom_flux_linear_profile():
    # linear profile in y should give constant flux at bottom
    H = 5
    W = 4
    dx = 1.0
    C = np.zeros((H, W), dtype=float)
    for y in range(H):
        C[y, :] = float(y)
    C_snap = C[None, :, :]

    J = compute_bottom_flux(C_snap, 2.0, dx)
    # dC/dy at bottom is 1, so flux is -D
    assert np.allclose(J[0], -2.0)


def test_compute_permeability():
    # simple ratio check
    P = compute_permeability(0.5, 1.0)
    assert P == 0.5
