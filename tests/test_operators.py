import numpy as np

from skin_diffusion.operators import step_constant_D, step_varD_conservative


def test_constant_D_matches_varD():
    # random field for a fair comparison
    rng = np.random.default_rng(0)
    C = rng.random((6, 6))

    # constant D in both paths
    D_scalar = 0.8
    D_field = np.full_like(C, D_scalar)

    # small step size for stability
    dt = 1e-4
    dx = 0.1

    # run both operators
    C_const = step_constant_D(C, D_scalar, dt, dx)
    C_var = step_varD_conservative(C, D_field, dt, dx)

    # compare interior only (both methods leave boundaries alone in different ways)
    assert np.allclose(C_const[1:-1, 1:-1], C_var[1:-1, 1:-1])
