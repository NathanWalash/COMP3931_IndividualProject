import numpy as np

from skin_diffusion.bc import apply_bc, make_patch_mask


def test_apply_bc_patch_and_sink():
    # small grid so it's easy to reason about
    H = 4
    W = 6
    C = np.zeros((H, W), dtype=float)

    # patch is half width, centered on the top row
    mask = make_patch_mask(H, W, 0.5, "center")
    Cpatch = 1.23

    # apply BCs once
    apply_bc(C, mask, Cpatch, bottom_sink=True, neumann_sides=True, top_offpatch="neumann")

    # top patch cells set to donor concentration
    assert np.all(C[mask] == Cpatch)

    # bottom row is sink (zero)
    assert np.all(C[-1, :] == 0.0)

    # sides are copied from inside (neumann)
    assert np.all(C[:, 0] == C[:, 1])
    assert np.all(C[:, -1] == C[:, -2])
