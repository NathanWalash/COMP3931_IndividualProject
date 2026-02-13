import numpy as np

from skin_diffusion.bc import apply_bc, make_patch_mask


def test_make_patch_mask_numeric_offset_edges():
    # simple top-row mask checks for numeric offset mode
    H = 4
    W = 8
    width_frac = 0.25  # width = 2 cells

    mask_left = make_patch_mask(H, W, width_frac, 0.0)
    mask_right = make_patch_mask(H, W, width_frac, 1.0)

    # left edge
    left_cols = np.where(mask_left[0])[0]
    assert left_cols.tolist() == [0, 1]

    # right edge
    right_cols = np.where(mask_right[0])[0]
    assert right_cols.tolist() == [6, 7]


def test_make_patch_mask_string_offsets_still_work():
    # old string behavior should remain unchanged
    H = 4
    W = 8
    width_frac = 0.25  # width = 2 cells

    mask_center = make_patch_mask(H, W, width_frac, "center")
    center_cols = np.where(mask_center[0])[0]
    assert center_cols.tolist() == [3, 4]


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
