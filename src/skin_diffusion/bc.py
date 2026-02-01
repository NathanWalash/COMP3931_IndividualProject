import numpy as np


# make patch mask on the top row
# true means the donor patch sits here
# width_frac is fraction of total width
# offset is left/center/right

def make_patch_mask(H, W, width_frac, offset):
    # start with all false
    mask = np.zeros((H, W), dtype=bool)

    # width in cells
    width = int(round(W * width_frac))
    if width < 1:
        width = 1
    if width > W:
        width = W

    # pick start index
    if offset == "left":
        start = 0
    elif offset == "right":
        start = W - width
    else:
        start = (W - width) // 2

    # mark patch on top row only
    end = start + width
    mask[0, start:end] = True
    return mask


# donor concentration over time
# modes: infinite_dose or time_decay

def patch_concentration(t, mode, C0, decay_rate):
    if mode == "time_decay":
        return C0 * np.exp(-decay_rate * t)
    return C0


# apply basic boundary conditions
# patch on top row gets fixed C
# off-patch top gets neumann copy
# bottom is sink if True
# sides are neumann if True

def apply_bc(C, mask, Cpatch, bottom_sink=True, neumann_sides=True, top_offpatch="neumann"):
    mask_row = mask[0]

    # top on patch
    C[mask] = Cpatch

    # top off patch
    if top_offpatch == "neumann":
        C[0, ~mask_row] = C[1, ~mask_row]

    # bottom sink
    if bottom_sink:
        C[-1, :] = 0.0

    # sides neumann
    if neumann_sides:
        C[:, 0] = C[:, 1]
        C[:, -1] = C[:, -2]

    return C
