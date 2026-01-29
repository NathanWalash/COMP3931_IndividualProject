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
