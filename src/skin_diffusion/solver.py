import numpy as np


def init_state(H, W):
    # start at zero everywhere
    return np.zeros((H, W), dtype=float)


def allocate_snapshots(Tsave, H, W):
    # store saved states
    return np.zeros((Tsave, H, W), dtype=float)
