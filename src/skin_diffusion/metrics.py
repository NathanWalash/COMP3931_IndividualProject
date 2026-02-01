import numpy as np


def compute_bottom_flux(C_snap, D, dx):
    # C_snap is [T, H, W]
    # use bottom two rows to estimate dC/dy
    dCdy = (C_snap[:, -1, :] - C_snap[:, -2, :]) / dx

    # pick D at the row just above the sink
    if np.isscalar(D):
        D_bottom_row = D
    else:
        D_bottom_row = D[-2, :]

    # flux profile along the bottom
    Jx_profile = -D_bottom_row * dCdy

    # average across x to get one value per time
    J = np.mean(Jx_profile, axis=1)
    return J
