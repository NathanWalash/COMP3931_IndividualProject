import numpy as np


def compute_bottom_flux(C_snapshots, D_field, dx):
    # C_snapshots is [T, H, W]
    # use bottom two rows to estimate dC/dy
    dCdy = (C_snapshots[:, -1, :] - C_snapshots[:, -2, :]) / dx

    # pick D at the row just above the sink
    if np.isscalar(D_field):
        D_bottom_row = D_field
    else:
        D_bottom_row = D_field[-2, :]

    # flux profile along the bottom
    flux_profile = -D_bottom_row * dCdy

    # average across x to get one value per time
    flux_curve = np.mean(flux_profile, axis=1)
    return flux_curve


def estimate_steady_state_flux(flux_curve, t, tail_fraction=0.2):
    # average the last part of the curve
    n = len(flux_curve)
    tail_count = max(1, int(n * tail_fraction))
    return float(np.mean(flux_curve[-tail_count:]))


def estimate_lag_time(flux_curve, t, tail_fraction=0.2):
    # fit a line to the tail and find where it hits zero
    n = len(flux_curve)
    tail_count = max(2, int(n * tail_fraction))
    flux_tail = flux_curve[-tail_count:]
    time_tail = t[-tail_count:]

    # linear fit: J = slope*time + intercept
    slope, intercept = np.polyfit(time_tail, flux_tail, 1)
    if slope == 0:
        return None

    # lag time is the x-intercept
    lag_time = -intercept / slope
    return float(lag_time)


def compute_permeability(J_ss, C0):
    # simple ratio
    donor_conc = C0
    if donor_conc == 0:
        return None
    return float(J_ss / donor_conc)
