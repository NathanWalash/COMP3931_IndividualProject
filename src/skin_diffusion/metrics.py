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
    # estimate lag time using cumulative mass (Q) vs time
    # Q(t) should be linear at steady state: Q = J_ss * (t - Tlag)
    n = len(flux_curve)
    tail_count = max(2, int(n * tail_fraction))
    time_tail = t[-tail_count:]

    # build cumulative mass curve Q(t) with simple trapezoid rule
    # Q has same length as t
    Q = np.zeros_like(t, dtype=float)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        Q[i] = Q[i - 1] + 0.5 * (flux_curve[i] + flux_curve[i - 1]) * dt

    Q_tail = Q[-tail_count:]

    # linear fit: Q = slope*time + intercept
    slope, intercept = np.polyfit(time_tail, Q_tail, 1)
    if slope <= 0:
        return None

    # lag time is the x-intercept
    lag_time = -intercept / slope
    if lag_time < 0:
        return None
    return float(lag_time)


def compute_permeability(J_ss, C0):
    # simple ratio
    donor_conc = C0
    if donor_conc == 0:
        return None
    return float(J_ss / donor_conc)
