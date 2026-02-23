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


def integrate_flux(flux_curve, t, t_end=None):
    # integrate J(t) using trapezoids; optional cutoff time for partial area
    flux_arr = np.asarray(flux_curve, dtype=float)
    time_arr = np.asarray(t, dtype=float)

    if flux_arr.size == 0 or time_arr.size == 0:
        return 0.0

    if flux_arr.size == 1 or time_arr.size == 1:
        return 0.0

    if t_end is None:
        return float(np.trapezoid(flux_arr, x=time_arr))

    cutoff = float(t_end)
    if cutoff <= float(time_arr[0]):
        return 0.0

    last_time = float(time_arr[-1])
    if cutoff >= last_time:
        return float(np.trapezoid(flux_arr, x=time_arr))

    right_idx = int(np.searchsorted(time_arr, cutoff, side="right"))
    if right_idx <= 0:
        return 0.0

    # if cutoff matches an existing saved time, no interpolation is needed
    if abs(float(time_arr[right_idx - 1]) - cutoff) <= 1e-12:
        return float(np.trapezoid(flux_arr[:right_idx], x=time_arr[:right_idx]))

    if right_idx >= len(time_arr):
        return float(np.trapezoid(flux_arr, x=time_arr))

    t_left = float(time_arr[right_idx - 1])
    t_right = float(time_arr[right_idx])
    j_left = float(flux_arr[right_idx - 1])
    j_right = float(flux_arr[right_idx])

    if t_right <= t_left:
        return float(np.trapezoid(flux_arr[:right_idx], x=time_arr[:right_idx]))

    weight = (cutoff - t_left) / (t_right - t_left)
    j_cutoff = j_left + weight * (j_right - j_left)

    time_short = np.concatenate([time_arr[:right_idx], np.array([cutoff], dtype=float)])
    flux_short = np.concatenate([flux_arr[:right_idx], np.array([j_cutoff], dtype=float)])
    return float(np.trapezoid(flux_short, x=time_short))


def compute_finite_dose_scalars(flux_curve, t, delivered_time_s=86400.0):
    # extra scalars that stay meaningful when donor concentration decays in time
    flux_arr = np.asarray(flux_curve, dtype=float)
    time_arr = np.asarray(t, dtype=float)

    if flux_arr.size == 0 or time_arr.size == 0:
        return {
            "AUC_J": 0.0,
            "J_peak": None,
            "t_peak": None,
            "M_delivered_24h": 0.0,
        }

    peak_idx = int(np.argmax(flux_arr))
    auc_j = integrate_flux(flux_arr, time_arr)
    delivered_24h = integrate_flux(flux_arr, time_arr, t_end=float(delivered_time_s))

    metrics = {}
    metrics["AUC_J"] = float(auc_j)
    metrics["J_peak"] = float(flux_arr[peak_idx])
    metrics["t_peak"] = float(time_arr[peak_idx])
    metrics["M_delivered_24h"] = float(delivered_24h)
    return metrics
