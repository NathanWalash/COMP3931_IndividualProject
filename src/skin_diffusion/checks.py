import warnings


# simple stability limit for 2D explicit diffusion
# dt_max = dx^2 / (4 * Dmax)

def stability_limit_diffusion(Dmax, dx):
    return (dx * dx) / (4.0 * Dmax)


# check dt against limit for explicit diffusion
# warn or raise if too large

def check_stability(dt, dt_max, mode="warn"):
    if dt <= dt_max:
        return True

    msg = f"dt={dt} is above dt_max={dt_max}"
    if mode == "raise":
        raise ValueError(msg)

    warnings.warn(msg)
    return False


# simple l2 error between fields
def l2_error(a, b):
    diff = a - b
    return float((diff * diff).mean() ** 0.5)


# total mass in the grid
def mass(C, dx):
    return float(C.sum() * dx * dx)


# minimum value in the grid
def min_value(C):
    return float(C.min())


# track mass and min(C) over time
def diagnostics_over_time(C_snap, dx):
    masses = []
    mins = []
    for i in range(C_snap.shape[0]):
        masses.append(mass(C_snap[i], dx))
        mins.append(min_value(C_snap[i]))
    return {"mass": masses, "min_value": mins}


# warn or raise if any negative values
def assert_nonnegative(C, tol=-1e-12, mode="warn"):
    min_c = float(C.min())
    if min_c >= tol:
        return True

    msg = f"min(C)={min_c} below tol={tol}"
    if mode == "raise":
        raise ValueError(msg)

    warnings.warn(msg)
    return False


# reaction stability (explicit)
# dt_max = 1 / kmax

def stability_limit_reaction(kmax):
    if kmax <= 0:
        return None
    return 1.0 / kmax


# check reaction stability
# returns (ok, dt_max)

def check_reaction_stability(dt, kmax, mode="warn"):
    dt_max = stability_limit_reaction(kmax)
    if dt_max is None:
        return True, None

    if dt <= dt_max:
        return True, dt_max

    msg = f"dt={dt} is above reaction dt_max={dt_max}"
    if mode == "raise":
        raise ValueError(msg)

    warnings.warn(msg)
    return False, dt_max
