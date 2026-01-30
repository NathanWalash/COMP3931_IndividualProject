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
