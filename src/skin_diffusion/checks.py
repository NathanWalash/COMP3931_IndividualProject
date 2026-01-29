import warnings


# simple stability limit for 2D explicit diffusion
# dt_max = dx^2 / (4 * Dmax)

def stability_limit_diffusion(Dmax, dx):
    return (dx * dx) / (4.0 * Dmax)


# check dt against limit for explicit diffusion

def check_stability(dt, dt_max, mode="warn"):
    if dt <= dt_max:
        return True

    msg = f"dt={dt} is above dt_max={dt_max}"
    if mode == "raise":
        raise ValueError(msg)

    warnings.warn(msg)
    return False
