import numpy as np


def create_coords(H, W, dx):
    # make x and y coords
    x = np.arange(W) * dx
    y = np.arange(H) * dx
    return x, y


def create_time(T, dt, save_every):
    # make time arrays
    n_steps = int(T / dt) + 1
    t_all = np.arange(n_steps) * dt
    t_save_idx = np.arange(0, n_steps, save_every)
    t_save = t_all[t_save_idx]
    return t_all, t_save_idx, t_save
