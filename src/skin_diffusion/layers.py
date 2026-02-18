import numpy as np


# build layer ids along y
# layer_rows is list of row counts (top to bottom)
# example: [8, 20, 36] for H=64
# result is a 1D array of layer ids

def build_layer_id(H, layer_rows):
    # start with all zeros
    layer_id = np.zeros(H, dtype=int)
    start = 0
    i = 0
    for n in layer_rows:
        # set this layer block
        end = start + int(n)
        if end > H:
            end = H
        layer_id[start:end] = i
        start = end
        i += 1
    # fill any leftover rows
    if start < H:
        layer_id[start:] = i - 1
    return layer_id


# assign values per layer to a 1D field
# values list should match number of layers

def assign_layer_field(layer_id, values):
    # start with zeros
    field = np.zeros_like(layer_id, dtype=float)
    i = 0
    for v in values:
        # fill where layer id matches
        field[layer_id == i] = float(v)
        i += 1
    return field


# expand 1D y-field to 2D
# useful for building D or k maps

def expand_to_2d(field_y, W):
    # repeat across x
    return np.tile(field_y[:, None], (1, W))


# build D field from layer values

def build_D_field(H, W, layer_id, D_values):
    # build D in 2D from layer values
    D_y = assign_layer_field(layer_id, D_values)
    return expand_to_2d(D_y, W)


# build k field from dermis rows
# dermis_rows is number of rows at bottom

def build_k_field(H, W, dermis_rows, k_dermis):
    # start with zeros
    k_y = np.zeros(H, dtype=float)
    if dermis_rows > 0:
        k_y[H - dermis_rows :] = float(k_dermis)
    return expand_to_2d(k_y, W)


# apply iid heterogeneity to a base D field
# sigma is the std in log-space (relative variability)
# clip keeps values in a safe range

def apply_iid_heterogeneity(D_base, sigma, seed, D_min, D_max):
    rng = np.random.default_rng(seed)
    log_perturb = rng.normal(0.0, sigma, size=D_base.shape)

    # multiplicative perturbation keeps units correct for D
    D_het = D_base * np.exp(log_perturb)
    D_het = np.clip(D_het, D_min, D_max)
    return D_het


# simple box blur for correlation
# repeats a few times to smooth noise

def apply_correlated_heterogeneity(D_base, sigma, seed, D_min, D_max, steps):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=D_base.shape)

    # start from noise then smooth
    field = noise.copy()
    for _ in range(int(steps)):
        # 5-point box blur with padding
        up = np.roll(field, -1, axis=0)
        down = np.roll(field, 1, axis=0)
        left = np.roll(field, -1, axis=1)
        right = np.roll(field, 1, axis=1)
        field = (field + up + down + left + right) / 5.0

    # normalize smoothed field to unit std so sigma has stable meaning
    field_std = np.std(field)
    if field_std > 0.0:
        field = field / field_std

    # sigma controls relative variability in log-space
    log_perturb = sigma * field

    # apply multiplicative perturbation and clip
    D_het = D_base * np.exp(log_perturb)
    D_het = np.clip(D_het, D_min, D_max)
    return D_het
