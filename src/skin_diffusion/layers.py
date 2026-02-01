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
