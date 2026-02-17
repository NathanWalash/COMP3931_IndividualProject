# Config Guide

This guide explains the YAML config fields used by the simulator.

## Top-level keys

- `seed`: random seed for repeatability.
- `output_dir`: where run bundles are written.
- `regime_name`: short label used in metadata.
- `grid`: spatial and temporal resolution.
- `boundary`: donor patch and boundary condition settings.
- `layers` (V2): layered diffusion and clearance.
- `heterogeneity` (V3): spatial noise on D.
- `literature`: targets for the compare script.

## Grid

```
H, W: number of grid cells (height/depth, width/lateral)
dx: cell size (cm)
dt: time step (s)
T: total simulated time (s)
save_every: save every N steps
```

## Boundary

```
mode: infinite_dose | time_decay
C0: donor concentration at t=0
decay_rate: exponential decay rate for time_decay
patch_width: fraction of domain width (0..1)
patch_offset: left | center | right
bottom: sink (zero)
sides: neumann (no-flux)
top_offpatch_mode: neumann (no-flux)
```

Notes:
- `patch_width=1.0` makes the problem effectively 1D in depth.
- Dataset workflow uses discrete patch placement (`left/center/right`).

## Layers (V2)

```
layer_rows: list of row counts from top to bottom
D_values: diffusion per layer (same length as layer_rows)
k_dermis: clearance coefficient in dermis rows
```

## Heterogeneity (V3)

```
mode: iid | correlated
sigma: noise magnitude
seed: random seed for the noise
D_min, D_max: clip range
steps: smoothing steps for correlated noise
```

## Literature (compare script)

```
source: citation string
P_target_cm_s: target permeability
Tlag_target_hours: target lag time
```
