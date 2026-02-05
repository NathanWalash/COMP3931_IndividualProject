# Figures Guide

This note explains what each figure is showing and how to interpret it.

## Validation V1 figures

Location: `figures/validation/v1/`

- Heatmaps (`heat_t*.png`):
  - 2D concentration snapshots at selected times.
  - Expect high values near the top patch, fading with depth.
  - As time increases, the field smooths and penetrates deeper.

- Depth profiles (`profile_t*.png`):
  - x-averaged concentration vs depth.
  - Curves should decrease with depth; later times shift upward.

## Validation V2 figures

Location: `figures/validation/v2/`

- D map (`D_map.png`):
  - Layered diffusion coefficients.
  - Low D in SC, higher D in deeper layers.

- k map (`k_map.png`):
  - Clearance in dermis (zero elsewhere if enabled).

- Heatmaps / profiles:
  - Penetration is slower across low-D layers.
  - Expect visible banding across depth from the layer structure.

## Validation V3 figures

Location: `figures/validation/v3/`

- Patch width/offset heatmaps:
  - Left/center/right patches show lateral effects.
  - Narrow patches show stronger lateral smoothing.

- Lateral profiles:
  - Concentration across x at a fixed depth.
  - Peaks under the patch and flattens as diffusion spreads.

- D field (if heterogeneity enabled):
  - IID noise looks speckled; correlated noise looks patchy/smooth.

## Convergence figure

Location: `figures/validation/v1_convergence.png`

- Three error curves (coarse vs fine grids).
- Convergence evidence: the finer-pair curve is lower than the coarser pair.
- A late-time plateau is normal once the shape stabilizes.

## Analytic 1D benchmark

Location: `outputs/sim/v1/benchmark/report_1d.json`

- Errors should be very small when compared to the finite-slab analytic
  series solution (Crank, 1975).
- This is a strong correctness check of the numerical solver.
