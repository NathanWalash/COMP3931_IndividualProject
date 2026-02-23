# Literature Layer Reference (V2/V3)

## Main papers used

1. Adamiak-Giera et al. (2023), Frontiers in Pharmacology, doi:10.3389/fphar.2023.1157977
- Used as the macro calibration anchor for lidocaine.
- Key targets used in this project:
  - `KP = 2.658e-3 cm/h` (converted to `P_target_cm_s = 7.38e-7 cm/s`)
  - `Tlag = 1.401 h`
- Also used for setup context:
  - Human skin in Franz diffusion cells
  - Skin thickness about `0.5 mm`
  - 24 hour run window

2. Prausnitz and Langer (2008), Nature Biotechnology, doi:10.1038/nbt.1504
- Used for model assumptions, not direct numeric layer coefficients.
- Supports:
  - Stratum corneum is the main barrier.
  - Transdermal transport is often modeled with layered diffusion logic.

3. Ellison et al. (2020), Toxicology in Vitro, doi:10.1016/j.tiv.2020.104990
- Used to support partition/diffusion framework and layer-wise analysis method.
- Not used as a direct lidocaine-specific coefficient source.

4. Telaprolu et al. (2025), AAPS PharmSciTech, doi:10.1208/s12249-025-03232-2
- Used as a lidocaine/prilocaine finite-dose human-skin PBPK context paper.
- Supports choice of layered structure and finite-dose behaviour focus.
- Provides dermal partition/diffusion-style parameters in a richer PBPK framework,
  but not a direct 1-to-1 first-order `k_dermis` value for this simulator.

5. Maciel Tabosa et al. (2021), Drug Delivery and Translational Research, doi:10.1007/s13346-020-00864-8
- Used for skin-clearance interpretation and a lidocaine terminal-rate anchor.
- Reports lidocaine `k_terminal = 0.12 h^-1` after patch removal, which maps to
  about `3.3e-5 s^-1` as a first-order sensitivity anchor in this project.
- Treated as a modelling prior/sensitivity point, not a direct calibrated truth.

## Values used in new configs

Layer layout (top to bottom):
- `SC`, `viable epidermis`, `dermis`
- `layer_rows: [1, 9, 54]` on `H=64`, `dx=0.0008 cm` gives:
  - SC ~ 8 um
  - viable epidermis ~ 72 um
  - dermis ~ 432 um
  - total ~ 512 um (~0.5 mm)

Layer diffusion ordering:
- `D_SC < D_VE < D_dermis`
- Default values used:
  - `D_values: [1.3e-9, 1.0e-8, 2.0e-7]` cm^2/s

Dermal clearance:
- `k_dermis` is kept small/simple in these configs.
- In the compare-style setup, `k_dermis = 0.0` is used to avoid overfitting unidentifiable sink terms from one paper.
- In clearance-sensitivity runs, `k_dermis` is varied around a lidocaine anchor
  (`~3.3e-5 s^-1`) to test robustness of surrogate behaviour and metrics.

## Important interpretation

- The paper gives macroscopic outputs (`P`, `Tlag`), not full per-layer `D` and `k`.
- So the layered `D` values here are best described as literature-constrained inverse-calibrated effective values.
- This is acceptable for dissertation scope if clearly stated.
