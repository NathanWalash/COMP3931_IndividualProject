# Metrics And Process Overview

This note is a plain-English summary of the workflow, what each regime is for,
and what the key metrics mean.

## 1) Overall process (high level)

1. Choose a config (V1, V2, or V3).
2. Build the grid (H, W, dx, dt, total time).
3. Build fields:
   - D field (diffusion) and optional k field (clearance).
   - Patch mask for the donor region.
4. Run the solver:
   - Apply diffusion step.
   - Apply reaction step (if k > 0).
   - Apply boundary conditions each step.
5. Save outputs:
   - Snapshots `C_snap` over time.
   - Diagnostics (mass, min C) when enabled.
   - Metrics (flux, permeability, lag time).

## 2) Versions (what each is for)

- **V1 (baseline, constant D)**
  - Purpose: simplest sanity check.
  - Shows pure diffusion with a patch and sink boundary.
  - Used for convergence and analytic validation.

- **V2 (layered D + optional clearance)**
  - Purpose: more realistic skin structure.
  - SC / epidermis / dermis layers with different D.
  - Optional k in dermis to model clearance.

- **V3 (2D patch + heterogeneity)**
  - Purpose: full 2D behavior and variability.
  - Different patch widths/offsets.
  - Optional IID or correlated D heterogeneity.

## 3) Core math in simple terms

- Diffusion follows Fick's law:
  - Concentration flows from high to low.
  - With variable D, the solver uses a conservative form
    so flux is continuous across layer boundaries.

- Reaction / clearance (if enabled):
  - Simple exponential decay: `C_new = C_old - dt * k * C_old`.

- Boundary conditions:
  - Top patch is fixed concentration (donor).
  - Bottom is a sink (zero).
  - Sides are no-flux (Neumann).

## 4) Metrics we measure

- **R² reporting policy for scalar targets**
  - For nearly constant targets, classic `R²` is mathematically undefined.
  - Reporting uses a finite fallback to keep summaries and plots stable:
    - `R² = 1.0` when prediction error is effectively zero.
    - `R² = 0.0` otherwise.

- **Bottom flux J(t)**
  - Rate of drug leaving the bottom boundary over time.
  - Computed from the concentration gradient at the bottom.

- **Steady-state flux J_ss**
  - Average of the tail of J(t).
  - In finite-dose runs, this is a tail summary, not a strict steady state.

- **Lag time Tlag**
  - Time offset from the linear fit of cumulative mass.
  - Represents how long it takes to reach steady permeation.
  - In finite-dose runs with strong decay, this can be `None`/NaN.

- **Permeability P**
  - Defined as `P = J_ss / C0`.
  - Standard permeability coefficient used in literature.
  - In finite-dose runs, treat this as a derived proxy (secondary),
    not a ground-truth constant permeability.

- **Finite-dose scalars (recommended for time-decay donor)**
  - `AUC_J`: area under `J(t)` over the saved time horizon.
  - `J_peak`: peak flux value.
  - `t_peak`: time at peak flux.
  - `M_delivered_24h`: integrated delivered mass up to 24 hours.
  - Note: if the run horizon is exactly 24h, `AUC_J` and `M_delivered_24h`
    will be numerically identical.

- **Diagnostics (optional)**
  - Total mass in the domain (sanity check).
  - Minimum concentration (to detect negative values).

## 5) How this supports ML later

- The simulator generates consistent, physics-based ground truth.
- Metrics provide targets for evaluation or conditioning.
- Dataset pipeline packages runs into train/val/test splits.
