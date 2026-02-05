# Tests Overview

This note explains what each test covers and why it exists.

Run all tests:
```
python -m pytest -q
```

## tests/test_bc.py

Purpose:
- Verify boundary condition behavior is correct.

What it checks:
- Patch region is set to the donor concentration.
- Bottom boundary is a sink (zero).
- Side boundaries enforce Neumann (copy interior).

## tests/test_operators.py

Purpose:
- Validate numerical operators.

What it checks:
- `step_constant_D` matches `step_varD_conservative` when D is uniform.

## tests/test_metrics.py

Purpose:
- Verify flux and metric calculations.

What it checks:
- Zero field produces zero flux.
- Linear depth profile produces constant, correct-sign flux.

## tests/test_checks.py

Purpose:
- Validate diagnostics and warning checks.

What it checks:
- `assert_nonnegative` warns on negative values.
- `diagnostics_over_time` returns correctly shaped arrays.

## tests/test_stability.py

Purpose:
- Confirm stability limit formulas.

What it checks:
- Diffusion dt limit.
- Reaction dt limit.

## Current test gaps

- No end-to-end test for saved run bundles or dataset assembly.
