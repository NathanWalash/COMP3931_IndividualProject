import numpy as np

from skin_diffusion.ml_scalar_diagnostics import safe_r2


def test_safe_r2_constant_target_perfect_prediction():
    # Degenerate target should return finite perfect score when error is zero.
    y_true = np.array([2.5, 2.5, 2.5, 2.5], dtype=float)
    y_pred = np.array([2.5, 2.5, 2.5, 2.5], dtype=float)
    assert safe_r2(y_true, y_pred) == 1.0


def test_safe_r2_constant_target_nonzero_error():
    # Degenerate target should return finite zero score when error is nonzero.
    y_true = np.array([2.5, 2.5, 2.5, 2.5], dtype=float)
    y_pred = np.array([2.4, 2.6, 2.5, 2.5], dtype=float)
    assert safe_r2(y_true, y_pred) == 0.0
