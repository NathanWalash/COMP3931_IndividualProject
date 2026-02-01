import warnings

import numpy as np

from skin_diffusion.checks import (
    assert_nonnegative,
    diagnostics_over_time,
    l2_error,
    mass,
    min_value,
)


def test_mass_and_min():
    # small array to make totals obvious
    C = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert mass(C, 1.0) == 10.0
    assert min_value(C) == 1.0


def test_l2_error_zero():
    # identical fields should give zero error
    C = np.ones((3, 3))
    assert l2_error(C, C) == 0.0


def test_diagnostics_over_time():
    # two snapshots should give two entries
    C_snap = np.zeros((2, 2, 2))
    diag = diagnostics_over_time(C_snap, 1.0)
    assert len(diag["mass"]) == 2
    assert len(diag["min_value"]) == 2


def test_assert_nonnegative_warns():
    # negative value should warn when tol is higher
    C = np.array([[-1.0]])
    with warnings.catch_warnings(record=True) as w:
        ok = assert_nonnegative(C, tol=-0.5, mode="warn")
        assert ok is False
        assert len(w) == 1
