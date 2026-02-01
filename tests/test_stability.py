import warnings

from skin_diffusion.checks import (
    check_reaction_stability,
    check_stability,
    stability_limit_diffusion,
    stability_limit_reaction,
)


def test_diffusion_limit():
    # simple known value check
    dt_max = stability_limit_diffusion(1.0, 1.0)
    assert dt_max == 0.25

    # below limit should pass
    assert check_stability(0.1, dt_max, mode="warn") is True

    # above limit should warn and fail
    with warnings.catch_warnings(record=True) as w:
        ok = check_stability(0.3, dt_max, mode="warn")
        assert ok is False
        assert len(w) == 1


def test_reaction_limit():
    # zero k gives no limit
    assert stability_limit_reaction(0.0) is None
    assert stability_limit_reaction(2.0) == 0.5

    # below limit should pass
    ok, dt_max = check_reaction_stability(0.4, 2.0, mode="warn")
    assert ok is True
    assert dt_max == 0.5

    # above limit should warn and fail
    with warnings.catch_warnings(record=True) as w:
        ok, dt_max = check_reaction_stability(0.6, 2.0, mode="warn")
        assert ok is False
        assert dt_max == 0.5
        assert len(w) == 1
