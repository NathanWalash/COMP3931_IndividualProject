import numpy as np
import pytest

from skin_diffusion.dataset import select_id_ood_runs, split_indices


def test_split_indices_tolerates_float_rounding():
    # 0.67 + 0.33 can become 1.0000000000000002 in floating point
    n = 10
    train_idx, val_idx, test_idx = split_indices(
        n=n,
        split_seed=123,
        train_frac=0.67,
        val_frac=0.33,
    )

    # should still be treated like a full train+val split
    assert len(test_idx) == 0

    # all rows should be covered exactly once
    combined = np.concatenate([train_idx, val_idx])
    assert len(combined) == n
    assert len(np.unique(combined)) == n


def test_split_indices_rejects_invalid_fractions():
    # clearly invalid split should raise
    with pytest.raises(ValueError):
        split_indices(
            n=10,
            split_seed=123,
            train_frac=0.8,
            val_frac=0.25,
        )


def test_split_indices_rejects_too_small_n():
    with pytest.raises(ValueError):
        split_indices(
            n=1,
            split_seed=123,
            train_frac=0.7,
            val_frac=0.15,
        )


def test_select_id_ood_runs_patch_width_holdout():
    runs = [
        {"patch_width": 0.25},
        {"patch_width": 0.50},
        {"patch_width": 1.00},
        {"patch_width": 0.25},
    ]

    id_runs, ood_runs = select_id_ood_runs(runs, ood_param="patch_width", ood_value=0.25)

    assert len(ood_runs) == 2
    assert len(id_runs) == 2

    for run in ood_runs:
        assert run["patch_width"] == 0.25

    for run in id_runs:
        assert run["patch_width"] != 0.25



