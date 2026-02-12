import numpy as np
import pytest

from skin_diffusion.dataset import split_indices


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
