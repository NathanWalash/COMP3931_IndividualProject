import numpy as np

from skin_diffusion.config import load_config
from skin_diffusion.dataset_spec import (
    apply_sample_to_config,
    is_dataset_spec,
    sample_dataset_parameters,
)


def test_apply_sample_to_config_writes_expected_fields():
    cfg = load_config("configs/sim/v3_layers_literature.yaml")

    sample = {
        "patch_width": 0.5,
        "patch_offset": "left",
        "C0": 1.1,
        "decay_rate": 0.15,
        "heterogeneity_mode": "correlated",
        "heterogeneity_sigma": 0.03,
        "heterogeneity_steps": 6,
    }
    apply_sample_to_config(cfg, sample, run_seed=1234)

    assert cfg.seed == 1234
    assert cfg.boundary.patch_width == 0.5
    assert cfg.boundary.patch_offset == "left"
    assert cfg.boundary.C0 == 1.1
    assert cfg.boundary.decay_rate == 0.15
    assert cfg.extras["heterogeneity"]["mode"] == "correlated"
    assert cfg.extras["heterogeneity"]["sigma"] == 0.03
    assert cfg.extras["heterogeneity"]["steps"] == 6
    assert cfg.extras["heterogeneity"]["seed"] == 1234
    assert cfg.extras["dataset_sample"]["run_seed"] == 1234

def make_min_spec():
    return {
        "sampling": {
            "parameters": {
                "patch_width": {"type": "discrete", "values": [0.25, 0.5, 1.0]},
                "patch_offset": {
                    "type": "conditional_discrete",
                    "values": ["left", "center", "right"],
                    "rule_when_patch_width_1p0": "center",
                },
                "C0": {"type": "uniform", "min": 0.8, "max": 1.2},
                "decay_rate": {"type": "uniform", "min": 0.05, "max": 0.3},
                "heterogeneity_mode": {"type": "fixed", "value": "correlated"},
                "heterogeneity_sigma": {"type": "uniform", "min": 0.01, "max": 0.08},
                "heterogeneity_steps": {"type": "randint_inclusive", "min": 3, "max": 9},
            }
        }
    }


def test_patch_offset_rule_when_width_is_full():
    # when width is full domain, offset must be forced to center
    spec = make_min_spec()
    rng = np.random.default_rng(7)

    seen_full_width = False
    for i in range(500):
        s = sample_dataset_parameters(spec, rng)
        if s["patch_width"] == 1.0:
            seen_full_width = True
            assert s["patch_offset"] == "center"
        else:
            assert s["patch_offset"] in ["left", "center", "right"]

    assert seen_full_width


