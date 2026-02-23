import numpy as np
import yaml


DEFAULT_SCALAR_TARGET_NAMES = ["P", "Tlag", "J_ss"]
SUPPORTED_SCALAR_TARGET_NAMES = [
    "P",
    "Tlag",
    "J_ss",
    "AUC_J",
    "J_peak",
    "t_peak",
    "M_delivered_24h",
]


def load_yaml_file(path):
    # read yaml into a plain dict
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_dataset_spec(raw):
    # dataset spec files must include a base config and sampling section
    if not isinstance(raw, dict):
        return False

    if "base_config" not in raw:
        return False

    if "sampling" not in raw:
        return False

    return True


def extract_primary_scalar_targets(spec):
    # select scalar targets from dataset-spec primary list
    if not isinstance(spec, dict):
        return list(DEFAULT_SCALAR_TARGET_NAMES)

    targets = spec.get("targets", {})
    if not isinstance(targets, dict):
        return list(DEFAULT_SCALAR_TARGET_NAMES)

    primary = targets.get("primary", [])
    if not isinstance(primary, list):
        return list(DEFAULT_SCALAR_TARGET_NAMES)

    selected = []
    for item in primary:
        name = str(item)
        if name in SUPPORTED_SCALAR_TARGET_NAMES and name not in selected:
            selected.append(name)

    if len(selected) == 0:
        return list(DEFAULT_SCALAR_TARGET_NAMES)
    return selected


def sample_dataset_parameters(spec, rng):
    # draw one dataset run from the spec ranges
    params = spec["sampling"]["parameters"]
    sample = {}

    # patch width from a fixed set
    width_cfg = params["patch_width"]
    width_idx = int(rng.integers(0, len(width_cfg["values"])))
    sample["patch_width"] = float(width_cfg["values"][width_idx])

    # patch offset is conditional discrete (left/center/right)
    offset_cfg = params["patch_offset"]
    offset_type = str(offset_cfg.get("type", ""))
    if offset_type == "conditional_discrete":
        if abs(sample["patch_width"] - 1.0) < 1e-12:
            sample["patch_offset"] = offset_cfg["rule_when_patch_width_1p0"]
        else:
            offset_idx = int(rng.integers(0, len(offset_cfg["values"])))
            sample["patch_offset"] = offset_cfg["values"][offset_idx]
    else:
        raise ValueError("Unsupported patch_offset type (expected conditional_discrete): " + offset_type)

    # donor concentration and decay
    c0_cfg = params["C0"]
    sample["C0"] = float(rng.uniform(float(c0_cfg["min"]), float(c0_cfg["max"])))
    decay_cfg = params["decay_rate"]
    sample["decay_rate"] = float(rng.uniform(float(decay_cfg["min"]), float(decay_cfg["max"])))

    # heterogeneity mode
    mode_cfg = params["heterogeneity_mode"]
    sample["heterogeneity_mode"] = mode_cfg["value"]

    # heterogeneity sigma
    sigma_cfg = params["heterogeneity_sigma"]
    sample["heterogeneity_sigma"] = float(
        rng.uniform(float(sigma_cfg["min"]), float(sigma_cfg["max"]))
    )

    # correlated smoothing steps
    steps_cfg = params["heterogeneity_steps"]
    steps_low = int(steps_cfg["min"])
    steps_high = int(steps_cfg["max"])
    sample["heterogeneity_steps"] = int(rng.integers(steps_low, steps_high + 1))

    # optional dermal clearance sampling
    if "k_dermis" in params:
        k_cfg = params["k_dermis"]
        k_type = str(k_cfg.get("type", ""))
        if k_type == "discrete":
            k_idx = int(rng.integers(0, len(k_cfg["values"])))
            sample["k_dermis"] = float(k_cfg["values"][k_idx])
        elif k_type == "uniform":
            sample["k_dermis"] = float(
                rng.uniform(float(k_cfg["min"]), float(k_cfg["max"]))
            )
        else:
            raise ValueError("Unsupported k_dermis type (expected discrete or uniform): " + k_type)

    return sample


def apply_sample_to_config(cfg, sample, run_seed):
    # copy sampled values into the runtime config
    cfg.seed = int(run_seed)

    # boundary values sampled per run
    cfg.boundary.patch_width = float(sample["patch_width"])
    cfg.boundary.patch_offset = sample["patch_offset"]
    cfg.boundary.C0 = float(sample["C0"])
    cfg.boundary.decay_rate = float(sample["decay_rate"])

    # make sure heterogeneity config exists before writing into it
    if "heterogeneity" not in cfg.extras:
        cfg.extras["heterogeneity"] = {}

    # heterogeneity values sampled per run
    cfg.extras["heterogeneity"]["mode"] = sample["heterogeneity_mode"]
    cfg.extras["heterogeneity"]["sigma"] = float(sample["heterogeneity_sigma"])
    cfg.extras["heterogeneity"]["steps"] = int(sample["heterogeneity_steps"])
    cfg.extras["heterogeneity"]["seed"] = int(run_seed)

    # optional per-run dermal clearance
    if "k_dermis" in sample:
        if "layers" not in cfg.extras:
            cfg.extras["layers"] = {}
        cfg.extras["layers"]["k_dermis"] = float(sample["k_dermis"])

    # keep one value for traceability even if k_dermis is fixed in base config
    if "layers" in cfg.extras and "k_dermis" in cfg.extras["layers"]:
        k_dermis_value = float(cfg.extras["layers"]["k_dermis"])
    else:
        k_dermis_value = 0.0

    # keep a copy of sampled inputs in metadata for traceability
    cfg.extras["dataset_sample"] = {
        "patch_width": float(sample["patch_width"]),
        "patch_offset": sample["patch_offset"],
        "C0": float(sample["C0"]),
        "decay_rate": float(sample["decay_rate"]),
        "heterogeneity_mode": sample["heterogeneity_mode"],
        "heterogeneity_sigma": float(sample["heterogeneity_sigma"]),
        "heterogeneity_steps": int(sample["heterogeneity_steps"]),
        "k_dermis": k_dermis_value,
        "run_seed": int(run_seed),
    }
