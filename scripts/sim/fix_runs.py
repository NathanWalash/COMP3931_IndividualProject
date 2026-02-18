import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.dataset import generate_run, validate_run
from skin_diffusion.dataset_spec import (
    apply_sample_to_config,
    is_dataset_spec,
    load_yaml_file,
    sample_dataset_parameters,
)
from skin_diffusion.run_index import in_index_range, run_index_from_path


def resolve_config_and_dataset_root(config_path):
    # Support both dataset spec input and plain simulation config input.
    raw = load_yaml_file(config_path)
    spec_mode = is_dataset_spec(raw)
    spec_path = Path(config_path).resolve()

    if spec_mode:
        base_cfg_path = Path(raw["base_config"])
        if not base_cfg_path.is_absolute():
            base_from_cwd = (Path.cwd() / base_cfg_path).resolve()
            if base_from_cwd.exists():
                base_cfg_path = base_from_cwd
            else:
                base_cfg_path = (spec_path.parent / base_cfg_path).resolve()

        output_root = Path(raw["output_root"])
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()

        cfg = load_config(str(base_cfg_path))
        cfg.output_dir = str(output_root)
        dataset_root = output_root / "dataset"
        return raw, spec_mode, cfg, dataset_root

    cfg = load_config(config_path)
    dataset_root = Path(cfg.output_dir) / "dataset"
    return raw, spec_mode, cfg, dataset_root


def load_json_file(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def check_run_integrity(run_dir):
    # Basic integrity checks for one run.
    # This function is intentionally simple so failures are easy to interpret.
    issues = []
    required = ["fields.npz", "meta.json", "metrics.json"]

    for name in required:
        if not (Path(run_dir) / name).exists():
            issues.append("missing_file:" + name)

    if len(issues) > 0:
        return issues

    try:
        meta_data = load_json_file(Path(run_dir) / "meta.json")
        if "seed" not in meta_data:
            issues.append("meta_missing_seed")
    except Exception as exc:
        issues.append("meta_parse_error:" + str(exc))

    try:
        metrics_data = load_json_file(Path(run_dir) / "metrics.json")
        for key in ["P", "Tlag", "J_ss"]:
            if key not in metrics_data:
                issues.append("metrics_missing_" + key)
    except Exception as exc:
        issues.append("metrics_parse_error:" + str(exc))

    try:
        validate_run(run_dir)
    except Exception as exc:
        issues.append("validate_run_error:" + str(exc))

    return issues


def infer_seed_offset(existing_runs):
    # Infer seed = run_index + offset from valid existing runs.
    # If every run was generated with seed_start + run_index,
    # this offset should be constant.
    offsets = []
    for run_dir in existing_runs:
        idx = run_index_from_path(run_dir)
        if idx is None:
            continue
        meta_path = Path(run_dir) / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = load_json_file(meta_path)
            seed = int(meta["seed"])
        except Exception:
            continue
        offsets.append(seed - idx)

    if len(offsets) == 0:
        return None, None

    offsets = np.asarray(offsets, dtype=int)
    offset = int(np.median(offsets))
    spread = int(np.max(offsets) - np.min(offsets))
    return offset, spread


def configure_run(cfg, raw, spec_mode, run_seed):
    # Apply sampled parameters for a specific seed.
    if spec_mode:
        rng = np.random.default_rng(run_seed)
        sample = sample_dataset_parameters(raw, rng)
        apply_sample_to_config(cfg, sample, run_seed)
        return

    cfg.seed = run_seed
    if "heterogeneity" in cfg.extras:
        cfg.extras["heterogeneity"]["seed"] = run_seed


def main():
    # CLI inputs define which run indices to inspect/fix.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_start_index", type=int, required=True)
    parser.add_argument("--run_end_index", type=int, required=True)
    parser.add_argument("--seed_offset", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out_path", default="outputs/qc/fix_runs_report.json")
    args = parser.parse_args()

    if args.run_end_index < args.run_start_index:
        raise ValueError("run_end_index must be >= run_start_index")

    # Resolve config and dataset root folder from the provided config path.
    raw, spec_mode, cfg, dataset_root = resolve_config_and_dataset_root(args.config)
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Gather existing run folders in the requested index range.
    existing_dirs = sorted(dataset_root.glob("run_*"))
    existing_by_idx = {}
    for run_dir in existing_dirs:
        idx = run_index_from_path(run_dir)
        if in_index_range(idx, args.run_start_index, args.run_end_index):
            existing_by_idx[idx] = run_dir

    broken = []
    good = []
    # Check integrity of each existing run in-range.
    for idx, run_dir in tqdm(sorted(existing_by_idx.items()), desc="checking existing runs"):
        issues = check_run_integrity(run_dir)
        if len(issues) == 0:
            good.append(idx)
        else:
            broken.append({"run_index": idx, "run_dir": str(run_dir), "issues": issues})

    missing = []
    # Identify indices that are missing run folders entirely.
    for idx in range(args.run_start_index, args.run_end_index + 1):
        if idx not in existing_by_idx:
            missing.append(idx)

    bad_indices = [item["run_index"] for item in broken]
    target_indices = sorted(set(missing + bad_indices))

    if args.seed_offset is None:
        # Try to infer seed offset from valid existing runs.
        inferred_offset, offset_spread = infer_seed_offset(existing_dirs)
    else:
        inferred_offset = int(args.seed_offset)
        offset_spread = 0

    report = {}
    report["dataset_root"] = str(dataset_root)
    report["range"] = {"start": args.run_start_index, "end": args.run_end_index}
    report["counts"] = {
        "existing_in_range": len(existing_by_idx),
        "good_in_range": len(good),
        "broken_in_range": len(broken),
        "missing_in_range": len(missing),
        "to_fix": len(target_indices),
    }
    report["broken_runs"] = broken
    report["missing_run_indices"] = missing
    report["target_run_indices"] = target_indices
    report["seed_offset"] = {"value": inferred_offset, "spread": offset_spread}
    report["applied"] = bool(args.apply)

    fixed = []
    failed = []
    # Apply mode: delete broken runs and regenerate all target indices.
    if args.apply and len(target_indices) > 0:
        if inferred_offset is None:
            raise ValueError(
                "Could not infer seed_offset from existing runs. "
                "Pass --seed_offset explicitly."
            )

        for item in broken:
            run_dir = Path(item["run_dir"])
            if run_dir.exists():
                shutil.rmtree(run_dir)

        # Regenerate each missing/broken run with deterministic seed mapping.
        for idx in tqdm(target_indices, desc="regenerating runs"):
            run_seed = int(idx + inferred_offset)
            run_id = f"run_{idx:03d}"
            run_dir = dataset_root / run_id
            try:
                configure_run(cfg, raw, spec_mode, run_seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                generate_run(cfg, run_dir)
                validate_run(run_dir)
                fixed.append({"run_index": idx, "run_seed": run_seed, "run_dir": str(run_dir)})
            except Exception as exc:
                failed.append({"run_index": idx, "run_seed": run_seed, "error": str(exc)})

    report["fixed_runs"] = fixed
    report["failed_runs"] = failed

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
