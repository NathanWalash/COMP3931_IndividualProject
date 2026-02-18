import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.dataset import validate_run
from skin_diffusion.dataset_spec import is_dataset_spec, load_yaml_file
from skin_diffusion.run_index import in_index_range, run_index_from_path


def resolve_dataset_root(config_path):
    # Support both dataset spec and normal simulation config.
    raw = load_yaml_file(config_path)
    spec_mode = is_dataset_spec(raw)
    spec_path = Path(config_path).resolve()

    if spec_mode:
        output_root = Path(raw["output_root"])
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()
        return output_root / "dataset"

    cfg = load_config(config_path)
    return Path(cfg.output_dir) / "dataset"


def load_json_file(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def quick_file_checks(run_dir):
    # Check required files before running deeper validation.
    required = ["fields.npz", "meta.json", "metrics.json"]
    missing = []
    for name in required:
        path = Path(run_dir) / name
        if not path.exists():
            missing.append(name)
    return missing


def quick_content_checks(run_dir):
    # Check that metadata and metrics can be parsed and include expected keys.
    issues = []

    try:
        meta = load_json_file(Path(run_dir) / "meta.json")
        boundary = meta.get("boundary", {})
        if "patch_width" not in boundary:
            issues.append("meta_missing_boundary_patch_width")
    except Exception as exc:
        issues.append("meta_parse_error: " + str(exc))

    try:
        metrics = load_json_file(Path(run_dir) / "metrics.json")
        for key in ["P", "Tlag", "J_ss"]:
            if key not in metrics:
                issues.append("metrics_missing_" + key)
    except Exception as exc:
        issues.append("metrics_parse_error: " + str(exc))

    return issues


def analyze_run(run_dir):
    # Run all checks for one run folder.
    issues = []

    missing = quick_file_checks(run_dir)
    for item in missing:
        issues.append("missing_file:" + item)

    if len(missing) == 0:
        content_issues = quick_content_checks(run_dir)
        for item in content_issues:
            issues.append(item)

        # validate_run checks fields shapes and key names in fields.npz.
        try:
            validate_run(run_dir)
        except Exception as exc:
            issues.append("validate_run_error: " + str(exc))

        # Try loading fields to catch zip corruption quickly.
        try:
            fields = np.load(Path(run_dir) / "fields.npz")
            t_shape = fields["t"].shape
            j_shape = fields["J"].shape
            if len(t_shape) == 0 or len(j_shape) == 0:
                issues.append("fields_empty_shape")
        except Exception as exc:
            issues.append("fields_load_error: " + str(exc))

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    parser.add_argument("--out_path", default="outputs/qc/run_integrity_report.json")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.config)
    run_dirs = sorted(dataset_root.glob("run_*"))

    selected = []
    for run_dir in run_dirs:
        idx = run_index_from_path(run_dir)
        if in_index_range(idx, args.run_start_index, args.run_end_index):
            selected.append(run_dir)

    if len(selected) == 0:
        raise ValueError("No runs found in requested range")

    good_runs = []
    bad_runs = []
    for run_dir in tqdm(selected, desc="checking runs"):
        idx = run_index_from_path(run_dir)
        issues = analyze_run(run_dir)
        if len(issues) == 0:
            good_runs.append({"run_index": idx, "run_dir": str(run_dir)})
        else:
            bad_runs.append(
                {
                    "run_index": idx,
                    "run_dir": str(run_dir),
                    "issues": issues,
                }
            )

    report = {}
    report["dataset_root"] = str(dataset_root)
    report["run_index_filter"] = {"start": args.run_start_index, "end": args.run_end_index}
    report["counts"] = {
        "selected": len(selected),
        "good": len(good_runs),
        "bad": len(bad_runs),
    }
    report["bad_runs"] = bad_runs

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
