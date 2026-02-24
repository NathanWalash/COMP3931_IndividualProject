import argparse
import time
from pathlib import Path

import numpy as np

from skin_diffusion.dataset_spec import load_yaml_file
from skin_diffusion.pinn_dataset import load_split_entries
from skin_diffusion.utils import ensure_dir, read_json, write_json


def load_split_shapes(path):
    # Read split array sizes without loading full tensors in memory.
    with np.load(path, mmap_mode="r") as data:
        shapes = {}
        shapes["rows"] = int(data["X"].shape[0])
        shapes["feature_count"] = int(data["X"].shape[1])
        shapes["scalar_target_count"] = int(data["y_scalar"].shape[1])
        shapes["curve_points"] = int(data["J"].shape[1])
    return shapes


def validate_ml_dir(ml_dir):
    # Check required files and collect split sizes.
    required_files = []
    required_files.append("meta.json")
    required_files.append("id_train.npz")
    required_files.append("id_val.npz")
    required_files.append("id_test.npz")

    for name in required_files:
        path = ml_dir / name
        if not path.exists():
            raise ValueError(f"Missing required file: {path}")

    meta = read_json(ml_dir / "meta.json")

    scalar_target_names = meta.get("scalar_target_names", [])
    feature_names = meta.get("feature_names", [])
    if not isinstance(scalar_target_names, list) or len(scalar_target_names) == 0:
        raise ValueError("meta.json is missing scalar_target_names")
    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("meta.json is missing feature_names")

    # Build split map explicitly for readability.
    splits = {}
    splits["id_train"] = load_split_shapes(ml_dir / "id_train.npz")
    splits["id_val"] = load_split_shapes(ml_dir / "id_val.npz")
    splits["id_test"] = load_split_shapes(ml_dir / "id_test.npz")

    # OOD split is optional.
    ood_path = ml_dir / "ood_primary.npz"
    if ood_path.exists():
        splits["ood_primary"] = load_split_shapes(ood_path)

    expected_feature_count = len(feature_names)
    expected_target_count = len(scalar_target_names)

    for split_name in splits:
        split_info = splits[split_name]
        if split_info["feature_count"] != expected_feature_count:
            raise ValueError(f"Feature count mismatch in {split_name}: npz has {split_info['feature_count']} but meta has {expected_feature_count}")
        if split_info["scalar_target_count"] != expected_target_count:
            raise ValueError(f"Target count mismatch in {split_name}: npz has {split_info['scalar_target_count']} but meta has {expected_target_count}")

    summary = {}
    summary["feature_names"] = feature_names
    summary["scalar_target_names"] = scalar_target_names
    summary["splits"] = splits
    return summary


def load_pinn_config(config_path):
    # Load YAML and confirm expected top-level blocks.
    cfg = load_yaml_file(str(config_path))
    if not isinstance(cfg, dict):
        raise ValueError(f"PINN config is not a dictionary: {config_path}")

    required_blocks = []
    required_blocks.append("model")
    required_blocks.append("training")
    required_blocks.append("optimizer")
    required_blocks.append("loss_weights")
    required_blocks.append("physics")
    required_blocks.append("runtime")

    for name in required_blocks:
        if name not in cfg:
            raise ValueError(f"Missing block in PINN config: {name}")

    return cfg


def detect_torch():
    # Detect torch availability and hardware info.
    info = {}
    info["available"] = False
    info["version"] = None
    info["cuda_available"] = False
    info["cuda_device_count"] = 0
    info["import_error"] = None

    try:
        import torch

        info["available"] = True
        info["version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        info["import_error"] = str(exc)

    return info


def choose_device(device_arg, config_device, torch_info):
    # Choose runtime device with simple precedence.
    if device_arg is not None:
        requested = str(device_arg)
    else:
        requested = str(config_device)

    if requested == "auto":
        if torch_info["available"] and torch_info["cuda_available"]:
            return "cuda"
        return "cpu"

    return requested


def main():
    # Validate inputs and write scaffold manifests
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default="configs/ml/pinn_v1.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fail_if_no_torch", action="store_true")
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    ml_dir = Path(args.ml_dir)
    if not ml_dir.exists():
        raise ValueError(f"ml_dir does not exist: {ml_dir}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise ValueError(f"config does not exist: {config_path}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    pinn_cfg = load_pinn_config(config_path)
    data_summary = validate_ml_dir(ml_dir)
    torch_info = detect_torch()

    if args.fail_if_no_torch and not torch_info["available"]:
        raise ValueError(f"torch is not available: {torch_info['import_error']}")

    # Use ML split mapping so PINN and black-box share the same split rows.
    train_entries = load_split_entries(
        ml_dir=ml_dir,
        split_name="id_train",
        run_start_index=args.run_start_index,
        run_end_index=args.run_end_index,
    )

    chosen_device = choose_device(args.device, pinn_cfg["runtime"].get("device", "auto"), torch_info)

    runtime = {}
    runtime["status"] = "scaffold_ready"
    runtime["ml_dir"] = str(ml_dir.resolve())
    runtime["config"] = str(config_path.resolve())
    runtime["device"] = chosen_device
    runtime["torch"] = torch_info
    runtime["run_index_filter"] = {
        "start": args.run_start_index,
        "end": args.run_end_index,
    }
    runtime["seconds"] = float(time.time() - t0)

    summary = {}
    summary["status"] = "ready_for_commit_2"
    summary["description"] = "PINN scaffold validated dataset paths and config."
    summary["feature_count"] = int(len(data_summary["feature_names"]))
    summary["scalar_target_count"] = int(len(data_summary["scalar_target_names"]))
    summary["train_rows"] = int(len(train_entries))
    summary["splits"] = data_summary["splits"]

    write_json(out_dir / "runtime.json", runtime)
    write_json(out_dir / "summary.json", summary)

    print("saved:", out_dir)


if __name__ == "__main__":
    main()
