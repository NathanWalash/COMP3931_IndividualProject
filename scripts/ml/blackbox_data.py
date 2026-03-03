from pathlib import Path

import numpy as np

from scripts.ml.common import extract_c0_feature, load_json, resolve_split_key
from skin_diffusion.ml_run_dataset import remap_run_dir
from skin_diffusion.run_index import in_index_range, run_index_from_path


def load_target_names(ml_dir):
    # Load scalar target names written by export_ml_dataset.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing ml/meta.json: " + str(meta_path))

    meta = load_json(meta_path)
    names = meta.get("scalar_target_names")
    if not isinstance(names, list):
        raise ValueError("meta.json is missing scalar_target_names list")
    if len(names) == 0:
        raise ValueError("scalar_target_names list is empty in meta.json")
    target_names = []
    for name in names:
        target_names.append(str(name))
    return target_names


def load_feature_names(ml_dir):
    # Load feature column names written by export_ml_dataset.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing ml/meta.json: " + str(meta_path))

    meta = load_json(meta_path)
    names = meta.get("feature_names")
    if not isinstance(names, list):
        raise ValueError("meta.json is missing feature_names list")
    if len(names) == 0:
        raise ValueError("feature_names list is empty in meta.json")
    feature_names = []
    for name in names:
        feature_names.append(str(name))
    return feature_names


def load_npz(path):
    # Load one dataset split from NPZ.
    data = np.load(path)
    result = {}
    result["X"] = data["X"]
    result["y_scalar"] = data["y_scalar"]
    result["J"] = data["J"]
    result["t"] = data["t"]
    return result


def filter_split_by_run_index(split_data, split_index_entries, start_idx, end_idx):
    # Filter one split by run index range using index entries from ml/meta.json.
    if start_idx is None and end_idx is None:
        return split_data, split_index_entries

    keep_rows = []
    for i in range(len(split_index_entries)):
        entry = split_index_entries[i]
        run_idx = run_index_from_path(entry["run_dir"])
        if in_index_range(run_idx, start_idx, end_idx):
            keep_rows.append(i)

    keep_rows = np.asarray(keep_rows, dtype=int)
    filtered = {}
    filtered["X"] = split_data["X"][keep_rows]
    filtered["y_scalar"] = split_data["y_scalar"][keep_rows]
    filtered["J"] = split_data["J"][keep_rows]
    filtered["t"] = split_data["t"][keep_rows]

    filtered_index = []
    for idx in keep_rows:
        filtered_index.append(split_index_entries[int(idx)])

    return filtered, filtered_index


def remap_entry_run_dirs(entries, run_root_override):
    # Keep split index rows but rewrite run_dir to staged-local location when requested.
    remapped = []
    for entry in entries:
        row = dict(entry)
        row["run_dir"] = remap_run_dir(entry["run_dir"], run_root_override)
        remapped.append(row)
    return remapped



