from pathlib import Path

import numpy as np

from skin_diffusion.run_index import in_index_range, run_index_from_path
from skin_diffusion.utils import read_json


def remap_run_dir(original_run_dir, run_root_override):
    # Optional remap from metadata paths to a faster staged dataset root.
    if run_root_override is None:
        return str(original_run_dir)

    run_name = Path(original_run_dir).name
    root = Path(run_root_override)
    candidates = [
        root / run_name,
        root / "dataset" / run_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise ValueError(
        "Could not remap run dir " + str(original_run_dir) + " under " + str(run_root_override)
    )


def load_run_fields(run_dir):
    # Load one run bundle from fields.npz.
    run_dir = Path(run_dir)
    fields_path = run_dir / "fields.npz"
    if not fields_path.exists():
        raise ValueError(f"Missing fields.npz for run: {run_dir}")

    with np.load(fields_path) as data:
        fields = {}
        fields["C_snap"] = data["C_snap"]
        fields["D"] = data["D"]
        fields["k"] = data["k"]
        fields["patch_mask"] = data["patch_mask"]
        fields["t"] = data["t"]
        fields["J"] = data["J"]
    return fields


def load_run_meta(run_dir):
    # Load meta.json for one run.
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing meta.json for run: {run_dir}")
    return read_json(meta_path)


def load_run_grid(meta):
    # Run-level metadata stores grid settings at top level.
    grid = meta.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Run metadata is missing grid settings")
    return grid


def load_run_bundle(run_dir):
    # Load one run's fields/meta.
    run_path = Path(run_dir)
    bundle = {}
    bundle["fields"] = load_run_fields(run_path)
    bundle["meta"] = load_run_meta(run_path)
    return bundle


def load_split_entries(
    ml_dir,
    split_name="train",
    run_start_index=None,
    run_end_index=None,
    allow_empty=False,
):
    # Load one split entry list from ml/meta.json.
    ml_dir = Path(ml_dir)
    meta_path = ml_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing ML meta file: {meta_path}")

    meta = read_json(meta_path)
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"meta.json is missing splits map: {meta_path}")

    split_info = splits.get(split_name)
    if split_info is None:
        raise ValueError(f"Split {split_name} is missing in ml/meta.json")

    entries = split_info.get("index", [])
    if not isinstance(entries, list):
        raise ValueError(f"Split index is not a list in ml/meta.json: {split_name}")

    split_entries = []
    row_idx = 0
    for entry in entries:
        run_idx = run_index_from_path(entry["run_dir"])
        keep = in_index_range(run_idx, run_start_index, run_end_index)
        if keep:
            row_entry = dict(entry)
            # split_row points back to the row position in split NPZ arrays.
            row_entry["split_row"] = int(row_idx)
            split_entries.append(row_entry)
        row_idx += 1

    if len(split_entries) == 0 and not bool(allow_empty):
        raise ValueError(f"No entries found for split {split_name} in {ml_dir}")
    return split_entries


def load_split_feature_matrix(ml_dir, split_name, entries):
    # Load X rows from split NPZ using split_row values in entries.
    return load_split_2d_array(ml_dir, split_name, entries, key="X")


def load_split_scalar_matrix(ml_dir, split_name, entries):
    # Load y_scalar rows from split NPZ using split_row values in entries.
    return load_split_2d_array(ml_dir, split_name, entries, key="y_scalar")


def load_split_2d_array(ml_dir, split_name, entries, key):
    # Generic split row loader for 2D arrays in one split NPZ file.
    ml_dir = Path(ml_dir)
    split_path = ml_dir / f"{split_name}.npz"
    if not split_path.exists():
        raise ValueError(f"Missing split array file: {split_path}")

    with np.load(split_path, mmap_mode="r") as data:
        if key not in data:
            raise ValueError(f"Missing key {key} in split file: {split_path}")
        matrix = np.asarray(data[key], dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"Split array for key {key} must be 2D: {split_path}")

    row_ids = []
    for entry in entries:
        if "split_row" not in entry:
            raise ValueError(f"Entry is missing split_row for split {split_name}")
        row_ids.append(int(entry["split_row"]))

    if len(row_ids) == 0:
        return np.zeros((0, int(matrix.shape[1])), dtype=np.float32)

    max_row = max(row_ids)
    if max_row >= int(matrix.shape[0]):
        raise ValueError(
            f"split_row index {max_row} is out of bounds for {split_path}"
        )

    row_array = np.asarray(row_ids, dtype=int)
    return np.asarray(matrix[row_array], dtype=np.float32)
