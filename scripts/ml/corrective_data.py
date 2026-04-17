import json
from pathlib import Path

import numpy as np
import torch

from scripts.ml.common import resolve_split_key
from skin_diffusion.ml_run_dataset import load_split_2d_array, load_split_entries, remap_run_dir


def build_entries(ml_dir, split_name, run_start_index=None, run_end_index=None, max_rows=None, run_root_override=None):
    # Load split rows from ml/meta and remap run paths when using staged storage.
    entries = load_split_entries(ml_dir=ml_dir, split_name=split_name, run_start_index=run_start_index, run_end_index=run_end_index)
    out = []
    for entry in entries:
        row = dict(entry)
        row["run_dir"] = remap_run_dir(entry["run_dir"], run_root_override)
        out.append(row)
    if max_rows is not None:
        out = out[: int(max_rows)]
    if len(out) == 0:
        raise ValueError("No rows selected for split " + str(split_name))
    return out


def load_split_matrix(ml_dir, split_name, entries, key):
    # Read one array block (X/J/t/...) for the selected split rows.
    return load_split_2d_array(ml_dir=ml_dir, split_name=split_name, entries=entries, key=key).astype(np.float32)


def load_split_name_map(ml_dir):
    # Build a stable train/val/test -> file/index key map from ml/meta.json.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing meta.json in " + str(ml_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_map = meta.get("splits", {})
    if not isinstance(split_map, dict):
        raise ValueError("meta.json is missing splits")

    result = {}
    result["train"] = resolve_split_key(split_map, "train")
    result["val"] = resolve_split_key(split_map, "val")
    result["test"] = resolve_split_key(split_map, "test")
    return result


def load_ml_meta_names(ml_dir):
    # Read feature and scalar target names used by scalar diagnostics.
    meta_path = Path(ml_dir) / "meta.json"
    if not meta_path.exists():
        raise ValueError("Missing meta.json in " + str(ml_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_names = meta.get("feature_names", [])
    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("meta.json is missing feature_names")

    scalar_target_names = meta.get("scalar_target_names", [])
    if not isinstance(scalar_target_names, list) or len(scalar_target_names) == 0:
        raise ValueError("meta.json is missing scalar_target_names")

    feature_out = []
    feature_index = 0
    while feature_index < len(feature_names):
        feature_out.append(str(feature_names[feature_index]))
        feature_index += 1

    scalar_out = []
    scalar_index = 0
    while scalar_index < len(scalar_target_names):
        scalar_out.append(str(scalar_target_names[scalar_index]))
        scalar_index += 1
    return feature_out, scalar_out


def load_run_physics_1d(entries):
    # Load only per-run time grids needed by the corrective trainer.
    t_norm_rows = []
    t_phys_rows = []

    t_count_ref = None
    row_count = len(entries)
    row_index = 0
    while row_index < row_count:
        run_dir = Path(entries[row_index]["run_dir"])
        fields_path = run_dir / "fields.npz"
        if not fields_path.exists():
            raise ValueError("Missing fields.npz: " + str(fields_path))

        with np.load(fields_path) as data:
            c_snap = np.asarray(data["C_snap"], dtype=np.float32)
            t_curve = np.asarray(data["t"], dtype=np.float32)

        if c_snap.ndim != 3 or t_curve.ndim != 1:
            raise ValueError("Invalid array shapes for run: " + str(run_dir))

        t_count = int(c_snap.shape[0])
        if t_count_ref is None:
            t_count_ref = t_count
        if t_count != int(t_count_ref):
            raise ValueError("All runs must share C_snap time length in selected split")
        t_end_value = float(t_curve[-1])
        if t_end_value <= 0.0:
            raise ValueError("Invalid t_end in run fields: " + str(run_dir))

        # Normalize time to [0,1] so one model can cover different t_end values.
        t_norm = np.asarray(t_curve / t_end_value, dtype=np.float32)

        t_norm_rows.append(t_norm)
        t_phys_rows.append(np.asarray(t_curve, dtype=np.float32))

        row_index += 1
        if row_index == row_count or row_index % 50 == 0:
            print("loaded run physics", row_index, "/", row_count)

    return {
        "t_norm": np.asarray(t_norm_rows, dtype=np.float32),
        "t_phys": np.asarray(t_phys_rows, dtype=np.float32),
    }


def pack_for_training(features, j_true, j_base, phys):
    # Keep only arrays consumed by the corrective trainer.
    return {
        "x_feat": np.asarray(features, dtype=np.float32),
        "j_true": np.asarray(j_true, dtype=np.float32),
        "j_base": np.asarray(j_base, dtype=np.float32),
        "t_norm": np.asarray(phys["t_norm"], dtype=np.float32),
        "t_phys": np.asarray(phys["t_phys"], dtype=np.float32),
    }


def tensorize_pack(np_pack, device):
    # Move numpy arrays to the selected device with one predictable dtype.
    out = {}
    for key in np_pack:
        value = np_pack[key]
        out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return out
