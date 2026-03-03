import json
from pathlib import Path

import numpy as np
import torch

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


def resolve_split_key(split_map, logical_name):
    # Use canonical split keys only.
    split_text = str(logical_name)
    if split_text in split_map:
        return split_text
    raise ValueError("Could not find split key " + split_text + ". Expected one of: train, val, test")


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


def extract_c0_feature(x_raw, feature_names):
    # C0 is needed to derive permeability-style scalar diagnostics from J(t).
    if "C0" not in feature_names:
        return np.full((int(x_raw.shape[0]),), np.nan, dtype=np.float32)
    c0_column = int(feature_names.index("C0"))
    return np.asarray(x_raw[:, c0_column], dtype=np.float32)


def load_run_physics_1d(entries):
    # Load 1D x-averaged concentration/material profiles from the exact simulator outputs.
    d1d_rows = []
    d1d_dy_rows = []
    k1d_rows = []
    c_star_rows = []
    t_norm_rows = []
    t_phys_rows = []
    patch_width_rows = []
    c0_rows = []
    decay_rows = []
    mode_rows = []
    depth_rows = []
    t_end_rows = []

    t_count_ref = None
    h_count_ref = None
    row_count = len(entries)
    row_index = 0
    while row_index < row_count:
        run_dir = Path(entries[row_index]["run_dir"])
        fields_path = run_dir / "fields.npz"
        meta_path = run_dir / "meta.json"
        if not fields_path.exists():
            raise ValueError("Missing fields.npz: " + str(fields_path))
        if not meta_path.exists():
            raise ValueError("Missing meta.json: " + str(meta_path))

        with np.load(fields_path) as data:
            c_snap = np.asarray(data["C_snap"], dtype=np.float32)
            d_field = np.asarray(data["D"], dtype=np.float32)
            k_field = np.asarray(data["k"], dtype=np.float32)
            t_curve = np.asarray(data["t"], dtype=np.float32)

        if c_snap.ndim != 3 or d_field.ndim != 2 or k_field.ndim != 2 or t_curve.ndim != 1:
            raise ValueError("Invalid array shapes for run: " + str(run_dir))

        t_count = int(c_snap.shape[0])
        h_count = int(c_snap.shape[1])
        if t_count_ref is None:
            t_count_ref = t_count
            h_count_ref = h_count
        if t_count != int(t_count_ref) or h_count != int(h_count_ref):
            raise ValueError("All runs must share C_snap shape [T,H,W] in selected split")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        grid = meta.get("grid", {})
        boundary = meta.get("boundary", {})
        dx_value = float(grid.get("dx", 0.0))
        if dx_value <= 0.0:
            raise ValueError("Invalid dx in run metadata: " + str(run_dir))
        depth_value = float((h_count - 1) * dx_value)
        t_end_value = float(t_curve[-1])
        if t_end_value <= 0.0:
            raise ValueError("Invalid t_end in run fields: " + str(run_dir))

        d1d = np.mean(d_field, axis=1).astype(np.float32)
        y_star = np.linspace(0.0, 1.0, h_count, dtype=np.float32)
        d1d_dy = np.gradient(d1d.astype(np.float64), y_star.astype(np.float64), edge_order=1).astype(np.float32)
        k1d = np.mean(k_field, axis=1).astype(np.float32)
        c1d = np.mean(c_snap, axis=2).astype(np.float32)

        # Convert simulator concentration into dimensionless c*=C/C0 for PINN training.
        c0_value = float(boundary.get("C0", 0.0))
        c0_safe = max(c0_value, 1e-12)
        c_star = np.clip(c1d / c0_safe, a_min=0.0, a_max=None).astype(np.float32)
        # Normalize time to [0,1] so one model can cover different t_end values.
        t_norm = np.asarray(t_curve / t_end_value, dtype=np.float32)

        mode_text = str(boundary.get("mode", "infinite_dose"))
        mode_rows.append(1.0 if mode_text == "time_decay" else 0.0)
        patch_width_rows.append(float(boundary.get("patch_width", 0.5)))
        c0_rows.append(c0_value)
        decay_rows.append(float(boundary.get("decay_rate", 0.0)))
        depth_rows.append(depth_value)
        t_end_rows.append(t_end_value)
        d1d_rows.append(d1d)
        d1d_dy_rows.append(d1d_dy)
        k1d_rows.append(k1d)
        c_star_rows.append(c_star)
        t_norm_rows.append(t_norm)
        t_phys_rows.append(np.asarray(t_curve, dtype=np.float32))

        row_index += 1
        if row_index == row_count or row_index % 50 == 0:
            print("loaded run physics", row_index, "/", row_count)

    return {
        "d1d": np.asarray(d1d_rows, dtype=np.float32),
        "d1d_dy_star": np.asarray(d1d_dy_rows, dtype=np.float32),
        "k1d": np.asarray(k1d_rows, dtype=np.float32),
        "c_star": np.asarray(c_star_rows, dtype=np.float32),
        "t_norm": np.asarray(t_norm_rows, dtype=np.float32),
        "t_phys": np.asarray(t_phys_rows, dtype=np.float32),
        "patch_width": np.asarray(patch_width_rows, dtype=np.float32),
        "c0": np.asarray(c0_rows, dtype=np.float32),
        "decay": np.asarray(decay_rows, dtype=np.float32),
        "mode_flag": np.asarray(mode_rows, dtype=np.float32),
        "depth": np.asarray(depth_rows, dtype=np.float32),
        "t_end": np.asarray(t_end_rows, dtype=np.float32),
    }


def pack_for_training(features, j_true, j_base, phys):
    # Keep all per-run arrays in one structure before moving to torch tensors.
    return {
        "x_feat": np.asarray(features, dtype=np.float32),
        "j_true": np.asarray(j_true, dtype=np.float32),
        "j_base": np.asarray(j_base, dtype=np.float32),
        "d1d": np.asarray(phys["d1d"], dtype=np.float32),
        "d1d_dy_star": np.asarray(phys["d1d_dy_star"], dtype=np.float32),
        "k1d": np.asarray(phys["k1d"], dtype=np.float32),
        "c_star": np.asarray(phys["c_star"], dtype=np.float32),
        "t_norm": np.asarray(phys["t_norm"], dtype=np.float32),
        "t_phys": np.asarray(phys["t_phys"], dtype=np.float32),
        "patch_width": np.asarray(phys["patch_width"], dtype=np.float32),
        "c0": np.asarray(phys["c0"], dtype=np.float32),
        "decay": np.asarray(phys["decay"], dtype=np.float32),
        "mode_flag": np.asarray(phys["mode_flag"], dtype=np.float32),
        "depth": np.asarray(phys["depth"], dtype=np.float32),
        "t_end": np.asarray(phys["t_end"], dtype=np.float32),
    }


def tensorize_pack(np_pack, device):
    # Move numpy arrays to the selected device with one predictable dtype.
    out = {}
    for key in np_pack:
        value = np_pack[key]
        out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return out
