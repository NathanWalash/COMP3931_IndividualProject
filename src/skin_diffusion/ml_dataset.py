import json
from pathlib import Path

import numpy as np

from skin_diffusion.utils import ensure_dir, write_json


def load_json_file(path):
    # small helper so json reads are consistent
    return json.loads(Path(path).read_text(encoding="utf-8"))


def patch_offset_to_float(value):
    # allow both numeric and left/center/right styles
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "left":
            return 0.0
        if text == "center":
            return 0.5
        if text == "right":
            return 1.0
        return float(text)
    return float(value)


def read_features_from_meta(meta):
    # require sampled values from the new dataset spec workflow
    extras = meta.get("extras", {})
    sample = extras.get("dataset_sample", {})

    if not sample:
        raise ValueError(
            "Missing extras.dataset_sample in meta.json. "
            "Rebuild runs with the dataset spec workflow."
        )

    patch_width = float(sample["patch_width"])
    patch_offset = patch_offset_to_float(sample["patch_offset"])
    c0 = float(sample["C0"])
    decay_rate = float(sample["decay_rate"])
    sigma = float(sample["heterogeneity_sigma"])
    steps = float(sample["heterogeneity_steps"])
    return [patch_width, patch_offset, c0, decay_rate, sigma, steps]


def read_scalar_targets(metrics):
    # some runs can have missing lag-time; store as NaN for filtering later
    p_val = metrics.get("P")
    tlag_val = metrics.get("Tlag")
    jss_val = metrics.get("J_ss")

    if p_val is None:
        p_val = np.nan
    if tlag_val is None:
        tlag_val = np.nan
    if jss_val is None:
        jss_val = np.nan

    return [
        float(p_val),
        float(tlag_val),
        float(jss_val),
    ]


def build_ml_split(split_npz_path, index_entries):
    # read curve targets from processed split arrays
    split_npz_path = Path(split_npz_path)
    fields = np.load(split_npz_path)

    # J and t are direct targets for curve models
    J = fields["J"]
    t = fields["t"]

    if len(index_entries) != J.shape[0]:
        raise ValueError("Index length does not match split size: " + str(split_npz_path))

    # build tabular inputs + scalar targets from per-run metadata
    X_rows = []
    y_scalar_rows = []
    row_meta = []

    for entry in index_entries:
        meta = load_json_file(entry["meta_path"])
        metrics = load_json_file(entry["metrics_path"])

        X_rows.append(read_features_from_meta(meta))
        y_scalar_rows.append(read_scalar_targets(metrics))
        row_meta.append(
            {
                "run_dir": entry["run_dir"],
                "meta_path": entry["meta_path"],
                "metrics_path": entry["metrics_path"],
            }
        )

    X = np.array(X_rows, dtype=float)
    y_scalar = np.array(y_scalar_rows, dtype=float)

    data = {}
    data["X"] = X
    data["y_scalar"] = y_scalar
    data["J"] = J
    data["t"] = t
    data["row_meta"] = row_meta
    return data


def save_ml_split(out_path, data):
    # save minimal training arrays for one split
    np.savez(
        out_path,
        X=data["X"],
        y_scalar=data["y_scalar"],
        J=data["J"],
        t=data["t"],
    )


def export_ml_ready_dataset(processed_dir, out_dir):
    # convert assembled ID/OOD outputs into model-ready split files
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    index = load_json_file(processed_dir / "index.json")

    # keep names next to arrays for training scripts
    feature_names = [
        "patch_width",
        "patch_offset",
        "C0",
        "decay_rate",
        "heterogeneity_sigma",
        "heterogeneity_steps",
    ]
    scalar_target_names = ["P", "Tlag", "J_ss"]

    id_index = index["id_index"]
    # split names mapped to source files + split index entries
    split_defs = [
        ("id_train", processed_dir / "id" / "v3_train.npz", id_index["train"]),
        ("id_val", processed_dir / "id" / "v3_val.npz", id_index["val"]),
        ("id_test", processed_dir / "id" / "v3_test.npz", id_index["test"]),
        ("ood_primary", processed_dir / "ood" / "v3_ood_primary.npz", index["ood_index"]),
    ]

    summary = {}
    summary["feature_names"] = feature_names
    summary["scalar_target_names"] = scalar_target_names
    summary["splits"] = {}

    for split_name, split_path, split_index in split_defs:
        # skip missing or empty splits
        if not split_path.exists():
            continue

        if len(split_index) == 0:
            continue

        # build and save one split
        split_data = build_ml_split(split_path, split_index)
        out_path = out_dir / f"{split_name}.npz"
        save_ml_split(out_path, split_data)

        summary["splits"][split_name] = {
            "rows": int(split_data["X"].shape[0]),
            "file": str(out_path),
            "index_rows": len(split_data["row_meta"]),
            "index": split_data["row_meta"],
        }

    # write one summary file with shapes and row mapping
    write_json(out_dir / "meta.json", summary)
    return out_dir
