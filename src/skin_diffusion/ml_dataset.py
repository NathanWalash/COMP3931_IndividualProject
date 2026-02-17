import json
from pathlib import Path

import numpy as np

from skin_diffusion.utils import ensure_dir, write_json


FEATURE_NAMES = [
    "patch_width",
    "patch_offset",
    "C0",
    "decay_rate",
    "heterogeneity_sigma",
    "heterogeneity_steps",
    "dose_proxy_c0_over_decay",
    "log_decay_rate",
    "width_times_sigma",
    "D_mean",
    "D_std",
    "D_p10",
    "D_p50",
    "D_p90",
    "D_top_mean",
    "D_bottom_mean",
]


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
    # read base sampled inputs from metadata
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
    base = {}
    base["patch_width"] = patch_width
    base["patch_offset"] = patch_offset
    base["C0"] = c0
    base["decay_rate"] = decay_rate
    base["heterogeneity_sigma"] = sigma
    base["heterogeneity_steps"] = steps
    return base


def read_d_stats_from_run(run_dir):
    # extract simple summary stats from run D field
    # these features help represent the sampled heterogeneity realization
    fields_path = Path(run_dir) / "fields.npz"
    if not fields_path.exists():
        raise ValueError("Missing fields.npz in run_dir: " + str(run_dir))

    fields = np.load(fields_path)
    D = np.asarray(fields["D"], dtype=float)
    flat = D.ravel()

    stats = {}
    stats["D_mean"] = float(np.mean(flat))
    stats["D_std"] = float(np.std(flat))
    stats["D_p10"] = float(np.percentile(flat, 10.0))
    stats["D_p50"] = float(np.percentile(flat, 50.0))
    stats["D_p90"] = float(np.percentile(flat, 90.0))
    stats["D_top_mean"] = float(np.mean(D[0, :]))
    stats["D_bottom_mean"] = float(np.mean(D[-1, :]))
    return stats


def build_feature_row(base, d_stats):
    # combine base inputs + derived terms + D summaries into one vector
    decay = base["decay_rate"]
    if decay <= 0.0:
        raise ValueError("decay_rate must be > 0 for feature building")

    derived = {}
    derived["dose_proxy_c0_over_decay"] = base["C0"] / decay
    derived["log_decay_rate"] = float(np.log(decay))
    derived["width_times_sigma"] = base["patch_width"] * base["heterogeneity_sigma"]

    values = []
    for name in FEATURE_NAMES:
        if name in base:
            values.append(float(base[name]))
        elif name in derived:
            values.append(float(derived[name]))
        else:
            values.append(float(d_stats[name]))
    return values


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
        base = read_features_from_meta(meta)
        d_stats = read_d_stats_from_run(entry["run_dir"])

        X_rows.append(build_feature_row(base, d_stats))
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
    feature_names = FEATURE_NAMES
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
