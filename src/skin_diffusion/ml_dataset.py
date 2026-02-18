import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

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
    "c0_times_width",
    "decay_times_width",
    "sigma_times_width",
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
    # extra low-cost interaction terms for better surrogate fit
    derived["c0_times_width"] = base["C0"] * base["patch_width"]
    derived["decay_times_width"] = base["decay_rate"] * base["patch_width"]
    derived["sigma_times_width"] = base["heterogeneity_sigma"] * base["patch_width"]

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


def patch_offset_bucket(value):
    # Keep offset categories coarse for stable stratification.
    if value <= 0.25:
        return "left"
    if value >= 0.75:
        return "right"
    return "center"


def jss_bucket(jss_values):
    # Two bins are usually enough and avoid tiny strata classes.
    finite = np.isfinite(jss_values)
    result = np.array(["nan"] * len(jss_values), dtype=object)
    if not np.any(finite):
        return result

    safe = np.clip(jss_values[finite], 1e-30, None)
    log_vals = np.log10(safe)
    threshold = np.median(log_vals)
    labels = np.where(log_vals <= threshold, "low", "high")
    result[finite] = labels
    return result


def build_id_strata(index_entries):
    # Build one simple stratification label per ID row.
    patch_width_labels = []
    patch_offset_labels = []
    jss_values = []

    for entry in index_entries:
        meta = load_json_file(entry["meta_path"])
        metrics = load_json_file(entry["metrics_path"])
        base = read_features_from_meta(meta)

        width_value = float(base["patch_width"])
        if width_value <= 0.75:
            patch_width_labels.append("w05")
        else:
            patch_width_labels.append("w10")

        patch_offset_labels.append(patch_offset_bucket(float(base["patch_offset"])))

        jss_value = metrics.get("J_ss")
        if jss_value is None:
            jss_values.append(np.nan)
        else:
            jss_values.append(float(jss_value))

    jss_values = np.asarray(jss_values, dtype=float)
    jss_labels = jss_bucket(jss_values)

    labels = []
    for i in range(len(index_entries)):
        parts = [patch_width_labels[i], patch_offset_labels[i], str(jss_labels[i])]
        labels.append("|".join(parts))
    return np.asarray(labels, dtype=object)


def stratified_id_split(index_entries, n_train, n_val, n_test, seed=42):
    # Rebuild ID train/val/test with stratification and original split counts.
    n_total = len(index_entries)
    if n_total != (n_train + n_val + n_test):
        raise ValueError("ID split counts do not sum to total ID rows")

    all_idx = np.arange(n_total, dtype=int)
    fine_strata = build_id_strata(index_entries)

    coarse_strata = []
    for label in fine_strata:
        parts = str(label).split("|")
        coarse_strata.append("|".join(parts[0:2]))
    coarse_strata = np.asarray(coarse_strata, dtype=object)

    width_only = []
    for label in fine_strata:
        parts = str(label).split("|")
        width_only.append(parts[0])
    width_only = np.asarray(width_only, dtype=object)

    # Try detailed strata first, then fall back to simpler options for tiny datasets.
    stratify_options = [fine_strata, coarse_strata, width_only, None]
    val_fraction = float(n_val) / float(n_val + n_test)

    for option in stratify_options:
        train_strat = option
        try:
            # First split: train vs temp (val+test).
            train_idx, temp_idx = train_test_split(
                all_idx,
                train_size=n_train,
                random_state=seed,
                shuffle=True,
                stratify=train_strat,
            )
        except ValueError:
            continue

        temp_strat = None
        if option is not None:
            temp_strat = option[temp_idx]

        try:
            # Second split: val vs test from temp.
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_fraction,
                random_state=seed,
                shuffle=True,
                stratify=temp_strat,
            )
            return train_idx, val_idx, test_idx
        except ValueError:
            continue

    raise ValueError("Unable to build ID split with requested sizes")


def subset_entries(entries, idx_array):
    selected = []
    for idx in idx_array:
        selected.append(entries[int(idx)])
    return selected


def subset_array(arr, idx_array):
    return arr[np.asarray(idx_array, dtype=int)]


def build_ml_split_from_arrays(J, t, index_entries):
    # Same as build_ml_split, but using in-memory J/t arrays.
    if len(index_entries) != J.shape[0]:
        raise ValueError("Index length does not match J rows")

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


def export_ml_ready_dataset(processed_dir, out_dir):
    # convert assembled ID/OOD outputs into model-ready split files
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    index = load_json_file(processed_dir / "index.json")

    # keep names next to arrays for training scripts
    feature_names = FEATURE_NAMES
    scalar_target_names = ["P", "Tlag", "J_ss"]

    # Load assembled ID arrays, then re-split them with stratification.
    # This keeps split sizes the same but improves balance across conditions.
    id_train_npz = np.load(processed_dir / "id" / "v3_train.npz")
    id_val_npz = np.load(processed_dir / "id" / "v3_val.npz")
    id_test_npz = np.load(processed_dir / "id" / "v3_test.npz")

    id_J_all = np.concatenate([id_train_npz["J"], id_val_npz["J"], id_test_npz["J"]], axis=0)
    id_t_all = np.concatenate([id_train_npz["t"], id_val_npz["t"], id_test_npz["t"]], axis=0)

    id_index = index["id_index"]
    id_entries_all = []
    for entry in id_index["train"]:
        id_entries_all.append(entry)
    for entry in id_index["val"]:
        id_entries_all.append(entry)
    for entry in id_index["test"]:
        id_entries_all.append(entry)

    n_train = len(id_index["train"])
    n_val = len(id_index["val"])
    n_test = len(id_index["test"])
    train_idx, val_idx, test_idx = stratified_id_split(id_entries_all, n_train, n_val, n_test, seed=42)

    split_defs = []
    split_defs.append(
        (
            "id_train",
            subset_array(id_J_all, train_idx),
            subset_array(id_t_all, train_idx),
            subset_entries(id_entries_all, train_idx),
        )
    )
    split_defs.append(
        (
            "id_val",
            subset_array(id_J_all, val_idx),
            subset_array(id_t_all, val_idx),
            subset_entries(id_entries_all, val_idx),
        )
    )
    split_defs.append(
        (
            "id_test",
            subset_array(id_J_all, test_idx),
            subset_array(id_t_all, test_idx),
            subset_entries(id_entries_all, test_idx),
        )
    )

    # OOD split stays exactly as assembled.
    ood_npz = np.load(processed_dir / "ood" / "v3_ood_primary.npz")
    split_defs.append(("ood_primary", ood_npz["J"], ood_npz["t"], index["ood_index"]))

    summary = {}
    summary["feature_names"] = feature_names
    summary["scalar_target_names"] = scalar_target_names
    summary["splits"] = {}

    for split_name, split_J, split_t, split_index in split_defs:
        # skip empty splits
        if len(split_index) == 0:
            continue

        # build and save one split
        split_data = build_ml_split_from_arrays(split_J, split_t, split_index)
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
