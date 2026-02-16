import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    # Read JSON from disk.
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def patch_offset_to_float(value):
    # Allow written offsets in configs (left/center/right).
    # We normalize everything to float so downstream code is simple.
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


def to_float_or_nan(value):
    # Keep missing values as NaN so stats code can ignore them cleanly.
    if value is None:
        return np.nan
    return float(value)


def copy_entry_with_split(entry, split_label):
    # Copy only fields we need from index.json entry and add split label.
    row = {}
    row["split"] = split_label
    row["row"] = entry["row"]
    row["run_dir"] = entry["run_dir"]
    row["meta_path"] = entry["meta_path"]
    row["metrics_path"] = entry["metrics_path"]
    row["patch_width"] = entry["patch_width"]
    return row


def collect_rows(index):
    # Flatten all split entries into one list.
    rows = []

    id_index = index["id_index"]
    for split_name in ["train", "val", "test"]:
        split_label = "id_" + split_name
        for entry in id_index[split_name]:
            rows.append(copy_entry_with_split(entry, split_label))

    ood_entries = index.get("ood_index", [])
    for entry in ood_entries:
        rows.append(copy_entry_with_split(entry, "ood_primary"))

    return rows


def enrich_row(entry):
    # Add sampled parameters and scalar targets for a single run.
    # Keep this mapping explicit so it is easy to debug missing fields.
    meta = load_json(entry["meta_path"])
    metrics = load_json(entry["metrics_path"])

    extras = meta.get("extras", {})
    sample = extras.get("dataset_sample", {})

    row = {}
    row["split"] = entry["split"]
    row["run_dir"] = entry["run_dir"]

    row["patch_width"] = to_float_or_nan(sample.get("patch_width"))
    row["patch_offset"] = patch_offset_to_float(sample.get("patch_offset", np.nan))
    row["C0"] = to_float_or_nan(sample.get("C0"))
    row["decay_rate"] = to_float_or_nan(sample.get("decay_rate"))
    row["heterogeneity_sigma"] = to_float_or_nan(sample.get("heterogeneity_sigma"))
    row["heterogeneity_steps"] = to_float_or_nan(sample.get("heterogeneity_steps"))

    row["P"] = to_float_or_nan(metrics.get("P"))
    row["Tlag"] = to_float_or_nan(metrics.get("Tlag"))
    row["J_ss"] = to_float_or_nan(metrics.get("J_ss"))
    return row


def scalar_stats(values):
    # Return simple summary stats for one numeric field.
    # NaNs are removed first so missing values do not distort QC.
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None}

    stats = {}
    stats["n"] = int(arr.size)
    stats["min"] = float(np.min(arr))
    stats["max"] = float(np.max(arr))
    stats["mean"] = float(np.mean(arr))
    stats["std"] = float(np.std(arr))
    return stats


def split_rows(rows):
    # Group rows by split name.
    # Fixed keys keep report shape stable across runs.
    out = {}
    out["id_train"] = []
    out["id_val"] = []
    out["id_test"] = []
    out["ood_primary"] = []

    for row in rows:
        split_name = row["split"]
        out[split_name].append(row)
    return out


def check_split_rules(index, rows_by_split):
    # Check split assumptions that training relies on.
    # if these checks fail, model metrics are not trustworthy
    checks = {}

    # 1) Split counts should match index.json.
    # This catches partial exports or bad indexing.
    counts = index.get("counts", {})
    expected = {}
    expected["id_train"] = int(counts.get("train", -1))
    expected["id_val"] = int(counts.get("val", -1))
    expected["id_test"] = int(counts.get("test", -1))
    expected["ood_primary"] = int(counts.get("ood_total", -1))

    actual = {}
    actual["id_train"] = len(rows_by_split["id_train"])
    actual["id_val"] = len(rows_by_split["id_val"])
    actual["id_test"] = len(rows_by_split["id_test"])
    actual["ood_primary"] = len(rows_by_split["ood_primary"])

    checks["count_match"] = {"expected": expected, "actual": actual}

    # 2) OOD policy: OOD patch width must be excluded from ID splits.
    # ID should not contain the OOD patch width value.
    ood_value = float(index["ood"]["value"])

    id_patch_widths = []
    for split_name in ["id_train", "id_val", "id_test"]:
        for row in rows_by_split[split_name]:
            id_patch_widths.append(row["patch_width"])

    ood_patch_widths = []
    for row in rows_by_split["ood_primary"]:
        ood_patch_widths.append(row["patch_width"])

    id_contains_ood_value = bool(np.any(np.isclose(id_patch_widths, ood_value)))

    if len(ood_patch_widths) == 0:
        ood_all_match_value = False
    else:
        ood_all_match_value = bool(np.all(np.isclose(ood_patch_widths, ood_value)))

    ood_rule = {}
    ood_rule["ood_value"] = ood_value
    ood_rule["id_contains_ood_value"] = id_contains_ood_value
    ood_rule["ood_all_match_value"] = ood_all_match_value
    checks["ood_rule"] = ood_rule

    # 3) Leakage check: a run must not appear in more than one split.
    # Same run in multiple splits would leak information.
    split_names = ["id_train", "id_val", "id_test", "ood_primary"]
    run_sets = {}
    for split_name in split_names:
        run_set = set()
        for row in rows_by_split[split_name]:
            run_set.add(row["run_dir"])
        run_sets[split_name] = run_set

    overlap = {}
    for i in range(len(split_names)):
        a = split_names[i]
        for j in range(i + 1, len(split_names)):
            b = split_names[j]
            inter = sorted(run_sets[a].intersection(run_sets[b]))
            if len(inter) > 0:
                key = a + "__" + b
                overlap[key] = inter
    checks["overlap"] = overlap

    return checks


def make_histograms(rows, out_dir):
    # Save one figure showing feature/target distributions per split.
    # This is a quick visual sanity check before training.
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = split_rows(rows)

    fields = [
        "patch_width",
        "patch_offset",
        "C0",
        "decay_rate",
        "heterogeneity_sigma",
        "heterogeneity_steps",
        "P",
        "Tlag",
        "J_ss",
    ]

    # Fixed bins for discrete fields keep plots stable across runs.
    bins_by_field = {}
    bins_by_field["patch_width"] = np.array([0.0, 0.3, 0.75, 1.1])
    bins_by_field["patch_offset"] = np.array([-0.1, 0.17, 0.67, 1.1])
    bins_by_field["heterogeneity_steps"] = np.arange(2.5, 10.6, 1.0)

    split_order = ["id_train", "id_val", "id_test", "ood_primary"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)

    for i in range(len(fields)):
        field = fields[i]
        # One subplot per field, overlaid by split.
        ax = axes[i // 3][i % 3]
        for split_name in split_order:
            values = []
            for row in rows_by_split[split_name]:
                values.append(row[field])
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            if field in bins_by_field:
                bins = bins_by_field[field]
            else:
                bins = 20
            ax.hist(vals, bins=bins, alpha=0.35, label=split_name)

        ax.set_title(field)
        ax.grid(alpha=0.2)

    axes[0][0].legend(fontsize=8)
    fig.suptitle("Dataset QC Distributions By Split")
    fig.savefig(out_dir / "qc_distributions.png", dpi=150)
    plt.close(fig)


def build_report(index, rows):
    # Build a machine-readable QC report.
    # Keep this JSON compact but enough for quick checks.
    rows_by_split = split_rows(rows)
    checks = check_split_rules(index, rows_by_split)

    # Same field list for each split report.
    fields = [
        "patch_width",
        "patch_offset",
        "C0",
        "decay_rate",
        "heterogeneity_sigma",
        "heterogeneity_steps",
        "P",
        "Tlag",
        "J_ss",
    ]

    stats = {}
    for split_name, split_rows_list in rows_by_split.items():
        # Basic numeric summary for each field within each split.
        split_stats = {}
        for field in fields:
            values = []
            for row in split_rows_list:
                values.append(row[field])
            split_stats[field] = scalar_stats(values)
        stats[split_name] = split_stats

    # Patch-width counts make OOD holdout checks easy to inspect.
    patch_counts = {}
    for split_name, split_rows_list in rows_by_split.items():
        values = []
        for row in split_rows_list:
            values.append(row["patch_width"])

        if len(values) == 0:
            keys = []
            counts = []
        else:
            rounded = np.round(values, 4)
            keys, counts = np.unique(rounded, return_counts=True)

        split_counts = {}
        for idx in range(len(keys)):
            key = str(float(keys[idx]))
            value = int(counts[idx])
            split_counts[key] = value
        patch_counts[split_name] = split_counts

    report = {}
    report["counts"] = index.get("counts", {})
    report["checks"] = checks
    report["patch_width_counts"] = patch_counts
    report["stats"] = stats
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    # Default output folder is inside the processed dataset folder.
    processed_dir = Path(args.processed_dir)
    if args.out_dir is None:
        out_dir = processed_dir / "qc"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # index.json is the source of truth for split membership.
    # We do not infer splits from folders.
    index = load_json(processed_dir / "index.json")
    base_rows = collect_rows(index)

    # Add sampled parameters and targets for each run.
    rows = []
    for entry in base_rows:
        rows.append(enrich_row(entry))

    # Write the QC report and distribution plot.
    report = build_report(index, rows)
    report_path = out_dir / "qc_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_histograms(rows, out_dir)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
