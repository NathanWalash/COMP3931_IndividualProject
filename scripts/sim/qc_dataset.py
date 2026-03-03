import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_index_from_path(run_dir):
    # Parse run index from run directory path.
    name = Path(run_dir).name
    if not name.startswith("run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def in_index_range(run_idx, start_idx, end_idx):
    # Inclusive range filter; None means unbounded.
    if run_idx is None:
        return False
    if start_idx is not None and run_idx < start_idx:
        return False
    if end_idx is not None and run_idx > end_idx:
        return False
    return True


def patch_offset_label(value):
    # Map numeric offset to a readable label for plots/reports.
    if not np.isfinite(value):
        return "nan"
    if np.isclose(value, 0.0):
        return "left"
    if np.isclose(value, 0.5):
        return "center"
    if np.isclose(value, 1.0):
        return "right"
    return f"{float(value):.2f}"


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

    split_index = index["splits"]
    for split_name in ["train", "val", "test"]:
        for entry in split_index[split_name]:
            rows.append(copy_entry_with_split(entry, split_name))

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
    if "k_dermis" in sample:
        row["k_dermis"] = to_float_or_nan(sample.get("k_dermis"))
    else:
        layers = extras.get("layers", {})
        row["k_dermis"] = to_float_or_nan(layers.get("k_dermis"))
    row["heterogeneity_sigma"] = to_float_or_nan(sample.get("heterogeneity_sigma"))
    row["heterogeneity_steps"] = to_float_or_nan(sample.get("heterogeneity_steps"))

    row["P"] = to_float_or_nan(metrics.get("P"))
    row["Tlag"] = to_float_or_nan(metrics.get("Tlag"))
    row["J_ss"] = to_float_or_nan(metrics.get("J_ss"))
    row["AUC_J"] = to_float_or_nan(metrics.get("AUC_J"))
    row["J_peak"] = to_float_or_nan(metrics.get("J_peak"))
    row["t_peak"] = to_float_or_nan(metrics.get("t_peak"))
    row["M_delivered_24h"] = to_float_or_nan(metrics.get("M_delivered_24h"))
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
    out["train"] = []
    out["val"] = []
    out["test"] = []

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
    expected["train"] = int(counts.get("train", -1))
    expected["val"] = int(counts.get("val", -1))
    expected["test"] = int(counts.get("test", -1))

    actual = {}
    actual["train"] = len(rows_by_split["train"])
    actual["val"] = len(rows_by_split["val"])
    actual["test"] = len(rows_by_split["test"])

    checks["count_match"] = {"expected": expected, "actual": actual}

    # 2) Leakage check: a run must not appear in more than one split.
    # Same run in multiple splits would leak information.
    split_names = ["train", "val", "test"]
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
        "k_dermis",
        "C0",
        "decay_rate",
        "heterogeneity_sigma",
        "heterogeneity_steps",
        "P",
        "J_ss",
        "AUC_J",
        "J_peak",
        "t_peak",
        "M_delivered_24h",
        "Tlag",
    ]

    # Fixed bins for discrete fields keep plots stable across runs.
    bins_by_field = {}
    bins_by_field["patch_width"] = np.array([0.0, 0.3, 0.75, 1.1])
    bins_by_field["patch_offset"] = np.array([-0.1, 0.17, 0.67, 1.1])
    bins_by_field["k_dermis"] = np.array([-1.0e-8, 0.5e-5, 2.0e-5, 6.0e-5, 1.5e-4])
    bins_by_field["heterogeneity_steps"] = np.arange(2.5, 10.6, 1.0)

    split_order = ["train", "val", "test"]
    ncols = 4
    nrows = int(np.ceil(float(len(fields)) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for i in range(len(fields)):
        field = fields[i]
        # One subplot per field, overlaid by split.
        ax = axes[i]
        if field == "patch_offset":
            labels = ["left", "center", "right"]
            x = np.arange(len(labels))
            width = 0.2
            for split_i in range(len(split_order)):
                split_name = split_order[split_i]
                values = []
                for row in rows_by_split[split_name]:
                    values.append(row[field])
                vals = np.asarray(values, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue

                counts = []
                for label in labels:
                    count = 0
                    for val in vals:
                        if patch_offset_label(val) == label:
                            count += 1
                    counts.append(count)
                pos = x + (split_i - 1.5) * width
                ax.bar(pos, counts, width=width, alpha=0.8, label=split_name)

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title("patch_offset (discrete)")
            ax.grid(alpha=0.2)
            continue

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

    for i in range(len(fields), len(axes)):
        axes[i].axis("off")

    axes[0].legend(fontsize=8)
    fig.suptitle("Dataset QC Distributions (train/val/test)")
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
        "k_dermis",
        "C0",
        "decay_rate",
        "heterogeneity_sigma",
        "heterogeneity_steps",
        "P",
        "J_ss",
        "AUC_J",
        "J_peak",
        "t_peak",
        "M_delivered_24h",
        "Tlag",
    ]
    scalar_targets = [
        "P",
        "Tlag",
        "J_ss",
        "AUC_J",
        "J_peak",
        "t_peak",
        "M_delivered_24h",
    ]

    stats = {}
    target_coverage = {}
    for split_name, split_rows_list in rows_by_split.items():
        # Basic numeric summary for each field within each split.
        split_stats = {}
        for field in fields:
            values = []
            for row in split_rows_list:
                values.append(row[field])
            split_stats[field] = scalar_stats(values)
        stats[split_name] = split_stats

        # Explicit target coverage helps catch sparse supervision (e.g., Tlag).
        split_cov = {}
        total_rows = len(split_rows_list)
        for target in scalar_targets:
            finite_count = split_stats[target]["n"]
            if total_rows == 0:
                finite_fraction = None
            else:
                finite_fraction = float(finite_count) / float(total_rows)
            split_cov[target] = {
                "finite_n": int(finite_count),
                "total_n": int(total_rows),
                "finite_fraction": finite_fraction,
            }
        target_coverage[split_name] = split_cov

    # Patch-width counts make split-balance checks easy to inspect.
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

    # Patch-offset counts make discrete placement balance easy to inspect.
    patch_offset_counts = {}
    for split_name, split_rows_list in rows_by_split.items():
        split_counts = {"left": 0, "center": 0, "right": 0}
        for row in split_rows_list:
            label = patch_offset_label(row["patch_offset"])
            if label in split_counts:
                split_counts[label] += 1
        patch_offset_counts[split_name] = split_counts

    # k_dermis counts are important for clearance-sensitivity studies.
    k_dermis_counts = {}
    for split_name, split_rows_list in rows_by_split.items():
        values = []
        for row in split_rows_list:
            values.append(row["k_dermis"])

        if len(values) == 0:
            keys = []
            counts = []
        else:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            rounded = np.round(arr, 8)
            keys, counts = np.unique(rounded, return_counts=True)

        split_counts = {}
        for idx in range(len(keys)):
            key = str(float(keys[idx]))
            value = int(counts[idx])
            split_counts[key] = value
        k_dermis_counts[split_name] = split_counts

    report = {}
    report["counts"] = index.get("counts", {})
    report["checks"] = checks
    report["patch_width_counts"] = patch_counts
    report["patch_offset_counts"] = patch_offset_counts
    report["k_dermis_counts"] = k_dermis_counts
    report["stats"] = stats
    report["target_coverage"] = target_coverage
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
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
    # Optional run-index filter for subset QC on large datasets.
    if args.run_start_index is not None or args.run_end_index is not None:
        filtered_rows = []
        for entry in base_rows:
            run_idx = run_index_from_path(entry["run_dir"])
            if in_index_range(run_idx, args.run_start_index, args.run_end_index):
                filtered_rows.append(entry)
        base_rows = filtered_rows

    if len(base_rows) == 0:
        raise ValueError("No rows found in requested run index range")

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
