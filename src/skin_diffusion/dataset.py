import numpy as np
from pathlib import Path
import json

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from skin_diffusion.metrics import compute_bottom_flux
from skin_diffusion.run_utils import run_simulation
from skin_diffusion.utils import ensure_dir, write_json


def to_dict(obj):
    # turn a small config object into a plain dict
    data = {}
    # copy fields one by one so json is clean
    for key, val in vars(obj).items():
        data[key] = val
    return data


def metrics_summary(J):
    # simple stats for the flux curve
    summary = {}
    # basic min/max/mean/sum
    summary["J_min"] = float(np.min(J))
    summary["J_max"] = float(np.max(J))
    summary["J_mean"] = float(np.mean(J))
    summary["J_sum"] = float(np.sum(J))
    return summary


def load_run_bundle(run_dir, include_full_fields=True):
    # load one saved run from disk
    # in lightweight mode, keep only J/t to reduce RAM usage
    run_dir = Path(run_dir)
    fields = np.load(run_dir / "fields.npz")
    meta_path = run_dir / "meta.json"
    metrics = run_dir / "metrics.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    data = {}
    if include_full_fields:
        data["C_snap"] = fields["C_snap"]
        data["D"] = fields["D"]
        data["k"] = fields["k"]
        data["patch_mask"] = fields["patch_mask"]
    data["t"] = fields["t"]
    data["J"] = fields["J"]
    data["meta_path"] = str(meta_path)
    data["metrics_path"] = str(metrics)
    data["meta"] = meta
    # keep width available for OOD split decisions
    data["patch_width"] = extract_patch_width(meta)

    return data


def extract_patch_width(meta):
    # support both meta styles used in this repo:
    # 1) dataset bundle meta (boundary at top level)
    # 2) run_sim style meta (nested under config)
    if "boundary" in meta and "patch_width" in meta["boundary"]:
        return float(meta["boundary"]["patch_width"])

    config = meta.get("config", {})
    boundary = config.get("boundary", {})
    if "patch_width" in boundary:
        return float(boundary["patch_width"])

    raise KeyError("patch_width not found in run metadata")


def save_run_bundle(out_dir, cfg, C_snap, t_save, D_field, k_field, patch_mask, metrics, stability_info):
    # make output folder
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # always store a full k field
    if k_field is None:
        k_save = np.zeros_like(D_field)
    else:
        k_save = k_field

    # make sure we have J and t
    # if metrics did not give J, compute it now
    if metrics is None or "J" not in metrics:
        J = compute_bottom_flux(C_snap, D_field, cfg.grid.dx)
        t = t_save
    else:
        J = np.array(metrics["J"], dtype=float)
        t = np.array(metrics.get("t", t_save), dtype=float)

    # save fields
    # fields.npz is the main data file for each run
    fields_path = out_dir / "fields.npz"
    np.savez(
        fields_path,
        C_snap=C_snap,
        D=D_field,
        k=k_save,
        patch_mask=patch_mask,
        t=t,
        J=J,
    )

    # meta info
    # meta.json holds the run settings
    meta = {}
    meta["grid"] = to_dict(cfg.grid)
    meta["boundary"] = to_dict(cfg.boundary)
    meta["seed"] = cfg.seed
    meta["regime"] = cfg.regime_name
    meta["extras"] = cfg.extras
    meta["stability"] = stability_info

    meta_path = out_dir / "meta.json"
    write_json(meta_path, meta)

    # metrics file
    # metrics.json is just a small summary file
    metrics_out = {}
    metric_names = [
        "P",
        "Tlag",
        "J_ss",
        "AUC_J",
        "J_peak",
        "t_peak",
        "M_delivered_24h",
    ]
    # pull metrics if they exist
    if metrics is None:
        for name in metric_names:
            metrics_out[name] = None
    else:
        for name in metric_names:
            metrics_out[name] = metrics.get(name)
    metrics_out.update(metrics_summary(J))

    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics_out)

    return fields_path, meta_path, metrics_path


def generate_run(cfg, out_dir):
    # run one simulation and save all files
    # this is the one-call helper used by the CLI
    C_snap, t_save, D_field, k_field, patch_mask, diagnostics, metrics, stability_info = run_simulation(cfg)

    fields_path, meta_path, metrics_path = save_run_bundle(
        out_dir,
        cfg,
        C_snap,
        t_save,
        D_field,
        k_field,
        patch_mask,
        metrics,
        stability_info,
    )

    return fields_path, meta_path, metrics_path


def validate_run(out_dir):
    # simple check for keys and shapes
    # this helps catch broken datasets early
    out_dir = Path(out_dir)
    fields_path = out_dir / "fields.npz"
    data = np.load(fields_path)

    needed = ["C_snap", "D", "k", "patch_mask", "t", "J"]
    # keys must exist
    for key in needed:
        if key not in data:
            raise ValueError("missing key: " + key)

    C_snap = data["C_snap"]
    D = data["D"]
    k = data["k"]
    patch_mask = data["patch_mask"]
    t = data["t"]
    J = data["J"]

    if C_snap.ndim != 3:
        raise ValueError("C_snap must be [T, H, W]")
    if D.shape != C_snap[0].shape:
        raise ValueError("D shape mismatch")
    if k.shape != C_snap[0].shape:
        raise ValueError("k shape mismatch")
    if patch_mask.shape != C_snap[0].shape:
        raise ValueError("patch_mask shape mismatch")
    if t.ndim != 1 or J.ndim != 1:
        raise ValueError("t and J must be 1D arrays")
    if len(t) != C_snap.shape[0] or len(J) != C_snap.shape[0]:
        raise ValueError("t/J length mismatch")

    return True


def stack_runs(runs, include_full_fields=True):
    # stack fields from a list of run dicts
    # full mode stacks all spatial fields; lightweight stacks only J/t
    if len(runs) == 0:
        return None

    if include_full_fields:
        C_list = []
        D_list = []
        k_list = []
        mask_list = []

    t_list = []
    J_list = []

    for r in runs:
        if include_full_fields:
            C_list.append(r["C_snap"])
            D_list.append(r["D"])
            k_list.append(r["k"])
            mask_list.append(r["patch_mask"])
        t_list.append(r["t"])
        J_list.append(r["J"])

    print("stacking arrays...")
    data = {}
    if include_full_fields:
        data["C_snap"] = np.stack(C_list, axis=0)
        data["D"] = np.stack(D_list, axis=0)
        data["k"] = np.stack(k_list, axis=0)
        data["patch_mask"] = np.stack(mask_list, axis=0)
    data["t"] = np.stack(t_list, axis=0)
    data["J"] = np.stack(J_list, axis=0)

    return data


def select_id_ood_runs(runs, ood_param="patch_width", ood_value=0.25, tol=1e-12):
    # simple holdout split: runs matching ood_value go to OOD
    if ood_param != "patch_width":
        raise ValueError("Only patch_width OOD split is supported")

    id_runs = []
    ood_runs = []
    for run in runs:
        val = float(run["patch_width"])
        if abs(val - float(ood_value)) <= tol:
            ood_runs.append(run)
        else:
            id_runs.append(run)

    return id_runs, ood_runs


def save_split_npz(path, split_data):
    # skip empty splits
    if split_data is None:
        return False

    # write only available keys so lightweight mode can skip large arrays
    save_data = {}
    if "C_snap" in split_data:
        save_data["C_snap"] = split_data["C_snap"]
    if "D" in split_data:
        save_data["D"] = split_data["D"]
    if "k" in split_data:
        save_data["k"] = split_data["k"]
    if "patch_mask" in split_data:
        save_data["patch_mask"] = split_data["patch_mask"]
    save_data["t"] = split_data["t"]
    save_data["J"] = split_data["J"]

    np.savez(path, **save_data)
    return True


def build_index_entries(runs):
    # keep index plain and easy to read in json
    entries = []
    i = 0
    for run in runs:
        entry = {}
        entry["row"] = i
        entry["run_dir"] = str(Path(run["meta_path"]).parent)
        entry["meta_path"] = run["meta_path"]
        entry["metrics_path"] = run["metrics_path"]
        entry["patch_width"] = float(run["patch_width"])
        entries.append(entry)
        i += 1
    return entries


def split_indices(n, split_seed, train_frac, val_frac):
    # make a deterministic split
    if n < 0:
        raise ValueError("n must be >= 0")

    if n < 2:
        raise ValueError("Need at least 2 samples to split")

    idx = np.arange(n)
    test_frac = 1.0 - train_frac - val_frac
    # tolerate tiny floating-point drift when fractions should sum to 1.0
    eps = 1e-12
    if test_frac < -eps:
        raise ValueError("train_frac + val_frac must be <= 1.0")
    if abs(test_frac) <= eps:
        test_frac = 0.0

    # split into train and temp (val+test)
    train_idx, temp_idx = train_test_split(
        idx,
        train_size=train_frac,
        random_state=split_seed,
        shuffle=True,
    )

    # split temp into val and test
    if test_frac == 0.0:
        val_idx = temp_idx
        test_idx = np.array([], dtype=int)
    else:
        # handle tiny temp sets cleanly
        if len(temp_idx) < 2:
            if val_frac > 0.0:
                val_idx = temp_idx
                test_idx = np.array([], dtype=int)
            else:
                val_idx = np.array([], dtype=int)
                test_idx = temp_idx
        else:
            # use integer count for tiny temp splits to avoid float rounding issues
            n_temp = len(temp_idx)
            val_ratio = val_frac / (val_frac + test_frac)
            n_val = int(round(val_ratio * n_temp))
            if n_val < 1:
                n_val = 1
            if n_val >= n_temp:
                n_val = n_temp - 1

            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=n_val,
                random_state=split_seed,
                shuffle=True,
            )

    return train_idx, val_idx, test_idx


def assemble_processed_dataset(run_dirs, out_dir, split_seed=123, train_frac=0.8, val_frac=0.1):
    # build processed train/val/test npz files
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # load all runs
    runs = []
    for rd in tqdm(run_dirs, desc="loading runs"):
        runs.append(load_run_bundle(rd))

    # split indices
    train_idx, val_idx, test_idx = split_indices(
        len(runs), split_seed, train_frac, val_frac
    )

    # index mapping so each row is traceable
    index = []
    i = 0
    for r in runs:
        entry = {}
        entry["row"] = i
        entry["run_dir"] = str(Path(r["meta_path"]).parent)
        entry["meta_path"] = r["meta_path"]
        entry["metrics_path"] = r["metrics_path"]
        index.append(entry)
        i += 1


    # make splits as explicit lists
    train_runs = []
    for i in train_idx:
        train_runs.append(runs[i])

    val_runs = []
    for i in val_idx:
        val_runs.append(runs[i])

    test_runs = []
    for i in test_idx:
        test_runs.append(runs[i])

    # stack into arrays
    print("stack train split")
    train_data = stack_runs(train_runs)
    print("stack val split")
    val_data = stack_runs(val_runs)
    print("stack test split")
    test_data = stack_runs(test_runs)

    # save processed datasets
    if train_data is not None:
        print("save train npz")
        np.savez(
            out_dir / "v3_train.npz",
            C_snap=train_data["C_snap"],
            D=train_data["D"],
            k=train_data["k"],
            patch_mask=train_data["patch_mask"],
            t=train_data["t"],
            J=train_data["J"],
        )
    if val_data is not None:
        print("save val npz")
        np.savez(
            out_dir / "v3_val.npz",
            C_snap=val_data["C_snap"],
            D=val_data["D"],
            k=val_data["k"],
            patch_mask=val_data["patch_mask"],
            t=val_data["t"],
            J=val_data["J"],
        )
    if test_data is not None:
        print("save test npz")
        np.savez(
            out_dir / "v3_test.npz",
            C_snap=test_data["C_snap"],
            D=test_data["D"],
            k=test_data["k"],
            patch_mask=test_data["patch_mask"],
            t=test_data["t"],
            J=test_data["J"],
        )

    # write index
    index_path = out_dir / "index.json"
    write_json(index_path, {"split_seed": split_seed, "index": index})

    return out_dir


def assemble_processed_dataset_with_ood(
    run_dirs,
    out_dir,
    split_seed=321,
    train_frac=0.7,
    val_frac=0.15,
    ood_param="patch_width",
    ood_value=0.25,
    include_full_fields=True,
    enable_ood=True,
):
    # build ID train/val/test plus OOD holdout set
    # ID split is done only on non-ood runs
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # load all run bundles
    runs = []
    for run_dir in tqdm(run_dirs, desc="loading runs"):
        runs.append(load_run_bundle(run_dir, include_full_fields=include_full_fields))

    # hold out OOD runs first, then split ID runs
    if enable_ood:
        id_runs, ood_runs = select_id_ood_runs(runs, ood_param=ood_param, ood_value=ood_value)
    else:
        id_runs = runs
        ood_runs = []

    train_idx, val_idx, test_idx = split_indices(
        len(id_runs),
        split_seed=split_seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    train_runs = []
    for i in train_idx:
        train_runs.append(id_runs[i])

    val_runs = []
    for i in val_idx:
        val_runs.append(id_runs[i])

    test_runs = []
    for i in test_idx:
        test_runs.append(id_runs[i])

    # stack splits into batched arrays
    train_data = stack_runs(train_runs, include_full_fields=include_full_fields)
    val_data = stack_runs(val_runs, include_full_fields=include_full_fields)
    test_data = stack_runs(test_runs, include_full_fields=include_full_fields)
    ood_data = stack_runs(ood_runs, include_full_fields=include_full_fields)

    # save split files under id/ and ood/
    id_dir = out_dir / "id"
    ood_dir = out_dir / "ood"
    ensure_dir(id_dir)
    ensure_dir(ood_dir)

    save_split_npz(id_dir / "v3_train.npz", train_data)
    save_split_npz(id_dir / "v3_val.npz", val_data)
    save_split_npz(id_dir / "v3_test.npz", test_data)
    save_split_npz(ood_dir / "v3_ood_primary.npz", ood_data)

    # keep root files so older notebooks/scripts still load as before
    save_split_npz(out_dir / "v3_train.npz", train_data)
    save_split_npz(out_dir / "v3_val.npz", val_data)
    save_split_npz(out_dir / "v3_test.npz", test_data)

    # write one full summary and split-specific index files
    report = {}
    report["split_seed"] = split_seed
    report["train_frac"] = train_frac
    report["val_frac"] = val_frac
    if enable_ood:
        report["ood"] = {"enabled": True, "parameter": ood_param, "value": ood_value}
    else:
        report["ood"] = {"enabled": False, "parameter": None, "value": None}
    if include_full_fields:
        report["assemble_mode"] = "full"
    else:
        report["assemble_mode"] = "lightweight"
    report["counts"] = {
        "total": len(runs),
        "id_total": len(id_runs),
        "ood_total": len(ood_runs),
        "train": len(train_runs),
        "val": len(val_runs),
        "test": len(test_runs),
    }
    report["id_index"] = {
        "train": build_index_entries(train_runs),
        "val": build_index_entries(val_runs),
        "test": build_index_entries(test_runs),
    }
    report["ood_index"] = build_index_entries(ood_runs)

    write_json(out_dir / "index.json", report)
    write_json(id_dir / "index.json", report["id_index"])
    write_json(ood_dir / "index.json", {"ood_index": report["ood_index"]})

    return out_dir
