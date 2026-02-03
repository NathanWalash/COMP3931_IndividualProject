import numpy as np
from pathlib import Path

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


def load_run_bundle(run_dir):
    # load one saved run from disk
    run_dir = Path(run_dir)
    fields = np.load(run_dir / "fields.npz")
    meta = run_dir / "meta.json"
    metrics = run_dir / "metrics.json"

    data = {}
    data["C_snap"] = fields["C_snap"]
    data["D"] = fields["D"]
    data["k"] = fields["k"]
    data["patch_mask"] = fields["patch_mask"]
    data["t"] = fields["t"]
    data["J"] = fields["J"]
    data["meta_path"] = str(meta)
    data["metrics_path"] = str(metrics)

    return data


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
    # pull metrics if they exist
    if metrics is None:
        metrics_out["P"] = None
        metrics_out["Tlag"] = None
        metrics_out["J_ss"] = None
    else:
        metrics_out["P"] = metrics.get("P")
        metrics_out["Tlag"] = metrics.get("Tlag")
        metrics_out["J_ss"] = metrics.get("J_ss")
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


def stack_runs(runs):
    # stack fields from a list of run dicts
    C_list = []
    D_list = []
    k_list = []
    mask_list = []
    t_list = []
    J_list = []

    for r in runs:
        C_list.append(r["C_snap"])
        D_list.append(r["D"])
        k_list.append(r["k"])
        mask_list.append(r["patch_mask"])
        t_list.append(r["t"])
        J_list.append(r["J"])

    data = {}
    data["C_snap"] = np.stack(C_list, axis=0)
    data["D"] = np.stack(D_list, axis=0)
    data["k"] = np.stack(k_list, axis=0)
    data["patch_mask"] = np.stack(mask_list, axis=0)
    data["t"] = np.stack(t_list, axis=0)
    data["J"] = np.stack(J_list, axis=0)

    return data


def split_indices(n, split_seed, train_frac, val_frac):
    # make a deterministic split
    idx = np.arange(n)
    test_frac = 1.0 - train_frac - val_frac
    if test_frac < 0.0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

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
        val_size = val_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
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
    for rd in run_dirs:
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
    train_data = stack_runs(train_runs)
    val_data = stack_runs(val_runs)
    test_data = stack_runs(test_runs)

    # save processed datasets
    np.savez(
        out_dir / "v3_train.npz",
        C_snap=train_data["C_snap"],
        D=train_data["D"],
        k=train_data["k"],
        patch_mask=train_data["patch_mask"],
        t=train_data["t"],
        J=train_data["J"],
    )
    np.savez(
        out_dir / "v3_val.npz",
        C_snap=val_data["C_snap"],
        D=val_data["D"],
        k=val_data["k"],
        patch_mask=val_data["patch_mask"],
        t=val_data["t"],
        J=val_data["J"],
    )
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
