import numpy as np
from pathlib import Path

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


def _metrics_summary(J):
    # simple stats for the flux curve
    summary = {}
    # basic min/max/mean/sum
    summary["J_min"] = float(np.min(J))
    summary["J_max"] = float(np.max(J))
    summary["J_mean"] = float(np.mean(J))
    summary["J_sum"] = float(np.sum(J))
    return summary


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
    metrics_out.update(_metrics_summary(J))

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
