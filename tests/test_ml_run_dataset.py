import json
from pathlib import Path

import numpy as np

from skin_diffusion.ml_run_dataset import (
    load_run_bundle,
    load_split_entries,
    load_split_feature_matrix,
    load_split_scalar_matrix,
    remap_run_dir,
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_run_bundle(run_dir, run_seed):
    # Build the smallest valid run bundle accepted by load_run_bundle.
    run_dir.mkdir(parents=True, exist_ok=True)

    t_count = 4
    height = 4
    width = 4
    c_snap = np.zeros((t_count, height, width), dtype=float)
    d_field = np.full((height, width), 1.0e-8, dtype=float)
    k_field = np.zeros((height, width), dtype=float)
    patch_mask = np.zeros((height, width), dtype=bool)
    patch_mask[0, 0:2] = True
    t_curve = np.linspace(0.0, 10.0, t_count, dtype=float)
    j_curve = np.linspace(0.0, 1.0, t_count, dtype=float)

    np.savez(
        run_dir / "fields.npz",
        C_snap=c_snap,
        D=d_field,
        k=k_field,
        patch_mask=patch_mask,
        t=t_curve,
        J=j_curve,
    )

    meta = {
        "grid": {"H": height, "W": width, "dx": 0.1},
        "boundary": {
            "mode": "time_decay",
            "C0": 1.0,
            "decay_rate": 0.2,
            "patch_width": 0.5,
            "patch_offset": "left",
            "bottom": "sink",
            "sides": "neumann",
            "top_offpatch_mode": "neumann",
        },
        "seed": int(run_seed),
        "regime": "test",
        "extras": {},
    }
    write_json(run_dir / "meta.json", meta)
    write_json(run_dir / "metrics.json", {"P": 1.0e-10, "J_ss": 2.0e-10})


def make_ml_split(ml_dir, run_dirs):
    # Build a minimal ML split with split index entries + matching NPZ arrays.
    ml_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in run_dirs:
        rows.append(
            {
                "run_dir": str(run_dir),
                "meta_path": str(run_dir / "meta.json"),
                "metrics_path": str(run_dir / "metrics.json"),
            }
        )

    meta = {
        "feature_names": ["f0", "f1"],
        "scalar_target_names": ["P"],
        "splits": {"train": {"rows": len(rows), "index": rows}},
    }
    write_json(ml_dir / "meta.json", meta)

    x = np.zeros((len(rows), 2), dtype=float)
    y = np.zeros((len(rows), 1), dtype=float)
    j = np.zeros((len(rows), 4), dtype=float)
    t = np.zeros((len(rows), 4), dtype=float)
    for i in range(len(rows)):
        x[i, 0] = float(i)
        x[i, 1] = float(i) + 1.0
        y[i, 0] = 10.0 + float(i)
    np.savez(ml_dir / "train.npz", X=x, y_scalar=y, J=j, t=t)


def test_load_split_entries_and_features(tmp_path):
    # Verify run-index filtering and split_row -> NPZ row alignment.
    run_a = tmp_path / "runs" / "run_000"
    run_b = tmp_path / "runs" / "run_101"
    make_run_bundle(run_a, run_seed=1)
    make_run_bundle(run_b, run_seed=2)

    ml_dir = tmp_path / "ml"
    make_ml_split(ml_dir, [run_a, run_b])

    entries = load_split_entries(
        ml_dir=ml_dir,
        split_name="train",
        run_start_index=100,
        run_end_index=200,
    )
    assert len(entries) == 1
    assert Path(entries[0]["run_dir"]).name == "run_101"
    assert int(entries[0]["split_row"]) == 1

    x = load_split_feature_matrix(ml_dir, "train", entries)
    assert x.shape == (1, 2)
    assert np.allclose(x[0], np.array([1.0, 2.0], dtype=float))

    y = load_split_scalar_matrix(ml_dir, "train", entries)
    assert y.shape == (1, 1)
    assert np.allclose(y[0], np.array([11.0], dtype=float))


def test_load_run_bundle(tmp_path):
    # Verify run fields/meta can be loaded as one bundle.
    run_a = tmp_path / "runs" / "run_000"
    make_run_bundle(run_a, run_seed=5)
    bundle = load_run_bundle(run_a)
    assert "fields" in bundle
    assert "meta" in bundle
    assert bundle["fields"]["C_snap"].shape == (4, 4, 4)
    assert bundle["meta"]["grid"]["H"] == 4


def test_remap_run_dir(tmp_path):
    # Verify both supported staged layouts: <root>/run_x and <root>/dataset/run_x.
    run_name = "run_007"
    original = "/tmp/original/" + run_name

    root_flat = tmp_path / "staged_flat"
    target_flat = root_flat / run_name
    target_flat.mkdir(parents=True, exist_ok=True)
    out_flat = remap_run_dir(original, root_flat)
    assert Path(out_flat) == target_flat

    root_nested = tmp_path / "staged_nested"
    target_nested = root_nested / "dataset" / run_name
    target_nested.mkdir(parents=True, exist_ok=True)
    out_nested = remap_run_dir(original, root_nested)
    assert Path(out_nested) == target_nested
