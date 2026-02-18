import json
from pathlib import Path

import numpy as np

from skin_diffusion.ml_dataset import FEATURE_NAMES, export_ml_ready_dataset


def write_json(path, data):
    # Small helper to keep test setup readable.
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_export_ml_ready_dataset_feature_width(tmp_path):
    # Build the smallest valid processed dataset structure.
    processed = tmp_path / "processed"
    out_dir = tmp_path / "ml"
    id_dir = processed / "id"
    ood_dir = processed / "ood"
    run_dir = tmp_path / "runs" / "run_000"
    id_dir.mkdir(parents=True, exist_ok=True)
    ood_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # fields.npz is needed for D summary feature extraction.
    D = np.full((4, 4), 1.0e-8, dtype=float)
    C_snap = np.zeros((3, 4, 4), dtype=float)
    k = np.zeros((4, 4), dtype=float)
    patch_mask = np.zeros((4, 4), dtype=bool)
    t_curve = np.array([0.0, 1.0, 2.0], dtype=float)
    J_curve = np.array([0.0, 0.1, 0.2], dtype=float)
    np.savez(run_dir / "fields.npz", C_snap=C_snap, D=D, k=k, patch_mask=patch_mask, t=t_curve, J=J_curve)

    # Meta and metrics contain the tabular inputs and scalar targets.
    sample = {
        "patch_width": 0.5,
        "patch_offset": 0.3,
        "C0": 1.0,
        "decay_rate": 0.2,
        "heterogeneity_sigma": 0.04,
        "heterogeneity_steps": 5,
    }
    metrics = {"P": 1e-10, "Tlag": 1000.0, "J_ss": 2e-10}
    write_json(run_dir / "meta.json", {"extras": {"dataset_sample": sample}})
    write_json(run_dir / "metrics.json", metrics)

    # Export reads J/t from processed split npz files.
    J_batch = np.zeros((1, 3), dtype=float)
    t_batch = np.tile(np.array([0.0, 1.0, 2.0], dtype=float), (1, 1))
    np.savez(id_dir / "v3_train.npz", J=J_batch, t=t_batch)
    np.savez(id_dir / "v3_val.npz", J=J_batch, t=t_batch)
    np.savez(id_dir / "v3_test.npz", J=J_batch, t=t_batch)
    np.savez(ood_dir / "v3_ood_primary.npz", J=J_batch, t=t_batch)

    # Index maps each split row to run metadata.
    row_ref = {
        "run_dir": str(run_dir),
        "meta_path": str(run_dir / "meta.json"),
        "metrics_path": str(run_dir / "metrics.json"),
    }
    index = {
        "id_index": {"train": [row_ref], "val": [row_ref], "test": [row_ref]},
        "ood_index": [row_ref],
    }
    write_json(processed / "index.json", index)

    # Run export and verify the richer feature schema is used.
    export_ml_ready_dataset(processed, out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == FEATURE_NAMES

    train = np.load(out_dir / "id_train.npz")
    assert train["X"].shape == (1, len(FEATURE_NAMES))
    assert train["y_scalar"].shape == (1, 3)

    # Verify the new interaction features are present and numerically correct.
    feature_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    x_row = train["X"][0]
    assert np.isclose(x_row[feature_idx["c0_times_width"]], 0.5)
    assert np.isclose(x_row[feature_idx["decay_times_width"]], 0.1)
    assert np.isclose(x_row[feature_idx["sigma_times_width"]], 0.02)

