import json
from pathlib import Path

import numpy as np

from skin_diffusion.pinn_dataset import load_split_entries


def write_json(path, data):
    # Helper to write JSON files with consistent formatting.
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_run_bundle(run_dir, run_seed):
    # Build one minimal run folder that matches real dataset structure.
    run_dir.mkdir(parents=True, exist_ok=True)

    # Small synthetic run with valid shapes for collocation and BC sampling.
    # We keep this tiny so tests stay fast.
    t_count = 6
    height = 4
    width = 4

    # Make C vary in space and time so samplers read non-constant values.
    c_snap = np.zeros((t_count, height, width), dtype=float)
    for ti in range(t_count):
        c_snap[ti] = (0.01 * float(ti)) + 0.001 * np.add.outer(np.arange(height), np.arange(width))

    # Keep D and k simple and stable for deterministic tests.
    d_field = np.full((height, width), 1.0e-8, dtype=float)
    k_field = np.zeros((height, width), dtype=float)

    # Patch occupies top-left half of the top boundary.
    patch_mask = np.zeros((height, width), dtype=bool)
    patch_mask[0, 0:2] = True

    # Short t and J curves are enough to validate loading/sampling logic.
    t_curve = np.linspace(0.0, 10.0, t_count, dtype=float)
    j_curve = np.linspace(0.0, 1.0, t_count, dtype=float)

    # Write fields in the same schema used by generated run bundles.
    np.savez(
        run_dir / "fields.npz",
        C_snap=c_snap,
        D=d_field,
        k=k_field,
        patch_mask=patch_mask,
        t=t_curve,
        J=j_curve,
    )

    # Write minimal metadata used by PINN boundary sampling code.
    meta = {}
    meta["grid"] = {"H": height, "W": width}
    meta["boundary"] = {
        "mode": "time_decay",
        "C0": 1.0,
        "decay_rate": 0.2,
        "patch_width": 0.5,
        "patch_offset": "left",
        "bottom": "sink",
        "sides": "neumann",
        "top_offpatch_mode": "neumann",
    }
    meta["seed"] = int(run_seed)
    meta["regime"] = "test"
    meta["extras"] = {}
    write_json(run_dir / "meta.json", meta)

    # metrics.json exists in real runs; include it for completeness.
    write_json(run_dir / "metrics.json", {"P": 1.0e-10, "J_ss": 2.0e-10})


def make_ml_meta(ml_dir, run_dirs):
    # Build a minimal ml/meta.json with id_train row mapping.
    ml_dir.mkdir(parents=True, exist_ok=True)

    split_index = []
    for run_dir in run_dirs:
        entry = {}
        entry["run_dir"] = str(run_dir)
        entry["meta_path"] = str(run_dir / "meta.json")
        entry["metrics_path"] = str(run_dir / "metrics.json")
        split_index.append(entry)

    meta = {}
    meta["feature_names"] = ["patch_width"]
    meta["scalar_target_names"] = ["P"]
    meta["splits"] = {}
    meta["splits"]["id_train"] = {"rows": len(split_index), "index": split_index}
    write_json(ml_dir / "meta.json", meta)


def test_load_split_entries_respects_run_index_filter(tmp_path):
    # Arrange: create two runs and an ML meta split with both runs.
    run_a = tmp_path / "runs" / "run_000"
    run_b = tmp_path / "runs" / "run_101"
    make_run_bundle(run_a, run_seed=1)
    make_run_bundle(run_b, run_seed=2)

    ml_dir = tmp_path / "ml"
    make_ml_meta(ml_dir, [run_a, run_b])

    # Act: filter to only the second run by run index.
    entries = load_split_entries(
        ml_dir=ml_dir,
        split_name="id_train",
        run_start_index=100,
        run_end_index=200,
    )

    # Assert: only run_101 should remain after filtering.
    assert len(entries) == 1
    assert Path(entries[0]["run_dir"]).name == "run_101"
