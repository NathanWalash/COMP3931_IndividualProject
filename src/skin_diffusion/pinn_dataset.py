from pathlib import Path

import numpy as np

from skin_diffusion.bc import patch_concentration
from skin_diffusion.run_index import in_index_range, run_index_from_path
from skin_diffusion.utils import read_json


def load_run_fields(run_dir):
    # Load one run bundle from fields.npz.
    run_dir = Path(run_dir)
    fields_path = run_dir / "fields.npz"
    if not fields_path.exists():
        raise ValueError(f"Missing fields.npz for run: {run_dir}")

    with np.load(fields_path) as data:
        # Keep explicit keys so downstream training code has a stable schema.
        fields = {}
        fields["C_snap"] = data["C_snap"]
        fields["D"] = data["D"]
        fields["k"] = data["k"]
        fields["patch_mask"] = data["patch_mask"]
        fields["t"] = data["t"]
        fields["J"] = data["J"]
    return fields


def load_run_meta(run_dir):
    # Load meta.json for one run.
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing meta.json for run: {run_dir}")
    return read_json(meta_path)


def load_run_boundary(meta):
    # Run-level metadata stores boundary settings at top level.
    boundary = meta.get("boundary")
    if not isinstance(boundary, dict):
        raise ValueError("Run metadata is missing boundary settings")
    return boundary


def load_run_grid(meta):
    # Run-level metadata stores grid settings at top level.
    grid = meta.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Run metadata is missing grid settings")
    return grid


def choose_run_rows(rng, row_count, run_count):
    # Sample run rows with replacement for stochastic mini-batches.
    picks = rng.integers(0,bint(row_count), size=int(run_count))
    out = []
    for value in picks:
        out.append(int(value))
    return out


def concat_arrays(parts):
    # Concatenate list of 1D arrays into one 1D array.
    if len(parts) == 0:
        return np.array([], dtype=float)
    return np.concatenate(parts, axis=0)


def concat_feature_arrays(parts, feature_count):
    # Concatenate list of 2D feature blocks into one 2D feature matrix.
    if len(parts) == 0:
        return np.zeros((0, int(feature_count)), dtype=float)
    return np.concatenate(parts, axis=0)


def make_feature_block(feature_row, count):
    # Repeat one run-level feature row so it aligns with sampled point rows.
    if feature_row is None:
        return np.zeros((int(count), 0), dtype=float)
    row = np.asarray(feature_row, dtype=float).reshape(1, -1)
    return np.repeat(row, int(count), axis=0)


def load_split_feature_matrix(ml_dir, split_name, entries):
    # Load X rows from split NPZ using the split_row values stored in entries.
    ml_dir = Path(ml_dir)
    split_path = ml_dir / f"{split_name}.npz"
    if not split_path.exists():
        raise ValueError(f"Missing split array file: {split_path}")

    with np.load(split_path, mmap_mode="r") as data:
        x_matrix = np.asarray(data["X"], dtype=float)

    if x_matrix.ndim != 2:
        raise ValueError(f"Split feature matrix must be 2D: {split_path}")

    row_ids = []
    for entry in entries:
        if "split_row" not in entry:
            raise ValueError(f"Entry is missing split_row for split {split_name}")
        row_ids.append(int(entry["split_row"]))

    if len(row_ids) == 0:
        return np.zeros((0, int(x_matrix.shape[1])), dtype=float)

    max_row = max(row_ids)
    if max_row >= int(x_matrix.shape[0]):
        raise ValueError(
            f"split_row index {max_row} is out of bounds for {split_path}"
        )

    row_array = np.asarray(row_ids, dtype=int)
    return np.asarray(x_matrix[row_array], dtype=float)


def load_split_entries(ml_dir, split_name="id_train", run_start_index=None, run_end_index=None):
    # Load one split entry list from ml/meta.json.
    # This keeps PINN rows aligned with black-box split rows.
    ml_dir = Path(ml_dir)
    meta_path = ml_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing ML meta file: {meta_path}")

    meta = read_json(meta_path)
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"meta.json is missing splits map: {meta_path}")

    split_info = splits.get(split_name)
    if split_info is None:
        raise ValueError(f"Split {split_name} is missing in ml/meta.json")

    # ml/meta.json stores row mapping under splits.<name>.index.
    entries = split_info.get("index", [])
    if not isinstance(entries, list):
        raise ValueError(f"Split index is not a list in ml/meta.json: {split_name}")

    # Keep split_row so we can align entry rows with X rows from split NPZ files.
    split_entries = []
    row_idx = 0
    for entry in entries:
        run_idx = run_index_from_path(entry["run_dir"])
        keep = in_index_range(run_idx, run_start_index, run_end_index)
        if keep:
            row_entry = dict(entry)
            row_entry["split_row"] = int(row_idx)
            split_entries.append(row_entry)
        row_idx += 1

    if len(split_entries) == 0:
        raise ValueError(f"No entries found for split {split_name} in {ml_dir}")
    return split_entries


def sample_collocation_batch(entries, run_count, points_per_run, seed=321, feature_matrix=None):
    # Sample interior points for PDE/collocation losses.
    rng = np.random.default_rng(int(seed))
    run_rows = choose_run_rows(rng, len(entries), run_count)
    feature_count = 0
    if feature_matrix is not None:
        feature_count = int(feature_matrix.shape[1])

    # Collect per-run chunks first, then concatenate once at the end.
    x_unit_parts = []
    y_unit_parts = []
    t_unit_parts = []
    c_now_parts = []
    d_parts = []
    k_parts = []
    x_scale_parts = []
    y_scale_parts = []
    t_scale_parts = []
    feature_parts = []

    for run_row in run_rows:
        entry = entries[int(run_row)]
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)
        meta = load_run_meta(run_dir)
        grid = load_run_grid(meta)

        c_snap = fields["C_snap"]
        d_field = fields["D"]
        k_field = fields["k"]
        t_curve = fields["t"]
        dx = float(grid["dx"])

        time_count = int(t_curve.shape[0])
        height = int(d_field.shape[0])
        width = int(d_field.shape[1])
        if height < 3 or width < 3:
            raise ValueError(f"Grid too small for interior sampling in run: {run_dir}")
        if time_count < 2:
            raise ValueError(f"Need at least 2 time points in run: {run_dir}")

        # Collocation points stay away from boundaries: x in [1, W-2], y in [1, H-2].
        x_idx = rng.integers(1, width - 1, size=int(points_per_run))
        y_idx = rng.integers(1, height - 1, size=int(points_per_run))
        # Time index starts at 1 because we also use t_idx-1 for previous step.
        t_idx = rng.integers(1, time_count, size=int(points_per_run))

        # Normalized coordinates are convenient for neural-network inputs.
        x_unit = x_idx.astype(float) / float(max(1, width - 1))
        y_unit = y_idx.astype(float) / float(max(1, height - 1))
        t_now = t_curve[t_idx]
        t_unit = t_now.astype(float) / float(max(1.0, float(t_curve[-1])))

        # Supervised collocation target uses current concentration only.
        c_now = c_snap[t_idx, y_idx, x_idx]
        # D and k are sampled at the same spatial points as concentration.
        d_vals = d_field[y_idx, x_idx]
        k_vals = k_field[y_idx, x_idx]
        x_scale_val = float(max(1, width - 1) * dx)
        y_scale_val = float(max(1, height - 1) * dx)
        t_scale_val = float(max(1.0, float(t_curve[-1])))
        run_feature = None
        if feature_matrix is not None:
            run_feature = feature_matrix[int(run_row)]

        x_unit_parts.append(x_unit)
        y_unit_parts.append(y_unit)
        t_unit_parts.append(t_unit)
        c_now_parts.append(c_now.astype(float))
        d_parts.append(d_vals.astype(float))
        k_parts.append(k_vals.astype(float))
        x_scale_parts.append(np.full(int(points_per_run), x_scale_val, dtype=float))
        y_scale_parts.append(np.full(int(points_per_run), y_scale_val, dtype=float))
        t_scale_parts.append(np.full(int(points_per_run), t_scale_val, dtype=float))
        feature_parts.append(make_feature_block(run_feature, int(points_per_run)))

    # Final flat arrays are what the training loop consumes.
    batch = {}
    batch["x_unit"] = concat_arrays(x_unit_parts)
    batch["y_unit"] = concat_arrays(y_unit_parts)
    batch["t_unit"] = concat_arrays(t_unit_parts)
    batch["C_now_true"] = concat_arrays(c_now_parts)
    batch["D"] = concat_arrays(d_parts)
    batch["k"] = concat_arrays(k_parts)
    batch["x_scale"] = concat_arrays(x_scale_parts)
    batch["y_scale"] = concat_arrays(y_scale_parts)
    batch["t_scale"] = concat_arrays(t_scale_parts)
    batch["run_features"] = concat_feature_arrays(feature_parts, feature_count)
    return batch


def sample_initial_batch(entries, run_count, points_per_run, seed=321, feature_matrix=None):
    # Sample points at t=0 for initial-condition losses.
    rng = np.random.default_rng(int(seed))
    run_rows = choose_run_rows(rng, len(entries), run_count)
    feature_count = 0
    if feature_matrix is not None:
        feature_count = int(feature_matrix.shape[1])

    # Same chunk pattern as collocation sampler.
    x_unit_parts = []
    y_unit_parts = []
    t_unit_parts = []
    c_parts = []
    feature_parts = []

    for run_row in run_rows:
        entry = entries[int(run_row)]
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)

        c_snap = fields["C_snap"]
        d_field = fields["D"]

        height = int(d_field.shape[0])
        width = int(d_field.shape[1])

        # Initial condition can use full grid, including boundaries.
        x_idx = rng.integers(0, width, size=int(points_per_run))
        y_idx = rng.integers(0, height, size=int(points_per_run))

        t0 = np.zeros(int(points_per_run), dtype=float)
        x_unit = x_idx.astype(float) / float(max(1, width - 1))
        y_unit = y_idx.astype(float) / float(max(1, height - 1))
        t_unit = t0.copy()
        c_vals = c_snap[0, y_idx, x_idx]
        run_feature = None
        if feature_matrix is not None:
            run_feature = feature_matrix[int(run_row)]

        x_unit_parts.append(x_unit)
        y_unit_parts.append(y_unit)
        t_unit_parts.append(t_unit)
        c_parts.append(c_vals.astype(float))
        feature_parts.append(make_feature_block(run_feature, int(points_per_run)))

    # Return flat arrays for direct tensor conversion in trainer.
    batch = {}
    batch["x_unit"] = concat_arrays(x_unit_parts)
    batch["y_unit"] = concat_arrays(y_unit_parts)
    batch["t_unit"] = concat_arrays(t_unit_parts)
    batch["C_true"] = concat_arrays(c_parts)
    batch["run_features"] = concat_feature_arrays(feature_parts, feature_count)
    return batch


def sample_boundary_batch(entries, run_count, points_per_run, seed=321, feature_matrix=None):
    # Sample boundary points used by BC losses.
    rng = np.random.default_rng(int(seed))
    run_rows = choose_run_rows(rng, len(entries), run_count)
    feature_count = 0
    if feature_matrix is not None:
        feature_count = int(feature_matrix.shape[1])

    # Four groups mirror the boundary conditions in the simulator.
    # Each group stores per-run chunks to keep sampling code simple.
    top_patch = {}
    top_patch["x_unit"] = []
    top_patch["y_unit"] = []
    top_patch["t_unit"] = []
    top_patch["C_target"] = []
    top_patch["run_features"] = []

    top_offpatch = {}
    top_offpatch["x_unit"] = []
    top_offpatch["y_boundary_unit"] = []
    top_offpatch["y_inner_unit"] = []
    top_offpatch["t_unit"] = []
    top_offpatch["run_features"] = []

    bottom_sink = {}
    bottom_sink["x_unit"] = []
    bottom_sink["y_unit"] = []
    bottom_sink["t_unit"] = []
    bottom_sink["C_target"] = []
    bottom_sink["run_features"] = []

    side_neumann = {}
    side_neumann["x_boundary_unit"] = []
    side_neumann["x_inner_unit"] = []
    side_neumann["y_unit"] = []
    side_neumann["t_unit"] = []
    side_neumann["run_features"] = []

    for run_row in run_rows:
        entry = entries[int(run_row)]
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)
        meta = load_run_meta(run_dir)
        boundary = load_run_boundary(meta)

        t_curve = fields["t"]
        mask = fields["patch_mask"]
        d_field = fields["D"]
        t_den = float(max(1.0, float(t_curve[-1])))

        height = int(d_field.shape[0])
        width = int(d_field.shape[1])
        time_count = int(t_curve.shape[0])
        if height < 2 or width < 2:
            raise ValueError(f"Grid too small for boundary sampling in run: {run_dir}")

        # Top row patch mask defines on-patch and off-patch x positions.
        mask_row = mask[0]
        patch_cols = np.where(mask_row)[0]
        off_cols = np.where(~mask_row)[0]
        x_den = float(max(1, width - 1))
        y_den = float(max(1, height - 1))

        mode = str(boundary.get("mode", "infinite_dose"))
        c0 = float(boundary.get("C0", 0.0))
        decay_rate = float(boundary.get("decay_rate", 0.0))
        run_feature = None
        if feature_matrix is not None:
            run_feature = feature_matrix[int(run_row)]

        if len(patch_cols) > 0:
            patch_pick = rng.integers(0, len(patch_cols), size=int(points_per_run))
            patch_x = patch_cols[patch_pick]
            patch_t_idx = rng.integers(0, time_count, size=int(points_per_run))
            patch_t = t_curve[patch_t_idx]
            patch_x_unit = patch_x.astype(float) / x_den
            patch_y_unit = np.zeros(int(points_per_run), dtype=float)
            patch_t_unit = patch_t.astype(float) / t_den
            # Target concentration on top patch comes from boundary schedule.
            patch_c_target = np.zeros(int(points_per_run), dtype=float)
            for i in range(int(points_per_run)):
                patch_c_target[i] = float(
                    patch_concentration(
                        float(patch_t[i]),
                        mode,
                        c0,
                        decay_rate,
                    )
                )

            top_patch["x_unit"].append(patch_x_unit)
            top_patch["y_unit"].append(patch_y_unit)
            top_patch["t_unit"].append(patch_t_unit)
            top_patch["C_target"].append(patch_c_target)
            top_patch["run_features"].append(make_feature_block(run_feature, int(points_per_run)))

        if len(off_cols) > 0:
            off_pick = rng.integers(0, len(off_cols), size=int(points_per_run))
            off_x = off_cols[off_pick]
            off_t_idx = rng.integers(0, time_count, size=int(points_per_run))
            off_t = t_curve[off_t_idx]
            off_x_unit = off_x.astype(float) / x_den
            off_y_boundary_unit = np.zeros(int(points_per_run), dtype=float)
            off_y_inner_unit = np.ones(int(points_per_run), dtype=float) / y_den
            off_t_unit = off_t.astype(float) / t_den

            top_offpatch["x_unit"].append(off_x_unit)
            top_offpatch["y_boundary_unit"].append(off_y_boundary_unit)
            top_offpatch["y_inner_unit"].append(off_y_inner_unit)
            top_offpatch["t_unit"].append(off_t_unit)
            top_offpatch["run_features"].append(make_feature_block(run_feature, int(points_per_run)))

        # Bottom sink target is always zero concentration.
        bottom_x = rng.integers(0, width, size=int(points_per_run))
        bottom_t_idx = rng.integers(0, time_count, size=int(points_per_run))
        bottom_t = t_curve[bottom_t_idx]
        bottom_x_unit = bottom_x.astype(float) / x_den
        bottom_y_unit = np.full(int(points_per_run), float(height - 1)) / y_den
        bottom_t_unit = bottom_t.astype(float) / t_den

        bottom_sink["x_unit"].append(bottom_x_unit)
        bottom_sink["y_unit"].append(bottom_y_unit)
        bottom_sink["t_unit"].append(bottom_t_unit)
        bottom_sink["C_target"].append(np.zeros(int(points_per_run), dtype=float))
        bottom_sink["run_features"].append(make_feature_block(run_feature, int(points_per_run)))

        # Side Neumann uses boundary vs inner-pair values on left or right edge.
        side_y = rng.integers(0, height, size=int(points_per_run))
        side_t_idx = rng.integers(0, time_count, size=int(points_per_run))
        side_t = t_curve[side_t_idx]
        side_pick = rng.integers(0, 2, size=int(points_per_run))
        side_x_boundary = np.zeros(int(points_per_run), dtype=int)
        side_x_inner = np.ones(int(points_per_run), dtype=int)
        # Randomly choose left or right side for each sampled boundary point.
        for i in range(int(points_per_run)):
            if int(side_pick[i]) == 1:
                side_x_boundary[i] = width - 1
                side_x_inner[i] = width - 2
        side_x_boundary_unit = side_x_boundary.astype(float) / x_den
        side_x_inner_unit = side_x_inner.astype(float) / x_den
        side_y_unit = side_y.astype(float) / y_den
        side_t_unit = side_t.astype(float) / t_den

        side_neumann["x_boundary_unit"].append(side_x_boundary_unit)
        side_neumann["x_inner_unit"].append(side_x_inner_unit)
        side_neumann["y_unit"].append(side_y_unit)
        side_neumann["t_unit"].append(side_t_unit)
        side_neumann["run_features"].append(make_feature_block(run_feature, int(points_per_run)))

    # Concatenate per-run chunks into one flat batch per boundary group.
    out = {}
    out["top_patch"] = {}
    out["top_patch"]["x_unit"] = concat_arrays(top_patch["x_unit"])
    out["top_patch"]["y_unit"] = concat_arrays(top_patch["y_unit"])
    out["top_patch"]["t_unit"] = concat_arrays(top_patch["t_unit"])
    out["top_patch"]["C_target"] = concat_arrays(top_patch["C_target"])
    out["top_patch"]["run_features"] = concat_feature_arrays(top_patch["run_features"], feature_count)

    out["top_offpatch"] = {}
    out["top_offpatch"]["x_unit"] = concat_arrays(top_offpatch["x_unit"])
    out["top_offpatch"]["y_boundary_unit"] = concat_arrays(top_offpatch["y_boundary_unit"])
    out["top_offpatch"]["y_inner_unit"] = concat_arrays(top_offpatch["y_inner_unit"])
    out["top_offpatch"]["t_unit"] = concat_arrays(top_offpatch["t_unit"])
    out["top_offpatch"]["run_features"] = concat_feature_arrays(top_offpatch["run_features"], feature_count)

    out["bottom_sink"] = {}
    out["bottom_sink"]["x_unit"] = concat_arrays(bottom_sink["x_unit"])
    out["bottom_sink"]["y_unit"] = concat_arrays(bottom_sink["y_unit"])
    out["bottom_sink"]["t_unit"] = concat_arrays(bottom_sink["t_unit"])
    out["bottom_sink"]["C_target"] = concat_arrays(bottom_sink["C_target"])
    out["bottom_sink"]["run_features"] = concat_feature_arrays(bottom_sink["run_features"], feature_count)

    out["side_neumann"] = {}
    out["side_neumann"]["x_boundary_unit"] = concat_arrays(side_neumann["x_boundary_unit"])
    out["side_neumann"]["x_inner_unit"] = concat_arrays(side_neumann["x_inner_unit"])
    out["side_neumann"]["y_unit"] = concat_arrays(side_neumann["y_unit"])
    out["side_neumann"]["t_unit"] = concat_arrays(side_neumann["t_unit"])
    out["side_neumann"]["run_features"] = concat_feature_arrays(side_neumann["run_features"], feature_count)
    return out
