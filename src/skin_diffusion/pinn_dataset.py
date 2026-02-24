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
    # Support both metadata styles used
    if "boundary" in meta and isinstance(meta["boundary"], dict):
        return meta["boundary"]
    if "config" in meta and isinstance(meta["config"], dict):
        config = meta["config"]
        boundary = config.get("boundary")
        if isinstance(boundary, dict):
            return boundary
    raise ValueError("Run metadata does not contain boundary settings")


def choose_run_rows(rng, row_count, run_count):
    # Sample run rows with replacement for stochastic mini-batches.
    picked = []
    for _ in range(int(run_count)):
        row = int(rng.integers(0, int(row_count)))
        picked.append(row)
    return picked


def concat_arrays(parts):
    # Concatenate list of 1D arrays into one 1D array.
    if len(parts) == 0:
        return np.array([], dtype=float)
    return np.concatenate(parts, axis=0)


def load_split_entries(
    ml_dir,
    split_name="id_train",
    run_start_index=None,
    run_end_index=None,
):
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

    # Optional run index filter is useful when training on a subset of runs.
    if run_start_index is None and run_end_index is None:
        split_entries = entries
    else:
        split_entries = []
        for entry in entries:
            run_idx = run_index_from_path(entry["run_dir"])
            if in_index_range(run_idx, run_start_index, run_end_index):
                split_entries.append(entry)

    if len(split_entries) == 0:
        raise ValueError(f"No entries found for split {split_name} in {ml_dir}")
    return split_entries


def get_entry(entries, row):
    # Access one row from an entry list.
    return entries[int(row)]


def sample_collocation_batch(entries, run_count, points_per_run, seed=321):
    # Sample interior points for PDE/collocation losses.
    rng = np.random.default_rng(int(seed))
    run_rows = choose_run_rows(rng, len(entries), run_count)

    # Collect per-run chunks first, then concatenate once at the end.
    x_parts = []
    y_parts = []
    t_parts = []
    x_unit_parts = []
    y_unit_parts = []
    t_unit_parts = []
    c_now_parts = []
    c_prev_parts = []
    d_parts = []
    k_parts = []
    dt_parts = []

    for run_row in run_rows:
        entry = get_entry(entries, run_row)
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)

        c_snap = fields["C_snap"]
        d_field = fields["D"]
        k_field = fields["k"]
        t_curve = fields["t"]

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

        # Keep current and previous concentration to support time-residual terms.
        c_now = c_snap[t_idx, y_idx, x_idx]
        c_prev = c_snap[t_idx - 1, y_idx, x_idx]
        # D and k are sampled at the same spatial points as concentration.
        d_vals = d_field[y_idx, x_idx]
        k_vals = k_field[y_idx, x_idx]
        dt_vals = t_curve[t_idx] - t_curve[t_idx - 1]

        x_parts.append(x_idx.astype(float))
        y_parts.append(y_idx.astype(float))
        t_parts.append(t_now.astype(float))
        x_unit_parts.append(x_unit)
        y_unit_parts.append(y_unit)
        t_unit_parts.append(t_unit)
        c_now_parts.append(c_now.astype(float))
        c_prev_parts.append(c_prev.astype(float))
        d_parts.append(d_vals.astype(float))
        k_parts.append(k_vals.astype(float))
        dt_parts.append(dt_vals.astype(float))

    # Final flat arrays are what the training loop consumes.
    batch = {}
    batch["x"] = concat_arrays(x_parts)
    batch["y"] = concat_arrays(y_parts)
    batch["t"] = concat_arrays(t_parts)
    batch["x_unit"] = concat_arrays(x_unit_parts)
    batch["y_unit"] = concat_arrays(y_unit_parts)
    batch["t_unit"] = concat_arrays(t_unit_parts)
    batch["C_now_true"] = concat_arrays(c_now_parts)
    batch["C_prev_true"] = concat_arrays(c_prev_parts)
    batch["D"] = concat_arrays(d_parts)
    batch["k"] = concat_arrays(k_parts)
    batch["dt"] = concat_arrays(dt_parts)
    return batch


def sample_initial_batch(entries, run_count, points_per_run, seed=321):
    # Sample points at t=0 for initial-condition losses.
    rng = np.random.default_rng(int(seed))
    run_rows = choose_run_rows(rng, len(entries), run_count)

    # Same chunk pattern as collocation sampler.
    x_parts = []
    y_parts = []
    t_parts = []
    x_unit_parts = []
    y_unit_parts = []
    t_unit_parts = []
    c_parts = []

    for run_row in run_rows:
        entry = get_entry(entries, run_row)
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

        x_parts.append(x_idx.astype(float))
        y_parts.append(y_idx.astype(float))
        t_parts.append(t0)
        x_unit_parts.append(x_unit)
        y_unit_parts.append(y_unit)
        t_unit_parts.append(t_unit)
        c_parts.append(c_vals.astype(float))

    # Return flat arrays for direct tensor conversion in trainer.
    batch = {}
    batch["x"] = concat_arrays(x_parts)
    batch["y"] = concat_arrays(y_parts)
    batch["t"] = concat_arrays(t_parts)
    batch["x_unit"] = concat_arrays(x_unit_parts)
    batch["y_unit"] = concat_arrays(y_unit_parts)
    batch["t_unit"] = concat_arrays(t_unit_parts)
    batch["C_true"] = concat_arrays(c_parts)
    return batch


def sample_boundary_batch(entries, run_count, points_per_run, seed=321):
    # Sample boundary points used by BC losses.
    rng = np.random.default_rng(int(seed))
    run_rows = choose_run_rows(rng, len(entries), run_count)

    # Four groups mirror the boundary conditions in the simulator.
    # Each group stores per-run chunks to keep sampling code simple.
    top_patch = {}
    top_patch["x"] = []
    top_patch["y"] = []
    top_patch["t"] = []
    top_patch["C_target"] = []
    top_patch["C_true"] = []

    top_offpatch = {}
    top_offpatch["x"] = []
    top_offpatch["y_boundary"] = []
    top_offpatch["y_inner"] = []
    top_offpatch["t"] = []
    top_offpatch["C_boundary_true"] = []
    top_offpatch["C_inner_true"] = []

    bottom_sink = {}
    bottom_sink["x"] = []
    bottom_sink["y"] = []
    bottom_sink["t"] = []
    bottom_sink["C_target"] = []
    bottom_sink["C_true"] = []

    side_neumann = {}
    side_neumann["x_boundary"] = []
    side_neumann["x_inner"] = []
    side_neumann["y"] = []
    side_neumann["t"] = []
    side_neumann["C_boundary_true"] = []
    side_neumann["C_inner_true"] = []

    for run_row in run_rows:
        entry = get_entry(entries, run_row)
        run_dir = Path(entry["run_dir"])
        fields = load_run_fields(run_dir)
        meta = load_run_meta(run_dir)
        boundary = load_run_boundary(meta)

        c_snap = fields["C_snap"]
        t_curve = fields["t"]
        mask = fields["patch_mask"]
        d_field = fields["D"]

        height = int(d_field.shape[0])
        width = int(d_field.shape[1])
        time_count = int(t_curve.shape[0])
        if height < 2 or width < 2:
            raise ValueError(f"Grid too small for boundary sampling in run: {run_dir}")

        # Top row patch mask defines on-patch and off-patch x positions.
        mask_row = mask[0]
        patch_cols = np.where(mask_row)[0]
        off_cols = np.where(~mask_row)[0]

        mode = str(boundary.get("mode", "infinite_dose"))
        c0 = float(boundary.get("C0", 0.0))
        decay_rate = float(boundary.get("decay_rate", 0.0))

        if len(patch_cols) > 0:
            patch_pick = rng.integers(0, len(patch_cols), size=int(points_per_run))
            patch_x = patch_cols[patch_pick]
            patch_t_idx = rng.integers(0, time_count, size=int(points_per_run))
            patch_t = t_curve[patch_t_idx]
            patch_c_true = c_snap[patch_t_idx, 0, patch_x]
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

            top_patch["x"].append(patch_x.astype(float))
            top_patch["y"].append(np.zeros(int(points_per_run), dtype=float))
            top_patch["t"].append(patch_t.astype(float))
            top_patch["C_target"].append(patch_c_target)
            top_patch["C_true"].append(patch_c_true.astype(float))

        if len(off_cols) > 0:
            off_pick = rng.integers(0, len(off_cols), size=int(points_per_run))
            off_x = off_cols[off_pick]
            off_t_idx = rng.integers(0, time_count, size=int(points_per_run))
            off_t = t_curve[off_t_idx]
            # Store boundary/inner pairs for no-flux checks on off-patch top.
            off_c_boundary = c_snap[off_t_idx, 0, off_x]
            off_c_inner = c_snap[off_t_idx, 1, off_x]

            top_offpatch["x"].append(off_x.astype(float))
            top_offpatch["y_boundary"].append(np.zeros(int(points_per_run), dtype=float))
            top_offpatch["y_inner"].append(np.ones(int(points_per_run), dtype=float))
            top_offpatch["t"].append(off_t.astype(float))
            top_offpatch["C_boundary_true"].append(off_c_boundary.astype(float))
            top_offpatch["C_inner_true"].append(off_c_inner.astype(float))

        # Bottom sink target is always zero concentration.
        bottom_x = rng.integers(0, width, size=int(points_per_run))
        bottom_t_idx = rng.integers(0, time_count, size=int(points_per_run))
        bottom_t = t_curve[bottom_t_idx]
        bottom_c_true = c_snap[bottom_t_idx, height - 1, bottom_x]

        bottom_sink["x"].append(bottom_x.astype(float))
        bottom_sink["y"].append(np.full(int(points_per_run), float(height - 1)))
        bottom_sink["t"].append(bottom_t.astype(float))
        bottom_sink["C_target"].append(np.zeros(int(points_per_run), dtype=float))
        bottom_sink["C_true"].append(bottom_c_true.astype(float))

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
        side_c_boundary = c_snap[side_t_idx, side_y, side_x_boundary]
        side_c_inner = c_snap[side_t_idx, side_y, side_x_inner]

        side_neumann["x_boundary"].append(side_x_boundary.astype(float))
        side_neumann["x_inner"].append(side_x_inner.astype(float))
        side_neumann["y"].append(side_y.astype(float))
        side_neumann["t"].append(side_t.astype(float))
        side_neumann["C_boundary_true"].append(side_c_boundary.astype(float))
        side_neumann["C_inner_true"].append(side_c_inner.astype(float))

    # Concatenate per-run chunks into one flat batch per boundary group.
    out = {}
    out["top_patch"] = {}
    out["top_patch"]["x"] = concat_arrays(top_patch["x"])
    out["top_patch"]["y"] = concat_arrays(top_patch["y"])
    out["top_patch"]["t"] = concat_arrays(top_patch["t"])
    out["top_patch"]["C_target"] = concat_arrays(top_patch["C_target"])
    out["top_patch"]["C_true"] = concat_arrays(top_patch["C_true"])

    out["top_offpatch"] = {}
    out["top_offpatch"]["x"] = concat_arrays(top_offpatch["x"])
    out["top_offpatch"]["y_boundary"] = concat_arrays(top_offpatch["y_boundary"])
    out["top_offpatch"]["y_inner"] = concat_arrays(top_offpatch["y_inner"])
    out["top_offpatch"]["t"] = concat_arrays(top_offpatch["t"])
    out["top_offpatch"]["C_boundary_true"] = concat_arrays(top_offpatch["C_boundary_true"])
    out["top_offpatch"]["C_inner_true"] = concat_arrays(top_offpatch["C_inner_true"])

    out["bottom_sink"] = {}
    out["bottom_sink"]["x"] = concat_arrays(bottom_sink["x"])
    out["bottom_sink"]["y"] = concat_arrays(bottom_sink["y"])
    out["bottom_sink"]["t"] = concat_arrays(bottom_sink["t"])
    out["bottom_sink"]["C_target"] = concat_arrays(bottom_sink["C_target"])
    out["bottom_sink"]["C_true"] = concat_arrays(bottom_sink["C_true"])

    out["side_neumann"] = {}
    out["side_neumann"]["x_boundary"] = concat_arrays(side_neumann["x_boundary"])
    out["side_neumann"]["x_inner"] = concat_arrays(side_neumann["x_inner"])
    out["side_neumann"]["y"] = concat_arrays(side_neumann["y"])
    out["side_neumann"]["t"] = concat_arrays(side_neumann["t"])
    out["side_neumann"]["C_boundary_true"] = concat_arrays(side_neumann["C_boundary_true"])
    out["side_neumann"]["C_inner_true"] = concat_arrays(side_neumann["C_inner_true"])
    return out
