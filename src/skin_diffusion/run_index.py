from pathlib import Path


def run_index_from_path(path):
    # Parse run index from names like run_123.
    name = Path(path).name
    if not name.startswith("run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def in_index_range(run_idx, start_idx, end_idx):
    # Inclusive index filter. None means unbounded.
    if run_idx is None:
        return False
    if start_idx is not None and run_idx < start_idx:
        return False
    if end_idx is not None and run_idx > end_idx:
        return False
    return True
