from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np


def set_seed(seed: int) -> None:
    # keep runs repeatable
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    # make a folder if missing
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    # small helper for metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
