from pathlib import Path
import json
import random
import numpy as np


def set_seed(seed):
    # keep runs repeatable
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    # make a folder if missing
    path.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    # small helper for metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
