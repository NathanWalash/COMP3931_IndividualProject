import json
from pathlib import Path

import numpy as np


def load_json(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def resolve_split_key(split_map, logical_name):
    # Use canonical split keys only.
    split_text = str(logical_name)
    if split_text in split_map:
        return split_text
    raise ValueError("Could not find split key " + split_text + ". Expected one of: train, val, test")


def extract_c0_feature(x_raw, feature_names):
    # C0 is needed for fair scalar comparison from predicted curves.
    if "C0" not in feature_names:
        return np.full((int(x_raw.shape[0]),), np.nan, dtype=np.float32)
    c0_col = int(feature_names.index("C0"))
    return np.asarray(x_raw[:, c0_col], dtype=np.float32)
