from dataclasses import dataclass
import yaml


# grid settings
@dataclass
class GridConfig:
    H: int
    W: int
    dx: float
    dt: float
    T: float
    save_every: int


# boundary settings
@dataclass
class BoundaryConfig:
    mode: str
    C0: float
    decay_rate: float
    patch_width: float
    patch_offset: object
    bottom: str
    sides: str
    top_offpatch_mode: str


# full run config
@dataclass
class RunConfig:
    seed: int
    output_dir: str
    regime_name: str
    grid: GridConfig
    boundary: BoundaryConfig
    extras: dict


def _grid_from_dict(d):
    # read grid section
    return GridConfig(
        H=int(d["H"]),
        W=int(d["W"]),
        dx=float(d["dx"]),
        dt=float(d["dt"]),
        T=float(d["T"]),
        save_every=int(d["save_every"]),
    )


def _boundary_from_dict(d):
    # read boundary section
    patch_offset = d["patch_offset"]
    if isinstance(patch_offset, str):
        # allow numeric strings in YAML too
        text = patch_offset.strip()
        try:
            patch_offset = float(text)
        except ValueError:
            patch_offset = text

    return BoundaryConfig(
        mode=str(d["mode"]),
        C0=float(d["C0"]),
        decay_rate=float(d["decay_rate"]),
        patch_width=float(d["patch_width"]),
        patch_offset=patch_offset,
        bottom=str(d["bottom"]),
        sides=str(d["sides"]),
        top_offpatch_mode=str(d["top_offpatch_mode"]),
    )


def load_config(path):
    # keep config loading plain and repeatable
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # parse sections
    grid = _grid_from_dict(raw["grid"])
    boundary = _boundary_from_dict(raw["boundary"])

    # keep any extra keys for later
    known = {"seed", "output_dir", "regime_name", "grid", "boundary"}
    extras = {}
    for key in raw:
        if key not in known:
            extras[key] = raw[key]

    # pack into one object
    return RunConfig(
        seed=int(raw["seed"]),
        output_dir=str(raw["output_dir"]),
        regime_name=str(raw["regime_name"]),
        grid=grid,
        boundary=boundary,
        extras=extras,
    )
