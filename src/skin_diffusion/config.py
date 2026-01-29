from dataclasses import dataclass
import yaml


@dataclass
class GridConfig:
    H: int
    W: int
    dx: float
    dt: float
    T: float
    save_every: int


@dataclass
class BoundaryConfig:
    mode: str
    C0: float
    decay_rate: float
    patch_width: float
    patch_offset: str
    bottom: str
    sides: str
    top_offpatch_mode: str


@dataclass
class RunConfig:
    seed: int
    output_dir: str
    regime_name: str
    grid: GridConfig
    boundary: BoundaryConfig
    extras: dict


def _grid_from_dict(d: dict) -> GridConfig:
    return GridConfig(
        H=int(d["H"]),
        W=int(d["W"]),
        dx=float(d["dx"]),
        dt=float(d["dt"]),
        T=float(d["T"]),
        save_every=int(d["save_every"]),
    )


def _boundary_from_dict(d: dict) -> BoundaryConfig:
    return BoundaryConfig(
        mode=str(d["mode"]),
        C0=float(d["C0"]),
        decay_rate=float(d["decay_rate"]),
        patch_width=float(d["patch_width"]),
        patch_offset=str(d["patch_offset"]),
        bottom=str(d["bottom"]),
        sides=str(d["sides"]),
        top_offpatch_mode=str(d["top_offpatch_mode"]),
    )


def load_config(path) -> RunConfig:
    # keep config loading plain and repeatable
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    grid = _grid_from_dict(raw["grid"])
    boundary = _boundary_from_dict(raw["boundary"])

    known = {"seed", "output_dir", "regime_name", "grid", "boundary"}
    extras = {}
    for key in raw:
        if key not in known:
            extras[key] = raw[key]

    return RunConfig(
        seed=int(raw["seed"]),
        output_dir=str(raw["output_dir"]),
        regime_name=str(raw["regime_name"]),
        grid=grid,
        boundary=boundary,
        extras=extras,
    )
