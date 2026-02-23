import argparse
from pathlib import Path
import numpy as np

from skin_diffusion.config import load_config
from skin_diffusion.dataset import generate_run, validate_run
from skin_diffusion.dataset_spec import (
    apply_sample_to_config,
    is_dataset_spec,
    load_yaml_file,
    sample_dataset_parameters,
)
from skin_diffusion.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=None)
    args = parser.parse_args()

    # support two inputs:
    # 1) normal simulation config
    # 2) dataset spec that points to a base sim config
    raw = load_yaml_file(args.config)
    spec_mode = is_dataset_spec(raw)
    spec_path = Path(args.config).resolve()

    if spec_mode:
        # base config path can be relative to repo root or to the spec file
        base_cfg_path = Path(raw["base_config"])
        if not base_cfg_path.is_absolute():
            base_from_cwd = (Path.cwd() / base_cfg_path).resolve()
            if base_from_cwd.exists():
                base_cfg_path = base_from_cwd
            else:
                base_cfg_path = (spec_path.parent / base_cfg_path).resolve()

        output_root = Path(raw["output_root"])
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()

        cfg = load_config(str(base_cfg_path))
        cfg.output_dir = str(output_root)
    else:
        cfg = load_config(args.config)

    # Each generated run is stored as output_dir/dataset/run_###
    base_out = Path(cfg.output_dir) / "dataset"
    ensure_dir(base_out)

    # Choose run index range.
    # If start_index is not provided, append after existing run_### folders.
    # lets generate batches without overwriting prior runs.
    if args.start_index is None:
        existing = sorted(base_out.glob("run_*"))
        if existing:
            max_idx = -1
            for p in existing:
                name = p.name
                if name.startswith("run_"):
                    try:
                        max_idx = max(max_idx, int(name.split("_", 1)[1]))
                    except ValueError:
                        continue
            start_index = max_idx + 1
        else:
            start_index = 0
    else:
        start_index = args.start_index

    if args.seed_start is None:
        # Keep default mapping stable across appended batches:
        # run_seed = cfg.seed + run_index
        seed_start = int(cfg.seed) + int(start_index)
    else:
        seed_start = int(args.seed_start)

    for i in range(args.num_runs):
        run_seed = seed_start + i

        if spec_mode:
            rng = np.random.default_rng(run_seed)
            sample = sample_dataset_parameters(raw, rng)
            apply_sample_to_config(cfg, sample, run_seed)
        else:
            cfg.seed = run_seed
            if "heterogeneity" in cfg.extras:
                cfg.extras["heterogeneity"]["seed"] = cfg.seed

        # Fixed run naming makes directory listing and batch organisation simple
        run_id = f"run_{start_index + i:03d}"
        run_dir = base_out / run_id
        ensure_dir(run_dir)

        generate_run(cfg, run_dir)

        validate_run(run_dir)
        print("saved:", run_dir)


if __name__ == "__main__":
    main()
