import argparse
from pathlib import Path

from skin_diffusion.config import load_config
from skin_diffusion.dataset import generate_run, validate_run
from skin_diffusion.utils import ensure_dir


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=None)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # where to save runs
    # each run gets its own folder
    base_out = Path(cfg.output_dir) / "dataset"
    ensure_dir(base_out)

    # seed control
    # seed is bumped per run for variation
    if args.seed_start is None:
        seed_start = cfg.seed
    else:
        seed_start = args.seed_start

    # generate runs
    # run_000, run_001, ...
    for i in range(args.num_runs):
        cfg.seed = seed_start + i
        # keep heterogeneity seed in sync if it exists
        if "heterogeneity" in cfg.extras:
            cfg.extras["heterogeneity"]["seed"] = cfg.seed
        run_id = f"run_{i:03d}"
        run_dir = base_out / run_id
        ensure_dir(run_dir)

        # run sim and save the bundle
        generate_run(cfg, run_dir)

        # quick sanity check
        validate_run(run_dir)
        print("saved:", run_dir)


if __name__ == "__main__":
    main()
