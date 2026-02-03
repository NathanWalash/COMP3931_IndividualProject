import argparse
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.dataset import assemble_processed_dataset
from skin_diffusion.utils import ensure_dir


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    # load config to find dataset folder
    cfg = load_config(args.config)
    dataset_root = Path(cfg.output_dir) / "dataset"

    # find run folders
    run_dirs = []
    for p in tqdm(sorted(dataset_root.glob("run_*")), desc="finding runs"):
        run_dirs.append(p)

    # output folder
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # build processed datasets
    assemble_processed_dataset(
        run_dirs,
        out_dir,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("saved:", out_dir)


if __name__ == "__main__":
    main()
