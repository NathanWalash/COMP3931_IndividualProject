import argparse
from pathlib import Path

from tqdm import tqdm

from skin_diffusion.config import load_config
from skin_diffusion.dataset import assemble_processed_dataset_with_ood
from skin_diffusion.dataset_spec import is_dataset_spec, load_yaml_file
from skin_diffusion.utils import ensure_dir


def run_index_from_dir(path):
    # Parse run index from folder name like run_123.
    name = Path(path).name
    if not name.startswith("run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def in_index_range(run_idx, start_idx, end_idx):
    # Inclusive range filter; None means unbounded.
    if run_idx is None:
        return False
    if start_idx is not None and run_idx < start_idx:
        return False
    if end_idx is not None and run_idx > end_idx:
        return False
    return True


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--split_seed", type=int, default=321)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--ood_param", default="patch_width")
    parser.add_argument("--ood_value", type=float, default=0.25)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--run_start_index", type=int, default=None)
    parser.add_argument("--run_end_index", type=int, default=None)
    args = parser.parse_args()

    # support both sim config and dataset spec input
    raw = load_yaml_file(args.config)
    spec_mode = is_dataset_spec(raw)
    spec_path = Path(args.config).resolve()

    if spec_mode:
        # for spec mode, load base sim config and use spec output_root
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
        # normal mode uses output_dir from the sim config
        cfg = load_config(args.config)

    # dataset runs are expected under output_dir/dataset
    dataset_root = Path(cfg.output_dir) / "dataset"

    # find run folders
    run_dirs = []
    for p in tqdm(sorted(dataset_root.glob("run_*")), desc="finding runs"):
        idx = run_index_from_dir(p)
        if in_index_range(idx, args.run_start_index, args.run_end_index):
            run_dirs.append(p)

    if len(run_dirs) == 0:
        raise ValueError("No run folders found in requested index range")

    # output folder
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # build ID and OOD processed datasets
    assemble_processed_dataset_with_ood(
        run_dirs,
        out_dir,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        ood_param=args.ood_param,
        ood_value=args.ood_value,
        include_full_fields=not bool(args.lightweight),
    )

    print("saved:", out_dir)


if __name__ == "__main__":
    main()
