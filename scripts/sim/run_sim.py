import argparse
from datetime import datetime
from pathlib import Path

from skin_diffusion.config import load_config
from skin_diffusion.utils import ensure_dir, set_seed, write_json


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # make output folder
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # build simple metadata
    meta = {}
    meta["timestamp"] = datetime.now().isoformat()
    meta["config"] = {
        "seed": cfg.seed,
        "output_dir": cfg.output_dir,
        "regime_name": cfg.regime_name,
        "grid": cfg.grid.__dict__,
        "boundary": cfg.boundary.__dict__,
        "extras": cfg.extras,
    }

    # save metadata
    meta_path = out_dir / "meta.json"
    write_json(meta_path, meta)

    print("loaded config:", cfg.regime_name)
    print("wrote:", meta_path)


if __name__ == "__main__":
    main()
