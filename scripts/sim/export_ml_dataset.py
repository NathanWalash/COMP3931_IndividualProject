import argparse

from skin_diffusion.ml_dataset import export_ml_ready_dataset


def main():
    # export training-ready splits from assembled processed data
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/processed/ml")
    args = parser.parse_args()

    out_dir = export_ml_ready_dataset(args.processed_dir, args.out_dir)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
