import argparse
from pathlib import Path


def remove_path(path):
    # delete file or folder
    if path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        for child in path.iterdir():
            remove_path(child)
        path.rmdir()
        return True
    return False


def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--outputs", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--subdir", default=None)
    args = parser.parse_args()

    # pick targets
    targets = []
    if args.figures:
        targets.append(Path("figures"))
    if args.outputs:
        targets.append(Path("outputs"))
    if args.data:
        targets.append(Path("data"))

    # optional subdir
    if args.subdir:
        new_targets = []
        for t in targets:
            new_targets.append(t / args.subdir)
        targets = new_targets

    # show what will be deleted
    print("targets:")
    for t in targets:
        print("-", t)

    # delete
    for t in targets:
        remove_path(t)
    print("done")


if __name__ == "__main__":
    main()
