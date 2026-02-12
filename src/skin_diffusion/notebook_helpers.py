import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def setup_repo_root(marker="pyproject.toml"):
    # walk up until we find repo marker, then cd there
    cwd = Path.cwd().resolve()
    candidates = [cwd]
    for parent in cwd.parents:
        candidates.append(parent)

    for candidate in candidates:
        if (candidate / marker).exists():
            os.chdir(candidate)
            return candidate

    return cwd


def run_module(module, args, repo_root=None):
    # run python -m module with current interpreter
    env = os.environ.copy()
    if repo_root is not None:
        src_path = str((repo_root / "src").resolve())
        prev = env.get("PYTHONPATH", "")
        if prev:
            env["PYTHONPATH"] = src_path + os.pathsep + prev
        else:
            env["PYTHONPATH"] = src_path

    cmd = [sys.executable, "-m", module, *args]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, check=True, env=env)


def list_images(fig_dir, pattern="*.png"):
    return sorted(Path(fig_dir).glob(pattern))


def show_images_grid(
    image_paths,
    cols=3,
    max_images=None,
    title=None,
):
    if max_images is not None:
        image_paths = image_paths[:max_images]

    if not image_paths:
        print("No images to show.")
        return

    count = len(image_paths)
    rows = (count + cols - 1) // cols

    # simple sizing so each tile is readable
    fig_w_per_col = 5.0
    fig_h_per_row = 4.0
    fig_w = fig_w_per_col * cols
    fig_h = fig_h_per_row * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)

    # normalize axes to a 2D list
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes_2d = []
        for ax in axes:
            axes_2d.append([ax])
        axes = axes_2d

    flat_axes = []
    for row in axes:
        for ax in row:
            flat_axes.append(ax)

    for ax, img_path in zip(flat_axes, image_paths):
        ax.imshow(plt.imread(img_path))
        ax.set_title(img_path.name, fontsize=10)
        ax.axis("off")

    for ax in flat_axes[len(image_paths):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.show()


def is_run_bundle_dir(path):
    return (
        (path / "fields.npz").exists()
        and (path / "meta.json").exists()
        and (path / "metrics.json").exists()
    )


def find_latest_run_bundle_dir(base_dir):
    # prefer latest child run bundle (used by timestamped runs)
    base_dir = Path(base_dir)
    candidates = []
    for child in base_dir.iterdir():
        if child.is_dir() and is_run_bundle_dir(child):
            candidates.append(child)

    if candidates:
        latest = candidates[0]
        for i in range(1, len(candidates)):
            current = candidates[i]
            if current.name > latest.name:
                latest = current
        return latest

    # fallback to base dir bundle
    if is_run_bundle_dir(base_dir):
        return base_dir

    raise FileNotFoundError(f"No run bundle found under {base_dir}")


def assert_paths_exist(paths):
    missing = []
    for p in paths:
        path_obj = Path(p)
        if not path_obj.exists():
            missing.append(str(path_obj))

    if missing:
        raise FileNotFoundError("Missing expected outputs:\n" + "\n".join(missing))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
