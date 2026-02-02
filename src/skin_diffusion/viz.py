import matplotlib.pyplot as plt
import numpy as np


# plot a heatmap of a 2D field
# bright = high concentration

def plot_heatmap(C, save_path, title=None, vmin=None, vmax=None):
    plt.figure()
    plt.imshow(C, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# plot average across x vs depth
# y-axis is depth index

def plot_depth_profile(C, save_path):
    depth = np.arange(C.shape[0])
    mean_x = C.mean(axis=1)
    plt.figure()
    plt.plot(mean_x, depth)
    plt.gca().invert_yaxis()
    plt.xlabel("C (x-avg)")
    plt.ylabel("depth index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# plot a mask as 0/1 image
# 1 means patch location

def plot_mask(mask, save_path):
    plt.figure()
    plt.imshow(mask.astype(int), origin="upper", aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
