"""Spatial error analysis (Figure 4 and the region statistics of the thesis).

Input: a CSV with columns gt_x, gt_y, pred_x, pred_y (normalized [0, 1]^2),
e.g. results/heatmap/base_loss1.csv (200 test samples, norm_subject split).

Screen partition (3 x 3 reference grid in the normalized space)
---------------------------------------------------------------
    centre  : (0.50, 0.50)
    edges   : (0.10, 0.50), (0.50, 0.10), (0.90, 0.50), (0.50, 0.90)
    corners : (0.10, 0.10), (0.10, 0.90), (0.90, 0.10), (0.90, 0.90)

A test sample belongs to a grid point g if ||t - g||_2 <= r (default r = 0.15).
For each grid point the mean Euclidean error of its samples is reported; a
region value (centre / edge / corner) is the mean over all samples assigned to
the grid points of that region.

Gaussian heat map (same method as notebooks/visualization.py)
-------------------------------------------------------------
For a cursor position c the mean error e(c) of all samples within radius r is
spread over the screen with an isotropic Gaussian kernel

    H(x, y) = e(c) * exp(-||(x, y) - c||^2 / (2 sigma^2)),   sigma = 0.20

so colour encodes the *magnitude* of the local mean error (same colour scale
in all panels), and the position of the blob shows *where* it was measured.

Usage:
    python spatial_analysis.py --input results/heatmap/base_loss1.csv \
        --compare results/heatmap/base_loss2.csv --out results/figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CENTER = [(0.50, 0.50)]
EDGES = [(0.10, 0.50), (0.50, 0.10), (0.90, 0.50), (0.50, 0.90)]
CORNERS = [(0.10, 0.10), (0.10, 0.90), (0.90, 0.10), (0.90, 0.90)]
REGIONS = {"center": CENTER, "edge": EDGES, "corner": CORNERS}


def load_predictions(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    gt = df[["gt_x", "gt_y"]].to_numpy(float)
    pred = df[["pred_x", "pred_y"]].to_numpy(float)
    return gt, pred


def grid_statistics(gt, pred, radius=0.15):
    """Per grid point: (mean Euclidean error, number of samples)."""
    err = np.linalg.norm(gt - pred, axis=1)
    rows = []
    for region, points in REGIONS.items():
        for p in points:
            mask = np.linalg.norm(gt - np.array(p), axis=1) <= radius
            rows.append({
                "region": region, "x": p[0], "y": p[1], "n": int(mask.sum()),
                "mean_error": float(err[mask].mean()) if mask.any() else np.nan,
            })
    return pd.DataFrame(rows)


def region_statistics(gt, pred, radius=0.15):
    """Pooled mean Euclidean error per region (centre / edge / corner)."""
    err = np.linalg.norm(gt - pred, axis=1)
    out = {}
    for region, points in REGIONS.items():
        mask = np.zeros(len(gt), bool)
        for p in points:
            mask |= np.linalg.norm(gt - np.array(p), axis=1) <= radius
        out[region] = {"n": int(mask.sum()), "mean_error": float(err[mask].mean())}
    return out


def gaussian_heatmap(gt, pred, cursor, radius=0.15, sigma=0.20, res=200):
    err = np.linalg.norm(gt - pred, axis=1)
    mask = np.linalg.norm(gt - np.array(cursor), axis=1) <= radius
    mean_err = float(err[mask].mean()) if mask.any() else 0.0
    g = np.linspace(0, 1, res)
    gx, gy = np.meshgrid(g, g)
    h = mean_err * np.exp(-((gx - cursor[0]) ** 2 + (gy - cursor[1]) ** 2) / (2 * sigma ** 2))
    return h, mean_err, int(mask.sum())


def plot_figure4(gt, pred, out_file, radius=0.15, sigma=0.20,
                 cursors=(("Centre", (0.50, 0.50)), ("Edge", (0.10, 0.50)), ("Corner", (0.10, 0.10)))):
    """Three heat-map panels (centre, edge, corner) + prediction scatter."""
    import matplotlib.pyplot as plt

    maps = [gaussian_heatmap(gt, pred, c, radius, sigma) for _, c in cursors]
    vmax = max(m[1] for m in maps)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1), constrained_layout=True)
    im = None
    for ax, (name, c), (h, e, n) in zip(axes, cursors, maps):
        im = ax.imshow(h, extent=[0, 1, 0, 1], origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax.add_patch(plt.Circle(c, radius, fill=False, ls="--", color="white", lw=1))
        ax.set_title(f"{name} {c}\nmean error = {e:.3f} (n = {n})", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("x (normalized)")
    axes[0].set_ylabel("y (normalized)")
    cb = fig.colorbar(im, ax=axes[:3], shrink=0.85, pad=0.01)
    cb.set_label("mean Euclidean error (normalized units)")

    ax = axes[3]
    ax.scatter(gt[:, 0], gt[:, 1], s=14, c="#1f77b4", label="ground truth", alpha=0.7)
    ax.scatter(pred[:, 0], pred[:, 1], s=14, c="#d62728", marker="x", label="prediction")
    for pts, col in ((CENTER, "white"), (EDGES, "#888"), (CORNERS, "#444")):
        for p in pts:
            ax.add_patch(plt.Circle(p, radius, fill=False, ls=":", color="k", lw=0.8))
    ax.set_title(f"Ground truth vs. prediction\n(dotted: grid neighbourhoods, r = {radius:.2f})", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x (normalized)")
    ax.legend(loc="upper right", fontsize=8)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--compare", default=None, help="second CSV (e.g. sensitive loss)")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--radius", type=float, default=0.15, help="grid neighbourhood radius")
    args = ap.parse_args()

    gt, pred = load_predictions(args.input)
    print(grid_statistics(gt, pred, args.radius).round(4).to_string(index=False))
    print(region_statistics(gt, pred, args.radius))
    plot_figure4(gt, pred, Path(args.out) / "fig4_spatial_heatmaps.png")
    if args.compare:
        gt2, pred2 = load_predictions(args.compare)
        a = grid_statistics(gt, pred, args.radius)
        b = grid_statistics(gt2, pred2, args.radius)
        a["compare"] = b["mean_error"]
        a["delta"] = a["compare"] - a["mean_error"]
        print(a.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
