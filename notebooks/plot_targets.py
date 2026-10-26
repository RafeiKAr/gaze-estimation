"""Figure 4.1 (a)/(b): raw vs. normalized GazeCapture gaze targets.

Usage (project root):  python plot_targets.py [--out results/figures]
Reads dataset/labels.csv (screen points) and dataset/norm_labels.csv ([0,1]^2).
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", default="results/figures")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(Path(a.data) / "labels.csv")
norm = pd.read_csv(Path(a.data) / "norm_labels.csv")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
CMAP = "viridis"

def style(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()                       # screen coordinates: y grows downwards
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for s in ax.spines.values():
        s.set_linewidth(0.6)

# (a) raw targets in screen points
fig, ax = plt.subplots(figsize=(4.8, 5.0), constrained_layout=True)
hb = ax.hexbin(raw.x, raw.y, gridsize=70, extent=(0, 1024, 0, 1024), cmap=CMAP,
               norm=LogNorm(vmin=1), mincnt=1, linewidths=0)
sizes = norm.groupby(["screen_w", "screen_h"]).size()
groups = {                                   # screen formats (points), grouped by device type
    "Phone, portrait":  ("#d62728", "--", lambda w, h: h > w and w < 700),
    "Phone, landscape": ("#ff7f0e", ":",  lambda w, h: w > h and h < 700),
    "Tablet":           ("#000000", "-.", lambda w, h: max(w, h) >= 1000),
}
for name, (col, ls, f) in groups.items():
    sel = [(w, h) for (w, h) in sizes.index if f(w, h)]
    share = 100 * sum(sizes[k] for k in sel) / len(norm)
    for i, (w, h) in enumerate(sel):
        ax.add_patch(Rectangle((0, 0), w, h, fill=False, ec=col, lw=1.0, ls=ls,
                               label=f"{name} ({share:.0f} %)" if i == 0 else None))
fig.legend(loc="outside lower center", ncol=3, fontsize=7, frameon=False, columnspacing=1.2,
           title="Screen outlines (share of frames)", title_fontsize=7, handlelength=2.2)
ax.set_xlim(0, 1060); ax.set_ylim(1060, 0)
style(ax); ax.set_ylim(1060, 0)
ax.set_xlabel("x (points)"); ax.set_ylabel("y (points)")
cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Frames per bin (log)")
fig.savefig(out / "raw_targets.png", dpi=250)

# (b) normalized targets
fig, ax = plt.subplots(figsize=(4.8, 4.6), constrained_layout=True)
hb = ax.hexbin(norm.x, norm.y, gridsize=50, extent=(0, 1, 0, 1), cmap=CMAP,
               norm=LogNorm(vmin=1), mincnt=1, linewidths=0)
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="#d62728", lw=0.9, ls="--"))
ax.set_xlim(-0.03, 1.03); style(ax); ax.set_ylim(1.03, -0.03)
ax.set_xlabel("x / screen width"); ax.set_ylabel("y / screen height")
cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Frames per bin (log)")
fig.savefig(out / "normalized_targets.png", dpi=250)
print(f"{len(raw)} frames, {norm.subject_ID.nunique()} subjects")
