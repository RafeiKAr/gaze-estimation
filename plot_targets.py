"""Figure 4.1 (a)/(b): raw vs. normalized GazeCapture gaze targets.

Both panels are drawn with exactly the same figure size, axes position, square
plot area, colour scale (1 ... 10^4 frames per bin) and legend area, so that they
look identical side by side in the thesis.

Usage (project root):
    python plot_targets.py --lang de --out results/figures      # German labels
    python plot_targets.py --lang en --out results/figures_en   # English labels
Reads dataset/labels.csv (screen points) and dataset/norm_labels.csv ([0,1]^2).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import pandas as pd

TXT = {
    "de": dict(xr="x (Punkte)", yr="y (Punkte)", xn="x / Bildschirmbreite", yn="y / Bildschirmhöhe",
               cb="Bilder pro Zelle (log)", legr="Bildschirmformate (Anteil der Bilder)",
               legn="Normierter Bildschirm", unit="Einheitsquadrat [0, 1]²",
               groups=("Smartphone hoch", "Smartphone quer", "Tablet"), dec=","),
    "en": dict(xr="x (points)", yr="y (points)", xn="x / screen width", yn="y / screen height",
               cb="Frames per bin (log)", legr="Screen formats (share of frames)",
               legn="Normalized screen", unit="unit square [0, 1]²",
               groups=("Phone, portrait", "Phone, landscape", "Tablet"), dec="."),
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", default="results/figures")
ap.add_argument("--lang", choices=("de", "en"), default="de")
a = ap.parse_args()
T = TXT[a.lang]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(Path(a.data) / "labels.csv")
norm = pd.read_csv(Path(a.data) / "norm_labels.csv")

plt.rcParams.update({"font.size": 11, "font.family": ["Inter", "DejaVu Sans"],
                     "axes.formatter.use_locale": False})
CMAP = "viridis"
NORM = dict(norm=LogNorm(vmin=1, vmax=1e4))
FIGSIZE = (5.0, 5.0)                       # inches, identical for both panels
AX_RECT = [0.15, 0.25, 0.60, 0.60 * FIGSIZE[0] / FIGSIZE[1]]   # square plot area
CB_RECT = [0.79, 0.25, 0.035, AX_RECT[3]]
# identical saved area (inches) for both panels: cut empty margin at top and bottom
SAVE_BOX = Bbox([[0, 0.07 * FIGSIZE[1]], [FIGSIZE[0], 0.98 * FIGSIZE[1]]])


def fmt(v, _=None):
    return f"{v:.1f}".replace(".", T["dec"])


def new_panel():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    cax = fig.add_axes(CB_RECT)
    return fig, ax, cax


def finish(fig, ax, cax, hb, handles, title):
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    cb = fig.colorbar(hb, cax=cax); cb.set_label(T["cb"])
    # same anchor in both panels: the legends start at the same height
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, 0.165), ncol=min(2, len(handles)),
               fontsize=9, frameon=False, handlelength=2.2, columnspacing=1.0, labelspacing=0.3)


# (a) raw targets in screen points
fig, ax, cax = new_panel()
hb = ax.hexbin(raw.x, raw.y, gridsize=60, extent=(0, 1060, 0, 1060), cmap=CMAP, mincnt=1,
               linewidths=0, **NORM)
sizes = norm.groupby(["screen_w", "screen_h"]).size()
styles = [("#d62728", "--", lambda w, h: h > w and w < 700),
          ("#ff7f0e", ":", lambda w, h: w > h and h < 700),
          ("#000000", "-.", lambda w, h: max(w, h) >= 1000)]
handles = []
for name, (col, ls, f) in zip(T["groups"], styles):
    sel = [(w, h) for (w, h) in sizes.index if f(w, h)]
    share = 100 * sum(sizes[k] for k in sel) / len(norm)
    for (w, h) in sel:
        ax.add_patch(Rectangle((0, 0), w, h, fill=False, ec=col, lw=1.0, ls=ls))
    handles.append(Line2D([], [], color=col, ls=ls, lw=1.2, label=f"{name} ({share:.0f} %)"))
ax.set_xlim(0, 1060); ax.set_ylim(1060, 0)
ax.set_xticks(range(0, 1001, 200)); ax.set_yticks(range(0, 1001, 200))
ax.set_xlabel(T["xr"]); ax.set_ylabel(T["yr"])
finish(fig, ax, cax, hb, handles, T["legr"])
fig.savefig(out / "raw_targets.png", dpi=250, bbox_inches=SAVE_BOX)

# (b) normalized targets
fig, ax, cax = new_panel()
hb = ax.hexbin(norm.x, norm.y, gridsize=50, extent=(0, 1, 0, 1), cmap=CMAP, mincnt=1,
               linewidths=0, **NORM)
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="#d62728", lw=1.0, ls="--"))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(1.03, -0.03)
ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel(T["xn"]); ax.set_ylabel(T["yn"])
finish(fig, ax, cax, hb, [Line2D([], [], color="#d62728", ls="--", lw=1.2, label=T["unit"])], T["legn"])
fig.savefig(out / "normalized_targets.png", dpi=250, bbox_inches=SAVE_BOX)
print(f"{len(raw)} frames, {norm.subject_ID.nunique()} subjects -> {out}")
