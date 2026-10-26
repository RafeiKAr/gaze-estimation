"""Session design analysis: spatial coverage and transition (invalid) samples.

Works only with the label/position files of the sessions (no images, no
training), so it can be run on any computer.

For every session it reports
  * number of labelled frames and recording length,
  * spatial coverage of the normalized screen [0,1]^2:
      - share of labels in the centre / edge / corner neighbourhoods of the
        3x3 reference grid (radius 0.15, same definition as spatial_analysis.py),
      - share of occupied cells of a 10x10 grid,
      - mean distance of the targets from the screen centre,
  * number of target jumps and the share of transition samples:
      - Gen 4: exact, from the logged phase ("wait") in <id>_position.txt,
      - Gen 2: estimated, labels within --transition-sec after a jump
               (jump = target displacement > 100 px between two frames),
      - Gen 1: no jumps (continuous movement).

Mapping label -> position row: label frame i (= image "iiii.jpg") belongs to
position frame i * FRAME_STEP (FRAME_STEP = 5, verified: the normalized label
times the screen size reproduces the logged pixel position exactly).

It can also (re)write the cleaned label files:
  python session_design.py --write-clean          # Gen 4: drop WAIT frames
  python session_design.py --write-clean --gen2   # also Gen 2 (heuristic)

Usage:
  python session_design.py                        # table + results/personal/session_design.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("personalization")
SESSIONS_GEN = {
    "Gen 1": ["01", "02", "03", "04"],
    "Gen 2": ["05", "06", "07", "08", "09"],
    "Gen 4": ["19", "20", "21", "22", "23", "24", "25", "26", "27"],
}
DESIGN = {"Gen 1": "standard (continuous)", "Gen 2": "random jumps", "Gen 4": "corner-weighted jumps (90 %)"}
FRAME_STEP = 5          # one label every 5 video frames
FPS = 30
JUMP_PX = 100           # displacement between two video frames that counts as a jump

CENTER = [(0.50, 0.50)]
EDGES = [(0.10, 0.50), (0.50, 0.10), (0.90, 0.50), (0.50, 0.90)]
CORNERS = [(0.10, 0.10), (0.10, 0.90), (0.90, 0.10), (0.90, 0.90)]


def read_positions(session):
    p = pd.read_csv(ROOT / session / f"{session}_position.txt", skipinitialspace=True)
    p.columns = [c.strip() for c in p.columns]
    return p


def label_rows(session, labels):
    """Position rows that belong to the label rows (same order)."""
    p = read_positions(session).set_index("frame")
    idx = labels["frame"].astype(str).str.replace(".jpg", "", regex=False).astype(int).to_numpy()
    return p.reindex(idx * FRAME_STEP)


def transition_mask(session, gen, labels, transition_sec=0.5):
    """True for labels recorded while the gaze is not on the target yet."""
    rows = label_rows(session, labels)
    if gen == "Gen 4":
        return (rows["phase"] == "wait").to_numpy()
    if gen == "Gen 2":
        p = read_positions(session)
        d = np.hypot(np.diff(p["x"]), np.diff(p["y"]))
        jump_frames = p["frame"].to_numpy()[1:][d > JUMP_PX]
        f = rows.index.to_numpy()
        mask = np.zeros(len(f), bool)
        for j in jump_frames:
            mask |= (f >= j) & (f < j + transition_sec * FPS)
        return mask
    return np.zeros(len(labels), bool)


def region_shares(xy, radius=0.15):
    out = {}
    for name, pts in (("center", CENTER), ("edge", EDGES), ("corner", CORNERS)):
        m = np.zeros(len(xy), bool)
        for g in pts:
            m |= np.linalg.norm(xy - np.array(g), axis=1) <= radius
        out[name] = m.mean()
    return out


def analyse(transition_sec=0.5):
    rows = []
    for gen, sessions in SESSIONS_GEN.items():
        for s in sessions:
            lab = pd.read_csv(ROOT / s / "norm_labels.csv")
            xy = lab[["x", "y"]].to_numpy(float)
            pos = read_positions(s)
            d = np.hypot(np.diff(pos["x"]), np.diff(pos["y"]))
            cells = np.clip((xy * 10).astype(int), 0, 9)
            reg = region_shares(xy)
            tm = transition_mask(s, gen, lab, transition_sec)
            rows.append({
                "session": s, "generation": gen, "design": DESIGN[gen],
                "labels": len(lab), "seconds": round(len(pos) / FPS, 1),
                "jumps": int((d > JUMP_PX).sum()),
                "transition_share": tm.mean(),
                "center_share": reg["center"], "edge_share": reg["edge"], "corner_share": reg["corner"],
                "cells_10x10": len(np.unique(cells[:, 0] * 10 + cells[:, 1])) / 100,
                "mean_dist_center": np.linalg.norm(xy - 0.5, axis=1).mean(),
            })
    return pd.DataFrame(rows)


def write_clean(include_gen2=False, transition_sec=0.5):
    """Write norm_labels_clean.csv = norm_labels.csv without transition samples."""
    for gen, sessions in SESSIONS_GEN.items():
        if gen == "Gen 1" or (gen == "Gen 2" and not include_gen2):
            continue
        for s in sessions:
            lab = pd.read_csv(ROOT / s / "norm_labels.csv")
            keep = ~transition_mask(s, gen, lab, transition_sec)
            lab[keep].to_csv(ROOT / s / "norm_labels_clean.csv", index=False)
            print(f"{s}: {len(lab)} -> {keep.sum()} labels ({100 * (1 - keep.mean()):.1f} % removed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transition-sec", type=float, default=0.5)
    ap.add_argument("--write-clean", action="store_true")
    ap.add_argument("--gen2", action="store_true", help="with --write-clean: also clean Gen 2 (heuristic)")
    ap.add_argument("--out", default="results/personal/session_design.csv")
    args = ap.parse_args()

    if args.write_clean:
        write_clean(args.gen2, args.transition_sec)
        return
    df = analyse(args.transition_sec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nMeans per generation:")
    print(df.groupby("generation").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
