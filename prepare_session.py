"""Prepare NEW personalization sessions: frames + label files from video and positions.

Replaces notebooks/personal_code.py for new recordings. Differences:
  * only the sessions given on the command line are processed
  * the screen size is taken per session (no fixed 1430 x 880)
  * nothing is overwritten: a session that already has images/ or label files
    is skipped with a message

Input  (personalization/<s>/):  <s>_video.mp4, <s>_position.txt (from base_Code_rand.py)
Output (personalization/<s>/):  images/0000.jpg ...   every 5th video frame (6 labels/s)
                                labels.csv            target in screen pixels
                                norm_labels.csv       target normalized to [0, 1]^2

Screen size (W x H in pixels, the area in which the point moves):
  * --size 1870x1000 sets it for all given sessions, or
  * a line "Screen: 1870x1000" (or "Screen: 1870×1000") in <s>_subject.txt, or
  * otherwise it is estimated as max(x) + radius, max(y) + radius (radius 8 px,
    as in base_Code_rand.py). The point touches all four walls in a 120 s
    recording, so this estimate is exact to about one pixel. It is printed;
    please check it.

Usage (from the project folder):
    python prepare_session.py 28 29
    python prepare_session.py 29 --size 1870x1000
"""
import argparse
import re
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path("personalization")
STEP = 5          # 30 fps video -> one label every 5 frames = 6 labels per second
RADIUS = 8        # radius of the point in base_Code_rand.py


def parse_size(text):
    m = re.search(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def screen_size(session, positions, override):
    if override:
        return override, "--size"
    subject = ROOT / session / f"{session}_subject.txt"
    if subject.exists():
        for line in subject.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("screen"):
                size = parse_size(line)
                if size:
                    return size, "subject.txt"
    w = int(round(positions["x"].max() + RADIUS))
    h = int(round(positions["y"].max() + RADIUS))
    return (w, h), "estimated from positions"


def prepare(session, override=None):
    folder = ROOT / session
    video = folder / f"{session}_video.mp4"
    pos_file = folder / f"{session}_position.txt"
    images = folder / "images"
    targets = [folder / "labels.csv", folder / "norm_labels.csv"]

    if not video.exists() or not pos_file.exists():
        print(f"[{session}] skipped: {video.name} or {pos_file.name} missing")
        return
    if (images.exists() and any(images.iterdir())) or any(t.exists() for t in targets):
        print(f"[{session}] skipped: images/ or label files already exist (nothing overwritten)")
        return

    pos = pd.read_csv(pos_file)
    pos.columns = pos.columns.str.strip()
    (w, h), source = screen_size(session, pos, override)
    print(f"[{session}] screen {w} x {h} ({source}); {len(pos)} position rows")

    images.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    rows, norm_rows = [], []
    frame_id = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_id >= len(pos):
            break
        if frame_id % STEP == 0:
            name = f"{saved:04d}.jpg"
            cv2.imwrite(str(images / name), frame)
            x, y = float(pos.loc[frame_id, "x"]), float(pos.loc[frame_id, "y"])
            rows.append({"frame": name, "x": round(max(0.0, x), 6), "y": round(max(0.0, y), 6)})
            norm_rows.append({"frame": name,
                              "x": round(min(max(0.0, x / w), 1.0), 6),
                              "y": round(min(max(0.0, y / h), 1.0), 6)})
            saved += 1
        frame_id += 1
    cap.release()

    if frame_id < len(pos):
        print(f"[{session}] note: video has fewer frames ({frame_id}) than position rows ({len(pos)})")
    pd.DataFrame(rows).to_csv(targets[0], index=False)
    pd.DataFrame(norm_rows).to_csv(targets[1], index=False)
    n = pd.DataFrame(norm_rows)
    print(f"[{session}] {saved} frames saved; normalized x {n.x.min():.3f}-{n.x.max():.3f}, "
          f"y {n.y.min():.3f}-{n.y.max():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session numbers, e.g. 28 29")
    ap.add_argument("--size", default="", help="screen size WxH for all given sessions, e.g. 1870x1000")
    args = ap.parse_args()
    override = parse_size(args.size) if args.size else None
    if args.size and not override:
        ap.error("--size must look like 1870x1000")
    for s in args.sessions:
        prepare(s.zfill(2), override)


if __name__ == "__main__":
    main()
