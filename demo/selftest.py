r"""Selbsttest der Demo OHNE Browser und OHNE Webcam.

Prüft, ob Modell und Personalisierung richtig funktionieren, indem eine eigene
Aufnahmesitzung der Masterarbeit so behandelt wird, als käme sie aus der Demo:
die Bilder werden in Aufnahmereihenfolge als "Kalibrierung" übergeben
(6 Bilder pro Sekunde), dann wird der Kopf genau wie in der Demo trainiert.

Aufruf (im Ordner demo):
    python selftest.py                     # Sitzung 02, ganze Aufnahme
    python selftest.py --session 04        # andere Sitzung
    python selftest.py --seconds 50        # nur die ersten 50 Sekunden
    python selftest.py --crop none         # ganzes Bild wie in der Masterarbeit
                                           # (Standard: --crop eyes wie in der Demo)

Es wird nur eine Sitzung gebraucht, deren Bilder auf diesem Rechner liegen
(Ordner images\ oder images2\). Fehlen sie, zeigt das Skript, welche
Sitzungen Bilder haben.

Erwartung: "nachher" deutlich kleiner als "vorher". Zum Vergleich: Mit dem
sequenziellen Split (Kapitel 4, E3) lag das Basismodell bei ca. 27-31 %;
gute Sitzungen wurden durch die Personalisierung deutlich besser.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--crop", choices=["eyes", "face", "none"], default=None,
                help="Bildausschnitt wie in der Demo (Standard: eyes)")
ap.add_argument("--session", default="02")
ap.add_argument("--seconds", type=float, default=0, help="0 = ganze Sitzung")
ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "personalization"))
args = ap.parse_args()
os.environ.pop("GAZE_DEBUG", None)          # Selbsttest überschreibt keine Demo-Berichte
if args.crop:
    os.environ["GAZE_CROP"] = args.crop
from gaze_model import GazeModel, Session  # noqa: E402  (nach dem Setzen von GAZE_CROP)

root = Path(args.root)
folder = root / args.session
images = folder / "images2" if (folder / "images2").is_dir() else folder / "images"
labels = pd.read_csv(folder / "norm_labels.csv")
first = images / str(labels.iloc[0]["frame"])
if not first.exists():
    have = sorted(d.name for d in root.iterdir()
                  if d.is_dir() and any((d / sub).is_dir() and any((d / sub).iterdir())
                                        for sub in ("images2", "images")))
    raise SystemExit(f"Für Sitzung {args.session} liegen auf diesem Rechner keine Bilder "
                     f"(gesucht: {first}).\nSitzungen mit Bildern: {', '.join(have) or 'keine'}\n"
                     f"Beispiel: python selftest.py --session {have[0] if have else '02'}")
if args.seconds > 0:
    labels = labels.iloc[: int(args.seconds * 6)]
print(f"Sitzung {args.session}: {len(labels)} Bilder aus {images}")

t0 = time.time()
gm = GazeModel()
print(f"Modell geladen in {time.time() - t0:.1f} s")

s = Session(gm)
t0 = time.time()
for i, row in labels.reset_index(drop=True).iterrows():
    with Image.open(images / str(row["frame"])) as img:
        s.add_calibration_frame(img, row["x"], row["y"], t=i / 6.0)
dt = time.time() - t0
print(f"Merkmale berechnet: {len(s.feats)} Bilder in {dt:.1f} s ({1000 * dt / max(1, len(labels)):.0f} ms pro Bild)")

r = s.calibrate()
print("\nErgebnis (Prüfung auf 2 von 10 Zeitblöcken, nicht trainiert):")
print(f"  Konstante Schätzung: {r['error_constant']:6.2f} %")
print(f"  Vorher (Basismodell): {r['error_before']:6.2f} %")
print(f"  Nachher (personal.):  {r['error_after']:6.2f} %")
print(f"  beste Epoche {r['best_epoch']} von {r['epochs']}, Training {r['seconds']} s")
print(f"  Verfahren: {r['method']}  (MLP-Kopf {r['error_mlp']:.2f} %, Ridge global {r['error_ridge']:.2f} %, "
      f"Ridge räumlich {r['error_spatial']:.2f} %, {r['spatial_dims']} Merkmale)")

# Gegenprobe: zufälliger Split wie in Kapitel 4 (E1: 70 % Training, 15 % Prüfung,
# Seed 42). Dieser Split ist optimistisch (fast gleiche Nachbarbilder), zeigt aber,
# ob die Pipeline dieselben Zahlen wie die Masterarbeit liefert.
import numpy as np
perm = np.random.default_rng(42).permutation(len(s.feats))
n_tr = int(0.70 * len(perm))
r2 = s.calibrate(split=(np.sort(perm[:n_tr]), np.sort(perm[n_tr:n_tr + int(0.15 * len(perm))])))
print("\nGegenprobe mit zufälligem Split (wie Kapitel 4, optimistisch):")
print(f"  Vorher {r2['error_before']:6.2f} %  ->  nachher {r2['error_after']:6.2f} %  "
      f"(beste Epoche {r2['best_epoch']})")
s.calibrate()                          # wieder das Demo-Modell (Blöcke) verwenden

base, personal = s.predict(Image.open(images / str(labels.iloc[-1]["frame"])))
print(f"\nBeispiel letzte Aufnahme: Ziel ({labels.iloc[-1]['x']:.2f}, {labels.iloc[-1]['y']:.2f}), "
      f"vorher ({base[0]:.2f}, {base[1]:.2f}), nachher ({personal[0]:.2f}, {personal[1]:.2f})")
print("\nSelbsttest erfolgreich." if r["error_after"] < r["error_before"] else
      "\nHinweis: keine Verbesserung – Bilder/Labels prüfen.")
