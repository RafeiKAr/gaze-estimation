r"""archivieren.py – alte, nicht mehr benötigte Dateien in den Ordner archiv\ verschieben.

Es wird NICHTS gelöscht. Jede Datei wird nur verschoben und behält ihren
Pfad innerhalb von archiv\ (z.B. notebooks\x.ipynb -> archiv\notebooks\x.ipynb).
Eine Liste aller verschobenen Dateien steht danach in archiv\INHALT.txt.

Aufruf im Projektordner:
    python archivieren.py               # nur anzeigen, was verschoben würde (Probelauf)
    python archivieren.py --ausfuehren  # wirklich verschieben

Rückgängig machen: die Dateien aus archiv\ einfach wieder zurückkopieren.
"""
import argparse
import shutil
import time
from pathlib import Path

ARCHIV = Path("archiv")

# Was archiviert wird (Pfade relativ zum Projektordner; * ist ein Platzhalter)
ALT = [
    # alte Skripte (ersetzt durch train_base.py, personal_exp.py und Masterarbeit.ipynb)
    "model_train.py", "train.py", "utils.py",
    # alte Notebooks (ersetzt durch Masterarbeit.ipynb)
    "notebooks",
    # alte Sicherungen, alte Abbildungen, TensorBoard-Logs, alte PDF-Kapitel
    "backup_2026-10-21", "figures", "runs", "docs",
    # alte Modelle (die Arbeit nutzt nur models/base/)
    "models/*.path", "models/personal",
    # alte Ergebnisdateien und Diagramme (nicht in der Arbeit verwendet)
    "results/baseline_predictions*.csv", "results/diag_error_*.png",
    "results/compare", "results/dataset",
    "results/personal/*.png", "results/personal/diagrams", "results/personal/compare_diagrams",
    "results/personal/predictions_10.csv",
    "results/heatmap/base_loss1_2.csv", "results/heatmap/base_loss2_2.csv",
    "results/heatmap/base_loss1_norm_random.csv",
    "results/exp_reproduction/exp_results.json",
    # Aufnahmeprogramm der verworfenen Generation 3
    "personalization/base_Code_corner.py",
    # automatisch erzeugte Zwischenordner
    ".ipynb_checkpoints", "__pycache__", "personalization/__pycache__",
]

# Diese Dateien und Ordner bleiben IMMER (Sicherheitsprüfung)
WICHTIG = [
    "Masterarbeit.ipynb", "README.md", "requirements.txt",
    "main.py", "dataset.py", "model.py", "metrics.py", "exp.py", "train_base.py",
    "personal_exp.py", "session_design.py", "spatial_analysis.py", "plot_targets.py", "auswertung.py",
    "dataset", "splits", "personalization", "models/base", "results/personal/personal_runs.jsonl",
    "results/personal/predictions", "results/exp_reproduction/exp_results_all.json",
    "results/heatmap/base_loss1.csv", "results/heatmap/base_loss2.csv", "demo",
]


def kandidaten():
    gefunden = []
    for muster in ALT:
        treffer = sorted(Path(".").glob(muster)) if "*" in muster else [Path(muster)]
        gefunden += [p for p in treffer if p.exists()]
    for p in gefunden:                                  # Sicherheitsprüfung
        for w in WICHTIG:
            if p == Path(w) or p in Path(w).parents:     # p ist oder enthält eine wichtige Datei
                raise SystemExit(f"Abbruch: {p} gehört zu den wichtigen Dateien ({w}).")
    return gefunden


def groesse(p):
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ausfuehren", action="store_true", help="wirklich verschieben (sonst nur anzeigen)")
    args = ap.parse_args()
    if not Path("personal_exp.py").exists():
        raise SystemExit("Bitte im Projektordner starten (dort, wo personal_exp.py liegt).")

    liste = kandidaten()
    if not liste:
        print("Nichts zu archivieren – das Projekt ist schon aufgeräumt.")
        return
    gesamt = 0
    print(("Verschiebe" if args.ausfuehren else "Probelauf – würde verschieben") + f" nach {ARCHIV}\\:\n")
    for p in liste:
        mb = groesse(p) / 1e6
        gesamt += mb
        print(f"  {str(p):60s} {mb:9.1f} MB")
    print(f"\n  {len(liste)} Einträge, zusammen {gesamt:.0f} MB")

    if not args.ausfuehren:
        print("\nEs wurde nichts verändert. Zum Verschieben:  python archivieren.py --ausfuehren")
        return

    protokoll = [f"Archiviert am {time.strftime('%d.%m.%Y %H:%M')}"]
    for p in liste:
        ziel = ARCHIV / p
        if ziel.exists():                               # nichts überschreiben
            ziel = ziel.with_name(ziel.name + time.strftime("_%Y%m%d_%H%M%S"))
        ziel.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(ziel))
        protokoll.append(f"{p}  ->  {ziel}")
    with (ARCHIV / "INHALT.txt").open("a", encoding="utf-8") as f:
        f.write("\n".join(protokoll) + "\n\n")
    print(f"\nFertig. Liste: {ARCHIV / 'INHALT.txt'}")


if __name__ == "__main__":
    main()
