# Dateneffiziente Personalisierung für webcam-basiertes Eye-Tracking

Masterarbeit (Technische Informatik), Khaled Rafei.

Ein auf ImageNet vortrainiertes **ResNet-18** bleibt eingefroren. Ein kleiner
**Regressionskopf** sagt den Blickpunkt auf dem Bildschirm voraus. Der Kopf wird
zuerst auf **GazeCapture** trainiert (Basismodell) und danach mit wenigen Minuten
eigener Webcam-Aufnahmen **personalisiert**.

---

## 1. Schnellstart

```powershell
cd F:\project\Venv\gaze-estimation
venv\Scripts\activate              # virtuelle Umgebung aktivieren
jupyter notebook notebooks\Masterarbeit.ipynb
```

Das Notebook liegt in `notebooks/` und arbeitet automatisch im Projektordner.
Im Notebook die drei Zellen nacheinander ausführen:

| Zelle | Inhalt | Ergebnis |
|---|---|---|
| 1 | Daten prüfen, **Basismodelle** auf GazeCapture trainieren (`norm_subject`, `norm_random`) und mit den gespeicherten vergleichen | `models/base/`, `models/base_neu/` |
| 2 | *(optional)* Hyperparameter-Tabellen und Abbildungen zu GazeCapture | Tabellen, `results/figures/` |
| 3 | **Personalisierung** der Sessions 01–09 und 19–27 (Tabelle 4.13), optional E1b/E2/E3, Vergleich mit den Werten der Arbeit | `results/personal/neu_runs.jsonl` |

---

## 2. Ordnerstruktur

```
gaze-estimation/
├── README.md              diese Datei
├── requirements.txt       benötigte Python-Pakete
│
├── main.py                GazeCapture laden, Ziele normieren, Splits erstellen
├── dataset.py             Daten laden, Merkmals-Cache, Teilmengen
├── model.py               Modell (ResNet-18 + Kopf), Verluste, Training
├── metrics.py             Diagonalfehler und Baselines
├── exp.py                 Hyperparameter-Studie (5 Wiederholungen, 98-%-KI)
├── train_base.py          E0: die zwei Basismodelle trainieren und speichern
├── personal_exp.py        E1–E3: Personalisierung der Sessions
├── session_design.py      Abdeckung, Übergangsbilder, bereinigte Labels
├── spatial_analysis.py    Fehler in Mitte / Rand / Ecke, Abbildung 4.3
├── plot_targets.py        Abbildung 4.1 (rohe und normierte Ziele)
├── auswertung.py          Tabellen und Vergleiche für das Notebook
│
├── notebooks/
│   ├── Masterarbeit.ipynb  Haupt-Notebook: alle Experimente
│   └── archivieren.py      alte Dateien nach archiv/ verschieben
│
├── dataset/               GazeCapture (images/ nicht in Git), labels.csv, norm_labels.csv
├── splits/                norm_subject_*.csv, norm_random_*.csv
├── personalization/       eigene Aufnahmen: <Session>/images/, norm_labels.csv,
│                          norm_labels_clean.csv, Aufnahmeprogramme base_Code*.py
├── models/base/           die zwei Basismodelle der Arbeit (.pt) + Angaben (.json)
├── results/
│   ├── exp_reproduction/  exp_results_all.json (Hyperparameter-Studie)
│   ├── heatmap/           Vorhersagen für die räumliche Analyse
│   ├── personal/          personal_runs.jsonl (Ergebnisse der Arbeit), predictions/
│   └── figures/           Abbildungen
├── cache/                 gespeicherte ResNet-Merkmale (automatisch, nicht in Git)
├── demo/                  Web-Demo (eigene README)
└── archiv/                alte Dateien (nach archivieren.py)
```

**Aufnahmeprogramme:** `base_Code.py` = Gen 1 (Sessions 01–04, Punkt bewegt sich
gleichmäßig), `base_Code_rand.py` = Gen 4 (Sessions 19–27, 0,5 s WAIT + 4,5 s MOVE,
90 % der Startpunkte in den Ecken). Gen 3 (Sessions 10–18) wurde verworfen.

---

## 3. Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

GazeCapture einmal vorbereiten (braucht `kaggle.json`):

```powershell
python main.py
```

---

## 4. Ohne Notebook (Kommandozeile)

| Schritt | Befehl |
|---|---|
| Basismodelle (E0) | `python train_base.py` (vorhandene Modelle bleiben; neu: `--retrain --out-dir models/base_neu`) |
| Alle 18 Sessions (E1) | `python personal_exp.py --sessions all --base norm_subject norm_random` |
| Bereinigte Labels (E1b) | `python personal_exp.py --sessions gen4 --labels clean` |
| Aufteilung der Daten (E2) | `python personal_exp.py --sessions 21 22 --split random --seeds 42 43 44`<br>`python personal_exp.py --sessions 21 22 --split sequential --seeds 42 43 44`<br>`python personal_exp.py --split two_videos --pairs 21:22,22:21 --seeds 42 43 44` |
| Session-Design (E3) | `python personal_exp.py --sessions all --split sequential` |
| Ergebnisse anzeigen | `python personal_exp.py --summary` |
| Hyperparameter-Studie | `python exp.py --experiment all --sample-seed 0 --val-size 200` |
| Bereinigte Labels schreiben | `python session_design.py --write-clean` |

Unter Windows bei `personal_exp.py` zusätzlich `--num-workers 0` angeben (sonst kann das
Laden der Bilder in mehreren Prozessen fehlschlagen; das Ergebnis ändert sich dadurch nicht).

Achtung: `personal_exp.py` schreibt standardmäßig in `results/personal/personal_runs.jsonl`
und überspringt Läufe, die dort schon stehen. Für einen echten Neu-Lauf `--out` mit
einer neuen Datei angeben (das Notebook macht das automatisch).

---

## 5. Wichtigste Ergebnisse (Kapitel 4)

Fehler = Diagonalfehler in % der Bildschirmdiagonale (kleiner ist besser).

| Experiment | Modell | Konstant-Baseline |
|---|---|---|
| GazeCapture, `norm_subject`, 1000/200, nur FC | 28,39 % | 28,46 % |
| Basismodell auf den 18 Sessions (ohne Personalisierung) | 30,57 % | 30,32 % |
| Personalisierung, 18 Sessions, zufälliger Split (E1) | **19,79 %** | 30,32 % |
| dasselbe mit Basismodell `norm_random` | 19,59 % | 30,32 % |
| Personalisierung, sequenzieller Split (E3) | 27,93 % | 31,84 % |
| Sessions 21/22: zufällig / sequenziell / zwei Videos (E2) | 13,85 / 15,55 / 17,24 % | 30,97 / 28,67 / 32,65 % |
| Gen 4, MOVE-Testbilder: Original / bereinigt (E1b) | 16,72 / 15,88 % | 30,51 % |
| Beste Session (S02) | 10,48 % | 28,49 % |

**Reproduzierbarkeit:** Basismodelle und Personalisierung nutzen feste Seeds.
Die Hyperparameter-Studie (`exp.py`) hat keine festen Seeds; beim Neurechnen
ändern sich dort die Einzelwerte leicht.

---

## 6. Aufräumen

```powershell
cd notebooks
python archivieren.py               # zeigt nur an, was verschoben würde
python archivieren.py --ausfuehren  # verschiebt alte Dateien nach archiv/
```

Es wird nichts gelöscht. Die Liste steht in `archiv/INHALT.txt`.

---

## 7. Datensatz

GazeCapture – Krafka et al., *Eye Tracking for Everyone*, CVPR 2016,
<https://gazecapture.csail.mit.edu/>. Der Datensatz ist **nicht** Teil dieses Projekts.
