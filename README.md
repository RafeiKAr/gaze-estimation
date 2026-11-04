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
├── moving_average.py      E4: zeitliche Glättung der gespeicherten Vorhersagen
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
│   ├── moving_average/    Ergebnisse der Glättung (E4)
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
| Glättung (E4) | `python moving_average.py` (kein Training, nutzt `results/personal/predictions/`) |
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
| dasselbe mit Moving Average, 0,5 s (E4) | 28,10 % (Springen −40 %) | 31,84 % |
| Sessions 21/22: zufällig / sequenziell / zwei Videos (E2) | 13,85 / 15,55 / 17,24 % | 30,97 / 28,67 / 32,65 % |
| Gen 4, MOVE-Testbilder: Original / bereinigt (E1b) | 16,72 / 15,88 % | 30,51 % |
| Beste Session (S02) | 10,48 % | 28,49 % |

**Reproduzierbarkeit:** Basismodelle und Personalisierung nutzen feste Seeds.
Die Hyperparameter-Studie (`exp.py`) hat keine festen Seeds; beim Neurechnen
ändern sich dort die Einzelwerte leicht.

---


## 6. Datensatz

GazeCapture – Krafka et al., *Eye Tracking for Everyone*, CVPR 2016,
<https://gazecapture.csail.mit.edu/>. Der Datensatz ist **nicht** Teil dieses Projekts.

Verwendet wurde eine Teilmenge von Kaggle
(<https://www.kaggle.com/datasets/dhruv413/gaze-capture-20gb-zip>), die `main.py`
herunterlädt. Wer die Experimente wiederholen möchte, muss die Daten selbst besorgen
und die Nutzungsbedingungen von GazeCapture akzeptieren (siehe Abschnitt 7).

---

## 7. Lizenzen und Nutzungsbedingungen

### 7.1 Eigener Code

Der Code in diesem Repository steht unter der **MIT-Lizenz** (siehe Datei `LICENSE`).
Das gilt nur für den eigenen Code, **nicht** für Daten oder Modelle Dritter.

### 7.2 GazeCapture (Daten)

- Die Daten, Modelle und der Code von GazeCapture dürfen laut den Autoren **nur für
  Forschungszwecke** verwendet werden (Lizenz: `LICENSE.md` im Repository
  <https://github.com/CSAILVision/GazeCapture>).
- Dieses Repository enthält **keine** GazeCapture-Daten: keine Bilder, keine
  `labels.csv`, keine Split-Dateien und keinen Merkmals-Cache (siehe `.gitignore`).
- Bei Nutzung bitte zitieren: K. Krafka et al., *Eye Tracking for Everyone*, CVPR 2016.

### 7.3 Trainierte Modelle

- In diesem Repository liegen **keine** trainierten Modelle (`*.pt` stehen in `.gitignore`).
- Das Basismodell liegt in einem eigenen Modell-Repository auf Hugging Face; die Demo lädt es
  beim ersten Start automatisch (siehe `demo/README.md`).
- Das Basismodell (`base_norm_subject.pt`) wurde auf GazeCapture trainiert. Es gelten
  daher ebenfalls die Bedingungen von GazeCapture: **nur Forschung, nicht kommerziell**.
- Der Backbone nutzt die ImageNet-Gewichte von torchvision
  (`ResNet18_Weights.DEFAULT`). torchvision selbst steht unter der BSD-3-Clause-Lizenz;
  für die vortrainierten Gewichte können eigene Bedingungen gelten (ImageNet:
  nur nicht kommerzielle Forschung).

### 7.4 Eigene Aufnahmen (Personalisierungsdaten)

- Die Webcam-Bilder und Videos der eigenen Sitzungen zeigen das Gesicht des Autors
  und sind personenbezogene Daten. Sie sind **nicht** in diesem Repository
  (`personalization/*/images/` und Videos stehen in `.gitignore`).
- Sie liegen in einem **privaten** Hugging-Face-Dataset und werden nicht öffentlich
  weitergegeben. Alle Rechte liegen beim Autor.

### 7.5 Web-Demo

- Der Code der Demo (`demo/`) steht unter der MIT-Lizenz.
- Die Demo verarbeitet Kamerabilder nur im Arbeitsspeicher und speichert sie nicht.
- Das Gesichtsmodell (`haarcascade_frontalface_default.xml`) stammt aus OpenCV und wird
  mit dem Paket `opencv-python-headless` installiert (Lizenz siehe OpenCV).

### 7.6 Verwendete Bibliotheken

Die Pakete werden nicht mitgeliefert, sondern mit `pip` installiert. Es gelten ihre
eigenen Lizenzen:

| Paket | Lizenz |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| NumPy, pandas, SciPy, scikit-learn | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache 2.0 |
| Pillow | MIT-CMU (HPND) |
| matplotlib | Matplotlib License (PSF-basiert) |
| TensorBoard, kaggle | Apache 2.0 |
| tqdm | MIT / MPL-2.0 |
| pygame | LGPL 2.1 |
| FastAPI | MIT |
| uvicorn | BSD-3-Clause |
| JupyterLab, ipykernel | BSD-3-Clause |
