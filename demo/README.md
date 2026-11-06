---
title: Gaze Personalization Demo
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Webcam eye tracking with a 60-second personal calibration
---

# Webcam-Eye-Tracking mit Personalisierung – Demo

Demo zur Masterarbeit *Dateneffiziente Personalisierung für webcam-basiertes Eye-Tracking
mithilfe vortrainierter Bildverarbeitungsmodelle* (Khaled Rafei, BHT Berlin).
Ein eingefrorenes ResNet-18 liefert Bild-Features; nur ein kleiner Regression Head wird
in wenigen Sekunden an die Person angepasst.

## Ablauf

1. **Kalibrierung (ca. 60 s):** Dem roten Punkt mit den Augen folgen. Er startet an einer
   zufälligen Stelle, fliegt mit konstanter Geschwindigkeit in eine zufällige Richtung und
   prallt an den Rändern ab. Der Weg besucht immer alle neun Felder des Bildschirms.
2. **Anpassung:** Das Modell wird auf die Person trainiert (einige Sekunden).
3. **Kurzer Test:** fünf feste Punkte, je 3 s anschauen. Die Demo zeigt den Fehler.
4. **Blickverfolgung:** Der Lichtfleck zeigt den Blickpunkt. Die Position wird über die
   letzten 0,5 s gemittelt (Moving Average), damit sie ruhig bleibt.
5. **Auswertung:** wann welches Feld angeschaut wurde; Download als PNG oder CSV.

## Starten (eigener Rechner)

Voraussetzung: Python 3.10 oder neuer, Webcam, Chrome oder Edge.

```bash
cd demo
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uvicorn app:app --port 7860
```

Dann im Browser öffnen: **http://localhost:7860**

**Basismodell:** Liegt `models/base_norm_subject.pt` nicht vor, lädt die Demo die Datei
beim ersten Start von Hugging Face (https://huggingface.co/RafeiKAr/demo_model; änderbar mit
der Umgebungsvariable `GAZE_MODEL_REPO`) und speichert sie in `models/`.

**Mit Docker:** `docker build -t gaze-demo .` und `docker run -p 7860:7860 gaze-demo`

## Einstellungen in der Adresse

| Adresse | Wirkung |
|---|---|
| `?calib=20` | Kalibrierung nur 20 s (zum Ausprobieren) |
| `?ma=0` | Moving Average aus (Rohwerte) |
| `?ma=1` | Moving Average über 1 s |
| `?grid=7x5` | feineres Raster für das Lichtfeld |

## Datenschutz

Kamerabilder werden nur im Arbeitsspeicher verarbeitet und nicht gespeichert.
Nur wer den Server selbst mit `GAZE_DEBUG=1` startet, speichert einzelne Bildausschnitte
lokal in `debug_frames/` (zur Fehlersuche; nicht hochladen).

## Lizenzen

- Code der Demo: MIT-Lizenz (https://github.com/RafeiKAr/gaze-estimation).
- Basismodell: auf GazeCapture trainiert (https://gazecapture.csail.mit.edu) –
  **nur für Forschung, nicht kommerziell**.
- ResNet-18 mit ImageNet-Gewichten von torchvision (BSD-3-Clause; für die Gewichte
  gelten ggf. die Bedingungen von ImageNet). Gesichtserkennung: Haar-Kaskade aus OpenCV.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Server (FastAPI) |
| `gaze_model.py` | Modell laden, Features berechnen, Head personalisieren |
| `static/` | Oberfläche im Browser (Kalibrierung, Moving Average, Auswertung) |
| `upload_model.py` | lädt nur das Basismodell nach Hugging Face |
| `MODEL_CARD.md` | Beschreibung des Modells für Hugging Face |
| `selftest.py` | kurzer Test des Servers ohne Webcam |
