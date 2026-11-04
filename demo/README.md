---
title: Gaze Personalization Demo
emoji: 👀
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Webcam eye tracking with a 50-second personal calibration
---

# Webcam-Eye-Tracking mit Personalisierung

Demo zur Masterarbeit *Dateneffiziente Personalisierung für webcam-basiertes Eye-Tracking
mithilfe vortrainierter Bildverarbeitungsmodelle* (Khaled Rafei).

1. **Kalibrierung (ca. 60 s):** dem roten Punkt mit den Augen folgen.
2. **Anpassung:** Der Regressionskopf eines eingefrorenen ResNet-18 wird auf die Person trainiert.
3. **Blickverfolgung:** Lichtfleck = Blickpunkt, Größe = gemessene Unsicherheit;
   grauer Ring = Basismodell ohne Kalibrierung.
4. **Auswertung:** Heatmaps vorher/nachher, Zeitleiste der Bildschirmbereiche, Download als PNG/CSV.

**Datenschutz:** Kamerabilder werden nur im Arbeitsspeicher verarbeitet und nicht gespeichert.

**Lizenzen:**
- Code der Demo: MIT-Lizenz (Repository: https://github.com/RafeiKAr/gaze-estimation).
- Basismodell `models/base_norm_subject.pt`: auf GazeCapture trainiert. Es gelten die
  Bedingungen von GazeCapture (https://gazecapture.csail.mit.edu): **nur Forschung,
  nicht kommerziell**. Das Modell darf nicht aus diesem Space weitergegeben werden.
- Backbone: ResNet-18 mit ImageNet-Gewichten von torchvision (BSD-3-Clause; für die
  Gewichte gelten ggf. die Bedingungen von ImageNet).
- Gesichtserkennung: Haar-Kaskade aus OpenCV.
Für die beste Darstellung die direkte Adresse `https://<name>-<space>.hf.space` öffnen.
