---
license: other
license_name: research-only
tags:
- gaze-estimation
- eye-tracking
- resnet-18
---

# Basismodell für die Webcam-Eye-Tracking-Demo

Basismodell der Masterarbeit *Dateneffiziente Personalisierung für webcam-basiertes
Eye-Tracking mithilfe vortrainierter Bildverarbeitungsmodelle* (Khaled Rafei,
BHT Berlin, 2026). Code und Demo: https://github.com/RafeiKAr/gaze-estimation

- **Datei:** `base_norm_subject.pt` (PyTorch `state_dict`)
- **Architektur:** ResNet-18 (ImageNet-Gewichte von torchvision, eingefroren) und ein
  Regression Head 512 → 512 → 256 → 128 → 2 mit Sigmoid
- **Training:** nur der Regression Head, auf einer Teilmenge von GazeCapture
  (personendisjunkter Split), L1-Loss, AdamW, Learning Rate 1e-4
- **Ausgabe:** Blickpunkt (x, y) im normierten Bildschirm [0, 1]²
- **Hinweis:** Allein ist das Modell kaum besser als eine Schätzung der Bildschirmmitte.
  Es ist der Startpunkt für die Personalisierung in der Demo.

## Lizenz und Nutzung

- Trainiert auf **GazeCapture** (Krafka et al., CVPR 2016, https://gazecapture.csail.mit.edu).
  Es gelten dessen Bedingungen: **nur für Forschung, nicht kommerziell**.
- Die ImageNet-Gewichte stammen von torchvision (BSD-3-Clause); für die Gewichte können die
  Bedingungen von ImageNet gelten.
- Bei Nutzung bitte die Masterarbeit und GazeCapture zitieren.
