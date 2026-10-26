# Anleitung: Webbasierte Eye-Tracking-Demo auf Hugging Face

Diese Anleitung beschreibt Schritt für Schritt, wie die Demo aufgebaut ist, wie du sie
auf deinem Rechner A (Windows) testest und wie du sie als **Hugging Face Space**
veröffentlichst.

**Den Programmcode (`app.py`, `gaze_model.py`, `static/…`) schreibe ich dir im nächsten
Schritt.** Diese Anleitung erklärt, was jede Datei macht und was du selbst tun musst.

---

## 0. Überblick: Wie funktioniert die Demo?

```
 Browser (deine Webcam, dein Bildschirm)            Server (Hugging Face Space, CPU)
 ───────────────────────────────────────            ────────────────────────────────
 1. Kalibrierung: Punkt bewegt sich 60 s   ──Bilder + Punktposition──►  ResNet-18 (eingefroren)
    über den ganzen Bildschirm                                          berechnet Merkmale,
                                                                        trainiert nur den Kopf
                                           ◄──Fehler vorher/nachher──  (wie in personal_exp.py)
 2. Blickverfolgung: alle ~150 ms ein Bild ──Bild──►                    Vorhersage mit
                                           ◄──(x, y) vorher + nachher──  Basismodell und
    Lichtmarkierung; Größe = Genauigkeit                                 personalisiertem Modell
 3. Vergleich: grauer Punkt = vorher,
    farbiger Lichtfleck = nachher
 4. Auswertung: Heatmap + Zeitleiste
    (welcher Bereich wann), Download PNG/CSV
```

Warum diese Aufteilung?
- Der **Server** verwendet denselben PyTorch-Code wie deine Experimente
  (Basismodell `base_norm_subject.pt`, eingefrorener Backbone, Training nur des Kopfes).
  Du musst also nichts Neues lernen, und die Demo passt zu Kapitel 4 deiner Arbeit.
- Der **Browser** braucht für Kamera, Vollbild und Grafik JavaScript. Das ist die
  einzige Stelle, an der kein Python verwendet wird.

Entscheidungen, die direkt aus deinen Ergebnissen folgen:

| Entscheidung | Begründung (Kapitel 4) |
|---|---|
| Basismodell `norm_subject` | gleich gut wie `norm_random`, aber ohne Information über Testpersonen |
| Kalibrierung mit **kontinuierlich** bewegtem Punkt | Standardvideo (Gen 1) war die beste Sitzungsgestaltung |
| Erste 0,5 s nach dem Start werden verworfen | WAIT-Frames haben falsche Labels (E1b) |
| 6 Bilder pro Sekunde während der Kalibrierung | gleiche Labelrate wie deine Aufnahmen |
| Genauigkeit = Fehler auf **2 von 10 zeitlichen Blöcken** der Kalibrierung (mit Lücke zum Training) | ehrlich wie der sequenzielle Split (keine Nachbarbilder), aber über den Bildschirm verteilt |

**Erwartung realistisch halten:** Das Basismodell allein sagt fast immer die
Bildschirmmitte vorher (Fehler ≈ Konstant-Baseline). Nach der Kalibrierung liegt der
Fehler je nach Licht und Hintergrund grob bei 15–28 % der Bildschirmdiagonale. Die
Lichtmarkierung ist deshalb **groß**: Sie zeigt einen Bereich, keinen exakten Punkt.
Genau das soll die Demo ehrlich zeigen.

---

## 1. Ordnerstruktur der Demo

```
gaze-estimation\demo\
├── app.py               Server (FastAPI): nimmt Bilder an, kalibriert, sagt vorher
├── gaze_model.py        Modell laden, Merkmale berechnen, Kopf trainieren
├── models\
│   └── base_norm_subject.pt   Basismodell (aus models\base\ kopieren, ca. 45 MB)
├── static\
│   ├── index.html       Oberfläche: Start, Kalibrierung, Verfolgung, Auswertung
│   ├── app.js           Kamera, bewegter Punkt, Lichtmarkierung, Heatmap
│   └── style.css        Aussehen
├── requirements.txt     Python-Pakete
├── Dockerfile           Bauanleitung für den Hugging-Face-Server
├── README.md            Pflicht für Hugging Face (Kopfzeilen mit Einstellungen)
└── upload_space.py      lädt den Ordner zu Hugging Face hoch
```

Die Bildvorverarbeitung in `gaze_model.py` muss **genau dieselbe** sein wie beim
Training (Größe, Normierung aus `dataset.py`). Sonst passen die Merkmale nicht zum
Modell.

---

## 2. Vorbereitung auf Rechner A (Windows)

**Schritt 2.1 – Basismodell kopieren**
```bat
cd F:\project\Venv\gaze-estimation
mkdir demo\models
copy models\base\base_norm_subject.pt demo\models\
```

**Schritt 2.2 – Python-Umgebung für die Demo**
```bat
cd demo
python -m venv .venv
.venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```
(Die CPU-Version von PyTorch ist kleiner und reicht für die Demo.)

**Schritt 2.3 – Lokal starten**
```bat
uvicorn app:app --port 7860
```
Dann im Browser (Chrome oder Edge) öffnen: **http://localhost:7860**

**Schritt 2.4 – Lokal testen (Checkliste)**
1. Der Browser fragt nach der Kamera → **Zulassen**.
2. „Kalibrierung starten“ → Vollbild → dem Punkt ca. 60 Sekunden mit den Augen folgen,
   Kopf möglichst ruhig.
3. Danach zeigt die Seite: Fehler **vorher** und **nachher** (in % der Diagonale).
4. Blickverfolgung: Der Lichtfleck folgt deinem Blick; der graue Punkt (Basismodell)
   bleibt meist in der Mitte.
5. „Beenden“ → Heatmap und Zeitleiste erscheinen → PNG/CSV herunterladen.
6. Im Terminal dürfen keine roten Fehlermeldungen erscheinen.

Tipp für gute Ergebnisse: helles, gleichmäßiges Licht von vorne, ruhiger Hintergrund,
Laptop nicht bewegen (das waren auch in deinen Sitzungen die wichtigsten Faktoren).

---

## 3. Hugging-Face-Konto vorbereiten (einmalig)

**Schritt 3.1 – Konto anlegen:** https://huggingface.co/join

**Schritt 3.2 – Zugriffstoken erstellen**
1. Profilbild → **Settings** → **Access Tokens** → **Create new token**.
2. Typ **Write**, Name z. B. `gaze-demo`.
3. Token kopieren und **geheim halten** (nicht in Code, nicht in Git, nicht in die Arbeit).

**Schritt 3.3 – Anmelden auf Rechner A**
```bat
pip install -U huggingface_hub
hf auth login
```
(Bei älteren Versionen heißt der Befehl `huggingface-cli login`.) Token einfügen,
Enter.

---

## 4. Space anlegen

1. https://huggingface.co/new-space öffnen.
2. **Space name:** z. B. `gaze-personalization-demo`.
3. **License:** passend wählen (siehe Abschnitt 8).
4. **SDK:** **Docker** → Vorlage **Blank**.
5. **Hardware:** **CPU basic (free)**.
6. **Visibility:** zuerst **Private** (siehe Abschnitt 8).
7. **Create Space**.

Die Adresse lautet danach `https://huggingface.co/spaces/<dein-name>/gaze-personalization-demo`.

---

## 5. Die Hugging-Face-Dateien

**`README.md`** – die Kopfzeilen zwischen `---` sind Pflicht:
```yaml
---
title: Gaze Personalization Demo
emoji: 👀
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
Webcam-basiertes Eye-Tracking mit Personalisierung (Masterarbeit Khaled Rafei).
Kamerabilder werden nur im Arbeitsspeicher verarbeitet und nicht gespeichert.
```

**`Dockerfile`** – so baut Hugging Face den Server:
```dockerfile
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY --chown=user . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**`upload_space.py`** – lädt alles hoch (große Dateien wie das Modell werden
automatisch richtig behandelt, du brauchst kein Git):
```python
from huggingface_hub import HfApi

REPO = "<dein-name>/gaze-personalization-demo"   # anpassen
HfApi().upload_folder(
    folder_path=".",            # der Ordner demo\
    repo_id=REPO,
    repo_type="space",
    ignore_patterns=[".venv/*", "__pycache__/*", "*.pyc"],
)
print("Hochgeladen:", f"https://huggingface.co/spaces/{REPO}")
```

---

## 6. Hochladen und starten

**Schritt 6.1 – Hochladen**
```bat
cd F:\project\Venv\gaze-estimation\demo
.venv\Scripts\activate
python upload_space.py
```

**Schritt 6.2 – Bauvorgang beobachten**
Auf der Space-Seite steht oben zuerst **Building**, dann **Running**
(beim ersten Mal 3–8 Minuten). Bei **Build error** oder **Runtime error**:
Reiter **Logs** öffnen und die letzte rote Meldung lesen.

**Schritt 6.3 – Demo benutzen**
Die Space-Seite zeigt die Demo in einem Rahmen. Für Kamera und Vollbild ist die
**direkte Adresse** besser:
```
https://<dein-name>-gaze-personalization-demo.hf.space
```
(Steht auch im Menü „⋮“ → **Embed this Space** → Direct URL.)

**Schritt 6.4 – Änderungen**
Datei lokal ändern → lokal testen (Abschnitt 2.3) → `python upload_space.py` erneut
ausführen. Der Space baut sich automatisch neu.

---

## 7. Häufige Probleme

| Problem | Ursache / Lösung |
|---|---|
| Kamera startet nicht | Kamera im Browser erlauben; anderes Programm (Teams, Zoom) schließen; direkte `.hf.space`-Adresse verwenden |
| Vollbild geht nicht | direkte `.hf.space`-Adresse verwenden (im eingebetteten Rahmen ist Vollbild oft blockiert) |
| Lichtfleck bleibt in der Mitte | Kalibrierung wiederholen, besseres Licht, Kopf ruhig halten; der Fehler „nachher“ zeigt, ob die Kalibrierung gewirkt hat |
| Sehr langsam | kostenlose CPU; Bildrate in `app.js` senken. Nach längerer Pause „schläft“ der Space und braucht ca. 1 Minute zum Aufwachen |
| `Build error` bei `pip install` | Paketname/Version in `requirements.txt` prüfen |
| `Runtime error` beim Start | meist Modelldatei nicht gefunden: liegt sie in `demo\models\` und wurde sie hochgeladen? |
| Ergebnis lokal gut, online schlechter | anderer Rechner/Bildschirm = andere Kamerageometrie; immer am selben Gerät kalibrieren und verfolgen |

---

## 8. Wichtig: Lizenz und Datenschutz

**GazeCapture-Lizenz.** GazeCapture darf nur für Forschungszwecke und nicht kommerziell
genutzt werden. Dein Basismodell wurde damit trainiert. Ob du das Modell **öffentlich**
anbieten darfst, ist nicht eindeutig. Deshalb:
- Space zuerst **privat** anlegen (nur du siehst ihn; du kannst ihn für die
  Präsentation trotzdem selbst zeigen).
- Vor dem Umschalten auf **Public** die Nutzungsbedingungen von GazeCapture lesen
  und mit deinem Betreuer klären.

**Datenschutz.** Webcam-Bilder sind personenbezogene Daten.
- Der Server verarbeitet die Bilder nur im Arbeitsspeicher und speichert sie nicht.
  Das steht auch auf der Startseite der Demo.
- Die personalisierten Gewichte gehören zu einer Sitzung und werden beim Beenden
  gelöscht.
- Keine Bilder anderer Personen ohne deren Einverständnis aufnehmen.

---

## 9. Für die Masterarbeit

- Die Demo passt in **Kapitel 6 (Fazit und Ausblick)** oder in den **Anhang**:
  ein Screenshot jeder Phase, die gemessenen Fehler vorher/nachher und ein Satz zu
  den Grenzen (CPU, Licht, große Unsicherheit).
- Die Zahlen der Demo sind **keine** Ergebnisse für Kapitel 4: Sie entstehen ohne
  festes Protokoll und ohne Wiederholungen.

---

## 10. Reihenfolge auf einen Blick

| # | Was | Wo |
|---|---|---|
| 1 | Code von Claude erhalten und in `demo\` speichern | Rechner A |
| 2 | Modell kopieren, Umgebung einrichten (2.1–2.2) | Rechner A |
| 3 | Lokal starten und alle Phasen testen (2.3–2.4) | Browser |
| 4 | Hugging-Face-Konto, Token, Anmeldung (3) | huggingface.co / Rechner A |
| 5 | Space anlegen: Docker, CPU basic, **Private** (4) | huggingface.co |
| 6 | `upload_space.py` ausführen (6.1) | Rechner A |
| 7 | Logs prüfen, Demo über die direkte Adresse öffnen (6.2–6.3) | Browser |
| 8 | Lizenz klären, erst dann ggf. **Public** (8) | Betreuer |
