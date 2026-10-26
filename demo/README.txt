==============================================================================
 README – Eye-Tracking-Demo (Kalibrierung, Blickverfolgung, Auswertung)
 Masterarbeit Khaled Rafei
 Ordner: F:\project\Venv\gaze-estimation\demo
==============================================================================

Inhalt
  1. Was macht die Demo?
  2. Die Dateien – was steht wo?
  3. Wie funktioniert alles? (technisch, einfach erklärt)
  4. Auf Rechner A ausführen (lokal)
  5. Selbsttest ohne Webcam
  6. Auf Hugging Face hochladen
  7. Die Demo benutzen
  8. Einstellungen ändern
  9. Fehler und Lösungen
 10. Datenschutz und Lizenz
 11. Für die Masterarbeit

(ANLEITUNG_DEMO.md enthält dieselben Schritte mit mehr Hintergrund und
Begründungen. Diese README.txt ist die kurze Arbeitsanleitung.)


------------------------------------------------------------------------------
 1. Was macht die Demo?
------------------------------------------------------------------------------

Eine Webseite mit vier Phasen:

  a) Kalibrierung (60 s): Ein roter Punkt bewegt sich als Schlangenlinie über
     den ganzen Bildschirm (erst Zeilen, dann Spalten). Die Person folgt ihm mit
     den Augen. Die Webcam nimmt 6 Bilder pro Sekunde auf; zu jedem Bild wird
     die Punktposition gespeichert (= Label).
  b) Anpassung: Der Server trainiert den Regressionskopf des Basismodells auf
     diesen Bildern (wenige Sekunden).
  c) Blickverfolgung: Ein Lichtfleck zeigt live den geschätzten Blickpunkt.
     Der Radius des Lichtflecks = gemessener Fehler nach der Kalibrierung.
     Ein grauer Ring zeigt die Schätzung des Basismodells OHNE Kalibrierung
     (Vorher-Nachher-Vergleich; Taste V schaltet ihn ein/aus).
  d) Auswertung: zwei Heatmaps (vorher / nachher), eine Zeitleiste, in welchem
     der neun Bildschirmbereiche wann geschaut wurde, und eine Tabelle mit dem
     Zeitanteil pro Bereich. Download als PNG (Bild) und CSV (Daten).


------------------------------------------------------------------------------
 2. Die Dateien – was steht wo?
------------------------------------------------------------------------------

  demo\
  ├── app.py            Server (FastAPI). Nimmt Bilder vom Browser an und ruft
  │                     gaze_model.py auf. Liefert auch die Webseite aus.
  ├── gaze_model.py     Das Modell: Basismodell laden, Bilder in Merkmale
  │                     umrechnen, Kopf trainieren (Kalibrierung), vorhersagen.
  ├── selftest.py       Test ohne Browser/Webcam mit einer eigenen Aufnahme
  │                     (z.B. Sitzung 02). Wird NICHT hochgeladen.
  ├── upload_space.py   Lädt den Ordner zu Hugging Face hoch.
  ├── requirements.txt  Python-Pakete (torch, torchvision, fastapi, uvicorn, ...)
  ├── Dockerfile        Bauanleitung für den Server auf Hugging Face.
  ├── README.md         Pflichtdatei für Hugging Face (Titel, Einstellungen).
  ├── README.txt        diese Datei
  ├── ANLEITUNG_DEMO.md ausführliche Anleitung mit Hintergrund
  ├── models\
  │   └── base_norm_subject.pt   Basismodell (selbst hineinkopieren, Schritt 4.1)
  └── static\           alles, was im Browser läuft
      ├── index.html    Aufbau der Seiten (Start, Ergebnis, Auswertung)
      ├── style.css     Farben und Layout
      └── app.js        Kamera, Punkt, Lichtfleck, Heatmap, Zeitleiste, Download


------------------------------------------------------------------------------
 3. Wie funktioniert alles? (einfach erklärt)
------------------------------------------------------------------------------

  Browser (app.js)                           Server (app.py + gaze_model.py)
  ----------------                           -------------------------------
  Start: /api/session  ------------------->  neue Sitzung (eigene ID)

  Kalibrierung, 6x pro Sekunde:
    Kamerabild (JPEG) + Punkt (x, y) + Zeit
                       /api/calib_frame -->  Bild -> ResNet-18 (eingefroren)
                                             -> 512 Merkmale speichern
                                             (das Bild selbst wird NICHT gespeichert)
  nach 60 s:           /api/calibrate  -->  Kopf trainieren:
                                             - erste 0,5 s verworfen (wie WAIT-Frames)
                                             - 10 zeitliche Blöcke (je ca. 5 s):
                                               8 Blöcke = Training, 2 Blöcke = Prüfung
                                               (Block 3 und 8), mit Lücke dazwischen ->
                                               keine fast gleichen Nachbarbilder,
                                               aber Prüfung an verschiedenen Stellen
                                             - AdamW, lr 1e-4, Batch 32, L1-Verlust,
                                               Early Stopping (Patience 10,
                                               siehe gaze_model.py)
                     <-- Fehler vorher / nachher / konstante Schätzung

  Verfolgung, so schnell wie möglich (eine Anfrage nach der anderen):
    Kamerabild         /api/predict   -->   Merkmale -> Basis-Kopf   = "vorher"
                                                     -> eigener Kopf = "nachher"
                     <-- zwei Punkte (x, y)
    Glättung + Zeichnen (Lichtfleck, Ring)

  Auswertung: nur im Browser (Heatmap, Zeitleiste, PNG, CSV).
  Beenden:             /api/end        -->  Sitzung löschen

Warum ist das genau das Verfahren der Masterarbeit?
  - gleiches Modell (ResNet-18 + Kopf 512-512-256-128-2, Sigmoid),
  - gleiches Basismodell (base_norm_subject.pt aus train_base.py, E0),
  - gleiche Bildvorverarbeitung (224x224, ImageNet-Normierung),
  - gleiches Training (nur Kopf, gleiche Hyperparameter, Merkmals-Cache),
  - gleiches Fehlermaß (E_diag in % der Bildschirmdiagonale),
  - Kamerabilder werden NICHT gespiegelt (wie bei den eigenen Aufnahmen);
    nur die kleine Vorschau auf der Startseite ist gespiegelt.

Was ist anders als in den Experimenten?
  - Keine Wiederholungen, kein festes Protokoll: Die Zahlen der Demo sind
    eine Vorführung, keine Ergebnisse für Kapitel 4.


------------------------------------------------------------------------------
 4. Auf Rechner A ausführen (lokal)
------------------------------------------------------------------------------

4.1 Basismodell kopieren (einmalig)

  cd F:\project\Venv\gaze-estimation
  mkdir demo\models
  copy models\base\base_norm_subject.pt demo\models\

4.2 Python-Umgebung anlegen (einmalig)

  cd F:\project\Venv\gaze-estimation\demo
  python -m venv .venv
  .venv\Scripts\activate
  pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

  (Du kannst auch deine bestehende Projekt-Umgebung benutzen, wenn dort torch
   und torchvision schon installiert sind; dann nur noch:
   pip install fastapi uvicorn)

4.3 Server starten

  .venv\Scripts\activate
  uvicorn app:app --port 7860

  Im Terminal erscheint: "Uvicorn running on http://127.0.0.1:7860".
  Das Fenster offen lassen (Beenden mit Strg + C).

4.4 Webseite öffnen

  Chrome oder Edge:  http://localhost:7860
  Kamera erlauben -> "Kalibrierung starten".

  Schneller Test mit kurzer Kalibrierung (nur zum Ausprobieren):
  http://localhost:7860/?calib=10       (10 statt 60 Sekunden)


------------------------------------------------------------------------------
 5. Selbsttest ohne Webcam
------------------------------------------------------------------------------

Prüft Modell und Training mit einer eigenen Aufnahme der Masterarbeit.

  cd F:\project\Venv\gaze-estimation\demo
  .venv\Scripts\activate
  python selftest.py                        (Sitzung 02, ganze Aufnahme)
  python selftest.py --session 04           (andere Sitzung)
  python selftest.py --seconds 50           (nur die ersten 50 Sekunden)

Erwartete Ausgabe (ungefähr):
  Vorher (Basismodell):  ca. 27-31 %
  Nachher (personal.):   deutlich kleiner (bei guten Sitzungen ca. 14-20 %)
  "Selbsttest erfolgreich."

Wenn das klappt, funktioniert der Python-Teil; Probleme liegen dann nur noch
an Browser, Kamera oder Licht.


------------------------------------------------------------------------------
 6. Auf Hugging Face hochladen
------------------------------------------------------------------------------

6.1 Konto und Token (einmalig)
  - Konto: https://huggingface.co/join
  - Token: Profilbild -> Settings -> Access Tokens -> Create new token,
    Typ "Write". Token geheim halten (nie in Code oder Git).

6.2 Anmelden (einmalig, auf Rechner A)
  .venv\Scripts\activate
  pip install -U huggingface_hub
  hf auth login                 (ältere Versionen: huggingface-cli login)
  -> Token einfügen, Enter.

6.3 Namen eintragen
  upload_space.py öffnen und diese Zeile anpassen:
    REPO = "DEIN-NAME/gaze-personalization-demo"
  z.B. REPO = "khaledrafei/gaze-personalization-demo"

6.4 Hochladen
  cd F:\project\Venv\gaze-estimation\demo
  python upload_space.py

  Das Skript legt den Space automatisch an (falls er noch nicht existiert):
  SDK Docker, PRIVAT. Danach lädt es alle Dateien hoch (das Modell mit ca.
  45 MB dauert etwas). selftest.py, README.txt und ANLEITUNG_DEMO.md werden
  nicht hochgeladen.

6.5 Bauvorgang beobachten
  https://huggingface.co/spaces/DEIN-NAME/gaze-personalization-demo
  Oben steht zuerst "Building" (beim ersten Mal 3-8 Minuten), dann "Running".
  Bei "Build error" oder "Runtime error": Reiter "Logs" öffnen.

6.6 Änderungen hochladen
  Datei lokal ändern -> lokal testen (4.3/4.4) -> python upload_space.py.
  Der Space baut sich danach automatisch neu.

6.7 Hardware
  Standard ist "CPU basic" (kostenlos). Nach ca. 48 Stunden ohne Nutzung
  schläft der Space; beim nächsten Aufruf wacht er in ca. 1 Minute auf.


------------------------------------------------------------------------------
 7. Die Demo benutzen
------------------------------------------------------------------------------

Am besten die DIREKTE Adresse öffnen (dort funktionieren Kamera und Vollbild
sicher):
  https://DEIN-NAME-gaze-personalization-demo.hf.space
(Im Space-Menü "⋮" -> "Embed this Space" steht die genaue Adresse.)

Ein privater Space ist nur sichtbar, wenn du bei Hugging Face angemeldet bist.

Ablauf:
  1. Kamera erlauben -> "Kalibrierung starten (Vollbild)".
  2. 3 Sekunden Countdown, dann 60 Sekunden dem Punkt mit den Augen folgen.
  3. Kurzer Test (12 s): 5 feste Punkte nacheinander ansehen. Daraus entstehen
     ein ehrlicher Testfehler (Bilder nicht im Training) und eine Offset-
     Korrektur (systematischer Versatz), die bei der Verfolgung abgezogen wird –
     nur wenn sie im Kreuzvalidierungs-Test wirklich hilft.
     Ergebnis: drei Balken (konstante Schätzung, vorher, nachher) und der
     Testfehler mit/ohne Offset-Korrektur.
     Wenn "nachher" nicht besser ist: Licht verbessern und neu kalibrieren.
  4. "Blickverfolgung starten": Der Bildschirmbereich (Viereck, 3 x 3 Raster),
     in dem der Blick liegt, wird hervorgehoben, dazu ein Lichtfleck;
     grauer Ring = vorher (Basismodell).
     Taste M: Lichtart wechseln:
       Bühnenlicht   – das Feld (3 x 3), in dem der Blick liegt, wird weich hell
       Scheinwerfer  – ein weiches Rechteck-Licht folgt dem Blick
       Lichtpunkt    – runder Lichtfleck (Größe = Unsicherheit)
     Taste V: Ring ein/aus. Taste Esc: Beenden.
  5. Auswertung: Heatmaps, Zeitleiste der Bereiche, Verlauf von x und y über die
     Zeit, Zeittabelle (für jeden Abschnitt von 2 s: Bereich und ungefähre
     Position), Anteil pro Bereich; PNG, alle Messpunkte (CSV), Zeittabelle (CSV).
     "Weiter verfolgen" setzt die Aufzeichnung fort, "Neu kalibrieren"
     beginnt von vorne.

Für gute Ergebnisse: helles Licht von vorne, ruhiger Hintergrund, Kopf und
Laptop nicht bewegen, Brille wenn möglich wie im Alltag (immer gleich).
Erwartung: Die Genauigkeit reicht für "welcher Bildschirmbereich", nicht für
einzelne Wörter oder Knöpfe. Der Lichtfleck ist deshalb bewusst groß.


------------------------------------------------------------------------------
 8. Einstellungen ändern
------------------------------------------------------------------------------

Browser (static\app.js, ganz oben in CONFIG):
  calibSeconds    Dauer der Kalibrierung (Standard 60; länger = langsamer)
  labelsPerSecond Bilder pro Sekunde in der Kalibrierung (6)
  rows            Zeilen/Spalten der Schlangenlinie (4; weniger = langsamer)

  Geschwindigkeit des Punktes = Länge der Bahn / calibSeconds.
  Ungefähr auf einem Full-HD-Bildschirm (1920 x 1080):
    rows 5, 50 s : ca. 325 Pixel/s   (alte Einstellung, schnell)
    rows 4, 60 s : ca. 255 Pixel/s   (Standard)
    rows 3, 60 s : ca. 180 Pixel/s   (langsam)
  Ohne die Datei zu ändern: http://localhost:7860/?rows=3  (oder ?rows=5, ?calib=75)
  margin          Abstand des Punktes zum Rand (0.05 = 5 %)
  smoothing       Glättung des Lichtflecks (0.35; kleiner = ruhiger, träger)
  minGlowPx       kleinster Radius des Lichtflecks in Pixeln

Server (gaze_model.py, oben):
  LR, WEIGHT_DECAY, BATCH, PATIENCE, MAX_EPOCHS   Training
  SKIP_START_SEC   verworfene Zeit am Anfang (0.5 s)
  N_BLOCKS, VAL_EVERY  Blöcke für die Prüfung (10 Blöcke, jeder 5. = Prüfung)
  MODEL_FILE       Pfad zum Basismodell

Anderes Basismodell (z.B. norm_random):
  Datei nach demo\models\ kopieren und MODEL_FILE in gaze_model.py ändern.


------------------------------------------------------------------------------
 9. Fehler und Lösungen
------------------------------------------------------------------------------

"Kamera nicht verfügbar"
  -> Zugriff im Browser erlauben (Schloss-Symbol links neben der Adresse),
     Teams/Zoom/Kamera-App schließen, Seite neu laden.
"Server nicht erreichbar" (lokal)
  -> läuft uvicorn noch? Richtiger Port (7860)?
"Modelldatei fehlt: ...base_norm_subject.pt"
  -> Schritt 4.1 vergessen, oder Datei beim Hochladen nicht dabei.
"Zu wenige Kalibrierbilder"
  -> Verbindung zu langsam oder Kalibrierung abgebrochen; neu kalibrieren.
Vollbild startet nicht
  -> direkte .hf.space-Adresse verwenden (nicht die eingebettete Seite).
Lichtfleck bleibt in der Mitte, "vorher" und "nachher" fast gleich
  -> häufigste Ursache: eingefrorenes Kamerabild (der Browser hatte das Video
     angehalten, alle Bilder waren gleich). Ist ab Version 26.10. behoben;
     die Demo warnt jetzt ("Kamera liefert keine neuen Bilder"), und der
     Server bricht die Kalibrierung mit einer Meldung ab.
  -> sonst: Kalibrierung hat nicht gewirkt ("beste Epoche 0"); Licht verbessern,
     Kopf ruhig halten, neu kalibrieren.
Sehr langsam online
  -> kostenlose CPU wird geteilt; nur eine Person gleichzeitig benutzen.
"Sitzung unbekannt oder abgelaufen"
  -> nach 30 Minuten ohne Nutzung gelöscht; Seite neu laden.
Build error auf Hugging Face
  -> Reiter "Logs": meist ein Paket in requirements.txt.
pip-Fehler beim Installieren von torch auf Windows
  -> Python 3.10-3.12 verwenden; Befehl aus 4.2 genau so ausführen.


Drei Verfahren für die Personalisierung
  (1) MLP-Kopf wie in der Masterarbeit (512 globale Merkmale),
  (2) Ridge-Regression auf denselben 512 Merkmalen,
  (3) Ridge-Regression auf RÄUMLICHEN Merkmalen (layer3-Karte auf ein Raster von
      3 x 8 Feldern gemittelt). Die globalen Merkmale mitteln über das ganze Bild
      und verlieren, WO die Iris liegt; die räumlichen behalten das. Das hilft vor
      allem bei der Unterscheidung links/rechts.
  Verwendet wird das Verfahren mit dem kleinsten Prüffehler; alle Werte stehen in
  der Diagnose-Zeile auf der Ergebnisseite und im Selbsttest.

Bildausschnitt (seit 26.10.)
  Bei der Webcam sitzt man weiter weg als bei den eigenen Aufnahmen, und die Augen
  sind im ganzen Bild nur wenige Pixel groß. Deshalb benutzt das personalisierte
  Modell einen FESTEN Ausschnitt, der beim ersten Bild mit Gesicht bestimmt wird:
      eyes (Standard)  Streifen um beide Augen, ca. 3-fach vergrößert
      face             Quadrat um das Gesicht
      none             ganzes Bild (wie in der Masterarbeit)
  Das Basismodell ("vorher") bekommt immer das ganze Bild.
  Einstellen beim Start:   $env:GAZE_CROP="face"; uvicorn app:app --port 7860
  Braucht opencv-python-headless Version 4:  pip install "opencv-python-headless<5"
  Beim Start steht im Terminal "Bildausschnitt: eyes".
  Selbsttest mit Ausschnitt:  python selftest.py --crop eyes   (oder face / none)

Fehlersuche bei schlechter Kalibrierung ("nachher" nicht besser als "vorher")
  1. Startseite: Unter dem Vorschaubild die richtige Kamera wählen (Frontkamera).
     Im Vorschaubild muss das Gesicht gut zu sehen sein.
  2. Ergebnisseite, Zeile "Diagnose":
       Trainingsfehler X % -> Y %   Y viel kleiner als X = das Modell lernt;
                                    Y fast gleich X = Bilder enthalten keine
                                    brauchbare Information (Kamera, Licht)
       Bildänderung                 sehr klein (< 0.01) = Bilder fast gleich
       Kamera                       Auflösung und Name der benutzten Kamera
  3. Kalibrierbilder ansehen (nur lokal!): Server so starten
       PowerShell:  $env:GAZE_DEBUG="1"; uvicorn app:app --port 7860
     Dann wird jedes 30. Kalibrierbild in demo\debug_frames\ gespeichert
     (Dateiname mit Zeit und Punktposition). Danach den Ordner löschen und
     ohne GAZE_DEBUG starten. Den Ordner nie hochladen.


------------------------------------------------------------------------------
 10. Datenschutz und Lizenz
------------------------------------------------------------------------------

Datenschutz
  - Bilder werden nur im Arbeitsspeicher in Merkmale umgerechnet und nicht
    gespeichert; Sitzungen werden beim Beenden bzw. nach 30 Minuten gelöscht.
  - Die Auswertung (Heatmap, CSV) entsteht nur im Browser.
  - Keine Personen ohne deren Einverständnis aufnehmen.

Lizenz
  - Das Basismodell wurde auf GazeCapture trainiert (nur Forschung, nicht
    kommerziell). Deshalb legt upload_space.py den Space PRIVAT an.
  - Vor dem Umstellen auf "Public" (Space -> Settings -> Change visibility)
    die Nutzungsbedingungen von GazeCapture lesen und mit dem Betreuer klären.


------------------------------------------------------------------------------
 11. Für die Masterarbeit
------------------------------------------------------------------------------

  - Passt in Kapitel 6 (Ausblick) oder in den Anhang: je ein Screenshot pro
    Phase, Fehler vorher/nachher, ein Satz zu den Grenzen.
  - Die Demo-Zahlen sind KEINE Ergebnisse für Kapitel 4 (kein festes
    Protokoll, keine Wiederholungen).
  - Den Code habe ich (Claude) geschrieben; bei der Angabe der KI-Hilfsmittel
    in der Erklärung mit aufführen.
