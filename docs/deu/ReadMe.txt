==============================================================================
 ReadMe – Masterarbeit (deutsche Fassung)
 Khaled Rafei · Dateneffiziente Personalisierung für webcam-basiertes
 Eye-Tracking mithilfe vortrainierter Bildverarbeitungsmodelle
==============================================================================

Inhalt dieser Datei
  1. Was brauche ich?
  2. Wie ist der Ordner aufgebaut?
  3. Wie erstelle ich die PDF-Dateien?
  4. Wie ändere ich etwas?
  5. Was tun bei Fehlern?
  6. Stand der Arbeit


------------------------------------------------------------------------------
 1. Was brauche ich?
------------------------------------------------------------------------------

  - MiKTeX        (enthält pdflatex und bibtex)
  - Python 3      (keine zusätzlichen Pakete nötig)

Test in der PowerShell:

  pdflatex --version
  python --version

Beide Befehle müssen eine Versionsnummer zeigen.
Beim ersten Bauen lädt MiKTeX fehlende Pakete nach. Fragt ein Fenster nach,
mit "Install" bestätigen.


------------------------------------------------------------------------------
 2. Wie ist der Ordner aufgebaut?
------------------------------------------------------------------------------

  F:\claude\@output\deutsch\
  │
  ├── masterarbeit.tex     Hauptdatei: die GANZE Arbeit
  ├── Abgabe.tex           Hauptdatei: nur Kapitel 2 und 4 (Auszug)
  ├── preamble.tex         Einstellungen, Angaben (Titel, Name, Prüfer, Datum)
  │                        und die BHT-Titelseite (\BHTTitelseite)
  ├── references.bib       alle Quellen
  ├── build_thesis.py      Skript, das die PDF-Dateien erstellt
  ├── ReadMe.txt           diese Datei
  │
  ├── kapitel\             der Text – ein Kapitel = eine Datei
  │     00_vorspann.tex                Titelblatt, Kurzfassung, Verzeichnisse
  │     01_einleitung.tex              Kapitel 1
  │     02_stand_der_technik.tex       Kapitel 2
  │     03_methodik.tex                Kapitel 3
  │     04_experimente_ergebnisse.tex  Kapitel 4
  │     05_diskussion.tex              Kapitel 5
  │     06_fazit_ausblick.tex          Kapitel 6
  │     07_anhang.tex                  Anhang
  │     08_erklaerung.tex              Eidesstattliche Erklärung
  │
  ├── figures\             alle Abbildungen; figures\bht\ = BHT-Logos der Titelseite
  │
  │   --- diese Ordner/Dateien entstehen beim Bauen: ---
  ├── masterarbeit.pdf     die ganze Arbeit
  ├── Abgabe\Abgabe.pdf    der Auszug (Kapitel 2 und 4)
  └── einzeln\             einzelne Kapitel als PDF

Merke:
  - Den TEXT schreibt man in kapitel\.
  - Die Kapiteldateien allein ergeben keine PDF. Eine PDF entsteht immer
    über eine Hauptdatei oder über das Skript.


------------------------------------------------------------------------------
 3. Wie erstelle ich die PDF-Dateien?
------------------------------------------------------------------------------

Schritt 1: PowerShell öffnen und in den Ordner wechseln

  cd F:\claude\@output\deutsch

Schritt 2: den passenden Befehl ausführen

  Ganze Arbeit
    python build_thesis.py masterarbeit.tex
    -> masterarbeit.pdf

  Auszug (Kapitel 2 + 4)
    python build_thesis.py Abgabe.tex --out Abgabe
    -> Abgabe\Abgabe.pdf

  Ein Kapitel, z.B. Kapitel 3
    python build_thesis.py --kapitel 03
    -> einzeln\kapitel_03.pdf

  Mehrere Kapitel, jedes als eigene PDF
    python build_thesis.py --kapitel 01 03 05
    -> einzeln\kapitel_01.pdf, kapitel_03.pdf, kapitel_05.pdf

  Alle Kapitel, jedes als eigene PDF
    python build_thesis.py --alle-kapitel
    -> einzeln\kapitel_00.pdf ... kapitel_08.pdf

Schritt 3: die Ausgabe prüfen. Am Ende stehen drei Zeilen:

  Undefinierte Verweise/Zitate: keine        <- sollte "keine" sein
  Zu breite Zeilen (Overfull hbox): 2        <- kleine Zahl ist in Ordnung
  PDF: F:\claude\@output\deutsch\masterarbeit.pdf

Gut zu wissen:
  - Bei einem einzelnen Kapitel stehen Verweise auf andere Kapitel als "??".
    Das ist normal. In der ganzen Arbeit sind alle Verweise richtig.
  - Die PDF muss geschlossen sein, sonst kann Windows sie nicht überschreiben.
  - Mit "--keep" bleibt der Ordner build\ mit der .log-Datei erhalten
    (hilfreich bei Fehlern).


------------------------------------------------------------------------------
 4. Wie ändere ich etwas?
------------------------------------------------------------------------------

Text eines Kapitels ändern
  -> Datei in kapitel\ öffnen (z.B. mit TeXstudio oder VS Code, UTF-8),
     ändern, speichern, dann nur dieses Kapitel bauen:
       python build_thesis.py --kapitel 03

Titel, Name, Hochschule, Prüfer, Abgabedatum
  -> einmal in preamble.tex eintragen.

Englisches Abstract und englischer Titel
  -> Standard: die Arbeit ist nur auf Deutsch. Wenn die Prüfungsordnung
     auch ein englisches Abstract verlangt, in preamble.tex
       \ZweiteSprachefalse   durch   \ZweiteSprachetrue   ersetzen.

Abbildungen
  -> figures\ enthält die deutschen Abbildungen (deutsche Beschriftung,
     Dezimalkomma). Die englische Fassung hat eigene Abbildungen.

Rote Stellen "[Noch zu schreiben: ...]"
  -> das sind Platzhalter. Den Befehl \placeholder{...} in der .tex-Datei
     durch eigenen Text ersetzen.

Neue Abbildung
  -> Bild nach figures\ kopieren und im Kapitel einfügen:
       \begin{figure}[htbp]
         \centering
         \includegraphics[width=0.8\textwidth]{mein_bild.png}
         \caption{Kurze Beschreibung.}
         \label{fig:mein_bild}
       \end{figure}
     Im Text darauf verweisen: \cref{fig:mein_bild}

Neue Quelle
  -> BibTeX-Eintrag in references.bib einfügen (z.B. aus Google Scholar:
     "Zitieren" -> "BibTeX"). Im Text: \cite{schluessel}

Neue Abkürzung
  -> in kapitel\00_vorspann.tex in die Tabelle "Abkürzungsverzeichnis"
     eine Zeile einfügen:   KÜRZEL & Bedeutung \\

Kapitel kurz ausblenden
  -> in masterarbeit.tex die Zeile mit % beginnen lassen:
       % \input{kapitel/03_methodik.tex}

Zahlen
  -> im deutschen Text mit Komma schreiben: 28,39
     In \SI{...} und \num{...} setzt LaTeX das Komma automatisch.


------------------------------------------------------------------------------
 5. Was tun bei Fehlern?
------------------------------------------------------------------------------

"pdflatex wird nicht erkannt"
  -> MiKTeX installieren (Haken bei "Add to PATH") und die PowerShell neu
     öffnen.

"FEHLER bei pdflatex ..."
  -> Häufige Ursachen:
     - eine Klammer { } oder ein \begin ... \end ist nicht geschlossen
     - Sonderzeichen ohne Backslash: % & _ # schreibt man \% \& \_ \#
     - ein Bild fehlt in figures\
  -> Mit --keep bauen und die Datei build\<name>\<name>.log ansehen.

"??" in der ganzen Arbeit
  -> Tippfehler in \label{...} oder \cref{...}, oder das Kapitel ist in
     masterarbeit.tex ausgeblendet.

"[?]" statt Zitatnummer
  -> der Schlüssel in \cite{...} fehlt in references.bib.


------------------------------------------------------------------------------
 6. Stand der Arbeit (26.10.2026)
------------------------------------------------------------------------------

  Fertig (erste Fassung):  alle Kapitel 1 bis 6, Kurzfassung, Abstract, Anhang
  Titelseite:              nach dem BHT-Muster (Logo oben, sonst schwarz), Angaben in
                           preamble.tex
  Noch offen (rot):        Kamera und Rechner (Kapitel 3), Angabe zu
                           KI-Werkzeugen (08_erklaerung.tex)
  Alte Fassung:            F:\claude\@output\_alt_2026-10-26
  Nicht mehr gebraucht:    betreuer_termin.tex, Ordner Betreuer_Termin\,
                           experiments_results.*, state_of_the_art.*,
                           build_chapter.py

Die englische Fassung liegt in F:\claude\@output\englisch und hat ihre eigene
ReadMe.txt. Änderungen werden NICHT automatisch übertragen: Wer etwas im
deutschen Text ändert, muss es auch im englischen Text ändern.
