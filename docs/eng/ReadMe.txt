==============================================================================
 ReadMe – Master's Thesis (English version)
 Khaled Rafei · Data-Efficient Personalization for Webcam-Based Eye Tracking
 Using Pretrained Vision Models
==============================================================================

Contents of this file
  1. What do I need?
  2. How is the folder organized?
  3. How do I create the PDF files?
  4. How do I change something?
  5. What to do when there are errors?
  6. Status of the thesis


------------------------------------------------------------------------------
 1. What do I need?
------------------------------------------------------------------------------

  - MiKTeX        (contains pdflatex and bibtex)
  - Python 3      (no extra packages needed)

Test in PowerShell:

  pdflatex --version
  python --version

Both commands must show a version number.
On the first build, MiKTeX downloads missing packages. If a window asks,
click "Install".


------------------------------------------------------------------------------
 2. How is the folder organized?
------------------------------------------------------------------------------

  F:\claude\@output\englisch\
  │
  ├── thesis.tex           main file: the WHOLE thesis
  ├── Submission.tex       main file: only Chapters 2 and 4 (excerpt)
  ├── preamble.tex         settings, details (title, name, examiners, date)
  │                        and the BHT title page (\BHTTitelseite)
  ├── references.bib       all sources
  ├── build_thesis.py      script that creates the PDF files
  ├── ReadMe.txt           this file
  │
  ├── chapters\            the text – one chapter = one file
  │     00_frontmatter.tex             title page, abstract, lists
  │     01_introduction.tex            Chapter 1
  │     02_state_of_the_art.tex        Chapter 2
  │     03_methodology.tex             Chapter 3
  │     04_experiments_results.tex     Chapter 4
  │     05_discussion.tex              Chapter 5
  │     06_conclusion_outlook.tex      Chapter 6
  │     07_appendix.tex                Appendix
  │     08_declaration.tex             Statutory declaration
  │
  ├── figures\             all figures; figures\bht\ = BHT logos for the title page
  │
  │   --- these folders/files are created when you build: ---
  ├── thesis.pdf                 the whole thesis
  ├── Submission\Submission.pdf  the excerpt (Chapters 2 and 4)
  └── single\                    single chapters as PDF

Remember:
  - You write the TEXT in chapters\.
  - The chapter files alone do not give a PDF. A PDF is always made from a
    main file or by the script.


------------------------------------------------------------------------------
 3. How do I create the PDF files?
------------------------------------------------------------------------------

Step 1: open PowerShell and go to the folder

  cd F:\claude\@output\englisch

Step 2: run the right command

  Whole thesis
    python build_thesis.py thesis.tex
    -> thesis.pdf

  Excerpt (Chapters 2 + 4)
    python build_thesis.py Submission.tex --out Submission
    -> Submission\Submission.pdf

  One chapter, e.g. Chapter 3
    python build_thesis.py --chapter 03
    -> single\chapter_03.pdf

  Several chapters, each as its own PDF
    python build_thesis.py --chapter 01 03 05
    -> single\chapter_01.pdf, chapter_03.pdf, chapter_05.pdf

  All chapters, each as its own PDF
    python build_thesis.py --all-chapters
    -> single\chapter_00.pdf ... chapter_08.pdf

Step 3: check the output. At the end there are three lines (the script prints
them in German):

  Undefinierte Verweise/Zitate: keine     <- undefined references: should be "keine" (none)
  Zu breite Zeilen (Overfull hbox): 2     <- lines too wide: a small number is fine
  PDF: F:\claude\@output\englisch\thesis.pdf

Good to know:
  - In a single chapter, references to other chapters show as "??". This is
    normal. In the whole thesis all references are correct.
  - The PDF must be closed, otherwise Windows cannot overwrite it.
  - With "--keep" the folder build\ with the .log file is kept (useful for
    errors).


------------------------------------------------------------------------------
 4. How do I change something?
------------------------------------------------------------------------------

Change the text of a chapter
  -> open the file in chapters\ (e.g. with TeXstudio or VS Code, UTF-8),
     change it, save it, then build only this chapter:
       python build_thesis.py --chapter 03

Title, name, university, examiners, submission date
  -> enter them once in preamble.tex.

German Kurzfassung, German title, German declaration text
  -> default: the thesis is in English only. If the examination regulations
     also require the German parts, replace in preamble.tex
       \ZweiteSprachefalse   with   \ZweiteSprachetrue

Figures
  -> figures\ contains the English figures. The German version has its
     own figures with German labels.

Red text "[To be written: ...]"
  -> these are placeholders. Replace the command \placeholder{...} in the
     .tex file with your own text.

New figure
  -> copy the image into figures\ and add it in the chapter:
       \begin{figure}[htbp]
         \centering
         \includegraphics[width=0.8\textwidth]{my_image.png}
         \caption{Short description.}
         \label{fig:my_image}
       \end{figure}
     Refer to it in the text: \cref{fig:my_image}

New source
  -> add a BibTeX entry to references.bib (e.g. from Google Scholar:
     "Cite" -> "BibTeX"). In the text: \cite{key}

New abbreviation
  -> in chapters\00_frontmatter.tex, add a line to the table
     "List of Abbreviations":   ABBR & Meaning \\

Hide a chapter for a moment
  -> in thesis.tex, start the line with %:
       % \input{chapters/03_methodology.tex}

Numbers
  -> in the English text, write them with a decimal point: 28.39


------------------------------------------------------------------------------
 5. What to do when there are errors?
------------------------------------------------------------------------------

"pdflatex is not recognized"
  -> install MiKTeX (tick "Add to PATH") and open PowerShell again.

"FEHLER bei pdflatex ..." (error in pdflatex)
  -> Common causes:
     - a bracket { } or a \begin ... \end is not closed
     - special characters without backslash: write % & _ # as \% \& \_ \#
     - an image is missing in figures\
  -> Build with --keep and look at build\<name>\<name>.log.

"??" in the whole thesis
  -> typo in \label{...} or \cref{...}, or the chapter is hidden in
     thesis.tex.

"[?]" instead of a citation number
  -> the key in \cite{...} is missing in references.bib.


------------------------------------------------------------------------------
 6. Status of the thesis (26 Oct 2026)
------------------------------------------------------------------------------

  Done (first draft):   all Chapters 1 to 6, abstract, appendix
  Title page:           BHT sample (logo at top, rest black), details in preamble.tex
  Still open (red):     camera and computer (Chapter 3), statement on AI
                        tools (08_declaration.tex)
  Old version:          F:\claude\@output\_alt_2026-10-26
  No longer needed:     supervisor_meeting.tex, folder Supervisor_Meeting\

The German version is in F:\claude\@output\deutsch and has its own ReadMe.txt.
Changes are NOT copied automatically: if you change the German text, you must
also change the English text.
