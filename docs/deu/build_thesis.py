#!/usr/bin/env python3
r"""Erzeugt PDF-Dateien der Masterarbeit: die ganze Arbeit oder einzelne Kapitel.

Aufruf (immer im Ordner dieser Datei, also deutsch\ oder englisch\):

  Ganze Arbeit:
    python build_thesis.py masterarbeit.tex          (englisch: thesis.tex)

  Auszug zur Abgabe (nur Kapitel 2 und 4):
    python build_thesis.py Abgabe.tex --out Abgabe
    (englisch: python build_thesis.py Submission.tex --out Submission)

  Ein einzelnes Kapitel (Nummer = Anfang des Dateinamens in kapitel\ bzw. chapters\):
    python build_thesis.py --kapitel 02              -> einzeln\kapitel_02.pdf
    python build_thesis.py --kapitel 02 04           -> mehrere Kapitel, je eine PDF

  Alle Kapitel einzeln:
    python build_thesis.py --alle-kapitel

  Zusätzlich:  --keep  (Hilfsdateien im Ordner build\ behalten, z.B. die .log-Datei)

Ablauf
  1. Alle nötigen Dateien (Hauptdatei, preamble.tex, kapitel\, figures\, *.bib, *.bst)
     werden nach build\ kopiert. Die Originaldateien werden nie verändert.
  2. Fehlt das Paket lmodern, wird in der Kopie auf Computer Modern umgestellt.
  3. pdflatex -> bibtex -> pdflatex -> pdflatex
  4. Die PDF wird gespeichert; danach werden undefinierte Verweise und zu
     breite Zeilen gemeldet. Der Ordner build\ wird wieder gelöscht.

Bei einzelnen Kapiteln erscheinen Verweise auf ANDERE Kapitel als "??".
Das ist normal; in der ganzen Arbeit sind sie korrekt.

Voraussetzung: TeX-Distribution mit pdflatex und bibtex (z.B. MiKTeX).
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCCLASS = (r"\documentclass[11pt, a4paper, oneside, DIV=12, BCOR=8mm, parskip=half," "\n"
            r"  headings=normal, bibliography=totoc, listof=totoc]{scrreprt}")


def have(sty):
    try:
        return bool(subprocess.run(["kpsewhich", sty], capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        return False


def fallback_fonts(build):
    """Ohne lmodern: Computer Modern verwenden (nur in der Build-Kopie)."""
    if have("lmodern.sty"):
        return False
    for tex in build.rglob("*.tex"):
        s = tex.read_text(encoding="utf-8")
        s2 = (s.replace("\\usepackage{lmodern}", "% lmodern nicht installiert")
               .replace("\\usepackage[T1]{fontenc}", "% \\usepackage[T1]{fontenc}")
               .replace("\\usepackage{microtype}", "\\usepackage[expansion=false]{microtype}"))
        if s2 != s:
            tex.write_text(s2, encoding="utf-8")
    return True


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout


def chapter_dir(root):
    for d in ("kapitel", "chapters"):
        if (root / d).is_dir():
            return root / d
    sys.exit("Ordner kapitel\\ bzw. chapters\\ nicht gefunden.")


def single_chapter_main(root, number):
    """Erzeugt den Text einer Hauptdatei, die nur ein Kapitel enthält."""
    kap = chapter_dir(root)
    files = sorted(kap.glob(f"{number}_*.tex"))
    if not files:
        sys.exit(f"Kein Kapitel mit der Nummer {number} in {kap} gefunden.")
    f = files[0]
    text = f.read_text(encoding="utf-8")
    german = "ngerman,capitalise" in (root / "preamble.tex").read_text(encoding="utf-8")
    lines = [DOCCLASS, r"\input{preamble.tex}", r"\begin{document}"]
    n = int(number)
    if 1 <= n <= 6:                                  # richtige Kapitelnummer behalten
        lines.append(rf"\setcounter{{chapter}}{{{n - 1}}}")
    lines.append(rf"\input{{{kap.name}/{f.name}}}")
    if "\\cite" in text:                             # Literatur nur, wenn zitiert wird
        lines += [rf"\renewcommand{{\bibname}}{{{'Literaturverzeichnis' if german else 'References'}}}",
                  r"\bibliographystyle{unsrtnat}", r"\bibliography{references}"]
    lines.append(r"\end{document}")
    return f.stem, "\n".join(lines) + "\n"


def build(root, name, out_dir, main_text=None, keep=False):
    """Kompiliert root/<name>.tex (oder main_text) und kopiert die PDF nach out_dir."""
    bdir = root / "build" / name
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True)
    for item in root.iterdir():
        if item.name == "build" or item.suffix == ".pdf":
            continue
        if item.is_dir() and item.name in ("kapitel", "chapters", "figures"):
            shutil.copytree(item, bdir / item.name)
        elif item.is_file() and item.suffix in (".tex", ".bib", ".bst", ".sty", ".cls"):
            shutil.copy(item, bdir / item.name)
    if main_text is not None:
        (bdir / f"{name}.tex").write_text(main_text, encoding="utf-8")
    elif not (bdir / f"{name}.tex").exists():
        sys.exit(f"Datei {name}.tex nicht gefunden.")
    font = "Computer Modern (lmodern fehlt)" if fallback_fonts(bdir) else "lmodern"
    print(f"\n=== {name}.tex  (Schrift: {font})")

    has_bib = "\\bibliography{" in (bdir / f"{name}.tex").read_text(encoding="utf-8")
    steps = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]]
    if has_bib:
        steps.append(["bibtex", name])
    steps += [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]] * 2
    for cmd in steps:
        code, out = run(cmd, bdir)
        if code != 0 and cmd[0] == "pdflatex":
            print(out[-3000:])
            sys.exit(f"FEHLER bei {' '.join(cmd)}  ->  Details: {bdir / (name + '.log')}")
        if cmd[0] == "bibtex" and (code or "Warning" in out):
            print(out[-1500:])

    log = (bdir / f"{name}.log").read_text(errors="replace")
    undefined = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)'.*undefined", log)))
    print("Undefinierte Verweise/Zitate:", undefined or "keine")
    print("Zu breite Zeilen (Overfull hbox):", log.count("Overfull \\hbox"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(bdir / f"{name}.pdf", out_dir / f"{name}.pdf")
    print("PDF:", out_dir / f"{name}.pdf")
    if not keep:
        shutil.rmtree(bdir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("main", nargs="?", default=None, help="Hauptdatei, z.B. masterarbeit.tex")
    ap.add_argument("--kapitel", "--chapter", nargs="+", metavar="NR",
                    help="nur diese Kapitel einzeln erzeugen, z.B. 02 04")
    ap.add_argument("--alle-kapitel", "--all-chapters", action="store_true",
                    help="jedes Kapitel als eigene PDF erzeugen")
    ap.add_argument("--out", default=None, help="Zielordner der PDF")
    ap.add_argument("--keep", action="store_true", help="Ordner build\\ mit Log-Dateien behalten")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    numbers = args.kapitel or []
    if args.alle_kapitel:
        numbers = sorted({p.name[:2] for p in chapter_dir(root).glob("[0-9][0-9]_*.tex")})
    if numbers:
        out_dir = Path(args.out).resolve() if args.out else root / ("einzeln" if chapter_dir(root).name == "kapitel" else "single")
        for nr in numbers:
            stem, text = single_chapter_main(root, f"{int(nr):02d}")
            build(root, f"kapitel_{int(nr):02d}" if chapter_dir(root).name == "kapitel" else f"chapter_{int(nr):02d}",
                  out_dir, main_text=text, keep=args.keep)
        if not args.keep and (root / "build").exists():
            shutil.rmtree(root / "build")
        return

    main_file = args.main or ("masterarbeit.tex" if (root / "masterarbeit.tex").exists() else "thesis.tex")
    src = (root / main_file).resolve()
    out_dir = Path(args.out).resolve() if args.out else root
    build(root, src.stem, out_dir, keep=args.keep)
    if not args.keep and (root / "build").exists():
        shutil.rmtree(root / "build")


if __name__ == "__main__":
    main()
