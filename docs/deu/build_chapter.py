#!/usr/bin/env python3
r"""Build a PDF from a thesis chapter source (.txt or .tex).

Usage:
    python build_chapter.py my_Work.txt                 # -> my_Work.pdf
    python build_chapter.py Experiments_Results.tex --keep

What it does
  1. copies the source to build/<name>.tex
  2. if the source uses biblatex, a build copy is made with natbib/bibtex instead
     (\cite keys are unchanged).  Reason: biblatex needs the packages
     biblatex + logreq, which are not installed everywhere.  Your original
     file is NOT modified - on Overleaf/Colab it still compiles with biblatex.
  3. if lmodern is missing, the build copy falls back to Computer Modern (OT1)
  4. runs  pdflatex -> bibtex -> pdflatex -> pdflatex  and copies the PDF next to
     the source.  Use --keep to keep the build directory (log files).

Requirements: a TeX distribution with pdflatex, bibtex, natbib, KOMA-Script,
babel (+ german), siunitx, cleveref, booktabs, csquotes, caption, subcaption.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def have(sty):
    try:
        return bool(subprocess.run(["kpsewhich", sty], capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        return False


def convert(tex):
    notes = []
    if "biblatex" in tex and not have("logreq.sty"):
        tex = re.sub(r"\\usepackage\[[^\]]*\]\{biblatex\}", r"\\usepackage[numbers,sort&compress]{natbib}", tex, flags=re.S)
        tex = re.sub(r"\\addbibresource\{([^}]*)\.bib\}", r"% bib: \1", tex)
        m = re.search(r"% bib: (\S+)", tex)
        bib = m.group(1) if m else "references"
        tex = re.sub(r"\\printbibliography(\[[^\]]*\])?",
                     lambda _: "\\bibliographystyle{unsrtnat}\n\\bibliography{" + bib + "}", tex)
        # biblatex-only citation commands -> natbib equivalents
        tex = tex.replace("\\textcite", "\\citet").replace("\\parencite", "\\citep")
        notes.append("biblatex -> natbib (unsrtnat)")
    if "\\usepackage{lmodern}" in tex and not have("lmodern.sty"):
        tex = tex.replace("\\usepackage{lmodern}", "% lmodern not installed - Computer Modern used")
        tex = tex.replace("\\usepackage[T1]{fontenc}", "% T1 disabled (no cm-super) \\usepackage[T1]{fontenc}")
        # some CM glyphs are only available as bitmap fonts -> no font expansion
        tex = tex.replace("\\usepackage{microtype}", "\\usepackage[expansion=false]{microtype}")
        notes.append("lmodern/T1 -> Computer Modern OT1")
    return tex, notes


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    name = src.stem
    build = src.parent / "build" / name
    build.mkdir(parents=True, exist_ok=True)

    tex, notes = convert(src.read_text(encoding="utf-8"))
    (build / f"{name}.tex").write_text(tex, encoding="utf-8")
    for f in src.parent.glob("*.bib"):
        shutil.copy(f, build / f.name)
    for d in ("figures",):
        if (src.parent / d).is_dir():
            shutil.copytree(src.parent / d, build / d, dirs_exist_ok=True)
    print("Build copy:", build / f"{name}.tex", "|", ", ".join(notes) or "unchanged")

    steps = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
             ["bibtex", name],
             ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
             ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]]
    for cmd in steps:
        code, out = run(cmd, build)
        if code != 0 and cmd[0] == "pdflatex":
            print(out[-3000:]); sys.exit(f"FAILED: {' '.join(cmd)}")
        if cmd[0] == "bibtex" and ("Warning" in out or code):
            print(out[-1500:])

    log = (build / f"{name}.log").read_text(errors="replace")
    undefined = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)'.*undefined", log)))
    print("Undefined refs/cites:", undefined or "none")
    print("Overfull boxes:", log.count("Overfull \\hbox"))
    shutil.copy(build / f"{name}.pdf", src.parent / f"{name}.pdf")
    print("PDF:", src.parent / f"{name}.pdf")
    if not args.keep:
        shutil.rmtree(src.parent / "build")


if __name__ == "__main__":
    main()
