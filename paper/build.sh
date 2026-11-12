#!/usr/bin/env bash
# Compile main.tex to main.pdf (pdfLaTeX + BibTeX, as AAAI-27 requires) and run the submission checks.
# Usage: bash paper/build.sh [--clean]     (from anywhere; --clean also removes aux files afterwards)
set -euo pipefail
cd "$(dirname "$0")"
TT=$(ls -d "$HOME"/.TinyTeX/bin/* 2>/dev/null | sed -n 1p || true)
[ -n "$TT" ] && export PATH="$TT:$PATH"
command -v pdflatex >/dev/null || { echo "pdflatex not found (install TinyTeX, see README.md)"; exit 1; }

run() { pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || { grep -n -A3 "^!" main.log | head -20; exit 1; }; }
run
bibtex main >/dev/null || { cat main.blg | tail -20; exit 1; }
run
run

echo "== built $(pwd)/main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
# where the main text ends: the page on which the References heading appears must be <= 9 (8 pages + references)
if command -v pdftotext >/dev/null; then
  ref=$(pdftotext main.pdf - | awk 'BEGIN{p=1} {n=gsub(/\f/,""); if (!f && $0=="References") {print p; f=1}; p+=n}' || true)
  echo "   References start on page ${ref:-?} (main text must fit in 8 pages)"
fi
ov=$(grep -c "Overfull" main.log || true); und=$(grep -c "undefined" main.log || true)
echo "   overfull boxes: $ov   undefined refs/cites: $und"
if command -v pdffonts >/dev/null; then
  bad=$(pdffonts main.pdf | awk 'NR>2 && ($(NF-4)!="yes" || $2=="Type" && $3=="3")' | wc -l)
  echo "   fonts not embedded or Type 3: $bad"
fi
echo "   open markers: TODO $(grep -c '\\todo{' main.tex)  DRAFT $(grep -c '\\devnote{}' main.tex)  ?? $(grep -o '\\tbd' main.tex | wc -l)"
if [ "${1:-}" = "--clean" ]; then
  rm -f main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk
fi
