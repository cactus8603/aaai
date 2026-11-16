"""Apply a list of exact find/replace edits to paper/main.tex.

Usage: python apply.py EDITS.py   (EDITS.py defines E = [(old, new, count_or_None), ...])
count None = must occur exactly once; an int = must occur exactly that many times (all replaced).
Nothing is written if any edit fails.
"""
import runpy
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parents[3] / 'paper' / 'main.tex'
E = runpy.run_path(sys.argv[1])['E']
t = TEX.read_text()
bad = []
for old, new, n in E:
    c = t.count(old)
    if c != (n or 1):
        bad.append((c, old[:90]))
        continue
    t = t.replace(old, new)
if bad:
    for c, o in bad:
        print(f'MISS count={c}: {o!r}')
    raise SystemExit(f'{len(bad)} edits failed; nothing written')
TEX.write_text(t)
print(f'applied {len(E)} edits')
