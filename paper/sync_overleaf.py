"""Copy the manuscript into a local clone of the Overleaf-linked repository (one-way: this repo -> Overleaf).

The \\author{...} and \\affiliations{...} blocks are kept from the Overleaf copy (they are maintained there and
never enter this repository); everything else in main.tex, plus refs.bib and figures/*.pdf, comes from here.
Refuses to run if the Overleaf main.tex differs from the last synced version outside the author block
(i.e. someone edited the paper on Overleaf): reconcile those edits first.

Usage: python paper/sync_overleaf.py OVERLEAF_CLONE   (then commit and push in the clone)
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTH = re.compile(r'(?:%[^\n]*\n)?\\author\{.*?\n\}\n\\affiliations\{.*?\n\}\n', re.S)


def strip_auth(t):
    return AUTH.sub('', t)


def main():
    dst = Path(sys.argv[1]).resolve()
    ov = (dst / 'main.tex').read_text()
    m = AUTH.search(ov)
    if not m:
        raise SystemExit('author/affiliations block not found in Overleaf main.tex')
    # last synced version = the file at the Overleaf repo's last commit that touched main.tex from this script
    base = subprocess.run(['git', '-C', str(dst), 'log', '--format=%H', '--grep', '^Sync from manuscript repo', '-1'],
                          capture_output=True, text=True).stdout.strip()
    if base:
        prev = subprocess.run(['git', '-C', str(dst), 'show', f'{base}:main.tex'], capture_output=True, text=True).stdout
        if strip_auth(prev) != strip_auth(ov):
            raise SystemExit('Overleaf main.tex was edited since the last sync (outside the author block); reconcile first.')
    src = (HERE / 'main.tex').read_text()
    ms = AUTH.search(src)
    if not ms:
        raise SystemExit('author/affiliations block not found in local main.tex')
    (dst / 'main.tex').write_text(src[:ms.start()] + m.group(0) + src[ms.end():])
    shutil.copy2(HERE / 'refs.bib', dst / 'refs.bib')
    (dst / 'figures').mkdir(exist_ok=True)
    for f in sorted((HERE / 'figures').glob('*.pdf')):
        shutil.copy2(f, dst / 'figures' / f.name)
    print('synced main.tex (author block kept), refs.bib, figures/*.pdf')


if __name__ == '__main__':
    main()
