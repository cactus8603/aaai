"""Build a self-contained tarball (code + episodes + only the referenced pet images).
python research/volatile_trust/portable/pack.py  ->  dist/volatile_trust_portable.tar.gz
"""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VT = ROOT / 'research/volatile_trust'


def main():
    files = set()
    for p in VT.glob('*.py'):
        files.add(p)
    for sub in ('portable', 'psych101'):
        d = VT / sub
        if d.exists():
            for p in d.rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.json', '.jsonl', '.md', '.txt', '.csv', '.sh') \
                        and 'examples' not in p.parts:
                    files.add(p)
    for data in ('episodes_v1', 'episodes_robust'):
        files.update(p for p in (VT / data).glob('*') if p.is_file())
    files.add(ROOT / 'research/source_trust/model_backends.py')
    files.add(VT / 'DESIGN.md')
    # referenced pet images (both resolutions)
    idx = json.loads((VT / 'episodes_v1/image_index.json').read_text())
    used = set()
    for data in ('episodes_v1', 'episodes_robust'):
        for line in open(VT / data / 'episodes.jsonl'):
            ep = json.loads(line)
            for v in ep['pet_image_ids'].values():
                used.update(v.values())
    for iid in used:
        for k in ('path224', 'path448'):
            files.add(ROOT / idx[iid][k])
    psy = ROOT / '.cache/datasets/psych101_xiong2023'
    if psy.exists():
        files.update(p for p in psy.rglob('*') if p.is_file())
    out = ROOT / 'dist/volatile_trust_portable.tar.gz'
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, 'w:gz') as tar:
        for p in sorted(files):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'{len(files)} files, {out.stat().st_size / 1e6:.1f} MB, sha256 {digest}\n{out}')


if __name__ == '__main__':
    main()
