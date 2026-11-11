"""Download pinned model snapshots: python download_models.py --models qwen3vl32b internvl38b [--root .cache/models]"""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--root', type=Path, default=ROOT / '.cache/models')
    a = ap.parse_args()
    reg = json.loads((HERE / 'models.json').read_text())
    for key in a.models:
        m = reg[key]
        dest = a.root / m['id'].split('/')[1] / m['revision']
        print(f'{key}: {m["id"]}@{m["revision"]} (~{m["gb"]} GB) -> {dest}', flush=True)
        snapshot_download(m['id'], revision=m['revision'], local_dir=str(dest), max_workers=16,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja',
                                          'merges.txt', 'vocab*'])
        print('done', key, flush=True)


if __name__ == '__main__':
    main()
