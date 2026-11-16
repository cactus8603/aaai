"""Download pinned model snapshots: python download_models.py --models qwen3vl32b internvl38b [--root .cache/models]

Gated repos (models.json "gated": true, e.g. google/gemma-3-*): accept the license on the model page with
your HF account, then export HF_TOKEN=<read token> (or `hf auth login`) before running this script."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

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
        try:
            snapshot_download(m['id'], revision=m['revision'], local_dir=str(dest), max_workers=16,
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja',
                                              'merges.txt', 'vocab*'])
        except GatedRepoError:
            raise SystemExit(f'{key}: {m["id"]} is gated. Accept its license on huggingface.co and set '
                             'HF_TOKEN (or run `hf auth login`), then rerun.')
        print('done', key, flush=True)


if __name__ == '__main__':
    main()
