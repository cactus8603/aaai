"""Download the xiong2023 inputs at pinned revisions and verify sha256 (no code from the data is executed).
  * Psych-101 prompts_training.jsonl (859 MB) is STREAMED; only lines of xiong2023neural/exp1.csv are kept.
  * marcelbinz/xiong2023neural exp1 parquet (raw per-trial values; no declared license: analysis only).
Usage: python -I research/volatile_trust/psych101/fetch_xiong.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / '.cache/datasets/psych101_xiong2023/raw'
PSY = ('https://huggingface.co/datasets/marcelbinz/Psych-101/resolve/'
       '611565c66395e2787cd7e3305149bb75dc138024/prompts_training.jsonl')
PSY_FULL_SHA = '7a370a94084acd6f73f20d6559214683cfadcab8a1f69e369df285ba75ccef01'
PSY_SLICE_SHA = '6018689512dce4bb3955df54909617b955ac314563b0ec54188df06168feb0ff'
PARQ = ('https://huggingface.co/datasets/marcelbinz/xiong2023neural/resolve/'
        '565dd050f4f146324c458ab7da664c2cf25e2310/exp1/train-00000-of-00001.parquet')
PARQ_SHA = 'a0d79f7630dcb59a87deeecbb5a1165ad6e647473433bb2ae0dac737d1aa0eca'
README = ('https://huggingface.co/datasets/marcelbinz/xiong2023neural/resolve/'
          '565dd050f4f146324c458ab7da664c2cf25e2310/README.md')
README_SHA = '02a23aa051beef748126d2bbf7da319a90f4c2b01a38d008701e7efb22ecbda4'
KEY = b'xiong2023neural/exp1.csv'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    slice_path = RAW / 'psych101_xiong2023neural_exp1.jsonl'
    if not (slice_path.exists() and sha(slice_path) == PSY_SLICE_SHA):
        full = hashlib.sha256()
        buf, kept = b'', []
        with urllib.request.urlopen(PSY, timeout=600) as r:
            while chunk := r.read(1 << 22):
                full.update(chunk)
                buf += chunk
                *lines, buf = buf.split(b'\n')
                kept += [l + b'\n' for l in lines if KEY in l]
        if buf and KEY in buf:
            kept.append(buf if buf.endswith(b'\n') else buf + b'\n')
        if full.hexdigest() != PSY_FULL_SHA:
            sys.exit('Psych-101 upstream sha256 mismatch')
        slice_path.write_bytes(b''.join(kept))
    assert sha(slice_path) == PSY_SLICE_SHA, 'Psych-101 slice sha256 mismatch'
    parq = RAW / 'xiong2023neural_exp1_train.parquet'
    if not (parq.exists() and sha(parq) == PARQ_SHA):
        urllib.request.urlretrieve(PARQ, parq)
    assert sha(parq) == PARQ_SHA, 'parquet sha256 mismatch'
    readme = RAW / 'README_xiong2023neural.md'
    if not (readme.exists() and sha(readme) == README_SHA):
        urllib.request.urlretrieve(README, readme)
    assert sha(readme) == README_SHA, 'README sha256 mismatch'
    print('xiong2023 inputs verified')


if __name__ == '__main__':
    main()
