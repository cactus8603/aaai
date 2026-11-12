"""H200 worker: take the next job from jobs_h200.jsonl in file (= priority) order and run it; repeat until nothing is left.

python research/volatile_trust/portable/h200/worker.py [--tier 1 2] [--models ...]
Run several workers (one Slurm job with one GPU each); they share the list through lock files, so the jobs follow the
priority order across models no matter how many GPUs the lab has free. Differences from portable/run_jobs.py:
- the list is re-read before every pick (edits to jobs_h200.jsonl apply to the next pick);
- a job is claimed with an exclusive create of <output>/_LOCK holding the Slurm job id; a lock whose Slurm job is no
  longer running is stale and is taken over (runners resume, so a killed job just continues);
- jobs whose model weights are not downloaded yet are skipped (download them and the next pick will see them);
- the GPU is the one Slurm assigned (CUDA_VISIBLE_DEVICES is inherited, not overwritten).
The runner command is identical to run_jobs.py.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ME = os.environ.get('SLURM_JOB_ID', f'{socket.gethostname()}-{os.getpid()}')


def running_slurm_jobs():
    try:
        out = subprocess.run(['squeue', '-h', '-t', 'R', '-o', '%A'], capture_output=True, text=True, check=True).stdout
        return set(out.split())
    except Exception:
        return None


def complete(j):
    return any((ROOT / j['output']).glob('_COMPLETE*'))


def claim(j, alive):
    out = ROOT / j['output']
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '_LOCK'
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ME.encode())
            os.close(fd)
            return True
        except FileExistsError:
            owner = lock.read_text().strip() if lock.exists() else ''
            if owner == ME:
                return True
            if alive is None or owner in alive:
                return False   # held by a live worker (or squeue unavailable: be conservative)
            lock.unlink(missing_ok=True)   # stale (its Slurm job ended); retry once
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, default=HERE / 'jobs_h200.jsonl')
    ap.add_argument('--tier', type=int, nargs='+')
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--model-root', type=Path, default=ROOT / '.cache/models')
    a = ap.parse_args()
    reg = json.loads((HERE.parent / 'models.json').read_text())
    print(f'worker {ME} root {ROOT}', flush=True)
    fails = {}
    while True:
        jobs = [json.loads(l) for l in open(a.jobs)]
        jobs = [j for j in jobs if (not a.tier or j.get('tier', 2) in a.tier) and (not a.models or j['model'] in a.models)
                and j.get('gpus', 1) == 1]
        alive = running_slurm_jobs()
        picked = None
        for j in jobs:
            m = reg[j['model']]
            mdir = a.model_root / m['id'].split('/')[1] / m['revision']
            if complete(j) or fails.get(j['id'], 0) >= 2 or not (mdir / 'config.json').exists():
                continue
            if claim(j, alive):
                picked = (j, mdir)
                break
        if picked is None:
            print(f'{time.strftime("%F %T")} nothing left to run; exit', flush=True)
            return
        j, mdir = picked
        m = reg[j['model']]
        cmd = [a.python, str(ROOT / j['script']), '--model-dir', str(mdir), '--model-id', m['id'], '--sharded',
               '--output', str(ROOT / j['output']), *j['args']]
        print(f'{time.strftime("%F %T")} start {j["id"]} (tier {j.get("tier")})', flush=True)
        t0 = time.time()
        with open(ROOT / j['output'] / 'launcher.log', 'a') as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and complete(j)
        (ROOT / j['output'] / '_LOCK').unlink(missing_ok=True)
        print(f'{time.strftime("%F %T")} {"done" if ok else "FAILED rc=" + str(rc)} {j["id"]} '
              f'{(time.time() - t0) / 3600:.2f} h', flush=True)
        if not ok:   # this worker gives up on a job after 2 failures; other workers may still try it
            fails[j['id']] = fails.get(j['id'], 0) + 1


if __name__ == '__main__':
    main()
