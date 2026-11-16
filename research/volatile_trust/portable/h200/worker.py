"""H200 worker: take the next job from jobs_h200.jsonl in file (= priority) order and run it; repeat until nothing is left.

python research/volatile_trust/portable/h200/worker.py [--tier 1 2] [--models ...]
Run several workers (one Slurm job with one GPU each); they share the list through lock files, so the jobs follow the
priority order across models no matter how many GPUs the lab has free. Differences from portable/run_jobs.py:
- the list is re-read before every pick (edits to jobs_h200.jsonl apply to the next pick);
- a job is claimed with an exclusive create of <output>/_LOCK holding the Slurm job id; a lock whose Slurm job is no
  longer running is stale and is taken over (runners resume, so a killed job just continues);
- jobs whose model weights are not fully downloaded yet are skipped (download them and the next pick will see them);
- housekeeping before every pick: delete the weights of a model whose jobs are all complete, and download a model
  that still has jobs once the group quota has room (one model at a time, list order);
- the GPU is the one Slurm assigned (CUDA_VISIBLE_DEVICES is inherited, not overwritten).
The runner command is identical to run_jobs.py.
"""
import argparse
import json
import os
import shutil
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


def weights_ready(mdir):
    """All shards named in the safetensors index exist and no download is in progress."""
    if not (mdir / 'config.json').exists():
        return False
    if any((mdir / '.cache').rglob('*.incomplete')):
        return False
    idx = mdir / 'model.safetensors.index.json'
    if idx.exists():
        return all((mdir / f).exists() for f in set(json.loads(idx.read_text())['weight_map'].values()))
    return (mdir / 'model.safetensors').exists()


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


def free_quota_gb():
    """Group quota headroom on /work from `quota -gs` (used and limit in G); None if it cannot be read."""
    try:
        out = subprocess.run(['quota', '-gs'], capture_output=True, text=True).stdout.split('\n')
        for i, line in enumerate(out):
            if '/work' in line:
                f = line.split()
                if len(f) < 3:
                    f = out[i + 1].split()
                else:
                    f = f[1:]
                g = lambda x: float(x.rstrip('*').rstrip('G'))
                return g(f[2]) - g(f[0])
    except Exception:
        pass
    return None


def housekeeping(jobs, reg, root, alive, margin_gb=15):
    """Swap weights so the list can finish within the quota (2026-11-13):
    1. a model whose jobs in the list are all complete and none is locked: delete its weights (re-downloadable);
    2. a model with pending jobs whose weights are missing: download it if quota headroom >= size + margin.
    Only one worker at a time (lock file holding the Slurm job id)."""
    lock = root / '.h200_housekeeping_LOCK'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, ME.encode()); os.close(fd)
    except FileExistsError:
        owner = lock.read_text().strip() if lock.exists() else ''
        if alive is None or owner in alive or owner == ME:
            return
        lock.unlink(missing_ok=True)
        return
    try:
        by_model = {}
        for j in jobs:
            by_model.setdefault(j['model'], []).append(j)
        for key, js in by_model.items():
            m = reg[key]
            mdir = root / m['id'].split('/')[1] / m['revision']
            if mdir.exists() and all(complete(j) for j in js) and not any((ROOT / j['output'] / '_LOCK').exists() for j in js):
                print(f'{time.strftime("%F %T")} housekeeping: all {len(js)} jobs of {key} complete; deleting {mdir}', flush=True)
                shutil.rmtree(mdir)
                if mdir.parent.exists() and not any(mdir.parent.iterdir()):
                    mdir.parent.rmdir()
        for key, js in by_model.items():   # list order = priority
            m = reg[key]
            mdir = root / m['id'].split('/')[1] / m['revision']
            if weights_ready(mdir) or all(complete(j) for j in js):
                continue
            free = free_quota_gb()
            if free is None or free < m['gb'] + margin_gb:
                print(f'{time.strftime("%F %T")} housekeeping: {key} needs {m["gb"]} GB + {margin_gb}, quota headroom {free}; wait', flush=True)
                break
            print(f'{time.strftime("%F %T")} housekeeping: downloading {key} ({m["gb"]} GB, headroom {free:.0f} GB)', flush=True)
            rc = subprocess.run([sys.executable, str(HERE.parent / 'download_models.py'), '--models', key, '--root', str(root)],
                                cwd=ROOT).returncode
            print(f'{time.strftime("%F %T")} housekeeping: download {key} rc={rc}', flush=True)
            break
    finally:
        lock.unlink(missing_ok=True)


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
        all_jobs = [json.loads(l) for l in open(a.jobs)]
        jobs = [j for j in all_jobs if (not a.tier or j.get('tier', 2) in a.tier) and (not a.models or j['model'] in a.models)
                and j.get('gpus', 1) == 1]
        alive = running_slurm_jobs()
        housekeeping(all_jobs, reg, a.model_root, alive)   # unfiltered: never delete weights a later tier needs
        picked = None
        for j in jobs:
            m = reg[j['model']]
            mdir = a.model_root / m['id'].split('/')[1] / m['revision']
            if complete(j) or fails.get(j['id'], 0) >= 2 or not weights_ready(mdir):
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
