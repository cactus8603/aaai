"""Greedy multi-GPU job launcher for jobs.jsonl (works on ROCm/MI300X and CUDA).

python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 [--models qwen3vl32b ...] [--only dev_ ho_]
- Each job gets `gpus` devices (from models.json). Jobs of the same model are queued together.
- A job is skipped if its runner wrote <output>/_COMPLETE* (written only after the runner verified the
  expected row count). Runners are resumable, so a crash or partial run is simply rerun.
- <output>/_LOCK holds the pid of a live runner; such jobs are skipped (prevents two writers).
- Logs: <output>/launcher.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, default=HERE / 'jobs.jsonl')
    ap.add_argument('--gpus', nargs='+', required=True)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--only', nargs='+', help='substring filters on job id')
    ap.add_argument('--tier', type=int, nargs='+', help='run only these tiers (1 = core evidence chain first)')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--model-root', type=Path, default=ROOT / '.cache/models')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    reg = json.loads((HERE / 'models.json').read_text())
    jobs = [json.loads(l) for l in open(a.jobs)]
    if a.models:
        jobs = [j for j in jobs if j['model'] in a.models]
    if a.only:
        jobs = [j for j in jobs if any(s in j['id'] for s in a.only)]
    if a.tier:
        jobs = [j for j in jobs if j.get('tier', 2) in a.tier]
    def complete(j):
        return any((ROOT / j['output']).glob('_COMPLETE*'))

    def locked(j):
        lock = ROOT / j['output'] / '_LOCK'
        if not lock.exists():
            return False
        try:
            os.kill(int(lock.read_text().strip()), 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False
    jobs = [j for j in jobs if not complete(j) and not locked(j)]
    print(f'{len(jobs)} pending jobs on GPUs {a.gpus}', flush=True)
    free, running = list(a.gpus), []
    while jobs or running:
        for p, j, g in list(running):
            if p.poll() is not None:
                running.remove((p, j, g))
                free += g
                ok = p.returncode == 0 and complete(j)
                (ROOT / j['output'] / '_LOCK').unlink(missing_ok=True)
                print(f'{time.strftime("%T")} {"done" if ok else "FAILED rc=" + str(p.returncode)} {j["id"]}', flush=True)
        for j in list(jobs):
            if j['gpus'] <= len(free):
                g, free = free[:j['gpus']], free[j['gpus']:]
                m = reg[j['model']]
                mdir = a.model_root / m['id'].split('/')[1] / m['revision']
                cmd = [a.python, str(ROOT / j['script']), '--model-dir', str(mdir), '--model-id', m['id'],
                       '--sharded', '--output', str(ROOT / j['output']), *j['args']]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g), HIP_VISIBLE_DEVICES=','.join(g))
                env.pop('ROCR_VISIBLE_DEVICES', None)   # a preset ROCR mask would remap the indices
                print(f'{time.strftime("%T")} start {j["id"]} on {g}', flush=True)
                if a.dry_run:
                    print('  ', ' '.join(cmd))
                    free += g
                    jobs.remove(j)
                    continue
                (ROOT / j['output']).mkdir(parents=True, exist_ok=True)
                log = open(ROOT / j['output'] / 'launcher.log', 'a')
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                (ROOT / j['output'] / '_LOCK').write_text(str(proc.pid))
                running.append((proc, j, g))
                jobs.remove(j)
        time.sleep(5)


if __name__ == '__main__':
    main()
