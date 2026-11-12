"""H200 job list: same jobs as portable/jobs.jsonl for the chosen models, output under runs_h200/ instead of runs_mi300x/.

2026-11-13 deviation from EXPERIMENTS.md (platform): every model in jobs.jsonl is rerun in full on H200 (user decision),
so all conditions of every model are on one platform. MI300X outputs stay untouched as a cross-platform check.
python research/volatile_trust/portable/h200/make_jobs_h200.py --models <all keys in models.json used by jobs.jsonl>
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', required=True)
a = ap.parse_args()
n = 0
with open(HERE / 'jobs_h200.jsonl', 'w') as f:
    for line in open(HERE.parent / 'jobs.jsonl'):
        j = json.loads(line)
        if j['model'] not in a.models:
            continue
        assert j['output'].startswith('research/volatile_trust/runs_mi300x/')
        assert not any('runs_mi300x' in x for x in j['args'])
        j['output'] = j['output'].replace('runs_mi300x/', 'runs_h200/', 1)
        j['gpus'] = 2 if j['model'] == 'qwen25vl72b' else 1   # 72B BF16 ~147 GB > one H200 (141 GB)
        f.write(json.dumps(j) + '\n')
        n += 1
print(n, 'jobs ->', HERE / 'jobs_h200.jsonl')
