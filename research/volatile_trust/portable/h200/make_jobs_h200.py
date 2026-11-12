"""H200 job list (prioritized, 2026-11-13): a subset of portable/jobs.jsonl, output under runs_h200/ instead of runs_mi300x/.

Deviations from EXPERIMENTS.md, decided before any held-out model output (see PLAN_H200.md):
- platform: every model is rerun on H200, so all conditions of a model are on one platform; MI300X outputs stay
  untouched as a cross-platform check;
- compute: models above 38B are not run, and jobs are re-tiered by what the paper needs, so the full list fits the
  time before the deadline. The `tier` field is the priority: 1 = primary test and the core evidence chain,
  2 = secondary tests and tables, 3 = only if time remains (revised the same day to a middle ground: cheap
  image/chart, Psych-101 and wording jobs, Gemma-3-12B and part of the reasoning mode added back; development
  main-task reruns dropped except dev_full_text, which is the per-model cross-platform check). Jobs not listed below are not run on H200.
Job ids, arguments and output names are unchanged from jobs.jsonl (only `output` moves to runs_h200/ and `tier`
changes), so a resumed run skips completed jobs.

python research/volatile_trust/portable/h200/make_jobs_h200.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VT = 'research/volatile_trust'
CORE = ['qwen3vl8b', 'internvl8b', 'qwen3vl32b', 'internvl38b', 'gemma3_27b']   # <= 38B
GEMMA12 = 'gemma3_12b'   # tier 3: its wording decomposition and masking jobs (dev only), as in jobs.jsonl
BIG = ['qwen3vl32b', 'internvl38b', 'gemma3_27b']

# priority 1: primary test (held-out summary - raw), its decomposition (coded, raw+correct), stated-reliability and
# binding controls (development-only by design), robustness set, and the cross-platform check: dev_full_text is
# compared item by item with the MI300X run of the same job (development work was done on MI300X / RTX 3090;
# every held-out and robustness evaluation is on H200)
P1 = ['xcheck', 'dev_full_text', 'dev_controls', 'dev_binding', 'dev_chartread',
      'ho_full_text', 'ho_sum_all', 'ho_full_rawcoded', 'ho_full_coded', 'rob_full', 'rob_sum_all']
# priority 2: secondary tests (TOST for change-point and dev-selected window), chart and image columns,
# Psych-101 primary fits, held-out confirmation of the masking result
P2 = ['ho_sum_cp', 'dev_sum_window5', 'dev_sum_window10', 'dev_sum_window20',
      'ho_sum_window5', 'ho_sum_window10', 'ho_sum_window20', 'ho_full_chart', 'ho_full_image',
      'psy_text_hidden', 'psy_text_told', 'psy_image_outcome_hidden', 'psy_chart_hidden',
      'psy_perc_outcome_hidden', 'psy_perc_chart_hidden',      # Psych-101: all formats (human comparison)
      'ho_mask_rep_ok', 'ho_mask_out_ok', 'ho_mask_mask_ok']
# priority 3: only if time remains
P3 = ['ho_sum_oracle', 'rob_sum_cp', 'rob_sum_window10', 'ho_window10', 'ho_cp_hard', 'rob_full_coded',
      'para_p2_raw', 'para_p2_raw_coded', 'para_p2_coded', 'para_p2_sum_all',
      'para_p1_raw', 'para_p1_raw_coded', 'para_p1_coded', 'para_p1_sum_all']
# reasoning mode (tier 3, after the above): the 8B thinking model in full (4 shards x raw/summary), then half of the
# 32B thinking model (shards 0-1); the 32B chain-of-thought jobs are not run
E9 = [f'qwen3vl8b_thinking__e9_thinking_{h}_s{i}' for i in range(4) for h in ('raw', 'summ')] + \
     [f'qwen3vl32b_thinking__e9_thinking_{h}_s{i}' for i in range(2) for h in ('raw', 'summ')]
PRIORITY = {**{k: 1 for k in P1}, **{k: 2 for k in P2}, **{k: 3 for k in P3}}
ORDER = {k: i for i, k in enumerate(P1 + P2 + P3)}

reg = json.load(open(HERE.parent / 'models.json'))


def mask_job(key, v):   # held-out masking (masking_probe.py; predictions in ADDENDUM_masking.md)
    return dict(model=key, gpus=1, id=f'{key}__ho_mask_{v}', script=f'{VT}/masking_probe.py',
                args=['--split', 'held_out', '--data-dir', f'{VT}/episodes_v3', '--history-mode', 'full',
                      '--history-format', v, '--domains', 'pet', 'scenario', '--formats', 'text', '--readouts',
                      'choice', '--pixels', str(reg[key]['pixels'])],
                output=f'{VT}/runs_h200/{key}/ho_mask_{v}')


jobs = []
for line in open(HERE.parent / 'jobs.jsonl'):
    j = json.loads(line)
    key, suffix = j['id'].split('__', 1)
    if j['id'] in E9:
        tier = 3
    elif key == GEMMA12 and j['tier'] == 4:
        tier = 3
    elif key in CORE and suffix in PRIORITY and (not suffix.startswith('para_') or key in BIG):
        tier = PRIORITY[suffix]
    else:
        continue
    assert j['output'].startswith(f'{VT}/runs_mi300x/') and not any('runs_mi300x' in x for x in j['args'])
    j.update(output=j['output'].replace('runs_mi300x/', 'runs_h200/', 1), gpus=1, tier=tier)
    jobs.append(j)
jobs += [dict(mask_job(k, v), tier=2) for k in CORE for v in ('rep_ok', 'out_ok', 'mask_ok')]
def rank(j):
    if j['id'] in E9:
        return 200 + E9.index(j['id'])
    return 100 if j['model'] == GEMMA12 else ORDER.get(j['id'].split('__', 1)[1], 99)


jobs.sort(key=lambda j: (j['tier'], rank(j)))
with open(HERE / 'jobs_h200.jsonl', 'w') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(len(jobs), 'jobs ->', HERE / 'jobs_h200.jsonl')
