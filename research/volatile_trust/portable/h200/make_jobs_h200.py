"""H200 job list: same jobs as portable/jobs.jsonl for the chosen models, output under runs_h200/ instead of runs_mi300x/.

2026-11-13 deviation from EXPERIMENTS.md (platform): every model in jobs.jsonl is rerun in full on H200 (user decision),
so all conditions of every model are on one platform. MI300X outputs stay untouched as a cross-platform check.
python research/volatile_trust/portable/h200/make_jobs_h200.py --models qwen3vl8b internvl8b qwen3vl32b internvl38b gemma3_27b gemma3_12b qwen3vl8b_thinking qwen3vl32b_thinking
2026-11-13: models above 38B are not run on H200 (Qwen2.5-VL-72B dropped from the paper, user decision; its lines are no
longer in jobs_h200.jsonl). H200-only exploratory jobs (tier 4) are appended at the end.
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
    # H200-only additions (2026-11-13; exploratory follow-ups, tier 4, appended after everything above; see
    # ADDENDUM_masking.md). Models above 38B are not run on H200. Order = priority within tier 4.
    reg = json.load(open(HERE.parent / 'models.json'))
    reg = reg.get('models', reg)
    SMALL, BIG = ['qwen3vl8b', 'internvl8b'], ['qwen3vl32b', 'internvl38b', 'gemma3_27b']
    VT = 'research/volatile_trust'

    def add(key, jid, script, args, out):
        global n
        if key not in a.models:
            return
        f.write(json.dumps(dict(model=key, gpus=1, id=f'{key}__{jid}', script=f'{VT}/{script}', args=args,
                                output=f'{VT}/runs_h200/{key}/{out}', tier=4)) + '\n')
        n += 1

    def common(key, split, mode, hfmt):
        return ['--split', split, '--data-dir', f'{VT}/episodes_v3', '--history-mode', mode, '--history-format',
                hfmt, '--domains', 'pet', 'scenario', '--formats', 'text', '--readouts', 'choice',
                '--pixels', str(reg[key]['pixels'])]

    MASKS = ('rep_ok', 'out_ok', 'mask_ok')
    # (1) held-out confirmation of the dev masking result (predictions written before running: ADDENDUM_masking.md)
    for key in SMALL + BIG:
        for v in MASKS:
            add(key, f'ho_mask_{v}', 'masking_probe.py', common(key, 'held_out', 'full', v), f'ho_mask_{v}')
    # (2) dev masking for the 8B models (run on the 3090s before; repeated here so every condition is on H200)
    for key in SMALL:
        for v in MASKS:
            add(key, f'mask_{v}', 'masking_probe.py', common(key, 'development', 'full', v), f'mask_{v}')
    # (3) wording key contrast for the 8B models (same reason; the 2x2x2 decomposition stays a 3090 result)
    FMTS = {'raw': ('full', 'raw'), 'raw_coded': ('full', 'raw_coded'), 'coded': ('full', 'coded'),
            'sum_all': ('sum_all', 'raw')}
    for key in SMALL:
        for w in ('p1', 'p2'):
            for fm, (mode, hfmt) in FMTS.items():
                add(key, f'para_{w}_{fm}', 'paraphrase_probe.py',
                    ['--wording', w] + common(key, 'development', mode, hfmt), f'para_{w}_{fm}')
print(n, 'jobs ->', HERE / 'jobs_h200.jsonl')
