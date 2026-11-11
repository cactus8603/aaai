"""Generate the frozen job manifest (jobs.jsonl). One line per job; launcher runs them resumably.

Design (frozen 2026-11-11, see discussion/long_paper_plan.md):
  dev      : controls, binding controls, full history (text/image/chart, + judgment on text), arms
  held_out : same arms as dev (no controls)
  robust   : IID/irregular episodes: full, sum_cp, sum_all, sum_window:10 (text)
  xcheck   : cross-platform consistency subset (2 dev episodes, text, full) for models also run on 3090
Arms: sum_cp, sum_all, sum_window:{5,10,20}, sum_oracle, cp_hard, window:10.
Primary analysis contrast: sum_cp vs dev-selected sum_window:K (K chosen on dev only) and vs sum_all.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VT = 'research/volatile_trust'

ARMS_TEXT_IMAGE = ['sum_cp']
ARMS_TEXT = ['sum_all', 'sum_window:5', 'sum_window:10', 'sum_window:20', 'sum_oracle', 'cp_hard', 'window:10']


def jobs_for(key, m, split_sets=('dev', 'held_out', 'robust')):
    px = ['--pixels', str(m['pixels'])]
    base = dict(model=key, gpus=m['gpus_mi300x'])
    out = []

    def vol(name, split, data, mode, formats, readouts=('choice',), extra=()):
        out.append(dict(base, id=f'{key}__{name}', script=f'{VT}/run_volatile.py',
                        args=['--split', split, '--data-dir', f'{VT}/{data}', '--history-mode', mode,
                              '--formats', *formats, '--readouts', *readouts, *px, *extra],
                        output=f'{VT}/runs_mi300x/{key}/{name}'))

    if 'dev' in split_sets:
        out.append(dict(base, id=f'{key}__dev_controls', script=f'{VT}/run_controls.py', args=px,
                        output=f'{VT}/runs_mi300x/{key}/dev_controls'))
        out.append(dict(base, id=f'{key}__dev_binding', script=f'{VT}/run_controls.py',
                        args=['--controls', 'explicit_rel', '--formats', 'image_named', 'image_composite',
                              'image_after', *px], output=f'{VT}/runs_mi300x/{key}/dev_binding'))
    for split, tag in (('development', 'dev'), ('held_out', 'ho')):
        if tag not in split_sets and not (tag == 'ho' and 'held_out' in split_sets):
            continue
        vol(f'{tag}_full_text', split, 'episodes_v1', 'full', ['text'], ('choice', 'judgment'))
        vol(f'{tag}_full_image', split, 'episodes_v1', 'full', ['image'])
        vol(f'{tag}_full_chart', split, 'episodes_v1', 'full', ['chart'])
        for arm in ARMS_TEXT_IMAGE:
            vol(f'{tag}_{arm.replace(":", "")}', split, 'episodes_v1', arm, ['text', 'image'])
        for arm in ARMS_TEXT:
            vol(f'{tag}_{arm.replace(":", "")}', split, 'episodes_v1', arm, ['text'])
    if 'robust' in split_sets:
        for arm in ['full', 'sum_cp', 'sum_all', 'sum_window:10']:
            vol(f'rob_{arm.replace(":", "")}', 'held_out', 'episodes_robust', arm, ['text'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['qwen3vl8b', 'internvl8b', 'qwen3vl32b', 'internvl38b', 'qwen25vl72b'])
    ap.add_argument('--out', type=Path, default=HERE / 'jobs.jsonl')
    a = ap.parse_args()
    reg = json.loads((HERE / 'models.json').read_text())
    jobs = []
    for key in a.models:
        jobs += jobs_for(key, reg[key])
    # cross-platform check (compare with 3090 pilot rows for the same prompts)
    for key in ('qwen3vl8b', 'qwen3vl32b'):
        if key in a.models:
            m = reg[key]
            jobs.append(dict(model=key, gpus=m['gpus_mi300x'], id=f'{key}__xcheck', script=f'{VT}/run_volatile.py',
                             args=['--split', 'development', '--max-episodes', '2', '--formats', 'text',
                                   '--readouts', 'choice', '--pixels', str(m['pixels'])],
                             output=f'{VT}/runs_mi300x/{key}/xcheck'))
    with open(a.out, 'w') as f:
        for j in jobs:
            f.write(json.dumps(j) + '\n')
    print(len(jobs), 'jobs ->', a.out)


if __name__ == '__main__':
    main()
