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


def jobs_for(key, m, split_sets=('dev', 'held_out', 'robust', 'psych101')):
    px = ['--pixels', str(m['pixels'])]
    base = dict(model=key, gpus=m['gpus_mi300x'])
    out = []

    def vol(name, split, data, mode, formats, readouts=('choice',), extra=(), pixels=None, hfmt='raw'):
        p = ['--pixels', str(pixels)] if pixels else px
        out.append(dict(base, id=f'{key}__{name}', script=f'{VT}/run_volatile.py',
                        args=['--split', split, '--data-dir', f'{VT}/{data}', '--history-mode', mode,
                              '--history-format', hfmt, '--domains', 'pet', 'scenario',
                              '--formats', *formats, '--readouts', *readouts, *p, *extra],
                        output=f'{VT}/runs_mi300x/{key}/{name}'))

    if 'dev' in split_sets:
        dv = ['--data-dir', f'{VT}/episodes_v3']
        out.append(dict(base, id=f'{key}__dev_controls', script=f'{VT}/run_controls.py', args=[*dv, *px],
                        output=f'{VT}/runs_mi300x/{key}/dev_controls'))
        out.append(dict(base, id=f'{key}__dev_binding', script=f'{VT}/run_controls.py',
                        args=[*dv, '--controls', 'explicit_rel', 'composite_read', '--formats', 'image_named',
                              'image_composite', 'image_after', *px], output=f'{VT}/runs_mi300x/{key}/dev_binding'))
        # chart legibility check at the same 448 px budget used by every chart-format job
        out.append(dict(base, id=f'{key}__dev_chartread', script=f'{VT}/run_controls.py',
                        args=[*dv, '--controls', 'chart_read', '--pixels', '448'],
                        output=f'{VT}/runs_mi300x/{key}/dev_chartread'))
    for split, tag in (('development', 'dev'), ('held_out', 'ho')):
        if tag not in split_sets and not (tag == 'ho' and 'held_out' in split_sets):
            continue
        vol(f'{tag}_full_text', split, 'episodes_v3', 'full', ['text'], ('choice', 'judgment'))
        vol(f'{tag}_full_image', split, 'episodes_v3', 'full', ['image'])
        vol(f'{tag}_full_chart', split, 'episodes_v3', 'full', ['chart'], pixels=448)
        # history format B: per-trial correctness stated (removes the "who was correct" step)
        vol(f'{tag}_full_coded', split, 'episodes_v3', 'full', ['text', 'image'], hfmt='coded')
        # A+: raw records plus the correctness statement (separates "adding correctness" from "removing distractors")
        vol(f'{tag}_full_rawcoded', split, 'episodes_v3', 'full', ['text'], hfmt='raw_coded')
        for arm in ARMS_TEXT_IMAGE:
            vol(f'{tag}_{arm.replace(":", "")}', split, 'episodes_v3', arm, ['text', 'image'])
        for arm in ARMS_TEXT:
            vol(f'{tag}_{arm.replace(":", "")}', split, 'episodes_v3', arm, ['text'])
    if 'psych101' in split_sets:   # external task: Psych-101 xiong2023 (human comparison)
        # hidden (hazard not stated) is primary; told is the secondary description contrast (text only)
        for fmt, hz in (('text', 'hidden'), ('text', 'told'), ('image_outcome', 'hidden'), ('chart', 'hidden'),
                        ('perc_outcome', 'hidden'), ('perc_chart', 'hidden')):
            out.append(dict(base, id=f'{key}__psy_{fmt}_{hz}', script=f'{VT}/psych101/run_xiong.py',
                            args=['--format', fmt, '--hazard-info', hz],
                            output=f'{VT}/runs_mi300x/{key}/psy_{fmt}_{hz}'))
    if 'robust' in split_sets:
        for arm in ['full', 'sum_cp', 'sum_all', 'sum_window:10']:
            vol(f'rob_{arm.replace(":", "")}', 'held_out', 'episodes_robust_v3', arm, ['text'])
        vol('rob_full_coded', 'held_out', 'episodes_robust_v3', 'full', ['text'], hfmt='coded')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['qwen3vl8b', 'internvl8b', 'qwen3vl32b', 'internvl38b',
                                                   'gemma3_27b', 'qwen25vl72b'])
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
                             args=['--split', 'development', '--data-dir', f'{VT}/episodes_v1',
                                   '--max-episodes', '2', '--formats', 'text',
                                   '--readouts', 'choice', '--pixels', str(m['pixels'])],
                             output=f'{VT}/runs_mi300x/{key}/xcheck'))
    # E9 reasoning mode (subset; long generations): Thinking models + CoT prompt on the instruct 32B
    for key, mode in (('qwen3vl8b_thinking', 'thinking'), ('qwen3vl32b_thinking', 'thinking'), ('qwen3vl32b', 'cot')):
        if key not in reg:
            continue
        m = reg[key]
        for hname, hargs in (('raw', ['--history-mode', 'full', '--history-format', 'raw']),
                             ('summ', ['--history-mode', 'sum_all', '--controls'])):
            for i in range(4):
                args = ['--data-dir', f'{VT}/episodes_v3', '--split', 'development', '--domain', 'scenario',
                        '--mode', mode, '--max-new-tokens', '16384', '--shard', f'{i}/4',
                        '--control-t', '40', '80', '--main-trials', '20', '40', '60', '80',
                        '--pixels', str(m['pixels'])]
                if hname == 'raw':   # controls (incl. coded variants) are run once, with the raw-history job
                    args += hargs + ['--controls', 'explicit_rel', 'easy_short', 'easy_short_coded',
                                     'easy_long', 'easy_long_coded']
                else:
                    args += hargs      # '--controls' with no values = main task only
                jobs.append(dict(model=key, gpus=m['gpus_mi300x'], id=f'{key}__e9_{mode}_{hname}_s{i}',
                                 script=f'{VT}/run_thinking.py', args=args,
                                 output=f'{VT}/runs_mi300x/{key}/e9_{mode}_{hname}/s{i}'))
    # Exploratory prompt-wording check (added 2026-11-12 after the freeze; development episodes only; see
    # EXPERIMENTS.md and discussion/paraphrase_analysis.md). Full 2x2x2 component decomposition on the Gemma-3 family's
    # smaller model (the 8B Qwen3-VL / InternVL3.5 decompositions were run on the 3090s); the larger models get the
    # key contrast only (reworded p1 and terse p2; their original-wording baseline is the dev_* jobs above).
    para = []
    PARA_FMTS = {'raw': ('full', 'raw'), 'raw_coded': ('full', 'raw_coded'), 'coded': ('full', 'coded'),
                 'sum_all': ('sum_all', 'raw')}
    MIXES = ('Ho-L2-Qo', 'H2-Lo-Qo', 'Ho-Lo-Q2', 'H2-L2-Qo', 'H2-Lo-Q2', 'Ho-L2-Q2')
    plan = [('gemma3_12b', w, f) for w in ('orig', 'p1', 'p2') for f in PARA_FMTS]
    plan += [('gemma3_12b', w, f) for w in MIXES for f in ('raw', 'coded', 'sum_all')]
    plan += [(k, w, f) for k in ('qwen3vl32b', 'internvl38b', 'gemma3_27b', 'qwen25vl72b') if k in a.models
             for w in ('p1', 'p2') for f in PARA_FMTS]
    for key, w, f in plan:
        m = reg[key]
        mode, hfmt = PARA_FMTS[f]
        name = f'para_{w}_{f}'
        para.append(dict(model=key, gpus=m['gpus_mi300x'], id=f'{key}__{name}',
                         script=f'{VT}/paraphrase_probe.py',
                         args=['--wording', w, '--split', 'development', '--data-dir', f'{VT}/episodes_v3',
                               '--history-mode', mode, '--history-format', hfmt, '--domains', 'pet', 'scenario',
                               '--formats', 'text', '--readouts', 'choice', '--pixels', str(m['pixels'])],
                         output=f'{VT}/runs_mi300x/{key}/{name}'))
    # Tiers (run tier 1 first): 1 = core evidence chain, 2 = secondary/explanatory, 3 = exploratory/extra.
    T1 = ('dev_controls', 'dev_binding', 'dev_chartread', 'xcheck', '_full_text', '_full_coded', '_full_rawcoded',
          '_sum_all', 'rob_full')
    T3 = ('_cp_hard', '_window10', 'psy_image_outcome', 'psy_chart', 'psy_perc', 'rob_full_coded')
    for j in jobs:
        jid = j['id'].split('__', 1)[1]
        tier = 1 if (any(k in jid for k in T1) and not jid.startswith('rob_full_coded')
                     and not jid.startswith('rob_sum_window')) else 2
        if any(k in jid for k in T3) or j['model'] == 'qwen25vl72b':
            tier = 3
        if jid.startswith('rob_sum_all') or jid == 'rob_full':
            tier = 1
        j['tier'] = tier
    for j in para:            # appended last, so run_jobs.py starts them after everything else
        j['tier'] = 4
    jobs += para
    with open(a.out, 'w') as f:
        for j in jobs:
            f.write(json.dumps(j) + '\n')
    print(len(jobs), 'jobs ->', a.out)


if __name__ == '__main__':
    main()
