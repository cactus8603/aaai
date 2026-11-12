"""Exploratory analysis of masking_probe.py (development episodes, text reports, choice readout).

Per model (3090: runs/paraphrase/orig_* and runs/masking/*; MI300X: runs_mi300x/<model>/dev_* and mask_*):
twin-averaged episode accuracy (t >= 2) for the raw, raw-plus-correctness, correctness-coded and summary
histories (same-machine runs in runs/paraphrase/orig_*) and the three masked histories (runs/masking/*), paired
episode contrasts with bootstrap CIs, the 2x2 (reports shown/masked x outcome shown/masked; all four with the
correctness statement) main effects and interaction, and the integration-kernel copy weight and the
copy-previous-outcome kappa per arm (analyze_mechanism.py).

Usage: python analyze_masking.py [--out-dir analysis/masking] [--partial]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402

MODELS = {'q': 'Qwen3-VL-8B', 'i': 'InternVL3.5-8B', 'gemma3_12b': 'Gemma-3-12B', 'qwen3vl32b': 'Qwen3-VL-32B',
          'internvl38b': 'InternVL3.5-38B', 'gemma3_27b': 'Gemma-3-27B', 'qwen25vl72b': 'Qwen2.5-VL-72B'}
AMD = {'A': 'dev_full_text', 'A+': 'dev_full_rawcoded', 'B': 'dev_full_coded', 'C': 'dev_sum_all'}
BASE = {'A': 'orig_raw', 'A+': 'orig_raw_coded', 'B': 'orig_coded', 'C': 'orig_sum_all'}
MASKED = ('rep_ok', 'out_ok', 'mask_ok')
ARM = {'A': 'A|full|text', 'A+': 'A+|full|text', 'B': 'B|full|text', 'C': 'C|sum_all|text',
       **{m: f'{m}|full|text' for m in MASKED}}
# (name, minuend, subtrahend)
CONTRASTS = [('A+ - A', 'A+', 'A'), ('B - A', 'B', 'A'), ('B - A+', 'B', 'A+'),
             ('mask_ok - B   (length/layout)', 'mask_ok', 'B'),
             ('rep_ok - A+   (hide outcome)', 'rep_ok', 'A+'),
             ('out_ok - A+   (hide reports)', 'out_ok', 'A+'),
             ('mask_ok - A+  (hide both)', 'mask_ok', 'A+')]
LABELS = {'A': 'raw', 'A+': 'raw+correct', 'B': 'coded', 'C': 'summary', 'rep_ok': 'reports+correct',
          'out_ok': 'outcome+correct', 'mask_ok': 'masked+correct'}


def dirs_for(m):
    if len(m) == 1:      # 3090 runs (q, i)
        d = {k: HERE / 'runs/paraphrase' / f'{v}_{m}' for k, v in BASE.items()}
        d.update({k: HERE / 'runs/masking' / f'{k}_{m}' for k in MASKED})
    else:                # MI300X runs
        d = {k: HERE / 'runs_mi300x' / m / v for k, v in AMD.items()}
        d.update({k: HERE / 'runs_mi300x' / m / f'mask_{k}' for k in MASKED})
    return {k: v for k, v in d.items() if v.exists() and any(v.glob('results_*'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=HERE / 'analysis/masking')
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out, lines = {}, []
    for m, name in MODELS.items():
        D = dirs_for(m)
        if not a.partial:
            D = {k: v for k, v in D.items() if any(v.glob('_COMPLETE*'))}
        if len(D) < 2:
            continue
        rows, _ = am.load_rows(list(D.values()), strict=not a.partial)
        rows = [r for r in rows if r['readout'] == 'choice']
        paired = am.pair_trials(rows, strict=not a.partial)
        group = sorted({c[:4] for c in paired})
        assert len(group) == 1, group      # one model x dataset x split x readout
        arms = [ARM[k] for k in D]
        per = am.arm_metrics(paired, group[0], arms, 'all', strict=not a.partial)
        acc = {k: {b: v['acc'] for b, v in per[ARM[k]].items()} for k in D}
        res = dict(acc={k: float(np.mean(list(v.values()))) for k, v in acc.items()}, contrasts={})
        for cname, x, y in CONTRASTS:
            if x in acc and y in acc:
                t = am.paired_test(acc[y], acc[x])
                res['contrasts'][cname] = dict(mean=t['mean'], ci95=t['ci95'], p=t['p_wilcoxon'], n=t['n'])
        if all(k in acc for k in ('A+', 'rep_ok', 'out_ok', 'mask_ok')):
            eps = sorted(set.intersection(*[set(acc[k]) for k in ('A+', 'rep_ok', 'out_ok', 'mask_ok')]))
            v = {k: np.array([acc[k][b] for b in eps]) for k in ('A+', 'rep_ok', 'out_ok', 'mask_ok')}
            fx = {'hide outcome (main)': (v['rep_ok'] + v['mask_ok'] - v['A+'] - v['out_ok']) / 2,
                  'hide reports (main)': (v['out_ok'] + v['mask_ok'] - v['A+'] - v['rep_ok']) / 2,
                  'interaction': v['mask_ok'] - v['out_ok'] - v['rep_ok'] + v['A+']}
            for k, d in fx.items():
                res['contrasts'][k] = dict(mean=float(d.mean()), ci95=am.boot_mean_ci(d, 10000, 0),
                                           p=am.wilcoxon_p(d), n=len(d))
        # mechanism: copy weight and kappa per arm
        tag = f'mech_{m}'
        with open(a.out_dir / f'log_{tag}.txt', 'w') as log:
            subprocess.run([sys.executable, str(HERE / 'analyze_mechanism.py'), *map(str, D.values()),
                            '--out-dir', str(a.out_dir), '--tag', tag, *(['--partial'] if a.partial else [])],
                           stdout=log, stderr=log)
        kj = a.out_dir / f'{tag}.json'
        K = json.loads(kj.read_text()) if kj.exists() else {}
        res['copy_w'], res['copy_kappa'], res['lag1_w'] = {}, {}, {}
        for k in D:
            kern = next((v for c, v in K.get('kernels', {}).items() if c.endswith('/' + ARM[k])), None)
            heur = next((v for c, v in K.get('heuristics', {}).items() if c.endswith('/' + ARM[k])), None)
            if kern:
                res['copy_w'][k] = kern['coef']['copy_prev_truth']['w']
                lag = next((kern['coef'][c]['w'] for c in kern['coef'] if c.endswith('1') and 'lag' in c), None)
                res['lag1_w'][k] = lag
            if heur:
                res['copy_kappa'][k] = heur['all']['strategies']['copy_prev_truth']['kappa']
        out[name] = res
        lines.append(f'== {name}')
        lines.append(f"{'history':18}{'acc':>7}{'copyW':>8}{'copyK':>8}{'lag1W':>8}")
        for k in D:
            f = lambda d: f"{d[k]:8.3f}" if d.get(k) is not None else f"{'-':>8}"  # noqa: E731
            lines.append(f"{LABELS[k]:18}{res['acc'][k]:7.3f}" + f(res['copy_w']) + f(res['copy_kappa'])
                         + f(res['lag1_w']))
        for c, r in res['contrasts'].items():
            lines.append(f"  {c:32}{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  p={r['p']:.1e}")
        lines.append('')
    (a.out_dir / 'masking.json').write_text(json.dumps(out, indent=1, default=am._json_default
                                                       if hasattr(am, '_json_default') else str))
    (a.out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
