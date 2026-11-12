"""Computational-model fits for VLM runs (EXPERIMENTS.md E2 learning-rate analysis; Appendix F).

Exports each episode pair's choice probabilities (averaged over the counterbalanced copies, label-aligned) per
model x history format x cover-story slot into the JSONL format of cogmodels.py, then runs cogmodels.fit_file:
per-episode fits, leave-one-episode-out cross-validated comparison of M0-M4 (the recommended comparison), and the
calibrated volatile-minus-stable effective learning rate. Text reports, choice readout.

The analysis plan restricts learning-rate interpretation to conditions in which the model tracks the better
advisor (stable-phase following >= 0.6) and uses stated reliability (>= 0.8); the gate is reported per group.

Usage: python analyze_cogmodels.py RUN_DIR [...] --out analysis/cogmodels_<tag>.json [--jobs 32] [--partial]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402
import cogmodels as cm  # noqa: E402

ARMS = {'A|full|text': 'raw', 'A+|full|text': 'raw_plus_correct', 'B|full|text': 'correct_only',
        'C|sum_all|text': 'summary'}


def export(run_dirs, partial):
    rows, _ = am.load_rows(run_dirs, strict=not partial)
    rows = [r for r in rows if r['readout'] == 'choice']
    paired = am.pair_trials(rows, strict=not partial)
    out = []
    for cell, per in paired.items():
        model, data, split, ro, arm, slot = cell
        if arm not in ARMS:
            continue
        for base, tr in per.items():
            tr = sorted(tr, key=lambda x: x['t'])
            if len(tr) != len(tr[0]['ep']['trials']):
                continue
            out.append(dict(model=model.split('@')[0], data=data, split=split, format=ARMS[arm], domain=slot,
                            episode_id=base, a_correct=[int(x['a_correct']) for x in tr],
                            phase=[x['phase'] for x in tr], p_a=[float(x['p_a']) for x in tr],
                            y=[float(x['py']) for x in tr]))
    return out


def stable_following(ep):
    y, o, ph = np.array(ep['y']), np.array(ep['a_correct']), np.array(ep['phase'])
    better_a = np.array(ep['p_a']) > 0.5
    m = (ph == 'stable') & (np.arange(len(y)) >= 1)
    return float(np.mean(np.where(better_a[m], y[m], 1 - y[m])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--restarts', type=int, default=4)
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    eps = export(a.runs, a.partial)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jl = a.out.with_suffix('.episodes.jsonl')
    with open(jl, 'w') as f:
        for e in eps:
            f.write(json.dumps(e) + '\n')
    res = cm.fit_file(str(jl), group_by=['model', 'format'], n_restarts=a.restarts, jobs=a.jobs, pooled=False,
                      cv=True, cv_restarts=2)
    gate = {}
    for e in eps:
        gate.setdefault(f"model={e['model']}|format={e['format']}", []).append(stable_following(e))
    res['stable_following'] = {g: float(np.mean(v)) for g, v in gate.items()}
    a.out.write_text(json.dumps(res, indent=1, default=cm._json_default))
    lines = []
    for g, d in sorted(res['cv'].items()):
        best = d['best_by_cv']
        ranked = sorted((m for m in d if m != 'best_by_cv'), key=lambda m: d[m]['heldout_nll_per_trial'])
        elr = res['eff_lr_calibrated_summary'].get(g, {})
        lines.append(f"{g:45} best {best:12} nll/trial {d[best]['heldout_nll_per_trial']:.3f}  2nd {ranked[1]:12} "
                     f"stable-following {res['stable_following'].get(g, float('nan')):.2f}  "
                     f"ELR {json.dumps(elr)[:120]}")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
