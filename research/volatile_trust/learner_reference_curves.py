"""Bayesian volatility learner on the held-out episodes, scored with the same reversal-curve and
integration-kernel definitions as the VLMs (analyze_main.episode_metrics, analyze_mechanism.kernel_cell).

Curve: share of trials on which the learner favors the currently better advisor (its choice, q > .5;
ties half credit), by trials since a true reversal, averaged over episodes. Also the soft version (q itself).
Kernel: soft-target logistic regression of q (P(A correct) before trial t) on A's correctness at lags 1..15,
with the same nuisance regressors as for the models. No VLM is involved.
Usage: python learner_reference_curves.py [--out analysis/h200_ho/learner_reference.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

import analyze_main as am
import analyze_mechanism as mech
import cogmodels as cm

HERE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else HERE / 'analysis/h200_ho/learner_reference.json'
    eps = mech.load_episodes(HERE / 'episodes_v3' / 'episodes.jsonl')
    per, curves_choice, curves_soft, acc = {}, [], [], []
    for eid, ep in sorted(eps.items()):
        if eid.endswith('t') or ep['split'] != 'held_out':
            continue
        o = np.array([t['a_correct'] for t in ep['trials']], float)
        q = cm.volatile_belief(o)['q']
        trials = [dict(x, ep=ep, py=float(q[i]), twinned=True) for i, x in enumerate(ep['trials'])]
        per[eid] = [trials]
        soft = am.episode_metrics(trials)
        hard = am.episode_metrics([dict(x, py=float(am.choice_credit(x['py']))) for x in trials])
        curves_soft.append(soft['curve'])
        curves_choice.append(hard['curve'])
        acc.append(hard['acc'])
    k = mech.kernel_cell(per, K=15, n_boot=1000, seed=0)
    res = dict(n_episodes=len(acc), acc=float(np.mean(acc)),
               curve_choice=np.nanmean(np.array(curves_choice, float), 0).round(4).tolist(),
               curve_soft=np.nanmean(np.array(curves_soft, float), 0).round(4).tolist(),
               kernel=k['kernel'], kernel_lo=k['kernel_lo'], kernel_hi=k['kernel_hi'],
               coef={n: k['coef'][n] for n in ('copy_prev_truth', 'label_prior', 'a_listed_first')},
               exp_fit=k['exp_fit'])
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res[x] for x in ('n_episodes', 'acc', 'curve_choice', 'curve_soft', 'exp_fit')}, indent=0))
    print('kernel', np.round(res['kernel'][:10], 3).tolist())


if __name__ == '__main__':
    main()
