"""Exploratory: reversal curves as share of discrete choices (argmax, ties 0.5) for open models and the
Bayesian volatility learner, overall and split by reversal direction, with episode-pair bootstrap CIs.
Same pairing and trial masks as analyze_main (twin-averaged logits). No new model runs.
Usage: python reversal_choice_curves.py [--out analysis/h200_ho/reversal_choice_curves.json]
"""
import json, sys
from pathlib import Path
import numpy as np
import analyze_main as am
import analyze_mechanism as mech
import cogmodels as cm

HERE = Path(__file__).resolve().parent
MODELS = ['qwen3vl8b', 'internvl8b', 'gemma3_27b', 'qwen3vl32b', 'internvl38b']
RUNS = ['ho_full_text', 'ho_full_rawcoded', 'ho_full_coded', 'ho_sum_all', 'ho_sum_cp', 'ho_sum_window5']
ARMS = ['A|full|text', 'A+|full|text', 'B|full|text', 'C|sum_all|text', 'C|sum_cp|text', 'C|sum_window:5|text']
N = am.N_CURVE


def curves(trials, hard=True):
    """Per-episode curves: overall, new-better=A, new-better=B (nan where no trials)."""
    tr = [x for x in trials if x['t'] >= 2]
    rev = am.true_reversals(tr[0]['ep'])
    S = {k: np.zeros(N) for k in ('all', 'toA', 'toB')}
    C = {k: np.zeros(N) for k in S}
    for x in tr:
        if x['p_a'] == 0.5:
            continue
        prior = [r for r in rev if r <= x['t']]
        if not prior:
            continue
        k = x['t'] - prior[-1]
        if k >= N:
            continue
        py = float(am.choice_credit(x['py'])) if hard else x['py']
        a_better = x['p_a'] > 0.5
        f = py if a_better else 1 - py
        for key in ('all', 'toA' if a_better else 'toB'):
            S[key][k] += f; C[key][k] += 1
    return {k: np.where(C[k] > 0, S[k] / np.maximum(C[k], 1), np.nan) for k in S}


def summarize(per_ep, n_boot=2000, seed=0):
    out = {}
    for key in ('all', 'toA', 'toB'):
        M = np.array([c[key] for c in per_ep])
        mean = np.nanmean(M, 0)
        rng = np.random.default_rng(seed)
        B = np.array([np.nanmean(M[rng.integers(0, len(M), len(M))], 0) for _ in range(n_boot)])
        out[key] = dict(mean=mean.round(4).tolist(), lo=np.nanquantile(B, .025, 0).round(4).tolist(),
                        hi=np.nanquantile(B, .975, 0).round(4).tolist())
    return out


def per_episode(paired, g, arm, hard):
    slots = sorted({c[5] for c in paired if c[:4] == g and c[4] == arm})
    by_base = {}
    for s in slots:
        for b, tr in paired[g + (arm, s)].items():
            by_base.setdefault(b, []).append(curves(tr, hard))
    out = []
    for b, lst in sorted(by_base.items()):
        if len(lst) != len(slots):
            continue
        out.append({k: np.nanmean(np.array([c[k] for c in lst]), 0) for k in ('all', 'toA', 'toB')})
    return out


def main():
    out_path = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else HERE / 'analysis/h200_ho/reversal_choice_curves.json'
    res = {'models': {}, 'mean': {}}
    allm = {}
    for m in MODELS:
        dirs = [HERE / 'runs_h200' / m / r for r in RUNS if (HERE / 'runs_h200' / m / r).exists()]
        rows, _ = am.load_rows(dirs, strict=False)
        paired = am.pair_trials(rows, strict=False)
        groups = sorted({c[:4] for c in paired if c[3] == 'choice' and c[2] == 'held_out'})
        assert len(groups) == 1, groups
        g = groups[0]
        res['models'][m] = {}
        for arm in ARMS:
            if not any(c[:4] == g and c[4] == arm for c in paired):
                continue
            pe = per_episode(paired, g, arm, True)
            res['models'][m][arm] = dict(n=len(pe), hard=summarize(pe),
                                         soft=summarize(per_episode(paired, g, arm, False), n_boot=200))
            allm.setdefault(arm, []).append(pe)
        print(m, sorted(res['models'][m]), file=sys.stderr)
    for arm, lst in allm.items():
        # model mean: average per-episode curves over models (same episode pairs), then bootstrap pairs
        n = min(len(x) for x in lst)
        pe = [{k: np.nanmean(np.array([x[i][k] for x in lst]), 0) for k in ('all', 'toA', 'toB')} for i in range(n)]
        res['mean'][arm] = summarize(pe)
    eps = mech.load_episodes(HERE / 'episodes_v3' / 'episodes.jsonl')
    pe = []
    for eid, ep in sorted(eps.items()):
        if eid.endswith('t') or ep['split'] != 'held_out':
            continue
        o = np.array([x['a_correct'] for x in ep['trials']], float)
        q = cm.volatile_belief(o)['q']
        pe.append(curves([dict(x, ep=ep, py=float(q[i])) for i, x in enumerate(ep['trials'])], True))
    res['learner'] = summarize(pe)
    out_path.write_text(json.dumps(res, indent=0))
    for arm, s in res['mean'].items():
        print(arm.ljust(22), 'all', np.round(s['all']['mean'], 2).tolist())
        print(' '*22, 'toA', np.round(s['toA']['mean'], 2).tolist(), 'toB', np.round(s['toB']['mean'], 2).tolist())
    s = res['learner']
    print('learner'.ljust(22), 'all', np.round(s['all']['mean'], 2).tolist())
    print(' '*22, 'toA', np.round(s['toA']['mean'], 2).tolist(), 'toB', np.round(s['toB']['mean'], 2).tolist())


if __name__ == '__main__':
    main()
