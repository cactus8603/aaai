"""Exploratory (added after the freeze): does a model weight recent outcomes more when accuracy is volatile?

Human signature in this task (Behrens et al. 2007, 2008): the learning rate is higher in the volatile phase.
For each model and history format, the integration kernel (analyze_mechanism.kernel_design; K lags) is fitted
separately on trials of the stable and of the volatile phase (phase of the current trial), and the
volatile-minus-stable difference in (i) the lag-1 weight and (ii) the summed weight of lags 1-3 is estimated with
a paired episode bootstrap (both phases refitted on the same resample). A constant-learning-rate delta agent run
through the same analysis on the same episodes gives the reference: its volatile/stable ratio of the summed
lag 1-3 weight is about 0.95 on these episodes (estimation alone pulls it slightly below 1), whereas a learner that
raises its learning rate under volatility, as humans do, gives a ratio above 1.

Usage: python analyze_volatility.py RUN_DIR [RUN_DIR ...] [--out analysis/volatility.json] [--K 8] [--n-boot 1000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402
import analyze_mechanism as mech  # noqa: E402

ARMS = ('A|full|text', 'A+|full|text', 'B|full|text', 'C|sum_all|text')


def design(per, phase, K):
    """Per-episode (X, y) blocks restricted to trials of one phase."""
    blocks = {}
    for b, trs in per.items():
        Xs, Ys = [], []
        for tr in trs:
            X, Y, names = mech.kernel_design(tr, K)
            ts = [x['t'] for x in tr if x['t'] >= 2]
            keep = np.array([tr[i]['phase'] == phase for i, x in enumerate(tr) if x['t'] >= 2])
            assert len(keep) == len(X) == len(ts)
            if keep.any():
                Xs.append(X[keep])
                Ys.append(Y[keep])
        if Xs:
            blocks[b] = (np.vstack(Xs), np.concatenate(Ys))
    return blocks


def stats(w, K):
    k = w[1:K + 1]
    return dict(lag1=float(k[0]), lag13=float(k[:3].sum()), kernel=[float(x) for x in k])


def fit(blocks, keys):
    X = np.vstack([blocks[b][0] for b in keys])
    Y = np.concatenate([blocks[b][1] for b in keys])
    return mech.fit_soft_logistic(X, Y)


def analyze_cell(per, K, n_boot, seed):
    st, vo = design(per, 'stable', K), design(per, 'volatile', K)
    eps = sorted(set(st) & set(vo))
    if len(eps) < 4:
        return None
    s0, v0 = stats(fit(st, eps), K), stats(fit(vo, eps), K)
    rng = np.random.default_rng(seed)
    D = {'lag1': [], 'lag13': []}
    ratio = []
    for _ in range(n_boot):
        j = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        s, v = stats(fit(st, j), K), stats(fit(vo, j), K)
        for m in D:
            D[m].append(v[m] - s[m])
        ratio.append(v['lag13'] / s['lag13'] if s['lag13'] > 0 else np.nan)
    out = dict(n_episodes=len(eps), stable=s0, volatile=v0)
    rr = np.array(ratio)
    out['ratio_lag13'] = dict(mean=v0['lag13'] / s0['lag13'] if s0['lag13'] > 0 else None,
                              ci95=[float(np.nanquantile(rr, .025)), float(np.nanquantile(rr, .975))])
    for m, d in D.items():
        d = np.array(d)
        out[f'diff_{m}'] = dict(mean=v0[m] - s0[m], ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                                p_boot_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    return out


def per_arm(run_dirs, partial):
    rows, _ = am.load_rows(run_dirs, strict=not partial)
    rows = [r for r in rows if r['readout'] == 'choice']
    paired = am.pair_trials(rows, strict=not partial)
    cells = {}
    for c, pe in paired.items():
        key = (c[0], c[1], c[2], c[4])
        for b, tr in pe.items():
            cells.setdefault(key, {}).setdefault(b, []).append(tr)
    return cells


def null_agent(ep_path, split, K, n_boot, alpha=0.3, beta=8.0):
    eps = {k: v for k, v in mech.load_episodes(Path(ep_path)).items() if v['split'] == split}
    per = {b: [v] for b, v in mech.synthetic_paired(eps, 'delta', alpha=alpha, beta=beta).items()}
    return analyze_cell(per, K, n_boot, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', type=Path, default=HERE / 'analysis/volatility.json')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    cells = per_arm(a.runs, a.partial)
    res, lines = {}, []
    for (model, data, split, arm), per in sorted(cells.items()):
        if arm not in ARMS:
            continue
        r = analyze_cell(per, a.K, a.n_boot, 0)
        if r is None:
            continue
        res[f'{model}/{data}/{split}/{arm}'] = r
        d1, d3 = r['diff_lag1'], r['diff_lag13']
        lines.append(f"{model.split('@')[0][:24]:24} {split[:8]:8} {arm:16} lag1 {r['stable']['lag1']:5.2f}->"
                     f"{r['volatile']['lag1']:5.2f} diff {d1['mean']:+.2f} [{d1['ci95'][0]:+.2f},{d1['ci95'][1]:+.2f}]"
                     f"  lags1-3 diff {d3['mean']:+.2f} [{d3['ci95'][0]:+.2f},{d3['ci95'][1]:+.2f}]"
                     f"  ratio {r['ratio_lag13']['mean']:.2f} [{r['ratio_lag13']['ci95'][0]:.2f},{r['ratio_lag13']['ci95'][1]:.2f}]")
    splits = sorted({k.split('/')[2] for k in res})
    res['null_delta_agent'] = {s: null_agent(HERE / 'episodes_v3/episodes.jsonl', s, a.K, min(a.n_boot, 200))
                               for s in splits if s in ('development', 'held_out')}
    for s, r in res['null_delta_agent'].items():
        if r:
            lines.append(f"null delta agent (alpha .3) {s}: lag1 diff {r['diff_lag1']['mean']:+.3f}, "
                         f"lags1-3 diff {r['diff_lag13']['mean']:+.3f}, ratio {r['ratio_lag13']['mean']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
