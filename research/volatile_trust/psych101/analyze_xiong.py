"""Learning-rate adaptation across hazard rates: humans vs frozen VLMs (xiong2023 changing bandit).

Learning model (reused from Xiong, Ji-An, Mattar & Wilson, CCN 2023, "myopic model-free agent": it
"updates the chosen action value with the immediate reward and gradually forgets the unchosen action
value"), with the paper's last-action dependence added as a perseveration term:
    Q_chosen   <- Q_chosen   + alpha * (r - Q_chosen)
    Q_unchosen <- Q_unchosen + phi   * (50 - Q_unchosen)            (50 = mean of U{1..99})
    P(slot1)_t  = sigmoid( beta * (Q1 - Q2)/10 + kappa * s_{t-1} + b ),   s = +1/-1 previous slot1/slot2
Q starts at 50. Models fit (same model for humans and VLMs, separately per hazard condition):
    delta     alpha, beta, kappa, b       (phi = 0; plain delta rule)
    xiong_mf  alpha, phi, beta, kappa, b  (primary)
    bayes_h   alpha = 1, phi = h; beta, kappa, b   (normative reference)
Why phi is the hazard-sensitive "learning rate": rewards in this task are noise-free (reward == chosen
machine's current value on 100% of trials). The exact Bayesian posterior mean is therefore Q_chosen = r
(alpha = 1 for every h) and, for the unobserved machine, E[v] = 50 + (1-h)^k (v_last - 50), i.e. the
recursion above with phi = h. A Kalman-style learner with a hazard-dependent volatility therefore shows
its adaptation in phi (and in alpha only if it wrongly treats outcomes as noisy). We report both.
Targets: humans = binary choices; VLMs = soft P(slot1) (soft cross-entropy, as in cogmodels.py).
The human's previous choice (shown in the history) drives the perseveration term for both.

Model-agnostic signature (paper Fig. 1g): P(shift | last reward < 50) and P(shift | last reward >= 50)
per hazard; humans show more shifting after low rewards when h is higher.

Usage:
  .venv/bin/python research/volatile_trust/psych101/analyze_xiong.py --results 'research/volatile_trust/psych101/runs/*/results_*.jsonl' \
      --output research/volatile_trust/psych101/analysis.json [--boot 200] [--human-long]
  .venv/bin/python research/volatile_trust/psych101/analyze_xiong.py --selftest
"""
import argparse
import collections
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # tiny arrays: BLAS threads only add overhead

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
HAZARDS = (0.1, 0.2, 0.4)
EPS = 1e-6


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


MODELS = {   # name -> (free parameter names)
    'delta': ('alpha', 'beta', 'kappa', 'b'),
    'xiong_mf': ('alpha', 'phi', 'beta', 'kappa', 'b'),
    'bayes_h': ('beta', 'kappa', 'b'),
}


def unpack(name, x, h):
    p = dict(zip(MODELS[name], x))
    out = dict(alpha=1.0, phi=0.0, beta=0.0, kappa=0.0, b=0.0)
    for k, v in p.items():
        out[k] = sig(v) if k in ('alpha', 'phi') else (np.logaddexp(0, v) if k == 'beta' else v)
    if name == 'bayes_h':
        out.update(alpha=1.0, phi=h)
    return out


def forward(prm, c, r):
    """c, r: (E, T) human choices (0 = slot1) and rewards. Returns P(slot1) (E, T)."""
    E, T = c.shape
    Q = np.full((E, 2), 50.0)
    s = np.zeros(E)
    P = np.empty((E, T))
    rows = np.arange(E)
    for t in range(T):
        P[:, t] = sig(prm['beta'] * (Q[:, 0] - Q[:, 1]) / 10 + prm['kappa'] * s + prm['b'])
        ch = c[:, t]
        Q[rows, ch] += prm['alpha'] * (r[:, t] - Q[rows, ch])
        Q[rows, 1 - ch] += prm['phi'] * (50 - Q[rows, 1 - ch])
        s = np.where(ch == 0, 1.0, -1.0)
    return P


def nll(x, name, h, c, r, y, mask):
    P = np.clip(forward(unpack(name, x, h), c, r), EPS, 1 - EPS)
    return -float(np.sum(mask * (y * np.log(P) + (1 - y) * np.log(1 - P))))


def fit(name, h, c, r, y, mask=None, x0=None):
    """Multi-start L-BFGS-B; with x0 (warm start, e.g. bootstrap) one extra start is added first."""
    mask = np.ones_like(y) if mask is None else mask
    starts = [] if x0 is None else [list(x0)]
    for a in ((0.5, 0.95) if x0 is None else (0.8,)):
        for ph in ((0.05, 0.3) if x0 is None else (0.15,)):
            st = {'alpha': logit(a), 'phi': logit(ph), 'beta': 1.0, 'kappa': 0.5, 'b': 0.0}
            starts.append([st[k] for k in MODELS[name]])
    best = None
    for x0 in starts:
        res = optimize.minimize(nll, x0, args=(name, h, c, r, y, mask), method='L-BFGS-B',
                                bounds=[(-8, 8) if k in ('alpha', 'phi') else (-8, 6) if k == 'beta' else (-10, 10)
                                        for k in MODELS[name]])
        if best is None or res.fun < best.fun:
            best = res
        if name == 'bayes_h':
            break
    prm = unpack(name, best.x, h)
    n = float(mask.sum())
    k = len(MODELS[name])
    ent = -float(np.sum(mask * (y * np.log(np.clip(y, EPS, 1)) + (1 - y) * np.log(np.clip(1 - y, EPS, 1)))))
    return dict(params={k_: float(v) for k_, v in prm.items()}, nll=float(best.fun), n=n,
                bic=k * np.log(n) + 2 * best.fun, excess_nll=float(best.fun) - ent, x=[float(v) for v in best.x])


def _fit_job(a):
    return fit(*a)


POOL = None


def pmap(jobs):
    return list(POOL.map(_fit_job, jobs, chunksize=4)) if POOL else [_fit_job(j) for j in jobs]


# ------------------------------------------------------------------ data
def load_episodes(path):
    return {e['episode_id']: e for e in map(json.loads, open(path))}


def arrays(eps, ids, targets=None):
    c = np.array([[tr['choice'] for tr in eps[i]['trials']] for i in ids])
    r = np.array([[tr['reward'] for tr in eps[i]['trials']] for i in ids], dtype=float)
    y = (c == 0).astype(float) if targets is None else np.array([targets[i] for i in ids])
    return c, r, y


def shift_signature(c, r, y):
    """P(shift away from previous human choice) split by last reward <50 / >=50. Uses soft y if given."""
    prev = c[:, :-1]
    p_same = np.where(prev == 0, y[:, 1:], 1 - y[:, 1:])
    low = r[:, :-1] < 50
    return dict(p_shift_low=float(np.mean(1 - p_same[low])), p_shift_high=float(np.mean(1 - p_same[~low])))


def analyse_source(eps, by_h, boot, rng, models=('delta', 'xiong_mf', 'bayes_h')):
    """by_h: h -> (ids, c, r, y). Pooled fits per h (+ participant bootstrap) and per-participant fits."""
    out = dict(pooled={}, per_participant={}, signature={}, tests={})
    pp = collections.defaultdict(dict)   # model -> h -> {pid: params}
    for h, (ids, c, r, y) in by_h.items():
        out['signature'][h] = shift_signature(c, r, y)
        out['pooled'][h] = {}
        for m in models:
            f = fit(m, h, c, r, y)
            if boot and m != 'bayes_h':
                idxs = [rng.integers(0, len(ids), len(ids)) for _ in range(boot)]
                bs = [b['params'] for b in pmap([(m, h, c[i], r[i], y[i], None, f['x']) for i in idxs])]
                f['ci95'] = {k: [float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))]
                             for k in ('alpha', 'phi', 'beta', 'kappa')}
            out['pooled'][h][m] = f
        for m in ('delta', 'xiong_mf'):
            res = pmap([(m, h, c[j:j + 1], r[j:j + 1], y[j:j + 1], None, out['pooled'][h][m]['x'])
                        for j in range(len(ids))])
            for eid, f in zip(ids, res):
                pp[m].setdefault(h, {})[eps[eid]['participant']] = f['params']
    hs = sorted(by_h)
    for m, dh in pp.items():
        common = sorted(set.intersection(*[set(dh[h]) for h in hs]))
        out['per_participant'][m] = {h: {k: float(np.median([dh[h][p][k] for p in common])) for k in ('alpha', 'phi', 'beta', 'kappa')}
                                     for h in hs}
        for k in ('alpha', 'phi'):
            if k == 'phi' and m == 'delta':
                continue
            lo = np.array([dh[hs[0]][p][k] for p in common])
            hi = np.array([dh[hs[-1]][p][k] for p in common])
            w = stats.wilcoxon(hi, lo) if np.any(hi != lo) else None
            hh = np.repeat(hs, len(common))
            vals = np.concatenate([[dh[h][p][k] for p in common] for h in hs])
            # within-participant trend: Spearman of h vs parameter after removing participant means
            cent = vals - np.tile([np.mean([dh[h][p][k] for h in hs]) for p in common], len(hs))
            rho = stats.spearmanr(hh, cent)
            out['tests'][f'{m}.{k}'] = dict(n=len(common), median_diff_hmax_minus_hmin=float(np.median(hi - lo)),
                                           frac_increase=float(np.mean(hi > lo)),
                                           wilcoxon_p=None if w is None else float(w.pvalue),
                                           spearman_rho_within=float(rho.statistic), spearman_p=float(rho.pvalue))
    pool = out['pooled']
    for m in ('delta', 'xiong_mf'):
        a = [pool[h][m]['params']['alpha'] for h in hs]
        out['tests'][f'{m}.pooled_alpha_increasing'] = bool(all(x < y_ for x, y_ in zip(a, a[1:])))
        if m == 'xiong_mf':
            p = [pool[h][m]['params']['phi'] for h in hs]
            out['tests'][f'{m}.pooled_phi_increasing'] = bool(all(x < y_ for x, y_ in zip(p, p[1:])))
    return out


def load_results(patterns):
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    groups = collections.defaultdict(dict)
    for r in rows:
        if r['format'] in ('text', 'image_outcome', 'chart'):
            groups[(r['model_id'], r['format'], r['hazard_info'])][(r['episode_id'], r['t'])] = r['p_slot1']
    perc = collections.defaultdict(list)
    for r in rows:
        if r['format'].startswith('perc'):
            perc[(r['model_id'], r['format'], r['subtask'])].append(r)
    return groups, perc


def selftest():
    """Closed-loop parameter recovery on the real value sequences (no VLM)."""
    eps = load_episodes(HERE / 'data/episodes.jsonl')
    rng = np.random.default_rng(0)
    res = {}
    for h in HAZARDS:
        ids = [i for i, e in eps.items() if e['hazard'] == h]
        V = np.array([[tr['v'] for tr in eps[i]['trials']] for i in ids])   # (E, T, 2)
        for name, truth in (('normative', dict(alpha=1.0, phi=h)), ('fixed', dict(alpha=0.6, phi=0.2))):
            prm = dict(truth, beta=1.5, kappa=0.5, b=0.0)
            E, T = V.shape[:2]
            c = np.zeros((E, T), int)
            r = np.zeros((E, T))
            Q = np.full((E, 2), 50.0)
            s = np.zeros(E)
            rows = np.arange(E)
            for t in range(T):
                p = sig(prm['beta'] * (Q[:, 0] - Q[:, 1]) / 10 + prm['kappa'] * s + prm['b'])
                ch = (rng.random(E) > p).astype(int)
                c[:, t], r[:, t] = ch, V[rows, t, ch]
                Q[rows, ch] += prm['alpha'] * (r[:, t] - Q[rows, ch])
                Q[rows, 1 - ch] += prm['phi'] * (50 - Q[rows, 1 - ch])
                s = np.where(ch == 0, 1.0, -1.0)
            y_soft = forward(prm, c, r)           # also check soft-target recovery (VLM-like)
            f_hard = fit('xiong_mf', h, c, r, (c == 0).astype(float))['params']
            f_soft = fit('xiong_mf', h, c, r, y_soft)['params']
            res[f'{name}_h{h}'] = dict(truth=truth, hard={k: round(f_hard[k], 3) for k in ('alpha', 'phi', 'beta')},
                                       soft={k: round(f_soft[k], 3) for k in ('alpha', 'phi', 'beta')})
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=Path, default=HERE / 'data/episodes.jsonl')
    ap.add_argument('--results', nargs='*', default=[])
    ap.add_argument('--output', type=Path, default=HERE / 'analysis.json')
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--human-long', action='store_true', help='also fit humans on the 1500-trial games (reference)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--jobs', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    global POOL
    if args.jobs > 1:
        POOL = ProcessPoolExecutor(args.jobs)
    if args.selftest:
        selftest()
        return
    rng = np.random.default_rng(20261111)
    eps = load_episodes(args.episodes)
    report = dict(model_doc=__doc__.split('Usage:')[0], sources={})

    def human(eps_):
        by_h = {}
        for h in HAZARDS:
            ids = sorted(i for i, e in eps_.items() if e['hazard'] == h)
            by_h[h] = (ids, *arrays(eps_, ids))
        return by_h

    report['sources']['human/short100'] = analyse_source(eps, human(eps), args.boot, rng)
    if args.human_long:
        long_eps = {i: e for i, e in load_episodes(HERE / 'data/episodes_long.jsonl').items() if e['n_trials'] == 1500}
        report['sources']['human/long1500'] = analyse_source(long_eps, human(long_eps), min(args.boot, 50), rng)

    groups, perc = load_results(args.results)
    for (mid, fmt, hz), tab in sorted(groups.items()):
        by_h = {}
        for h in HAZARDS:
            ids = sorted(i for i, e in eps.items() if e['hazard'] == h
                         and all((i, t) in tab for t in range(1, e['n_trials'] + 1)))
            if ids:
                targets = {i: [tab[(i, t)] for t in range(1, eps[i]['n_trials'] + 1)] for i in ids}
                by_h[h] = (ids, *arrays(eps, ids, targets))
        if len(by_h) < 2:
            continue
        src = analyse_source(eps, by_h, args.boot, rng)
        # how well the VLM predicts the human's actual choice (same trials)
        ll = []
        for h, (ids, c, r, y) in by_h.items():
            pc = np.where(c == 0, y, 1 - y)
            ll.append(np.log(np.clip(pc, EPS, 1)).ravel())
        src['human_choice_loglik_per_trial'] = float(np.mean(np.concatenate(ll)))
        src['n_complete_episodes'] = {h: len(v[0]) for h, v in by_h.items()}
        report['sources'][f'{mid}/{fmt}/{hz}'] = src
    report['perception'] = {f'{m}/{f}/{s}': dict(n=len(v), acc=float(np.mean([r['correct_argmax'] for r in v])),
                                                 mean_p_correct=float(np.mean([r['correct_prob'] for r in v])))
                            for (m, f, s), v in sorted(perc.items())}
    args.output.write_text(json.dumps(report, indent=1, default=float))
    # compact console summary
    for name, src in report['sources'].items():
        pool = src['pooled']
        a = ' '.join(f"h{h}:a={pool[h]['xiong_mf']['params']['alpha']:.2f},phi={pool[h]['xiong_mf']['params']['phi']:.2f}"
                     for h in sorted(pool))
        sg = ' '.join(f"h{h}:{src['signature'][h]['p_shift_low']:.2f}/{src['signature'][h]['p_shift_high']:.2f}"
                      for h in sorted(src['signature']))
        print(f'{name:45s} xiong_mf pooled {a} | P(shift|low/high) {sg} | '
              f"phi trend rho={src['tests'].get('xiong_mf.phi', {}).get('spearman_rho_within', float('nan')):.2f} "
              f"p={src['tests'].get('xiong_mf.phi', {}).get('spearman_p', float('nan')):.3g}")
    for k, v in report['perception'].items():
        print('perception', k, v)


if __name__ == '__main__':
    main()
