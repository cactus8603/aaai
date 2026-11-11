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


def trend_tests(X, hs, P, rng, n_boot=2000):
    """X: (N participants, 3 hazards) parameter matrix; P: matching game positions (1..3) or None.
    One observation per participant per test (no pseudo-replication):
      * slope of the parameter on h per participant (OLS over the 3 conditions) -> Wilcoxon vs 0
      * Friedman (any difference) and Page's L (ordered increase h=0.1<0.2<0.4)
      * position covariate: within-participant (fixed-effects) OLS  x_c = b_h * h_c + b_pos * pos_c,
        CI of b_h by participant bootstrap; plus medians per (h, position) cell (position-balanced view).
    """
    N = len(X)
    hc = hs - hs.mean()
    slopes = (X - X.mean(1, keepdims=True)) @ hc / (hc @ hc)
    w = stats.wilcoxon(slopes) if np.any(slopes != 0) else None
    fr = stats.friedmanchisquare(*X.T) if N > 2 else None
    pg = stats.page_trend_test(X) if N > 2 else None
    out = dict(n=N, median_slope_per_unit_h=float(np.median(slopes)), frac_positive_slope=float(np.mean(slopes > 0)),
               slope_wilcoxon_p=None if w is None else float(w.pvalue),
               friedman_p=None if fr is None else float(fr.pvalue),
               page_L=None if pg is None else float(pg.statistic), page_p=None if pg is None else float(pg.pvalue),
               median_diff_hmax_minus_hmin=float(np.median(X[:, -1] - X[:, 0])))
    if P is not None and not np.any(P == None):  # noqa: E711
        P = P.astype(float)

        def fe(idx):
            Xc = X[idx] - X[idx].mean(1, keepdims=True)
            Hc = np.tile(hs, (len(idx), 1)) - hs.mean()
            Pc = P[idx] - P[idx].mean(1, keepdims=True)
            A = np.stack([Hc.ravel(), Pc.ravel()], 1)
            return np.linalg.lstsq(A, Xc.ravel(), rcond=None)[0]
        b = fe(np.arange(N))
        bs = np.array([fe(rng.integers(0, N, N)) for _ in range(n_boot)])
        out['fe_position_adjusted'] = dict(b_h=float(b[0]), b_h_ci95=[float(v) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                                           b_h_boot_p_le0=float(np.mean(bs[:, 0] <= 0)),
                                           b_position=float(b[1]), b_position_ci95=[float(v) for v in np.percentile(bs[:, 1], [2.5, 97.5])])
        out['median_by_h_position'] = {f'h{h}_pos{q}': float(np.median(X[:, j][P[:, j] == q])) if np.any(P[:, j] == q) else None
                                       for j, h in enumerate(hs) for q in (1, 2, 3)}
    return out


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
    pos = {(eps[i]['participant'], h): eps[i].get('position') for h, (ids, *_) in by_h.items() for i in ids}
    out['per_participant_params'] = {m: {str(h): {str(p_): v for p_, v in dh[h].items()} for h in hs} for m, dh in pp.items()}
    for m, dh in pp.items():
        common = sorted(set.intersection(*[set(dh[h]) for h in hs]))
        out['per_participant'][m] = {h: {k: float(np.median([dh[h][p_][k] for p_ in common])) for k in ('alpha', 'phi', 'beta', 'kappa')}
                                     for h in hs}
        for k in ('alpha', 'phi'):
            if k == 'phi' and m == 'delta':
                continue
            out['tests'][f'{m}.{k}'] = trend_tests(np.array([[dh[h][p_][k] for h in hs] for p_ in common]),
                                                   np.array(hs), np.array([[pos[(p_, h)] for h in hs] for p_ in common]),
                                                   rng)
    pool = out['pooled']
    for m in ('delta', 'xiong_mf'):
        a = [pool[h][m]['params']['alpha'] for h in hs]
        out['tests'][f'{m}.pooled_alpha_increasing'] = bool(all(x < y_ for x, y_ in zip(a, a[1:])))
        if m == 'xiong_mf':
            p = [pool[h][m]['params']['phi'] for h in hs]
            out['tests'][f'{m}.pooled_phi_increasing'] = bool(all(x < y_ for x, y_ in zip(p, p[1:])))
    return out


def read_jsonl(path):
    """Rows of a results file; a truncated LAST line (crash mid-write) is skipped, anything else is an error."""
    lines = Path(path).read_text().split('\n')
    rows = []
    for j, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if j == len(lines) - 1:           # no trailing newline -> partial last row
                print(f'warning: skipping truncated last line of {path}')
            else:
                raise
    return rows


def load_results(patterns):
    """Groups: (model, format, hazard_info, image_px, n_img) -> {(episode, t): P(slot1)}; duplicates are errors."""
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            rows += read_jsonl(f)
    groups = collections.defaultdict(dict)
    perc = collections.defaultdict(dict)
    for r in rows:
        if r['format'] in ('text', 'image_outcome', 'chart'):
            g = (r['model_id'], r['format'], r['hazard_info'], r.get('image_px'), r.get('n_img'))
            key = (r['episode_id'], r['t'])
            assert key not in groups[g], f'duplicate row {key} in group {g}'
            groups[g][key] = r['p_slot1']
        elif r['format'].startswith('perc'):
            g = (r['model_id'], r['format'], r['subtask'], r.get('image_px'))
            assert r['item'] not in perc[g], f"duplicate perception item {r['item']} in {g}"
            perc[g][r['item']] = r
    return groups, {g: list(v.values()) for g, v in perc.items()}


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
    res['null_constant_phi'] = null_trend_test(eps)
    print(json.dumps(res, indent=1))
    return res


def null_trend_test(eps, seeds=(0, 1, 2, 3, 4), prm=dict(alpha=1.0, phi=0.2, beta=0.4, kappa=0.2, b=0.0)):
    """Null check: every simulated participant has the SAME phi = 0.2 at all h (human-like beta ~ 0.4).
    Closed-loop simulation on each participant's real value sequences, per-participant fits, then the
    same trend tests as for real data. A valid pipeline should give no spurious h trend (p > 0.05)."""
    out = {}
    pids = sorted({e['participant'] for e in eps.values()})
    hs = np.array(HAZARDS)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X = np.zeros((len(pids), len(hs)))
        jobs = []
        for j, h in enumerate(hs):
            ids = [next(i for i, e in eps.items() if e['participant'] == p_ and e['hazard'] == h) for p_ in pids]
            V = np.array([[tr['v'] for tr in eps[i]['trials']] for i in ids])
            E, T = V.shape[:2]
            c, r = np.zeros((E, T), int), np.zeros((E, T))
            Q, s, rows = np.full((E, 2), 50.0), np.zeros(E), np.arange(E)
            for t in range(T):
                p = sig(prm['beta'] * (Q[:, 0] - Q[:, 1]) / 10 + prm['kappa'] * s + prm['b'])
                ch = (rng.random(E) > p).astype(int)
                c[:, t], r[:, t] = ch, V[rows, t, ch]
                Q[rows, ch] += prm['alpha'] * (r[:, t] - Q[rows, ch])
                Q[rows, 1 - ch] += prm['phi'] * (50 - Q[rows, 1 - ch])
                s = np.where(ch == 0, 1.0, -1.0)
            jobs += [('xiong_mf', float(h), c[k:k + 1], r[k:k + 1], (c[k:k + 1] == 0).astype(float)) for k in range(E)]
        fits = pmap(jobs)
        for idx, f in enumerate(fits):
            X[idx % len(pids), idx // len(pids)] = f['params']['phi']
        tt = trend_tests(X, hs, None, rng)
        out[f'seed{seed}'] = dict(median_phi={str(h): float(np.median(X[:, j])) for j, h in enumerate(hs)},
                                  slope_wilcoxon_p=tt['slope_wilcoxon_p'], page_p=tt['page_p'], friedman_p=tt['friedman_p'])
    out['n_seeds_slope_p_gt_0.05'] = sum(v['slope_wilcoxon_p'] > 0.05 for k, v in out.items() if k.startswith('seed'))
    out['n_seeds_page_p_gt_0.05'] = sum(v['page_p'] > 0.05 for k, v in out.items() if k.startswith('seed'))
    return out


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
        # side-by-side: humans on the replayed 100-trial games vs their 1500-trial games (same h)
        report['human_short_vs_long'] = {
            n: {'pooled_phi': {h: src['pooled'][h]['xiong_mf']['params']['phi'] for h in src['pooled']},
                'pooled_alpha': {h: src['pooled'][h]['xiong_mf']['params']['alpha'] for h in src['pooled']},
                'median_phi': {h: v['phi'] for h, v in src['per_participant']['xiong_mf'].items()},
                'phi_tests': src['tests']['xiong_mf.phi']}
            for n, src in report['sources'].items() if n.startswith('human/')}

    groups, perc = load_results(args.results)
    for (mid, fmt, hz, px, nimg), tab in sorted(groups.items(), key=str):
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
        report['sources'][f'{mid}/{fmt}/{hz}/px{px}' + (f'/n{nimg}' if fmt == 'image_outcome' else '')] = src
    report['perception'] = {f'{m}/{f}/{s}/px{px_}': dict(n=len(v), acc=float(np.mean([r['correct_argmax'] for r in v])),
                                                 mean_p_correct=float(np.mean([r['correct_prob'] for r in v])))
                            for (m, f, s, px_), v in sorted(perc.items(), key=str)}
    args.output.write_text(json.dumps(report, indent=1, default=float))
    # compact console summary
    for name, src in report['sources'].items():
        pool = src['pooled']
        a = ' '.join(f"h{h}:a={pool[h]['xiong_mf']['params']['alpha']:.2f},phi={pool[h]['xiong_mf']['params']['phi']:.2f}"
                     for h in sorted(pool))
        sg = ' '.join(f"h{h}:{src['signature'][h]['p_shift_low']:.2f}/{src['signature'][h]['p_shift_high']:.2f}"
                      for h in sorted(src['signature']))
        t_ = src['tests'].get('xiong_mf.phi', {})
        fe_ = t_.get('fe_position_adjusted', {})
        pm = ' '.join(f"{src['per_participant']['xiong_mf'][h]['phi']:.2f}" for h in sorted(pool))
        print(f'{name:45s} xiong_mf pooled {a} | per-participant median phi {pm} | P(shift|low/high) {sg} | '
              f"phi slope Wilcoxon p={t_.get('slope_wilcoxon_p')} Page p={t_.get('page_p')} "
              f"pos-adjusted b_h={fe_.get('b_h', float('nan')):.3f} CI={fe_.get('b_h_ci95')}")
    for k, v in report['perception'].items():
        print('perception', k, v)


if __name__ == '__main__':
    main()
