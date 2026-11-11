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

Human learning-rate fits are OUR re-analysis; Xiong et al. (2023) report no per-h learning rates.
Model and human share only the within-game choice/feedback history: humans were told h and had prior task
experience, so model-hidden vs human-told is a descriptive cross-condition difference (see PROTOCOL.md).

Model-agnostic signature (paper Fig. 1g): P(shift | last reward < 50) and P(shift | last reward >= 50)
per hazard; humans show more shifting after low rewards when h is higher.

Usage:
  .venv/bin/python research/volatile_trust/psych101/analyze_xiong.py --results 'research/volatile_trust/psych101/runs/*/results_*.jsonl' \
      --output research/volatile_trust/psych101/analysis.json [--boot 200] [--human-long]
  .venv/bin/python research/volatile_trust/psych101/analyze_xiong.py --selftest
  .venv/bin/python research/volatile_trust/psych101/analyze_xiong.py --calibrate 200 --jobs 40   (H02 calibration)
"""
import argparse
import collections
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # tiny arrays: BLAS threads only add overhead

import numpy as np
from scipy import optimize, special, stats

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


BOUNDS = {'alpha': (-8, 8), 'phi': (-8, 8), 'beta': (-8, 6), 'kappa': (-10, 10), 'b': (-10, 10)}
BOUND_TOL = 1e-3      # |x - bound| below this (unconstrained scale) = boundary fit


def nll(x, name, h, c, r, y, mask):
    """Reference implementation (sequential forward); fit() uses the equivalent closed form in nll_grad()."""
    P = np.clip(forward(unpack(name, x, h), c, r), EPS, 1 - EPS)
    return -float(np.sum(mask * (y * np.log(P) + (1 - y) * np.log(1 - P))))


def prep(c, r):
    """History-only quantities for the closed-form Q recursion. With q = Q - 50, an unchosen update is
    q <- (1 - phi) q and a chosen one q <- (1 - alpha) q + alpha (r - 50), so
    q_k(t) = alpha * sum_{j<t, c_j=k} (r_j - 50) (1-alpha)^{DA_k[t,j]} (1-phi)^{DB_k[t,j]},
    DA/DB = number of chosen/unchosen updates of machine k strictly between j and t."""
    E, T = c.shape
    ti = np.arange(T)
    low = (ti[None, :] < ti[:, None]).astype(float)                      # [t, j] = j < t
    gap = (ti[:, None] - ti[None, :] - 1) * low
    ks = []
    for k in (0, 1):
        ch = (c == k)
        A = np.concatenate([np.zeros((E, 1)), np.cumsum(ch, 1)], 1)        # A[:, t] = #chosen-k before t
        DA = (A[:, :T, None] - A[:, None, 1:]) * low
        ks.append((DA, gap[None] - DA, ((r - 50) * ch)[..., None]))
    s = np.zeros((E, T))
    s[:, 1:] = np.where(c[:, :-1] == 0, 1.0, -1.0)
    return dict(k=ks, low=low, s=s)


CLOSED_FORM_MAX = 2_000_000     # E*T*T above this (e.g. 1500-trial games): O(T)-memory sequential gradient


def prep_any(c, r):
    E, T = c.shape
    return prep(c, r) if E * T * T <= CLOSED_FORM_MAX else dict(seq=True, c=c, r=r)


def nll_grad_seq(x, name, h, pre, y, mask):
    """Same NLL and gradient as nll_grad, by forward-mode differentiation of the sequential recursion."""
    c, r = pre['c'], pre['r']
    p = dict(zip(MODELS[name], x))
    alpha = sig(p['alpha']) if 'alpha' in p else 1.0
    phi = sig(p['phi']) if 'phi' in p else (h if name == 'bayes_h' else 0.0)
    beta = np.logaddexp(0, p['beta'])
    kappa, b = p.get('kappa', 0.0), p.get('b', 0.0)
    da, dp = alpha * (1 - alpha), phi * (1 - phi)          # d alpha / d x_alpha, d phi / d x_phi
    E, T = c.shape
    rows = np.arange(E)
    Q, Qa, Qp, s = np.full((E, 2), 50.0), np.zeros((E, 2)), np.zeros((E, 2)), np.zeros(E)
    f, g = 0.0, dict(alpha=0.0, phi=0.0, beta=0.0, kappa=0.0, b=0.0)
    for t in range(T):
        dq = Q[:, 0] - Q[:, 1]
        P = sig(beta * dq / 10 + kappa * s + b)
        Pc = np.clip(P, EPS, 1 - EPS)
        yt, mt = y[:, t], mask[:, t]
        f -= float(np.sum(mt * (yt * np.log(Pc) + (1 - yt) * np.log(1 - Pc))))
        gz = mt * (P - yt) * ((P > EPS) & (P < 1 - EPS))
        g['b'] += np.sum(gz)
        g['kappa'] += np.sum(gz * s)
        g['beta'] += np.sum(gz * sig(p['beta']) * dq / 10)
        g['alpha'] += np.sum(gz * beta / 10 * (Qa[:, 0] - Qa[:, 1]))
        g['phi'] += np.sum(gz * beta / 10 * (Qp[:, 0] - Qp[:, 1]))
        ch = c[:, t]
        u = 1 - ch
        err = r[:, t] - Q[rows, ch]
        Qa[rows, ch] = (1 - alpha) * Qa[rows, ch] + ('alpha' in p) * da * err
        Qp[rows, ch] = (1 - alpha) * Qp[rows, ch]
        Q[rows, ch] += alpha * err
        dec = 50 - Q[rows, u]
        Qa[rows, u] = (1 - phi) * Qa[rows, u]
        Qp[rows, u] = (1 - phi) * Qp[rows, u] + ('phi' in p) * dp * dec
        Q[rows, u] += phi * dec
        s = np.where(ch == 0, 1.0, -1.0)
    return f, np.array([g[k] for k in MODELS[name]], dtype=float)


def nll_grad(x, name, h, pre, y, mask):
    """Exact NLL (same clipping as nll()) and its analytic gradient on the unconstrained scale."""
    if pre.get('seq'):
        return nll_grad_seq(x, name, h, pre, y, mask)
    p = dict(zip(MODELS[name], x))
    alpha = sig(p['alpha']) if 'alpha' in p else 1.0
    phi = sig(p['phi']) if 'phi' in p else (h if name == 'bayes_h' else 0.0)
    beta = np.logaddexp(0, p['beta'])
    kappa, b = p.get('kappa', 0.0), p.get('b', 0.0)
    lp = np.log1p(-phi)
    q, dqa, dqp = [], [], []
    for DA, DB, W in pre['k']:
        if alpha < 1:
            Em = np.exp(np.log1p(-alpha) * DA + lp * DB) * pre['low']
        else:                                      # alpha = 1 (bayes_h): only DA = 0 terms survive
            Em = (DA == 0) * np.exp(lp * DB) * pre['low']
        S0 = (Em @ W)[..., 0]
        q.append(alpha * S0)
        if 'alpha' in p:
            dqa.append(alpha * (1 - alpha) * S0 - alpha ** 2 * ((Em * DA) @ W)[..., 0])
        if 'phi' in p:
            dqp.append(-alpha * phi * ((Em * DB) @ W)[..., 0])
    dq = q[0] - q[1]
    P = sig(beta * dq / 10 + kappa * pre['s'] + b)
    Pc = np.clip(P, EPS, 1 - EPS)
    f = -float(np.sum(mask * (y * np.log(Pc) + (1 - y) * np.log(1 - Pc))))
    gz = mask * (P - y) * ((P > EPS) & (P < 1 - EPS))
    g = {'b': np.sum(gz), 'kappa': np.sum(gz * pre['s']), 'beta': np.sum(gz * sig(p['beta']) * dq / 10)}
    if 'alpha' in p:
        g['alpha'] = np.sum(gz * beta / 10 * (dqa[0] - dqa[1]))
    if 'phi' in p:
        g['phi'] = np.sum(gz * beta / 10 * (dqp[0] - dqp[1]))
    return f, np.array([g[k] for k in MODELS[name]], dtype=float)


def fit(name, h, c, r, y, mask=None, x0=None):
    """Multi-start L-BFGS-B (analytic gradient); with x0 (warm start, e.g. bootstrap / sampled draws) one extra
    start is added first. Returns params plus optimizer diagnostics: converged (scipy success of the best
    start), at_bound (parameters within BOUND_TOL of a box bound, unconstrained scale)."""
    mask = np.ones_like(y) if mask is None else mask
    starts = [] if x0 is None else [list(x0)]
    for a in ((0.5, 0.95) if x0 is None else (0.8,)):
        for ph in ((0.05, 0.3) if x0 is None else (0.15,)):
            st = {'alpha': logit(a), 'phi': logit(ph), 'beta': 1.0, 'kappa': 0.5, 'b': 0.0}
            starts.append([st[k] for k in MODELS[name]])
    pre = prep_any(c, r)
    bounds = [BOUNDS[k] for k in MODELS[name]]
    best, n_conv = None, 0
    for x0_ in starts:
        res = optimize.minimize(nll_grad, x0_, args=(name, h, pre, y, mask), jac=True, method='L-BFGS-B',
                                bounds=bounds)
        n_conv += bool(res.success)
        if best is None or res.fun < best.fun:
            best = res
        if name == 'bayes_h':
            break
    prm = unpack(name, best.x, h)
    n = float(mask.sum())
    k = len(MODELS[name])
    ent = -float(np.sum(mask * (y * np.log(np.clip(y, EPS, 1)) + (1 - y) * np.log(np.clip(1 - y, EPS, 1)))))
    at_bound = [f"{q}_{'lo' if abs(v - lo) < BOUND_TOL else 'hi'}" for q, v, (lo, hi) in zip(MODELS[name], best.x, bounds)
                if abs(v - lo) < BOUND_TOL or abs(v - hi) < BOUND_TOL]
    return dict(params={k_: float(v) for k_, v in prm.items()}, nll=float(best.fun), n=n,
                bic=k * np.log(n) + 2 * best.fun, excess_nll=float(best.fun) - ent, x=[float(v) for v in best.x],
                diag=dict(converged=bool(best.success), n_starts=len(starts), n_starts_converged=n_conv,
                          nit=int(best.nit), at_bound=at_bound))


def diag_summary(fits):
    """Failed / boundary-fit rates over a list of fit() results."""
    n = len(fits)
    if not n:
        return dict(n=0)
    tags = collections.Counter(t for f in fits for t in f['diag']['at_bound'])
    return dict(n=n, frac_failed=float(np.mean([not f['diag']['converged'] for f in fits])),
                frac_any_boundary=float(np.mean([bool(f['diag']['at_bound']) for f in fits])),
                frac_boundary_excl_alpha_hi=float(np.mean([bool(set(f['diag']['at_bound']) - {'alpha_hi'}) for f in fits])),
                frac_phi_or_beta_boundary=float(np.mean([any(t.split('_')[0] in ('phi', 'beta') for t in f['diag']['at_bound'])
                                                          for f in fits])),
                frac_by_bound={t: v / n for t, v in sorted(tags.items())},
                note='alpha_hi (alpha -> 1) is the normative value for noise-free rewards, not a failure')


BASE_SEED = 20261111


def rng_for(*parts):
    """Independent, fixed RNG per (condition, purpose, episode ...): adding a condition never changes another."""
    hsh = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return np.random.default_rng([BASE_SEED, int.from_bytes(hsh[:8], 'little')])


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
    """X: (N participants, K hazards) parameter matrix; P: matching game positions (1..3) or None.
    One observation per participant per test (no pseudo-replication):
      * slope of the parameter on h per participant (OLS over the K conditions) -> Wilcoxon vs 0
      * Friedman (any difference) and Page's L (ordered increase h=0.1<0.2<0.4); need K >= 3, else None
      * position covariate: within-participant (fixed-effects) OLS  x_c = b_h * h_c + b_pos * pos_c,
        CI of b_h by participant bootstrap; plus medians per (h, position) cell (position-balanced view).
    """
    N, K = X.shape
    hc = hs - hs.mean()
    slopes = (X - X.mean(1, keepdims=True)) @ hc / (hc @ hc)
    w = stats.wilcoxon(slopes) if np.any(slopes != 0) else None
    fr = stats.friedmanchisquare(*X.T) if N > 2 and K >= 3 else None
    pg = stats.page_trend_test(X) if N > 2 and K >= 3 else None
    out = dict(n=N, n_hazards=K, median_slope_per_unit_h=float(np.median(slopes)), frac_positive_slope=float(np.mean(slopes > 0)),
               slope_wilcoxon_p=None if w is None else float(w.pvalue),
               friedman_p=None if fr is None else float(fr.pvalue),
               page_L=None if pg is None else float(pg.statistic), page_p=None if pg is None else float(pg.pvalue),
               median_diff_hmax_minus_hmin=float(np.median(X[:, -1] - X[:, 0])))
    if K < 3:
        out['note'] = 'Friedman / Page need >= 3 hazard conditions: not computed'
    if P is not None and not np.any(P == None) and K >= 3:  # noqa: E711
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


def analyse_source(name, eps, by_h, boot, models=('delta', 'xiong_mf', 'bayes_h')):
    """by_h: h -> (ids, c, r, y). Pooled fits per h (+ participant bootstrap) and per-participant fits.
    Every random draw uses rng_for(name, ...), so results do not depend on which other sources are analysed."""
    out = dict(pooled={}, per_participant={}, signature={}, tests={}, fit_diagnostics={})
    pp = collections.defaultdict(dict)   # model -> h -> {pid: params}
    for h, (ids, c, r, y) in by_h.items():
        out['signature'][h] = shift_signature(c, r, y)
        out['pooled'][h] = {}
        for m in models:
            f = fit(m, h, c, r, y)
            if boot and m != 'bayes_h':
                rng = rng_for(name, 'pooled_boot', h, m)
                idxs = [rng.integers(0, len(ids), len(ids)) for _ in range(boot)]
                bfits = pmap([(m, h, c[i], r[i], y[i], None, f['x']) for i in idxs])
                bs = [b['params'] for b in bfits]
                f['ci95'] = {k: [float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))]
                             for k in ('alpha', 'phi', 'beta', 'kappa')}
                f['boot_diagnostics'] = diag_summary(bfits)
            out['pooled'][h][m] = f
        for m in ('delta', 'xiong_mf'):
            res = pmap([(m, h, c[j:j + 1], r[j:j + 1], y[j:j + 1], None, out['pooled'][h][m]['x'])
                        for j in range(len(ids))])
            out['fit_diagnostics'][f'per_participant.{m}.h{h}'] = diag_summary(res)
            for eid, f in zip(ids, res):
                pp[m].setdefault(h, {})[eps[eid]['participant']] = dict(f['params'], converged=f['diag']['converged'],
                                                                        at_bound=f['diag']['at_bound'])
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
                                                   rng_for(name, 'trend', m, k))
    out['fit_diagnostics']['all_per_participant'] = diag_summary(
        [dict(diag=dict(converged=v['converged'], at_bound=v['at_bound'])) for dh in pp.values() for d in dh.values() for v in d.values()])
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
    """Groups: (model, format, hazard_info, image_px, n_img) -> {(episode, t): P(slot1)}; duplicates are errors.
    P(slot1) = sigmoid(logit_diff) from the saved raw letter logits when present (exact, no saturation);
    older rows without logits fall back to the stored probability. Returns also the readout counts per group."""
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            rows += read_jsonl(f)
    groups = collections.defaultdict(dict)
    readout = collections.defaultdict(collections.Counter)
    fingerprints = collections.defaultdict(set)
    perc = collections.defaultdict(dict)
    for r in rows:
        if r['format'] in ('text', 'image_outcome', 'chart'):
            g = (r['model_id'], r['format'], r['hazard_info'], r.get('image_px'), r.get('n_img'))
            key = (r['episode_id'], r['t'])
            assert key not in groups[g], f'duplicate row {key} in group {g}'
            if r.get('logit_diff') is not None:
                groups[g][key] = float(special.expit(r['logit_diff']))
                readout[g]['logit_diff'] += 1
            else:
                groups[g][key] = r['p_slot1']
                readout[g]['probability'] += 1
            fingerprints[g].add(r.get('run_fingerprint'))
        elif r['format'].startswith('perc'):
            g = (r['model_id'], r['format'], r['subtask'], r.get('image_px'))
            assert r['item'] not in perc[g], f"duplicate perception item {r['item']} in {g}"
            perc[g][r['item']] = r
    for g, fps in fingerprints.items():
        assert len(fps) == 1, f'group {g} mixes run fingerprints {fps}: analyse them separately'
    return groups, {g: list(v.values()) for g, v in perc.items()}, {g: dict(v) for g, v in readout.items()}


# ------------------------------------------------------------------ primary model analysis
PRIMARY_TEST = ('Per-participant OLS slope of phi (xiong_mf) on h, model minus human, paired over the participants '
                '(same within-game choice and feedback histories). Model phi = median over S sampled binary choice '
                'sequences per episode; human phi = fit to the actual binary choices. Wilcoxon signed-rank on the '
                'paired differences + participant-bootstrap 95% CI of the median difference. Primary model condition: '
                'hazard_info = hidden. Humans were TOLD h and had prior task experience (1,600 / 3,200 trials before '
                'games 3 / 5), so the hidden-model vs told-human difference is a DESCRIPTIVE cross-condition difference, '
                'not by itself evidence of a human-model learning gap. Reported with it: model told vs human told '
                '(hazard information matched, experience still not) and the same-model told - hidden contrast.')
COMPARISON = {
    'hidden': 'descriptive cross-condition difference: model hazard-HIDDEN vs human hazard-TOLD (hazard information '
              'and prior experience both differ)',
    'told': 'hazard-information-matched: model hazard-TOLD vs human hazard-TOLD (prior experience still differs)'}
HUMAN_PROVENANCE = ('Human learning-rate (phi) fits and trend tests are OUR re-analysis of the Psych-101 / xiong2023 '
                    'data; Xiong et al. (2023) do not report per-hazard learning rates (their human finding is more '
                    'shifting after low rewards when h is higher).')


def sampled_fits(cond, eps, by_h, n_samples, x0_by_h=None):
    """Binary-choice fits comparable to humans: for each episode draw n_samples choice sequences
    c_s ~ Bernoulli(P_model) (history stays the human's), fit xiong_mf to each, take the per-episode median.
    Draws for an episode come from rng_for(cond, 'draws', episode): independent of every other episode/condition,
    and the first k draws are the same for any n_samples >= k. Returns (median params, all draws, all fits)."""
    jobs, keys = [], []
    for h, (ids, c, r, y) in by_h.items():
        for j, eid in enumerate(ids):
            rng = rng_for(cond, 'draws', eid)
            for _ in range(n_samples):
                ys = (rng.random(y.shape[1]) < y[j]).astype(float)[None]
                jobs.append(('xiong_mf', h, c[j:j + 1], r[j:j + 1], ys, None, None if x0_by_h is None else x0_by_h[h]))
                keys.append((h, eps[eid]['participant']))
    acc = collections.defaultdict(list)
    fits = pmap(jobs)
    for k, f in zip(keys, fits):
        acc[k].append(f['params'])
    out = collections.defaultdict(dict)
    for (h, pid), lst in acc.items():
        out[h][pid] = {q: float(np.median([d[q] for d in lst])) for q in ('alpha', 'phi', 'beta', 'kappa', 'b')}
    return out, dict(acc), fits


def slopes_of(pp, hs, pids, k='phi'):
    X = np.array([[pp[h][p_][k] for h in hs] for p_ in pids])
    hc = np.asarray(hs) - np.mean(hs)
    return (X - X.mean(1, keepdims=True)) @ hc / (hc @ hc)


def paired_test(a, b, rng, n_boot=5000):
    d = np.asarray(a) - np.asarray(b)
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    bs = np.median(d[rng.integers(0, len(d), (n_boot, len(d)))], 1)
    return dict(n=len(d), median_diff=float(np.median(d)), mean_diff=float(np.mean(d)),
                median_diff_ci95=[float(v) for v in np.percentile(bs, [2.5, 97.5])],
                frac_positive=float(np.mean(d > 0)), wilcoxon_p=None if w is None else float(w.pvalue))


def mc_error(draws, hs, pids, human_slopes, cond, n_rep=200):
    """Monte Carlo error of the S-draw median: re-sample the S draws of every episode with replacement,
    recompute the per-episode medians, model slopes and the paired model - human statistics."""
    rng = rng_for(cond, 'mc_error')
    phis = {k: np.array([d['phi'] for d in v]) for k, v in draws.items()}
    hc = np.asarray(hs) - np.mean(hs)
    med, mean, pv, ep_sd = [], [], [], []
    reps = {k: np.median(v[rng.integers(0, len(v), (n_rep, len(v)))], 1) for k, v in phis.items()}
    for k, v in reps.items():
        ep_sd.append(np.std(v))
    for b in range(n_rep):
        X = np.array([[reps[(h, p_)][b] for h in hs] for p_ in pids])
        d = (X - X.mean(1, keepdims=True)) @ hc / (hc @ hc) - human_slopes
        med.append(np.median(d))
        mean.append(np.mean(d))
        pv.append(stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
    half = len(next(iter(phis.values()))) // 2
    split = {}
    for nm, sl in (('first_half', slice(0, half)), ('second_half', slice(half, None))):
        X = np.array([[np.median(phis[(h, p_)][sl]) for h in hs] for p_ in pids])
        split[nm] = float(np.median((X - X.mean(1, keepdims=True)) @ hc / (hc @ hc) - human_slopes))
    return dict(method='within-episode bootstrap of the S draws (n_rep=%d)' % n_rep,
                median_diff_mc_sd=float(np.std(med)), mean_diff_mc_sd=float(np.std(mean)),
                wilcoxon_p_mc_range_5_95=[float(v) for v in np.percentile(pv, [5, 95])],
                episode_phi_median_mc_sd_median=float(np.median(ep_sd)),
                split_half_median_diff=split)


def simulate_fit(eps, phi_of_h, beta_of_h, kappa_of_h, seed, alpha=1.0):
    """Closed-loop simulation on each participant's real value sequences; per-participant xiong_mf fits.
    Returns (pids, X[pid, h] fitted phi, B[pid, h] fitted beta)."""
    rng = np.random.default_rng(seed)
    pids = sorted({e['participant'] for e in eps.values()})
    jobs = []
    for h in HAZARDS:
        ids = [next(i for i, e in eps.items() if e['participant'] == p_ and e['hazard'] == h) for p_ in pids]
        V = np.array([[tr['v'] for tr in eps[i]['trials']] for i in ids])
        E, T = V.shape[:2]
        c, r = np.zeros((E, T), int), np.zeros((E, T))
        Q, s_, rows = np.full((E, 2), 50.0), np.zeros(E), np.arange(E)
        for t in range(T):
            p = sig(beta_of_h(h) * (Q[:, 0] - Q[:, 1]) / 10 + kappa_of_h(h) * s_)
            ch = (rng.random(E) > p).astype(int)
            c[:, t], r[:, t] = ch, V[rows, t, ch]
            Q[rows, ch] += alpha * (r[:, t] - Q[rows, ch])
            Q[rows, 1 - ch] += phi_of_h(h) * (50 - Q[rows, 1 - ch])
            s_ = np.where(ch == 0, 1.0, -1.0)
        jobs += [('xiong_mf', float(h), c[k:k + 1], r[k:k + 1], (c[k:k + 1] == 0).astype(float)) for k in range(E)]
    fits = pmap(jobs)
    n = len(pids)
    X = np.array([[fits[j * n + k]['params']['phi'] for j in range(len(HAZARDS))] for k in range(n)])
    B = np.array([[fits[j * n + k]['params']['beta'] for j in range(len(HAZARDS))] for k in range(n)])
    return pids, X, B


def recovery_at_beta(eps, beta_by_h, kappa_by_h, seed=7):
    """Is phi identifiable at this (model's) beta? (a) phi = h -> recovered medians; (b) phi = 0.2 constant ->
    slope test must not be significant. beta/kappa per h = the model's median sampled-fit values."""
    hs = np.array(HAZARDS)
    out = {}
    for name, phi_fn in (('phi_eq_h', lambda h: h), ('phi_const_0.2', lambda h: 0.2)):
        _, X, B = simulate_fit(eps, phi_fn, lambda h: beta_by_h[h], lambda h: kappa_by_h[h], seed)
        tt = trend_tests(X, hs, None, np.random.default_rng(seed), n_boot=10)
        out[name] = dict(median_phi={str(h): float(np.median(X[:, j])) for j, h in enumerate(hs)},
                         median_beta={str(h): float(np.median(B[:, j])) for j, h in enumerate(hs)},
                         slope_wilcoxon_p=tt['slope_wilcoxon_p'], median_slope=tt['median_slope_per_unit_h'])
    return out


# ------------------------------------------------------------------ H02: calibration of the paired slope test
_CAL = {}
CAL_PHI = (0.01, 0.9)       # truth phi clipped to this range (human fits at the phi boundary)
CAL_BETA = (0.05, 4.0)
CAL_KB = 3.0                # |kappa|, |b| clipped to this


def _cal_init(episodes_path, truths):
    _CAL['eps'] = load_episodes(episodes_path)
    _CAL['truths'] = truths


def _simulate_one(th, V, rng):
    T = V.shape[0]
    c, r = np.zeros((1, T), int), np.zeros((1, T))
    Q, s_ = np.array([50.0, 50.0]), 0.0
    for t in range(T):
        p = sig(th['beta'] * (Q[0] - Q[1]) / 10 + th['kappa'] * s_ + th['b'])
        ch = int(rng.random() > p)
        c[0, t], r[0, t] = ch, V[t, ch]
        Q[ch] += th['alpha'] * (r[0, t] - Q[ch])
        Q[1 - ch] += th['phi'] * (50 - Q[1 - ch])
        s_ = 1.0 if ch == 0 else -1.0
    return c, r


def calib_task(task):
    """One replicate: simulated humans (known strategy, closed loop) -> complete human fit (pooled warm start,
    per-participant fits); model = the same strategy (phi shifted by delta * (h - mean h)) replayed teacher-forced
    on the simulated human's history -> pooled soft fit warm start -> S-draw sampled fits -> paired slope test."""
    seed, delta, n_samples = task
    eps, truths = _CAL['eps'], _CAL['truths']
    hs = list(HAZARDS)
    hbar = float(np.mean(hs))
    pids = sorted({e['participant'] for e in eps.values() if e['n_trials'] == 100})
    hum, mod, single, true_h, true_m, fits = {}, {}, {}, {}, {}, []
    for h in hs:
        ids = [next(i for i, e in eps.items() if e['participant'] == p_ and e['hazard'] == h) for p_ in pids]
        C, R, PM = [], [], []
        for p_, eid in zip(pids, ids):
            th = truths[(p_, h)]
            V = np.array([tr['v'] for tr in eps[eid]['trials']])
            c, r = _simulate_one(th, V, rng_for('calib', seed, 'human', eid))
            thm = dict(th, phi=float(np.clip(th['phi'] + delta * (h - hbar), 0.005, 0.95)))
            C.append(c[0]), R.append(r[0]), PM.append(forward(thm, c, r)[0])
            true_h[(h, p_)], true_m[(h, p_)] = th['phi'], thm['phi']
        C, R, PM = np.array(C), np.array(R), np.array(PM)
        x0h = fit('xiong_mf', h, C, R, (C == 0).astype(float))['x']             # human pipeline: pooled warm start
        x0m = fit('xiong_mf', h, C, R, PM)['x']                                 # model pipeline: pooled soft fit
        for j, (p_, eid) in enumerate(zip(pids, ids)):
            fh = fit('xiong_mf', h, C[j:j + 1], R[j:j + 1], (C[j:j + 1] == 0).astype(float), None, x0h)
            hum[(h, p_)] = fh['params']['phi']
            rng = rng_for('calib', seed, 'draws', eid)
            ph = []
            for s in range(n_samples):
                ys = (rng.random(R.shape[1]) < PM[j]).astype(float)[None]
                fm = fit('xiong_mf', h, C[j:j + 1], R[j:j + 1], ys, None, x0m)
                ph.append(fm['params']['phi'])
                fits.append(fm)
            fits.append(fh)
            mod[(h, p_)], single[(h, p_)] = float(np.median(ph)), ph[0]

    def sl(dct):
        X = np.array([[dct[(h, p_)] for h in hs] for p_ in pids])
        hc = np.array(hs) - hbar
        return (X - X.mean(1, keepdims=True)) @ hc / (hc @ hc)
    true_d = sl(true_m) - sl(true_h)
    hum_sl = sl(hum)
    out = dict(seed=seed, delta=delta, true_median_diff=float(np.median(true_d)),
               true_human_slope_mean=float(np.mean(sl(true_h))), est_human_slope_mean=float(np.mean(hum_sl)),
               true_model_slope_mean=float(np.mean(sl(true_m))), est_model_slope_mean=float(np.mean(sl(mod))),
               diag=dict(n=len(fits), failed=sum(not f['diag']['converged'] for f in fits),
                         phi_or_beta_boundary=sum(any(t.split('_')[0] in ('phi', 'beta') for t in f['diag']['at_bound'])
                                                  for f in fits)))
    for nm, m in (('median20', mod), ('single_draw', single)):
        out[nm] = paired_test(sl(m), hum_sl, rng_for('calib', seed, delta, nm, 'paired'), n_boot=2000)
    return out


def calibrate(episodes_path, n_seeds, deltas, n_samples, jobs, budget_s=None):
    """H02. Truth = the real humans' per-participant xiong_mf fits (realistic alpha/phi/beta/kappa/b per h, clipped)."""
    import time
    eps = load_episodes(episodes_path)
    truths = {}
    for h in HAZARDS:
        ids = sorted(i for i, e in eps.items() if e['hazard'] == h)
        c, r, y = arrays(eps, ids)
        x0 = fit('xiong_mf', h, c, r, y)['x']
        for j, eid in enumerate(ids):
            p = fit('xiong_mf', h, c[j:j + 1], r[j:j + 1], y[j:j + 1], None, x0)['params']
            truths[(eps[eid]['participant'], h)] = dict(
                alpha=p['alpha'], phi=float(np.clip(p['phi'], *CAL_PHI)), beta=float(np.clip(p['beta'], *CAL_BETA)),
                kappa=float(np.clip(p['kappa'], -CAL_KB, CAL_KB)), b=float(np.clip(p['b'], -CAL_KB, CAL_KB)))
    tasks = [(s, d, n_samples) for s in range(n_seeds) for d in deltas]     # seed-major: every seed covers all deltas
    t0, res = time.time(), []
    with ProcessPoolExecutor(jobs, initializer=_cal_init, initargs=(str(episodes_path), truths)) as ex:
        futs = [ex.submit(calib_task, t) for t in tasks]
        for f in futs:
            res.append(f.result())
            if budget_s and time.time() - t0 > budget_s:
                for g in futs:
                    g.cancel()
                break
    done_seeds = collections.Counter(r['seed'] for r in res)
    full = {s for s, k in done_seeds.items() if k == len(deltas)}
    res = [r for r in res if r['seed'] in full]
    summary = {}
    for d in deltas:
        rs = [r for r in res if r['delta'] == d]
        if not rs:
            continue
        tr = np.array([r['true_median_diff'] for r in rs])
        s = dict(n_seeds=len(rs), true_median_diff_mean=float(tr.mean()),
                 human_slope_bias=float(np.mean([r['est_human_slope_mean'] - r['true_human_slope_mean'] for r in rs])),
                 model_slope_bias=float(np.mean([r['est_model_slope_mean'] - r['true_model_slope_mean'] for r in rs])),
                 frac_fits_failed=float(np.sum([r['diag']['failed'] for r in rs]) / np.sum([r['diag']['n'] for r in rs])),
                 frac_fits_phi_or_beta_boundary=float(np.sum([r['diag']['phi_or_beta_boundary'] for r in rs]) /
                                                      np.sum([r['diag']['n'] for r in rs])))
        for nm in ('median20', 'single_draw'):
            est = np.array([r[nm]['median_diff'] for r in rs])
            p = np.array([r[nm]['wilcoxon_p'] if r[nm]['wilcoxon_p'] is not None else 1.0 for r in rs])
            cov = np.mean([lo <= t <= hi for (lo, hi), t in zip([r[nm]['median_diff_ci95'] for r in rs], tr)])
            rate = float(np.mean(p < 0.05))
            s[nm] = dict(reject_rate_p05=rate, reject_rate_mc_se=float(np.sqrt(rate * (1 - rate) / len(rs))),
                         bias_median_diff=float(np.mean(est - tr)), rmse_median_diff=float(np.sqrt(np.mean((est - tr) ** 2))),
                         ci95_coverage=float(cov), mean_est_median_diff=float(est.mean()))
        summary[str(d)] = s
    return dict(design=dict(truth='per-participant xiong_mf fits of the real humans per h (phi clipped to %s, beta to %s, '
                                  '|kappa|,|b| <= %s); simulated humans play closed loop on the real value sequences; '
                                  'model = same strategy with phi_h + delta*(h - mean h), replayed teacher-forced on the '
                                  'simulated human history; complete pipeline (pooled warm starts, %d-draw medians, '
                                  'Wilcoxon + bootstrap CI of the median paired slope difference). single_draw = '
                                  'the same test using only the first draw.' % (CAL_PHI, CAL_BETA, CAL_KB, n_samples),
                            n_seeds_requested=n_seeds, n_seeds_completed=len(full), deltas=list(deltas),
                            n_samples=n_samples, seconds=time.time() - t0),
                summary=summary, replicates=res)


def selftest():
    """Closed-loop parameter recovery on the real value sequences (no VLM) + numerical/plumbing checks."""
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
    res['sampled_fit_recovery'] = sampled_fit_selftest(eps)
    res['plumbing'] = plumbing_selftest(eps)
    print(json.dumps(res, indent=1))
    return res


def plumbing_selftest(eps):
    """Analytic gradient == finite differences; closed-form NLL == sequential forward; two hazards handled;
    per-condition seeds independent of other conditions; logit_diff readout used without saturation."""
    import tempfile
    out = {}
    rng = np.random.default_rng(3)
    ids = sorted(i for i, e in eps.items() if e['hazard'] == 0.4)[:3]
    c, r, y = arrays(eps, ids)
    ys = rng.random(y.shape)
    err_f, err_g = 0.0, 0.0
    for name in MODELS:
        x = rng.normal(size=len(MODELS[name]))
        f, g = nll_grad(x, name, 0.4, prep(c, r), ys, np.ones_like(ys))
        err_f = max(err_f, abs(f - nll(x, name, 0.4, c, r, ys, np.ones_like(ys))))
        gn = np.array([(nll(x + e, name, 0.4, c, r, ys, np.ones_like(ys)) - nll(x - e, name, 0.4, c, r, ys, np.ones_like(ys))) / 2e-6
                       for e in np.eye(len(x)) * 1e-6])
        err_g = max(err_g, float(np.max(np.abs(g - gn)) / (1 + np.max(np.abs(gn)))))
    assert err_f < 1e-8 and err_g < 1e-5, (err_f, err_g)
    out['nll_abs_err'], out['grad_rel_err'] = err_f, err_g
    # two hazards: no Friedman/Page call
    tt = trend_tests(rng.random((10, 2)), np.array([0.1, 0.4]), None, rng)
    assert tt['friedman_p'] is None and tt['page_p'] is None and tt['slope_wilcoxon_p'] is not None
    out['two_hazards'] = 'trend_tests returns slope test only; main() skips sources without all 3 hazards'
    # seeds: a condition's sampled fits do not depend on other conditions being analysed
    sub = {}
    for h in (0.1, 0.4):
        ids_ = sorted(i for i, e in eps.items() if e['hazard'] == h)[:2]
        c_, r_, _ = arrays(eps, ids_)
        sub[h] = (ids_, c_, r_, forward(dict(alpha=1.0, phi=h, beta=1.0, kappa=0.0, b=0.0), c_, r_))
    a1, _, _ = sampled_fits('condA', eps, sub, 2)
    sampled_fits('condB', eps, sub, 2)
    a2, _, _ = sampled_fits('condA', eps, {0.4: sub[0.4]}, 2)
    assert all(a1[0.4][p_] == a2[0.4][p_] for p_ in a2[0.4]), 'per-condition seeds not independent'
    out['seed_independence'] = 'ok'
    # logit_diff readout: a saturated stored probability does not limit the target
    with tempfile.TemporaryDirectory() as td:
        rows = [dict(format='text', model_id='m', hazard_info='hidden', image_px=448, n_img=5, episode_id='e', t=1,
                     p_slot1=1.0, logit_diff=-30.0 + 45.0, run_fingerprint='f'),
                dict(format='text', model_id='m', hazard_info='hidden', image_px=448, n_img=5, episode_id='e', t=2,
                     p_slot1=0.25, run_fingerprint='f')]
        Path(td, 'r.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
        g, _, ro = load_results([str(Path(td, 'r.jsonl'))])
        v = g[('m', 'text', 'hidden', 448, 5)]
        assert abs(v[('e', 1)] - special.expit(15.0)) < 1e-15 and v[('e', 2)] == 0.25
        assert ro[('m', 'text', 'hidden', 448, 5)] == {'logit_diff': 1, 'probability': 1}
    out['logit_readout'] = 'ok'
    return out


def null_trend_test(eps, seeds=(0, 1, 2, 3, 4), prm=dict(alpha=1.0, phi=0.2, beta=0.4, kappa=0.2)):
    """Null check: every simulated participant has the SAME phi = 0.2 at all h (human-like beta ~ 0.4).
    Per-participant fits + the same trend tests as for real data; expect no spurious h trend (p > 0.05)."""
    out = {}
    hs = np.array(HAZARDS)
    for seed in seeds:
        _, X, _ = simulate_fit(eps, lambda h: prm['phi'], lambda h: prm['beta'], lambda h: prm['kappa'], seed, prm['alpha'])
        tt = trend_tests(X, hs, None, np.random.default_rng(seed), n_boot=10)
        out[f'seed{seed}'] = dict(median_phi={str(h): float(np.median(X[:, j])) for j, h in enumerate(hs)},
                                  slope_wilcoxon_p=tt['slope_wilcoxon_p'], page_p=tt['page_p'], friedman_p=tt['friedman_p'])
    out['n_seeds_slope_p_gt_0.05'] = sum(v['slope_wilcoxon_p'] > 0.05 for k, v in out.items() if k.startswith('seed'))
    out['n_seeds_page_p_gt_0.05'] = sum(v['page_p'] > 0.05 for k, v in out.items() if k.startswith('seed'))
    return out


def sampled_fit_selftest(eps, n_samples=20):
    """Teacher-forced soft agent on the HUMAN histories (like a VLM): P from xiong_mf with known phi, then the
    sampled-choice pipeline. Checks recovery of phi = h and absence of a trend for constant phi, at two betas."""
    hs = list(HAZARDS)
    out = {}
    for beta in (0.4, 1.5):
        for name, phi_fn in (('phi_eq_h', lambda h: h), ('phi_const_0.2', lambda h: 0.2)):
            by_h = {}
            for h in hs:
                ids = sorted(i for i, e in eps.items() if e['hazard'] == h)
                c, r, _ = arrays(eps, ids)
                y = forward(dict(alpha=1.0, phi=phi_fn(h), beta=beta, kappa=0.2, b=0.0), c, r)
                by_h[h] = (ids, c, r, y)
            pp, _, fits = sampled_fits(f'selftest_beta{beta}_{name}', eps, by_h, n_samples)
            pids = sorted(set.intersection(*[set(pp[h]) for h in hs]))
            sl = slopes_of(pp, hs, pids)
            w = stats.wilcoxon(sl)
            out[f'beta{beta}_{name}'] = dict(median_phi={str(h): float(np.median([pp[h][p_]['phi'] for p_ in pids])) for h in hs},
                                             median_beta={str(h): float(np.median([pp[h][p_]['beta'] for p_ in pids])) for h in hs},
                                             slope_wilcoxon_p=float(w.pvalue), fit_diagnostics=diag_summary(fits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=Path, default=HERE / 'data/episodes.jsonl')
    ap.add_argument('--results', nargs='*', default=[])
    ap.add_argument('--output', type=Path, default=HERE / 'analysis.json')
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--human-long', action='store_true', help='also fit humans on the 1500-trial games (reference)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--calibrate', type=int, metavar='N_SEEDS', help='H02 calibration of the paired slope test')
    ap.add_argument('--calib-deltas', type=float, nargs='+', default=[0.0, -0.5, 0.5])
    ap.add_argument('--calib-budget-min', type=float, default=35.0)
    ap.add_argument('--calib-output', type=Path, default=HERE / 'calibration.json')
    ap.add_argument('--primary-hazard-info', default='hidden', choices=('hidden', 'told'),
                    help='PRIMARY = hidden (volatility must be inferred from experience); told = secondary')
    ap.add_argument('--n-samples', type=int, default=20, help='sampled choice sequences per episode (primary fit)')
    ap.add_argument('--jobs', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    global POOL
    if args.calibrate:
        cal = calibrate(args.episodes, args.calibrate, args.calib_deltas, args.n_samples, args.jobs,
                        args.calib_budget_min * 60)
        args.calib_output.write_text(json.dumps(cal, indent=1))
        print(json.dumps(dict(cal['design'], summary=cal['summary']), indent=1))
        return
    if args.jobs > 1:
        POOL = ProcessPoolExecutor(args.jobs)
    if args.selftest:
        selftest()
        return
    eps = load_episodes(args.episodes)
    report = dict(model_doc=__doc__.split('Usage:')[0], human_results_provenance=HUMAN_PROVENANCE, sources={},
                  skipped_sources={})

    def human(eps_):
        by_h = {}
        for h in HAZARDS:
            ids = sorted(i for i, e in eps_.items() if e['hazard'] == h)
            by_h[h] = (ids, *arrays(eps_, ids))
        return by_h

    report['sources']['human/short100'] = analyse_source('human/short100', eps, human(eps), args.boot)
    if args.human_long:
        long_eps = {i: e for i, e in load_episodes(HERE / 'data/episodes_long.jsonl').items() if e['n_trials'] == 1500}
        report['sources']['human/long1500'] = analyse_source('human/long1500', long_eps, human(long_eps), min(args.boot, 50))
        # side-by-side: humans on the replayed 100-trial games vs their 1500-trial games (same h)
        report['human_short_vs_long'] = {
            n: {'pooled_phi': {h: src['pooled'][h]['xiong_mf']['params']['phi'] for h in src['pooled']},
                'pooled_alpha': {h: src['pooled'][h]['xiong_mf']['params']['alpha'] for h in src['pooled']},
                'median_phi': {h: v['phi'] for h, v in src['per_participant']['xiong_mf'].items()},
                'phi_tests': src['tests']['xiong_mf.phi']}
            for n, src in report['sources'].items() if n.startswith('human/')}

    groups, perc, readout = load_results(args.results)
    hum = report['sources']['human/short100']['per_participant_params']['xiong_mf']
    for (mid, fmt, hz, px, nimg), tab in sorted(groups.items(), key=str):
        name = f'{mid}/{fmt}/{hz}/px{px}' + (f'/n{nimg}' if fmt == 'image_outcome' else '')
        by_h = {}
        for h in HAZARDS:
            ids = sorted(i for i, e in eps.items() if e['hazard'] == h
                         and all((i, t) in tab for t in range(1, e['n_trials'] + 1)))
            if ids:
                targets = {i: [tab[(i, t)] for t in range(1, eps[i]['n_trials'] + 1)] for i in ids}
                by_h[h] = (ids, *arrays(eps, ids, targets))
        if set(by_h) != set(HAZARDS):
            # slope / Friedman / Page / recovery are defined on all three hazards: refuse partial hazard sets
            report['skipped_sources'][name] = f'complete episodes only for h in {sorted(by_h)}; all of {list(HAZARDS)} required'
            continue
        src = analyse_source(name, eps, by_h, args.boot)
        src.update(model_id=mid, format=fmt, hazard_info=hz, image_px=px, n_img=nimg, readout=readout[(mid, fmt, hz, px, nimg)])
        # how well the VLM predicts the human's actual choice (same trials)
        ll = []
        for h, (ids, c, r, y) in by_h.items():
            pc = np.where(c == 0, y, 1 - y)
            ll.append(np.log(np.clip(pc, EPS, 1)).ravel())
        src['human_choice_loglik_per_trial'] = float(np.mean(np.concatenate(ll)))
        src['n_complete_episodes'] = {h: len(v[0]) for h, v in by_h.items()}
        # PRIMARY: sampled binary-choice fits (comparable to humans); soft fits above are secondary
        hs = sorted(by_h)
        pp, draws, sfits = sampled_fits(name, eps, by_h, args.n_samples,
                                        {h: src['pooled'][h]['xiong_mf']['x'] for h in hs})
        pids = sorted(set.intersection(*[set(pp[h]) for h in hs]))
        src['sampled_fit'] = dict(
            n_samples=args.n_samples, fit_diagnostics=diag_summary(sfits),
            median={str(h): {q: float(np.median([pp[h][p_][q] for p_ in pids])) for q in ('alpha', 'phi', 'beta', 'kappa')} for h in hs},
            phi_trend=trend_tests(np.array([[pp[h][p_]['phi'] for h in hs] for p_ in pids]), np.array(hs),
                                  np.array([[eps[next(i for i in by_h[h][0] if eps[i]['participant'] == p_)]['position'] for h in hs] for p_ in pids]),
                                  rng_for(name, 'sampled_phi_trend')),
            beta_trend=trend_tests(np.array([[pp[h][p_]['beta'] for h in hs] for p_ in pids]), np.array(hs), None,
                                   rng_for(name, 'sampled_beta_trend'), n_boot=10),
            per_participant={str(h): {str(p_): v for p_, v in pp[h].items()} for h in hs})
        src['sampled_fit']['phi_recovery_at_model_beta'] = recovery_at_beta(
            eps, {h: src['sampled_fit']['median'][str(h)]['beta'] for h in hs},
            {h: src['sampled_fit']['median'][str(h)]['kappa'] for h in hs})
        hpp = {h: {int(k): v for k, v in hum[str(h)].items()} for h in hs}
        common = [p_ for p_ in pids if all(p_ in hpp[h] for h in hs)]
        hsl = slopes_of(hpp, hs, common)
        src['primary'] = hz == args.primary_hazard_info
        src['model_vs_human_phi_slope'] = dict(
            definition=PRIMARY_TEST, comparison=COMPARISON[hz],
            **paired_test(slopes_of(pp, hs, common), hsl, rng_for(name, 'model_vs_human')),
            mc_error_of_draws=mc_error({k: v for k, v in draws.items() if k[1] in common}, hs, common, hsl, name))
        src['_pp_sampled'] = pp
        report['sources'][name] = src
    # SECONDARY: description-experience contrast (told - hidden), same model/format/px/n_img
    contrasts = {}
    for name, src in report['sources'].items():
        if '/told/' in name and name.replace('/told/', '/hidden/') in report['sources']:
            a, b = src['_pp_sampled'], report['sources'][name.replace('/told/', '/hidden/')]['_pp_sampled']
            hs = sorted(a)
            pids = sorted(set.intersection(*[set(a[h]) & set(b[h]) for h in hs]))
            cname = name.replace('/told/', '/told-minus-hidden/')
            contrasts[cname] = dict(
                phi_slope=paired_test(slopes_of(a, hs, pids), slopes_of(b, hs, pids), rng_for(cname, 'slope')),
                mean_phi={str(h): paired_test([a[h][p_]['phi'] for p_ in pids], [b[h][p_]['phi'] for p_ in pids],
                                              rng_for(cname, 'phi', h)) for h in hs})
    report['told_vs_hidden'] = contrasts
    # H01: the three human/model comparisons side by side, with what each one does and does not match
    comp = {}
    for name, src in report['sources'].items():
        if 'model_vs_human_phi_slope' not in src:
            continue
        base = name.replace(f"/{src['hazard_info']}/", '/')
        mv = src['model_vs_human_phi_slope']
        comp.setdefault(base, {})[f"model_{src['hazard_info']}_vs_human_told"] = dict(
            comparison=mv['comparison'], median_diff=mv['median_diff'], median_diff_ci95=mv['median_diff_ci95'],
            wilcoxon_p=mv['wilcoxon_p'], primary=src['primary'])
    for cname, v in contrasts.items():
        comp.setdefault(cname.replace('/told-minus-hidden/', '/'), {})['model_told_minus_hidden'] = dict(
            comparison='same model, same histories: effect of stating h', **{k: v['phi_slope'][k] for k in
                                                                            ('median_diff', 'median_diff_ci95', 'wilcoxon_p')})
    report['human_model_comparisons'] = dict(
        note=('Model and human see the same within-game choice and feedback history only. Humans were told h and had '
              'prior experience (1,600 / 3,200 trials before games 3 / 5); models had neither. The hidden-model vs '
              'told-human difference is descriptive (cross-condition); the told-vs-told comparison matches the hazard '
              'information but not the experience.'), by_model=comp)
    for src in report['sources'].values():
        src.pop('_pp_sampled', None)
    report['primary_test'] = PRIMARY_TEST
    report['p_value_legend'] = {
        'tests.xiong_mf.phi.slope_wilcoxon_p': 'PRIMARY human trend test (our re-analysis): per-participant OLS slope of phi on h, Wilcoxon vs 0',
        'tests.xiong_mf.phi.page_p': "secondary: Page's L ordered trend h=0.1<0.2<0.4",
        'tests.xiong_mf.phi.friedman_p': 'secondary: Friedman, any difference across h',
        'tests.xiong_mf.phi.fe_position_adjusted': 'secondary: within-participant OLS phi ~ h + position, bootstrap CI of b_h',
        'superseded': 'Earlier drafts quoted p=0.015 = Wilcoxon of phi(h=0.4) - phi(h=0.1); no longer reported.',
        'model_vs_human_phi_slope.wilcoxon_p': 'PRIMARY model-vs-human test (see primary_test); descriptive cross-condition '
                                               'difference in the hidden condition; see calibration.json for its error rates'}
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
        fd = src['fit_diagnostics']['all_per_participant']
        print(f'{name:45s} xiong_mf pooled {a} | per-participant median phi {pm} | P(shift|low/high) {sg} | '
              f"phi slope Wilcoxon p={t_.get('slope_wilcoxon_p')} Page p={t_.get('page_p')} "
              f"pos-adjusted b_h={fe_.get('b_h', float('nan')):.3f} CI={fe_.get('b_h_ci95')} | "
              f"fits failed {fd['frac_failed']:.3f}, boundary(excl. alpha->1) {fd['frac_boundary_excl_alpha_hi']:.3f}")
        if 'sampled_fit' in src:
            sf = src['sampled_fit']
            mv = src['model_vs_human_phi_slope']
            print(f"{'':45s} [{'PRIMARY' if src['primary'] else 'secondary'}] sampled-fit median phi "
                  f"{[round(sf['median'][h]['phi'], 3) for h in sorted(sf['median'])]} beta "
                  f"{[round(sf['median'][h]['beta'], 3) for h in sorted(sf['median'])]} | model-human phi slope "
                  f"{mv['median_diff']:.3f} CI {mv['median_diff_ci95']} p={mv['wilcoxon_p']} "
                  f"(MC sd {mv['mc_error_of_draws']['median_diff_mc_sd']:.3f}) [{mv['comparison'].split(':')[0]}]")
    for k, v in report['told_vs_hidden'].items():
        print('told-hidden', k, v['phi_slope'])
    for k, v in report['skipped_sources'].items():
        print('skipped', k, v)
    for k, v in report['perception'].items():
        print('perception', k, v)


if __name__ == '__main__':
    main()
