"""Mechanism analysis (plan v3): heuristic matching and integration kernels.

(a) Heuristic matching. Model choice c_t = 1[twin-averaged y_t > 0.5] (choose A's report) is compared
    with simple strategies (all use trials 1..t-1 only):
      copy_prev_truth   choose the reporter whose current report equals the previous true label
      follow_prev_correct  choose the reporter who was correct on t-1 (win-stay/lose-shift on reporters)
      count_all         majority of A-correct over all past trials (ties dropped)
      last3/last5/last10  majority over the last K trials (ties dropped)
      bocpd             controller.bocpd posterior p_A > 0.5
      oracle            true current reliability p_A(t) > 0.5
      label_prior       choose the reporter whose current report is the domain's first label (red / cat) in
                        the original episode's display coordinates (constant label preference)
    Reported per arm (model x history format x condition): agreement, chance agreement
    p_e = pm ps + (1-pm)(1-ps), Cohen's kappa (pooled over trials, episode-bootstrap CI), agreement
    by phase, and the per-episode most consistent strategy (max kappa). Because strategies are correlated on
    this schedule, every strategy is also scored on ONE common trial mask (trials where the model and all
    strategies make a non-tied choice; ``kappa_common``) and the strategy-by-strategy agreement / kappa
    matrix on that same mask is reported. All of this is descriptive: "behaviour consistent with" a
    strategy, never evidence that the model internally runs it.
(b) Integration kernel. Soft-target logistic regression of the twin-averaged choice probability on
      s_{t-k} = +1 if A correct at t-k else -1 (k = 1..K, 0 if t-k < 1),
      copy  = +1 if A's current report equals the previous true label else -1,
      label = +1 if A's current report is the first label of the domain (red / cat) in the original
              episode's display coordinates (label prior; twin averaging should cancel it),
      first = +1 if A is listed first on the current trial (position prior), plus an intercept.
    Pooled per model x arm (domains pooled), episode-bootstrap CIs; an exponential A*lam^(k-1) is
    fitted to the kernel for a summary. For C arms the kernel describes the joint effect of the external
    summary and the model, not a memory kernel learned by the VLM. Runs on different datasets (e.g. the IID
    robust set) are separate cells, so kernels are estimated on each dataset.
(c) ``--validate`` recovers synthetic agents (delta rule, copy-last-label, constant label prior, mixture of
    copy and delta, last-K window majority, count-all majority) on every ``--validate-episodes`` set
    (default: episodes_v3 and the IID episodes_robust_v3), checking that the analysis tells them apart.

CLI:
  python analyze_mechanism.py RUN_DIR [RUN_DIR ...] --out-dir DIR [--n-boot 1000] [--K 15] [--partial]
      [--validate [--validate-episodes PATH ...]]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_main import (AnalysisError, _json_default, choice_credit, load_episodes, load_rows,  # noqa: E402
                          pair_trials, sigmoid)
from controller import bocpd  # noqa: E402
from cogmodels import delta_rule  # noqa: E402

STRATEGIES = ('copy_prev_truth', 'follow_prev_correct', 'count_all', 'last3', 'last5', 'last10',
              'bocpd', 'oracle', 'label_prior')
DEFAULT_VALIDATE = (HERE / 'episodes_v3' / 'episodes.jsonl', HERE / 'episodes_robust_v3' / 'episodes.jsonl')


# ----------------------------------------------------------------------------- strategies

def label_bit(ep, i):
    """1 if A's report on trial i+1 is the domain's first label (original display coordinates)."""
    return int((ep['trials'][i]['a_report_bit'] ^ int(ep.get('label_polarity', 0))) == 0)


def strategy_predictions(ep, hazard=1 / 15):
    """{name: array over t=1..T of P(choose A) in {0, 1, nan}} (nan = undefined / tie)."""
    tr = ep['trials']
    T = len(tr)
    o = np.array([x['a_correct'] for x in tr], float)
    out = {k: np.full(T, np.nan) for k in STRATEGIES}
    cs = np.concatenate([[0.0], np.cumsum(o)])
    bp = [s['p_a'] for s in bocpd(o, h=hazard)]    # bp[t-1] = state before trial t
    for i in range(T):          # i = t - 1
        out['oracle'][i] = 1.0 if tr[i]['p_a'] > 0.5 else (0.0 if tr[i]['p_a'] < 0.5 else np.nan)
        if i == 0:
            continue
        out['copy_prev_truth'][i] = float(tr[i]['a_report_bit'] == tr[i - 1]['truth_bit'])
        out['follow_prev_correct'][i] = o[i - 1]
        for name, k in (('count_all', None), ('last3', 3), ('last5', 5), ('last10', 10)):
            lo = 0 if k is None else max(0, i - k)
            s, n = cs[i] - cs[lo], i - lo
            out[name][i] = np.nan if 2 * s == n else float(2 * s > n)
        out['bocpd'][i] = np.nan if bp[i] == 0.5 else float(bp[i] > 0.5)
        out['label_prior'][i] = float(label_bit(ep, i))
    return out


def kappa_from_counts(c):
    """c = [n11, n10, n01, n00] (model, strategy). Returns (agreement, chance, kappa)."""
    n11, n10, n01, n00 = c
    n = n11 + n10 + n01 + n00
    if n == 0:
        return np.nan, np.nan, np.nan
    po = (n11 + n00) / n
    pm, ps = (n11 + n10) / n, (n11 + n01) / n
    pe = pm * ps + (1 - pm) * (1 - ps)
    k = np.nan if pe >= 1 else (po - pe) / (1 - pe)
    return po, pe, k


def counts(m, s):
    ok = np.isfinite(m) & np.isfinite(s)
    m, s = m[ok], s[ok]
    return np.array([(m * s).sum(), (m * (1 - s)).sum(), ((1 - m) * s).sum(), ((1 - m) * (1 - s)).sum()])


def _trial_arrays(trials):
    sp = strategy_predictions(trials[0]['ep'], trials[0]['hazard'])
    ts = np.array([x['t'] for x in trials])
    keep = ts >= 2
    ts = ts[keep]
    m = choice_credit([x['py'] for x in trials])[keep]
    m = np.where(m == 0.5, np.nan, m)
    ph = np.array([x['phase'] for x in trials])[keep]
    S = {k: sp[k][ts - 1] for k in STRATEGIES}
    common = np.isfinite(m) & np.all([np.isfinite(S[k]) for k in STRATEGIES], axis=0)
    return m, S, ph, common


def heuristic_cell(per, n_boot=1000, seed=0):
    """per: {base: [trial list, ...]} (one list per domain); the episode is the bootstrap unit."""
    eps = sorted(per)
    C = {k: np.zeros((len(eps), 4)) for k in STRATEGIES}
    Cc = {k: np.zeros((len(eps), 4)) for k in STRATEGIES}         # on the common mask
    phase_C = {ph: {k: np.zeros(4) for k in STRATEGIES} for ph in ('stable', 'volatile')}
    pair = {(a, b): np.zeros(4) for a in STRATEGIES for b in STRATEGIES}
    n_common, n_all = 0, 0
    best = Counter()
    per_ep = {}
    for i, b in enumerate(eps):
        for trials in per[b]:
            m, S, ph, common = _trial_arrays(trials)
            n_common += int(common.sum())
            n_all += len(m)
            for k in STRATEGIES:
                s = S[k]
                C[k][i] += counts(m, s)
                Cc[k][i] += counts(m[common], s[common])
                for p in phase_C:
                    phase_C[p][k] += counts(m[ph == p], s[ph == p])
            for a in STRATEGIES:
                for c in STRATEGIES:
                    pair[(a, c)] += counts(S[a][common], S[c][common])
        kap = {k: kappa_from_counts(C[k][i])[2] for k in STRATEGIES}
        valid = {k: v for k, v in kap.items() if np.isfinite(v)}
        bk = max(valid, key=valid.get) if valid else 'none'
        best[bk] += 1
        per_ep[b] = dict(best=bk, kappa={k: (None if not np.isfinite(v) else round(float(v), 3)) for k, v in kap.items()})
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(eps), size=(n_boot, len(eps)))
    res = {}
    for k in STRATEGIES:
        po, pe, kp = kappa_from_counts(C[k].sum(0))
        bs = np.array([kappa_from_counts(C[k][j].sum(0))[2] for j in idx])
        bs = bs[np.isfinite(bs)]
        ci = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))] if len(bs) else [np.nan, np.nan]
        poc, _, kc = kappa_from_counts(Cc[k].sum(0))
        res[k] = dict(agreement=po, chance=pe, kappa=kp, kappa_ci95=ci, n=int(C[k].sum()),
                      agreement_common=poc, kappa_common=kc,
                      agreement_by_phase={p: kappa_from_counts(phase_C[p][k])[0] for p in phase_C},
                      kappa_by_phase={p: kappa_from_counts(phase_C[p][k])[2] for p in phase_C})
    matrix = {a: {c: dict(agreement=kappa_from_counts(pair[(a, c)])[0], kappa=kappa_from_counts(pair[(a, c)])[2])
                  for c in STRATEGIES} for a in STRATEGIES}
    return dict(strategies=res, best_matching=dict(best), per_episode=per_ep, n_episodes=len(eps),
                strategy_matrix=matrix, n_common_mask=n_common, n_trials=n_all,
                common_mask_frac=n_common / n_all if n_all else float('nan'))


# ----------------------------------------------------------------------------- kernel

def kernel_design(trials, K=15):
    """Rows for t >= 2. Returns X (n x (K+4)), y (soft), names."""
    ep = trials[0]['ep']
    tr = ep['trials']
    pol = int(ep.get('label_polarity', 0))
    s = np.array([2 * x['a_correct'] - 1 for x in tr], float)
    X, Y = [], []
    for x in trials:
        t = x['t']
        if t < 2:
            continue
        cur, prev = tr[t - 1], tr[t - 2]
        lags = [s[t - 1 - k] if t - k >= 1 else 0.0 for k in range(1, K + 1)]
        copy = 1.0 if cur['a_report_bit'] == prev['truth_bit'] else -1.0
        label = 1.0 if (cur['a_report_bit'] ^ pol) == 0 else -1.0
        first = 1.0 if cur.get('a_listed_first', True) else -1.0
        X.append([1.0] + lags + [copy, label, first])
        Y.append(x['py'])
    names = ['intercept'] + [f's_lag{k}' for k in range(1, K + 1)] + ['copy_prev_truth', 'label_prior', 'a_listed_first']
    return np.array(X), np.array(Y), names


def fit_soft_logistic(X, y, ridge=1e-4, iters=50):
    """Newton/IRLS for soft-target logistic regression with a tiny ridge (not on the intercept)."""
    y = np.clip(y, 1e-6, 1 - 1e-6)
    w = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + R @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def exp_fit(w):
    k = np.arange(1, len(w) + 1, dtype=float)
    try:
        (A, lam), _ = curve_fit(lambda k, A, lam: A * lam ** (k - 1), k, w, p0=[w[0] if w[0] else 0.1, 0.6],
                                bounds=([-np.inf, 0.0], [np.inf, 1.0]), maxfev=20000)
        r2 = 1 - np.sum((w - A * lam ** (k - 1)) ** 2) / max(np.sum((w - w.mean()) ** 2), 1e-12)
        return dict(A=float(A), lam=float(lam), half_life=float(np.log(0.5) / np.log(lam)) if 0 < lam < 1 else None,
                    r2=float(r2))
    except (RuntimeError, ValueError):
        return dict(A=None, lam=None, half_life=None, r2=None)


def kernel_cell(per, K=15, n_boot=1000, seed=0):
    eps = sorted(per)
    blocks = []
    names = None
    for b in eps:
        Xs, Ys = [], []
        for trials in per[b]:
            X, Y, names = kernel_design(trials, K)
            if len(X):
                Xs.append(X)
                Ys.append(Y)
        if Xs:
            blocks.append((np.vstack(Xs), np.concatenate(Ys)))
    if not blocks:
        return None
    X = np.vstack([b[0] for b in blocks])
    Y = np.concatenate([b[1] for b in blocks])
    w = fit_soft_logistic(X, Y)
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(n_boot):
        j = rng.integers(0, len(blocks), len(blocks))
        B.append(fit_soft_logistic(np.vstack([blocks[i][0] for i in j]), np.concatenate([blocks[i][1] for i in j])))
    B = np.array(B)
    lo, hi = np.quantile(B, .025, 0), np.quantile(B, .975, 0)
    coef = {n: dict(w=float(w[i]), ci95=[float(lo[i]), float(hi[i])]) for i, n in enumerate(names)}
    kern = w[1:K + 1]
    return dict(coef=coef, kernel=kern.tolist(), kernel_lo=lo[1:K + 1].tolist(), kernel_hi=hi[1:K + 1].tolist(),
                exp_fit=exp_fit(kern), n_trials=int(len(Y)), n_episodes=len(blocks))


# ----------------------------------------------------------------------------- synthetic agents

AGENTS = ('delta', 'copy', 'label_prior', 'mixture', 'window', 'count_all')


def _majority_logit(o, i, k, w):
    lo = 0 if k is None else max(0, i - k)
    s, n = o[lo:i].sum(), i - lo
    return 0.0 if 2 * s == n else w * (1.0 if 2 * s > n else -1.0)


def synthetic_paired(episodes, agent='delta', alpha=0.3, beta=8.0, w_copy=3.0, noise=0.0, seed=0,
                     hazard=1 / 15, w_label=3.0, mix=0.5, K=5, w_major=3.0):
    """Build a {base: [trial dicts]} cell from a synthetic agent's twin-averaged probabilities.

    agents: delta (delta rule on A-correct), copy (copy the previous true label), label_prior (constant
    preference for the domain's first label), mixture (per trial: copy with prob ``mix``, else delta),
    window (majority over the last K outcomes), count_all (majority over all past outcomes).
    """
    rng = np.random.default_rng(seed)
    per = {}
    for eid, ep in sorted(episodes.items()):
        if eid.endswith('t'):
            continue
        o = np.array([x['a_correct'] for x in ep['trials']], float)
        T = len(o)
        copy_lg = np.zeros(T)
        for i in range(1, T):
            tr, pv = ep['trials'][i], ep['trials'][i - 1]
            copy_lg[i] = w_copy * (1 if tr['a_report_bit'] == pv['truth_bit'] else -1)
        if agent == 'delta':
            lg = beta * (delta_rule(o, np.full(T, alpha)) - 0.5)
        elif agent == 'copy':
            lg = copy_lg
        elif agent in ('label_prior', 'constant'):
            lg = np.array([w_label * (1 if label_bit(ep, i) else -1) for i in range(T)], float)
        elif agent == 'mixture':
            dl = beta * (delta_rule(o, np.full(T, alpha)) - 0.5)
            lg = np.where(rng.random(T) < mix, copy_lg, dl)
        elif agent == 'window':
            lg = np.array([_majority_logit(o, i, K, w_major) for i in range(T)])
        elif agent == 'count_all':
            lg = np.array([_majority_logit(o, i, None, w_major) for i in range(T)])
        else:
            raise ValueError(agent)
        lg = lg + noise * rng.standard_normal(T)
        per[eid] = [dict(t=i + 1, l=float(lg[i]), py=float(sigmoid(lg[i])), a_correct=int(o[i]),
                         p_a=ep['trials'][i]['p_a'], phase=ep['trials'][i]['phase'], twinned=True, exact=True,
                         window_start=None, ep=ep, hazard=hazard, history_mode='full', scenario='na')
                    for i in range(T)]
    return per


VALIDATION_SPECS = (('delta', dict(alpha=0.3, beta=8.0)), ('copy', dict(w_copy=3.0)),
                    ('label_prior', dict(w_label=3.0)), ('mixture', dict(mix=0.5, alpha=0.3, beta=8.0, w_copy=3.0)),
                    ('window', dict(K=5)), ('count_all', dict()))


def validate(episodes, K=15, n_boot=100):
    out = {}
    for agent, kw in VALIDATION_SPECS:
        per = synthetic_paired(episodes, agent, **kw)
        kc = kernel_cell({b: [v] for b, v in per.items()}, K, n_boot)
        hc = heuristic_cell({b: [v] for b, v in per.items()}, n_boot)
        kap = {k: v['kappa'] for k, v in hc['strategies'].items()}
        top = max((k for k in kap if np.isfinite(kap[k])), key=lambda k: kap[k], default='none')
        out[agent] = dict(params=kw, kernel=[round(x, 3) for x in kc['kernel']], exp_fit=kc['exp_fit'],
                          copy_w=kc['coef']['copy_prev_truth']['w'], label_w=kc['coef']['label_prior']['w'],
                          kappa={k: (None if not np.isfinite(v) else round(float(v), 3)) for k, v in kap.items()},
                          most_consistent=top, expected_lam=1 - kw['alpha'] if agent == 'delta' else None)
    return out


# ----------------------------------------------------------------------------- pipeline

def analyze(run_dirs, K=15, n_boot=1000, seed=0, episodes=None, readout='choice', do_validate=False,
            partial=False, validate_episodes=None):
    rows, info = load_rows(run_dirs, episodes, strict=not partial)
    rows = [r for r in rows if r['readout'] == readout]
    paired = pair_trials(rows, strict=not partial)
    # group by (model, data, split, readout, arm): heuristics per slot and pooled; kernel pooled over slots
    by_arm = defaultdict(dict)
    for cell, per in paired.items():
        by_arm[cell[:5]][cell[5]] = per
    res = dict(mode='partial' if partial else 'formal', info=info,
               settings=dict(K=K, n_boot=n_boot, seed=seed, readout=readout), heuristics={}, kernels={})
    for key5, doms in sorted(by_arm.items()):
        key = '/'.join(key5)
        res['heuristics'][key] = {}
        for dom, per in sorted(doms.items()):
            res['heuristics'][key][dom] = heuristic_cell({b: [tr] for b, tr in per.items()}, n_boot, seed)
        pooled = defaultdict(list)
        for dom, per in sorted(doms.items()):
            for b, tr in per.items():
                pooled[b].append(tr)
        res['heuristics'][key]['all'] = heuristic_cell(pooled, n_boot, seed)
        res['kernels'][key] = kernel_cell(pooled, K, n_boot, seed)
    if do_validate:
        res['validation'] = run_validation(validate_episodes, K, min(n_boot, 200))
    return res


def run_validation(paths=None, K=15, n_boot=100):
    out = {}
    for p in (paths or DEFAULT_VALIDATE):
        p = Path(p)
        if not p.exists():
            continue
        eps = {k: v for k, v in load_episodes(p).items()}
        out[p.parent.name] = validate(eps, K, n_boot)
    return out


def _f(x, d=3):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def to_markdown(res):
    L = ['# Mechanism analysis (analyze_mechanism.py)', '']
    if res.get('mode') == 'partial':
        L += ['> **EXPLORATORY (--partial): incomplete data.**', '']
    L += ['All results are behavioural descriptions: "behaviour consistent with" a strategy and "effective weight '
          'on past outcomes", not evidence about internal computation. C-arm kernels reflect the external summary '
          'and the model jointly.', '',
          '## Behaviour consistent with simple strategies (domains pooled): agreement / chance / kappa [95% CI]', '',
          '| model/data/split/readout/arm | ' + ' | '.join(STRATEGIES) + ' | most consistent (episodes) |',
          '|---|' + '---|' * (len(STRATEGIES) + 1)]
    for key, d in res['heuristics'].items():
        h = d['all']['strategies']
        cells = [f"{_f(h[k]['agreement'], 2)}/{_f(h[k]['chance'], 2)}/**{_f(h[k]['kappa'], 2)}** "
                 f"[{_f(h[k]['kappa_ci95'][0], 2)}, {_f(h[k]['kappa_ci95'][1], 2)}]" for k in STRATEGIES]
        best = ', '.join(f'{k}:{v}' for k, v in sorted(d['all']['best_matching'].items(), key=lambda x: -x[1]))
        L.append(f'| {key} | ' + ' | '.join(cells) + f' | {best} |')
    L += ['', '## Same trial mask: kappa_common (model vs strategy) and strategy-by-strategy kappa', '']
    for key, d in res['heuristics'].items():
        a = d['all']
        L += [f"**{key}** (common mask: {a['n_common_mask']} of {a['n_trials']} trials, "
              f"{_f(a['common_mask_frac'], 2)})", '',
              '| | model | ' + ' | '.join(STRATEGIES) + ' |', '|---|---|' + '---|' * len(STRATEGIES)]
        for s in STRATEGIES:
            row = [_f(a['strategies'][s]['kappa_common'], 2)] + [_f(a['strategy_matrix'][s][c]['kappa'], 2) for c in STRATEGIES]
            L.append(f'| {s} | ' + ' | '.join(row) + ' |')
        L.append('')
    L += ['## Effective weight on past outcomes (soft logistic kernel, domains pooled, episode-bootstrap 95% CI)', '',
          '| model/data/split/readout/arm | w(s_1) | w(s_2) | w(s_3) | w(s_5) | w(s_10) | sum w(s_1..K) | exp fit lam (R2) | copy_prev_truth | label_prior | a_listed_first | intercept |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key, k in res['kernels'].items():
        if k is None:
            continue
        c = k['coef']

        def wc(n):
            return f"{_f(c[n]['w'], 2)} [{_f(c[n]['ci95'][0], 2)}, {_f(c[n]['ci95'][1], 2)}]"
        ef = k['exp_fit']
        L.append(f"| {key} | {wc('s_lag1')} | {wc('s_lag2')} | {wc('s_lag3')} | {wc('s_lag5')} | {wc('s_lag10')} | "
                 f"{_f(sum(k['kernel']), 2)} | {_f(ef['lam'], 2)} ({_f(ef['r2'], 2)}) | {wc('copy_prev_truth')} | "
                 f"{wc('label_prior')} | {wc('a_listed_first')} | {wc('intercept')} |")
    if 'validation' in res:
        L += ['', '## Synthetic recovery (can the analysis tell the agents apart?)', '']
        for ds, agents in res['validation'].items():
            L.append(f'### {ds}')
            for a, v in agents.items():
                L.append(f"- **{a}** {v['params']}: most consistent = {v['most_consistent']}; kernel[1:6]={v['kernel'][:6]}, "
                         f"exp lam={_f(v['exp_fit']['lam'])} (expected {v['expected_lam']}), copy w={_f(v['copy_w'], 2)}, "
                         f"label w={_f(v['label_w'], 2)}")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out-dir', default=str(HERE / 'analysis'))
    ap.add_argument('--tag', default='mechanism')
    ap.add_argument('--K', type=int, default=15)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', default=None)
    ap.add_argument('--readout', default='choice')
    ap.add_argument('--partial', action='store_true', help='exploratory: allow incomplete data')
    ap.add_argument('--validate', action='store_true', help='also run synthetic-agent recovery')
    ap.add_argument('--validate-episodes', nargs='+', default=None,
                    help='episodes.jsonl files for --validate (default: episodes_v3 and episodes_robust_v3)')
    a = ap.parse_args(argv)
    try:
        res = analyze(a.runs, a.K, a.n_boot, a.seed, a.episodes, a.readout, a.validate, a.partial,
                      a.validate_episodes)
    except AnalysisError as e:
        sys.exit(f'ERROR: {e}')
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1, default=_json_default))
    md = to_markdown(res)
    (od / f'{a.tag}.md').write_text(md)
    print(md)
    print(f'wrote {od / a.tag}.json / .md')


if __name__ == '__main__':
    main()
