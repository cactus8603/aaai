"""Behavioural analysis of the API subset (exploratory; Appendix K).

Question: do the proprietary reasoning models' raw-history choices track the advisors, or follow a shortcut
(repeat the previous outcome, or the generator's 5/5 outcome quota per 10-trial block)? The open models are
analysed on exactly the same items (scenario cover story, text reports, query trials t = 4, 8, ..., 80,
episodes hel000-hel023 and their counterbalanced copies) with per-copy choices (y > .5), the same scoring
as the API answers.

Per agent x history format:
  * accuracy (per-copy choices, averaged within episode pair, then over pairs; 95% bootstrap CI over pairs)
  * accuracy by phase; separately, accuracy and hard-choice follow-better in the volatile phase
    on the first 1-3, middle 4-7, and last 8-10 trials after a genuine reversal (offsets 0-2, 3-6, 7-9)
  * agreement (%, Cohen's kappa) with strategies: repeat the previous outcome, win-stay/lose-shift on advisors
    (follow the advisor correct on the previous trial), all-trial count, last-5 count, Bayesian volatility
    learner, and two diagnostics that use hidden generator information (outcome quota, always follow A)
  * on items where the quota rule and the Bayesian learner disagree: share of choices that follow the learner
  * logistic regression of following A on A's correctness at lags 1-10 (+1/-1) and on the repeat-previous-
    outcome regressor (+1 if A's current report equals the previous outcome, else -1); 95% CIs from a
    bootstrap over episode pairs

Usage: python analyze_api_behavior.py [--out analysis/api_behavior]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cogmodels as cm  # noqa: E402
from analyze_main import true_reversals  # noqa: E402

EPISODES = [f'hel{i:03d}' for i in range(24)]
TRIALS = list(range(4, 81, 4))
API = {'Claude Fable 5.2': 'runs_api/planA_fable52.jsonl', 'GPT-6-Astra': 'runs_api/planA_gpt6astra.jsonl'}
OPEN = {'Qwen3-VL-8B': 'qwen3vl8b', 'InternVL3.5-8B': 'internvl8b', 'Gemma-3-27B': 'gemma3_27b',
        'Qwen3-VL-32B': 'qwen3vl32b', 'InternVL3.5-38B': 'internvl38b'}
OPEN_RUNS = {'raw': 'ho_full_text', 'correct_only': 'ho_full_coded', 'summary': 'ho_sum_all'}
CONDS = ['raw', 'correct_only', 'summary']
LAGS = 10
REVERSAL_WINDOWS = ((1, 3), (4, 7), (8, 10))
RULES = ['repeat_outcome', 'wsls', 'count_all', 'last5', 'bayes', 'quota (hidden)', 'always_A (hidden)']


def load_eps():
    eps = {}
    for line in open(HERE / 'episodes_v3/episodes.jsonl'):
        e = json.loads(line)
        if e['episode_id'].rstrip('t') in EPISODES:
            eps[e['episode_id']] = e
    return eps


def rules_for(ep):
    """{rule: array over t-1 of P(follow A) in {0, .5, 1} (nan where undefined)}, using only trials < t."""
    tr = ep['trials']; T = len(tr)
    o = np.array([x['a_correct'] for x in tr], float)
    q = cm.volatile_belief(o)['q']
    out = {k: np.full(T, np.nan) for k in RULES}
    for i in range(1, T):
        x, prev = tr[i], tr[i - 1]
        out['repeat_outcome'][i] = float(x['a_report_bit'] == prev['truth_bit'])
        out['wsls'][i] = o[i - 1]
        for name, seg in (('count_all', o[:i]), ('last5', o[max(0, i - 5):i])):
            s, n = seg.sum(), len(seg)
            out[name][i] = 0.5 if 2 * s == n else float(2 * s > n)
        out['bayes'][i] = 0.5 if q[i] == 0.5 else float(q[i] > 0.5)
        b0 = i // 10 * 10
        seen = [tr[j]['truth_bit'] for j in range(b0, i)]
        r1, r0 = 5 - sum(seen), 5 - (len(seen) - sum(seen))
        out['quota (hidden)'][i] = 0.5 if r1 == r0 else float(x['a_report_bit'] == int(r1 > r0))
        out['always_A (hidden)'][i] = 1.0
    return out


def since_reversal(ep):
    """Zero-based offset from the latest genuine reversal; None before the first.

    Match analyze_main: r > 1 excludes the initial block, and r <= t includes the
    reversal trial itself at offset 0. Window selection below retains offsets 0-9.
    """
    rev = true_reversals(ep)
    return [t - max(prior) if (prior := [r for r in rev if r <= t]) else None
            for t in range(1, len(ep['trials']) + 1)]


def reversal_summary(rows):
    """Distinct accuracy and follow-better estimates, with episode-pair CIs."""
    out = {}
    for lo, hi in REVERSAL_WINDOWS:
        sel = [r for r in rows if r['phase'] == 'volatile' and r['sr'] is not None
               and lo - 1 <= r['sr'] <= hi - 1]
        entry = dict(trials_after_reversal=[lo, hi], offsets=[lo - 1, hi - 1], n_items=len(sel))
        for metric in ('correct', 'follow_better'):
            by_pair = collections.defaultdict(list)
            for r in sel:
                if r[metric] is not None:
                    by_pair[r['base']].append(r[metric])
            v = [float(np.mean(x)) for x in by_pair.values()]
            entry['accuracy' if metric == 'correct' else metric] = dict(
                estimate_ci95=boot_ci(v) if v else None, n_pairs=len(v),
                n_items=sum(map(len, by_pair.values())))
        out[f'{lo}_{hi}'] = entry
    return out


def load_api(path):
    """{cond: {(eid, t): follow_A in {0,1} or None}}"""
    d = collections.defaultdict(dict)
    for line in open(HERE / path):
        r = json.loads(line)
        ok = r['parse'] in ('exact', 'final_line')
        d[r['condition']][(r['episode_id'], r['t'])] = r['y'] if ok else None
    return d


def load_open(key):
    d = collections.defaultdict(dict)
    items = {(e + c, t) for e in EPISODES for c in ('', 't') for t in TRIALS}
    for cond, run in OPEN_RUNS.items():
        for line in open(HERE / f'runs_h200/{key}/{run}/results_shard0of1.jsonl'):
            r = json.loads(line)
            if r['readout'] != 'choice' or r['domain_slot'] != 'scenario' or r['format'] != 'text':
                continue
            k = (r['episode_id'], r['t'])
            if k in items:
                d[cond][k] = float(r['y'] > 0.5) if r['y'] != 0.5 else None
    return d


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def boot_ci(per_pair, n=10000, seed=0):
    v = np.array(per_pair, float); rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def logit_fit(X, y, ridge=1e-3, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def analyse(choices, eps, R, SR):
    """choices: {(eid, t): 0/1/None} for one agent x condition."""
    rows = []
    for (eid, t), c in choices.items():
        if c is None or t < 2:
            continue
        ep = eps[eid]; tr = ep['trials'][t - 1]
        rows.append(dict(eid=eid, base=eid.rstrip('t'), t=t, c=c, correct=float(c == tr['a_correct']),
                         follow_better=float(c == (tr['p_a'] > 0.5)) if tr['p_a'] != 0.5 else None,
                         phase=tr['phase'], sr=SR[eid][t - 1], **{k: R[eid][k][t - 1] for k in RULES}))
    n_items = len(choices); n_valid = len(rows)
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r['base']].append(r)
    out = dict(n_items=n_items, valid_rate=n_valid / max(n_items, 1),
               acc=boot_ci([np.mean([r['correct'] for r in v]) for v in by_pair.values()]))
    for ph in ('stable', 'volatile'):
        out[f'acc_{ph}'] = boot_ci([np.mean([r['correct'] for r in v if r['phase'] == ph]) for v in by_pair.values()
                                    if any(r['phase'] == ph for r in v)])
    out['volatile_reversal_windows'] = reversal_summary(rows)
    agree = {}
    for k in RULES:
        sel = [r for r in rows if not np.isnan(r[k]) and r[k] != 0.5]
        a = [r['c'] for r in sel]; b = [r[k] for r in sel]
        agree[k] = dict(agree=float(np.mean(np.array(a) == np.array(b))), kappa=kappa(a, b) if k != 'always_A (hidden)' else None,
                        n=len(sel))
    out['agreement'] = agree
    dis = [r for r in rows if r['quota (hidden)'] in (0.0, 1.0) and r['bayes'] in (0.0, 1.0) and r['quota (hidden)'] != r['bayes']]
    out['quota_vs_bayes_disagree'] = dict(n=len(dis), follow_bayes=float(np.mean([r['c'] == r['bayes'] for r in dis])) if dis else None)
    dis = [r for r in rows if r['repeat_outcome'] != r['bayes'] and r['bayes'] in (0.0, 1.0)]
    out['repeat_vs_bayes_disagree'] = dict(n=len(dis), follow_bayes=float(np.mean([r['c'] == r['bayes'] for r in dis])) if dis else None)

    # lagged regression on per-copy choices
    def design(rs):
        X, y = [], []
        for r in rs:
            ep = eps[r['eid']]; t = r['t']
            if t <= LAGS:
                continue
            o = [ep['trials'][t - 1 - k]['a_correct'] for k in range(1, LAGS + 1)]
            X.append([1.0] + [2 * v - 1 for v in o] + [2 * r['repeat_outcome'] - 1]); y.append(r['c'])
        return np.array(X), np.array(y)
    X, y = design(rows)
    w = logit_fit(X, y)
    rng = np.random.default_rng(1); pairs = list(by_pair); bs = []
    for _ in range(1000):
        samp = [r for p in rng.choice(pairs, len(pairs)) for r in by_pair[p]]
        Xb, yb = design(samp)
        bs.append(logit_fit(Xb, yb))
    bs = np.array(bs)
    names = ['intercept'] + [f'lag{k}' for k in range(1, LAGS + 1)] + ['repeat_outcome']
    out['regression'] = {nm: [float(w[i]), float(np.quantile(bs[:, i], .025)), float(np.quantile(bs[:, i], .975))]
                         for i, nm in enumerate(names)}
    out['regression']['n'] = int(len(y))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'analysis/api_behavior')
    a = ap.parse_args()
    eps = load_eps()
    R = {e: rules_for(ep) for e, ep in eps.items()}
    SR = {e: since_reversal(ep) for e, ep in eps.items()}
    agents = {k: load_api(v) for k, v in API.items()}
    agents.update({k: load_open(v) for k, v in OPEN.items()})
    # rules alone on the same items (each rule's choice; ties count .5)
    res = dict(rules={}, agents={}, reversal_definition=dict(
        genuine_reversal='r > 1; latest r <= t; offset = t - r; initial block excluded',
        windows='first 1-3 / middle 4-7 / last 8-10 trials = offsets 0-2 / 3-6 / 7-9',
        accuracy='hard choice agrees with realized outcome (a_correct)',
        follow_better='hard choice follows scheduled better advisor (p_a > 0.5); p_a = 0.5 excluded',
        aggregation='per-copy scores averaged within episode pair, then equal weight over pairs',
        uncertainty='95% percentile bootstrap CI over episode pairs; 10000 resamples, seed 0'))
    for k in RULES:
        acc = collections.defaultdict(list)
        for e, ep in eps.items():
            for t in TRIALS:
                p = R[e][k][t - 1]; ac = ep['trials'][t - 1]['a_correct']
                acc[e.rstrip('t')].append(p if ac else 1 - p)
        res['rules'][k] = boot_ci([np.mean(v) for v in acc.values()])
    for ag, d in agents.items():
        res['agents'][ag] = {c: analyse(d[c], eps, R, SR) for c in CONDS if c in d}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'api_behavior.json').write_text(json.dumps(res, indent=1))
    L = ['rules alone (same items): ' + ', '.join(f'{k} {v[0]:.3f}' for k, v in res['rules'].items()),
         'Reversal windows: r > 1, latest r <= t; trials 1-3/4-7/8-10 = offsets 0-2/3-6/7-9; initial block excluded.',
         'Reversal accuracy and hard-choice follow-better are distinct; means and 95% CIs use episode pairs.', '']
    for ag, dc in res['agents'].items():
        for c, r in dc.items():
            ag_ = r['agreement']; rg = r['regression']
            L.append(f"{ag:18s} {c:12s} acc {r['acc'][0]:.3f} [{r['acc'][1]:.3f}, {r['acc'][2]:.3f}]  valid {r['valid_rate']:.3f}  "
                     f"stable {r['acc_stable'][0]:.3f} volatile {r['acc_volatile'][0]:.3f}")
            for metric in ('accuracy', 'follow_better'):
                values = []
                for lo, hi in REVERSAL_WINDOWS:
                    ent = r['volatile_reversal_windows'][f'{lo}_{hi}'][metric]
                    ci = ent['estimate_ci95']
                    values.append(f"{lo}-{hi}: {ci[0]:.3f} [{ci[1]:.3f},{ci[2]:.3f}] n={ent['n_items']} pairs={ent['n_pairs']}"
                                  if ci else f'{lo}-{hi}: undefined')
                L.append(f"    reversal {metric}: " + '; '.join(values))
            L.append('    kappa: ' + ', '.join(f"{k} {v['kappa']:.2f}" if v['kappa'] is not None else f"{k} agree {v['agree']:.2f}"
                                           for k, v in ag_.items()))
            L.append(f"    quota-vs-bayes disagree n={r['quota_vs_bayes_disagree']['n']} follow bayes {r['quota_vs_bayes_disagree']['follow_bayes']:.2f}; "
                     f"repeat-vs-bayes n={r['repeat_vs_bayes_disagree']['n']} follow bayes {r['repeat_vs_bayes_disagree']['follow_bayes']:.2f}")
            L.append('    regression: ' + ', '.join(f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in rg.items()
                                                if k in ('lag1', 'lag2', 'lag3', 'lag5', 'lag10', 'repeat_outcome')))
        L.append('')
    (a.out / 'summary.txt').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
