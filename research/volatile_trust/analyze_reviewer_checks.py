"""Three exploratory checks on existing held-out outputs (no new runs).

1. judgment: on raw-history trials the models were also asked which advisor is currently more likely to be
   correct. Twin-averaged judgment (A named if mean logit_diff > 0) is scored against the scheduled better
   advisor and compared with the choice on the same items; agreement (kappa) with the all-trial count, the
   last-5 count, the Bayesian learner and win-stay/lose-shift (name the advisor correct on the previous
   trial); and how often the choice follows the advisor the model itself judged better.
2. position: image vs text reports (raw and correctness-only histories). Share of choices for the
   first-listed report, and accuracy when the correct advisor's report is listed first vs second.
3. psych101: model-free value sensitivity in the changing bandit. For each trial with both machines seen,
   logistic regression of the choice of machine 1 (humans: binary; models: soft P) on the difference of the
   last observed values (per 10 points) and on the previous human choice (perseveration), per hazard rate.
   If models were insensitive to value, their fitted forgetting rate would not be identified.

All open-model items: held-out, twin-averaged as in the main analysis. Usage: python analyze_reviewer_checks.py
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

MODELS = {'Qwen3-VL-8B': 'qwen3vl8b', 'InternVL3.5-8B': 'internvl8b', 'Gemma-3-27B': 'gemma3_27b',
          'Qwen3-VL-32B': 'qwen3vl32b', 'InternVL3.5-38B': 'internvl38b'}
OUT = HERE / 'analysis/h200_ho/reviewer_checks'


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def boot(per_ep, n=10000, seed=0):
    v = np.array(per_ep, float); rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def episodes():
    E = {}
    for line in open(HERE / 'episodes_v3/episodes.jsonl'):
        e = json.loads(line)
        if e['split'] == 'held_out':
            E[e['episode_id']] = e
    return E


def twin(path, readout, fmt):
    """{(base, slot, t): mean logit_diff over the two copies} (only items present in both copies)."""
    g = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r['readout'] != readout or r['format'] != fmt or r['t'] < 2:
            continue
        g[(r['episode_id'].rstrip('t'), r['domain_slot'], r['t'])].append(r['logit_diff'])
    return {k: float(np.mean(v)) for k, v in g.items() if len(v) == 2}


def rules(ep):
    o = np.array([x['a_correct'] for x in ep['trials']], float); T = len(o)
    q = cm.volatile_belief(o)['q']
    R = {k: np.full(T, np.nan) for k in ('count_all', 'last5', 'bayes', 'wsls')}
    for i in range(1, T):
        for nm, seg in (('count_all', o[:i]), ('last5', o[max(0, i - 5):i])):
            s, n = seg.sum(), len(seg); R[nm][i] = 0.5 if 2 * s == n else float(2 * s > n)
        R['bayes'][i] = 0.5 if q[i] == 0.5 else float(q[i] > 0.5)
        R['wsls'][i] = o[i - 1]
    return R


def judgment(E):
    RU = {e: rules(ep) for e, ep in E.items() if not e.endswith('t')}
    res = {}
    for name, key in MODELS.items():
        f = HERE / f'runs_h200/{key}/ho_full_text/results_shard0of1.jsonl'
        J, C = twin(f, 'judgment', 'text'), twin(f, 'choice', 'text')
        rows = []
        for (b, slot, t), lj in J.items():
            if (b, slot, t) not in C or lj == 0 or C[(b, slot, t)] == 0:
                continue
            tr = E[b]['trials'][t - 1]
            jA, cA = float(lj > 0), float(C[(b, slot, t)] > 0)
            rows.append(dict(b=b, jA=jA, cA=cA, betterA=float(tr['p_a'] > 0.5), aC=tr['a_correct'],
                             **{k: RU[b][k][t - 1] for k in RU[b]}))
        per = collections.defaultdict(list)
        for r in rows:
            per[r['b']].append(r)
        out = dict(n=len(rows),
                   judgment_names_better=boot([np.mean([r['jA'] == r['betterA'] for r in v]) for v in per.values()]),
                   choice_follows_better=boot([np.mean([r['cA'] == r['betterA'] for r in v]) for v in per.values()]),
                   choice_acc=boot([np.mean([r['cA'] == r['aC'] for r in v]) for v in per.values()]),
                   judgment_names_A=float(np.mean([r['jA'] for r in rows])),
                   choice_follows_own_judgment=float(np.mean([r['cA'] == r['jA'] for r in rows])),
                   kappa_judgment_choice=kappa([r['jA'] for r in rows], [r['cA'] for r in rows]))
        for k in ('count_all', 'last5', 'bayes', 'wsls'):
            sel = [r for r in rows if r[k] in (0.0, 1.0)]
            out[f'kappa_judgment_{k}'] = kappa([r['jA'] for r in sel], [r[k] for r in sel])
            out[f'kappa_choice_{k}'] = kappa([r['cA'] for r in sel], [r[k] for r in sel])
        res[name] = out
    return res


def position(E):
    res = {}
    for name, key in MODELS.items():
        res[name] = {}
        for hist, run_t, run_i in (('raw', 'ho_full_text', 'ho_full_image'), ('correct_only', 'ho_full_coded', 'ho_full_coded')):
            for fmt, run in (('text', run_t), ('image', run_i)):
                C = twin(HERE / f'runs_h200/{key}/{run}/results_shard0of1.jsonl', 'choice', fmt)
                first, acc_first, acc_second = [], collections.defaultdict(list), collections.defaultdict(list)
                for (b, slot, t), ld in C.items():
                    if ld == 0:
                        continue
                    tr = E[b]['trials'][t - 1]
                    cA = ld > 0
                    chose_first = cA == tr['a_listed_first']
                    first.append(chose_first)
                    correct_first = bool(tr['a_correct']) == tr['a_listed_first']
                    (acc_first if correct_first else acc_second)[b].append(float(cA == bool(tr['a_correct'])))
                res[name][f'{hist}/{fmt}'] = dict(
                    n=len(first), chose_first=float(np.mean(first)),
                    acc_correct_listed_first=boot([np.mean(v) for v in acc_first.values()]),
                    acc_correct_listed_second=boot([np.mean(v) for v in acc_second.values()]))
    return res


def logit_fit(X, y, ridge=1e-4, iters=100, weights=None):
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (weights * (y - p)) - ridge * w
        H = (X * (weights * p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        st = np.linalg.solve(H, g); w += st
        if np.abs(st).max() < 1e-9:
            break
    return w


def cluster_weight_draws(X, y, participant, n_boot=2000, seed=20261113):
    """Refit after resampling whole participants; sorted IDs keep paired draws aligned."""
    ids, inverse = np.unique(participant, return_inverse=True)
    if len(ids) < 2:
        raise ValueError('participant bootstrap requires at least two participants')
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        draws[b] = logit_fit(X, y, weights=counts[inverse])
    return ids.tolist(), draws


def psych101():
    eps = {}
    for line in open(HERE / 'psych101/data/episodes.jsonl'):
        e = json.loads(line); eps[e['episode_id']] = e

    def design(eid):
        """per item t (1-based, predicting choice t from trials < t): (dv, prev) or None."""
        ep = eps[eid]; last = [None, None]; out = {}
        for i, tr in enumerate(ep['trials']):
            t = i + 1
            if i > 0 and None not in last:
                prev = 1.0 if ep['trials'][i - 1]['choice'] == 0 else -1.0
                out[t] = ((last[0] - last[1]) / 10.0, prev)
            last[tr['choice']] = tr['reward']
        return out
    D = {e: design(e) for e in eps}
    agents = {'Human': None}
    for name, key in MODELS.items():
        for info in ('hidden', 'told'):
            fs = sorted((HERE / f'runs_h200/{key}/psy_text_{info}').glob('results_*.jsonl'))
            if fs:
                agents[f'{name} ({info})'] = fs[0]
    res = {}
    for ag, f in agents.items():
        Y = {}
        if f is None:
            for e, ep in eps.items():
                for i, tr in enumerate(ep['trials']):
                    Y[(e, i + 1)] = 1.0 if tr['choice'] == 0 else 0.0
        else:
            for line in open(f):
                r = json.loads(line)
                lab = eps[r['episode_id']]['labels']
                lg = r['logits']; Y[(r['episode_id'], r['item'])] = 1 / (1 + np.exp(-(lg[lab[0]] - lg[lab[1]])))
        res[ag] = {}
        for h in (0.1, 0.2, 0.4):
            X, y, pid = [], [], []
            for (e, t), v in Y.items():
                if eps[e]['hazard'] != h or t not in D[e]:
                    continue
                dv, prev = D[e][t]; X.append([1.0, dv, prev]); y.append(v); pid.append(eps[e]['participant'])
            X, y, pid = np.array(X), np.array(y), np.array(pid)
            w = logit_fit(X, y)
            # per-participant value slope
            ps = []
            for p_ in np.unique(pid):
                m = pid == p_
                ps.append(logit_fit(X[m], y[m])[1])
            res[ag][str(h)] = dict(n=int(len(y)), value_per10=float(w[1]), persev=float(w[2]),
                                   value_per10_median_participant=float(np.median(ps)),
                                   frac_participants_value_pos=float(np.mean(np.array(ps) > 0)))
    return res


def psych101_staleness(n_boot=2000, seed=20261113):
    """Value weight when the older of the two last-observed values is fresh (seen <= 2 trials ago) vs stale
    (>= 4 trials ago), per hazard rate. A learner that forgets the unchosen machine weights stale differences less."""
    eps = {}
    for line in open(HERE / 'psych101/data/episodes.jsonl'):
        e = json.loads(line); eps[e['episode_id']] = e

    def design(eid):
        ep = eps[eid]; last = [None, None]; when = [None, None]; out = {}
        for i, tr in enumerate(ep['trials']):
            t = i + 1
            if i > 0 and None not in last:
                prev = 1.0 if ep['trials'][i - 1]['choice'] == 0 else -1.0
                out[t] = ((last[0] - last[1]) / 10.0, prev, t - min(when))
            last[tr['choice']] = tr['reward']; when[tr['choice']] = t
        return out
    D = {e: design(e) for e in eps}
    agents = {'Human': None}
    for name, key in MODELS.items():
        for info in ('hidden', 'told'):
            fs = sorted((HERE / f'runs_h200/{key}/psy_text_{info}').glob('results_*.jsonl'))
            if fs:
                agents[f'{name} ({info})'] = fs[0]
    res, sampled, human_masks = {}, {}, {}
    for ag, f in agents.items():
        Y = {}
        if f is None:
            for e, ep in eps.items():
                for i, tr in enumerate(ep['trials']):
                    Y[(e, i + 1)] = 1.0 if tr['choice'] == 0 else 0.0
        else:
            for line in open(f):
                r = json.loads(line); lab = eps[r['episode_id']]['labels']; lg = r['logits']
                Y[(r['episode_id'], r['item'])] = 1 / (1 + np.exp(-(lg[lab[0]] - lg[lab[1]])))
        res[ag] = {}
        for h in (0.1, 0.2, 0.4):
            X, y, pid, item_keys = [], [], [], []
            for (e, t), v in Y.items():
                if eps[e]['hazard'] != h or t not in D[e]:
                    continue
                dv, prev, st = D[e][t]
                fresh, stale = float(st <= 2), float(st >= 4)
                if fresh or stale:
                    X.append([1.0, dv * fresh, dv * stale, prev, stale]); y.append(v)
                    pid.append(eps[e]['participant'])
                    item_keys.append((e, t))
            if ag == 'Human':
                human_masks[h] = set(item_keys)
            elif set(item_keys) != human_masks[h]:
                raise ValueError(f'{ag}, h={h}: trial mask differs from the human replay comparison')
            X, y = np.array(X), np.array(y)
            w = logit_fit(X, y)
            ids, draws = cluster_weight_draws(X, y, pid, n_boot=n_boot, seed=seed)
            delta = draws[:, 2] - draws[:, 1]
            sampled[(ag, str(h))] = (ids, delta)
            res[ag][str(h)] = dict(fresh=float(w[1]), stale=float(w[2]), ratio=float(w[2] / w[1]), n=len(y),
                                   n_participants=len(ids), stale_minus_fresh=float(w[2] - w[1]),
                                   stale_minus_fresh_ci95=np.quantile(delta, [.025, .975]).tolist(),
                                   bootstrap_replicates=n_boot, bootstrap_seed=seed)
    # Resampling the same participant IDs couples each model to the human histories it saw.
    for ag in res:
        if ag == 'Human':
            continue
        for h, row in res[ag].items():
            ids, delta = sampled[(ag, h)]
            human_ids, human_delta = sampled[('Human', h)]
            if ids != human_ids:
                raise ValueError(f'{ag}, h={h}: participant sets differ from the human comparison')
            row['stale_minus_fresh_model_minus_human'] = row['stale_minus_fresh'] - res['Human'][h]['stale_minus_fresh']
            row['stale_minus_fresh_model_minus_human_ci95'] = np.quantile(delta - human_delta, [.025, .975]).tolist()
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staleness-only', action='store_true', help='update only staleness in the existing report')
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100')
    E = episodes()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.staleness_only:
        res = json.loads((OUT / 'reviewer_checks.json').read_text())
    else:
        res = dict(judgment=judgment(E), position=position(E), psych101=psych101())
    res['psych101_staleness'] = psych101_staleness(n_boot=args.bootstrap)
    (OUT / 'reviewer_checks.json').write_text(json.dumps(res, indent=1))
    L = ['== judgment (raw history, held-out, twin-averaged)']
    for m, r in res['judgment'].items():
        L.append(f"{m:16s} judgment names better {r['judgment_names_better'][0]:.3f}  choice follows better {r['choice_follows_better'][0]:.3f}"
                 f"  choice acc {r['choice_acc'][0]:.3f}  choice follows own judgment {r['choice_follows_own_judgment']:.3f}"
                 f"  kappa(j,c) {r['kappa_judgment_choice']:.2f}  names A {r['judgment_names_A']:.2f}")
        L.append('    kappa judgment: ' + ', '.join(f"{k} {r['kappa_judgment_'+k]:.2f}" for k in ('count_all', 'last5', 'bayes', 'wsls'))
                 + ' | choice: ' + ', '.join(f"{k} {r['kappa_choice_'+k]:.2f}" for k in ('count_all', 'last5', 'bayes', 'wsls')))
    L.append('\n== position (share choosing first-listed report; accuracy when correct advisor listed first / second)')
    for m, d in res['position'].items():
        for k, r in d.items():
            L.append(f"{m:16s} {k:20s} first {r['chose_first']:.3f}  acc first {r['acc_correct_listed_first'][0]:.3f}"
                     f"  second {r['acc_correct_listed_second'][0]:.3f}")
    L.append('\n== psych101 value sensitivity (logistic weight per 10 points of last-observed value difference; perseveration)')
    for ag, d in res['psych101'].items():
        L.append(f"{ag:26s} " + '  '.join(f"h={h}: v {r['value_per10']:+.2f} (pp median {r['value_per10_median_participant']:+.2f},"
                                           f" {r['frac_participants_value_pos']:.0%}>0) pers {r['persev']:+.2f}" for h, r in d.items()))
    L.append('\n== psych101 staleness (value weight fresh <= 2 vs stale >= 4 trials; ratio stale/fresh)')
    for ag, d in res['psych101_staleness'].items():
        L.append(f"{ag:26s} " + '  '.join(f"h={h}: fresh {r['fresh']:+.2f} stale {r['stale']:+.2f} ratio {r['ratio']:.2f} "
                   f"delta {r['stale_minus_fresh']:+.2f} CI {r['stale_minus_fresh_ci95']}" for h, r in d.items()))
    (OUT / 'summary.txt').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
