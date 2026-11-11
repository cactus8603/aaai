"""Development analysis: raw and twin-averaged (aligned-logit) tracking, reversal curves, accuracy."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EPISODE_FILES = [Path(__file__).parent / f'{d}/episodes.jsonl'
                 for d in ('episodes_v1', 'episodes_robust', 'episodes_v2', 'episodes_robust_v2')]


def load_reversals():
    rev = {}
    for f in EPISODE_FILES:
        if f.exists():
            for line in open(f):
                e = json.loads(line)
                rev[e['episode_id'].rstrip('t')] = e['reversals']
    return rev


REV = load_reversals()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(run):
    rows = []
    for p in sorted(Path(run).glob('results_shard*.jsonl')):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def summarize(rows, mode_filter=None):
    by = defaultdict(dict)
    for r in rows:
        if mode_filter and r.get('history_mode', 'full') != mode_filter:
            continue
        base = r['episode_id'].rstrip('t')
        twin = r['episode_id'].endswith('t')
        by[(r['readout'], r['domain'], r['format'])][(base, r['t'], twin)] = r
    out = {}
    for key, d in sorted(by.items()):
        recs = []
        for (base, t, twin), r in d.items():
            if twin:
                continue
            r2 = d.get((base, t, True))
            sign = 1 if r['p_a'] > 0.5 else -1
            raw = r['y'] if sign > 0 else 1 - r['y']
            pair = None
            l = None
            if r2 is not None:
                l = (logit(r['y']) + logit(r2['y'])) / 2
                pair = 1 / (1 + np.exp(-sign * l))
            prior = [x for x in REV[base] if x <= t]
            since = (t - max(prior)) if prior else None      # None before the first reversal
            # accuracy: chosen reporter (by P>0.5) was correct; twin-averaged when the twin exists
            acc = (r['y'] > 0.5) == bool(r['a_correct'])
            pair_y = None if r2 is None else 1 / (1 + np.exp(-l))
            acc_pair = None if pair_y is None else float((pair_y > 0.5) == bool(r['a_correct']))
            recs.append(dict(t=t, phase=r['phase'], raw=raw, pair=pair, since=since, acc=acc, acc_pair=acc_pair,
                             logl=np.log(max(r['y'] if r['a_correct'] else 1 - r['y'], 1e-9))))
        R = lambda f, k: np.array([x[k] for x in recs if f(x) and x[k] is not None], dtype=float)
        allx = lambda x: x['t'] > 1
        res = dict(n=len(recs), acc_pair=R(lambda x: x['t'] > 1, 'acc_pair').mean() if any(x['acc_pair'] is not None for x in recs) else None,
                   raw_stable=R(lambda x: allx(x) and x['phase'] == 'stable', 'raw').mean(),
                   raw_volatile=R(lambda x: allx(x) and x['phase'] == 'volatile', 'raw').mean(),
                   acc=R(allx, 'acc').mean(), loglik=R(allx, 'logl').mean())
        if any(x['pair'] is not None for x in recs):
            res.update(pair_stable=R(lambda x: allx(x) and x['phase'] == 'stable', 'pair').mean(),
                       pair_volatile=R(lambda x: allx(x) and x['phase'] == 'volatile', 'pair').mean(),
                       pair_curve=[round(float(R(lambda x, k=k: x['since'] is not None and x['since'] == k, 'pair').mean()), 3)
                                   for k in range(10)])
        out['/'.join(key)] = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                              for k, v in res.items()}
    return out


if __name__ == '__main__':
    for run in sys.argv[1:]:
        print('==', run)
        for k, v in summarize(load(run)).items():
            print(k, json.dumps(v))
