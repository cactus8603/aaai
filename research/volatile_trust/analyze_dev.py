"""Development analysis: raw and twin-averaged (aligned-logit) tracking, reversal curves, accuracy."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REV = {'SV': [41, 51, 61, 71], 'VS': [11, 21, 31, 41]}


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
            if r2 is not None:
                l = (logit(r['y']) + logit(r2['y'])) / 2
                pair = 1 / (1 + np.exp(-sign * l))
            since = t - max([x for x in REV[r['order']] if x <= t], default=1)
            # accuracy of following model's argmax: correct if chosen reporter correct
            acc = (r['y'] > 0.5) == bool(r['a_correct'])
            recs.append(dict(t=t, phase=r['phase'], raw=raw, pair=pair, since=since, acc=acc,
                             logl=np.log(max(r['y'] if r['a_correct'] else 1 - r['y'], 1e-9))))
        R = lambda f, k: np.array([x[k] for x in recs if f(x) and x[k] is not None], dtype=float)
        allx = lambda x: x['t'] > 1
        res = dict(n=len(recs),
                   raw_stable=R(lambda x: allx(x) and x['phase'] == 'stable', 'raw').mean(),
                   raw_volatile=R(lambda x: allx(x) and x['phase'] == 'volatile', 'raw').mean(),
                   acc=R(allx, 'acc').mean(), loglik=R(allx, 'logl').mean())
        if any(x['pair'] is not None for x in recs):
            res.update(pair_stable=R(lambda x: allx(x) and x['phase'] == 'stable', 'pair').mean(),
                       pair_volatile=R(lambda x: allx(x) and x['phase'] == 'volatile', 'pair').mean(),
                       pair_curve=[round(float(R(lambda x, k=k: allx(x) and x['since'] == k, 'pair').mean()), 3)
                                   for k in range(10)])
        out['/'.join(key)] = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                              for k, v in res.items()}
    return out


if __name__ == '__main__':
    for run in sys.argv[1:]:
        print('==', run)
        for k, v in summarize(load(run)).items():
            print(k, json.dumps(v))
