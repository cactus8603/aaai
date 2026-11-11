"""Summarize positive controls: mean P(correct/follow-A), twin-averaged (aligned logit) where available."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(Path(run) / 'controls.jsonl') if l.strip()]
    print('==', Path(run).name, len(rows))
    perc = defaultdict(list)
    for r in rows:
        if r['control'] == 'perception':
            perc[r['domain']].append(r['y'])
    for d, v in perc.items():
        print(f'  perception/{d}: meanP={np.mean(v):.3f} acc={np.mean(np.array(v) > .5):.3f} n={len(v)}')
    g = defaultdict(dict)
    for r in rows:
        if r['control'] == 'perception':
            continue
        g[(r['control'], r['domain'], r['format'])][(r['episode_id'], r['t'])] = r['y']
    for k, d in sorted(g.items()):
        raw = np.array(list(d.values()))
        pairs = [1 / (1 + np.exp(-(lg(y) + lg(d[(e + 't', t)])) / 2)) for (e, t), y in d.items()
                 if not e.endswith('t') and (e + 't', t) in d]
        print(f'  {"/".join(k)}: raw={raw.mean():.3f} follow>0.5={np.mean(raw > .5):.3f} '
              f'twin_avg={np.mean(pairs) if pairs else float("nan"):.3f} n={len(raw)}')
