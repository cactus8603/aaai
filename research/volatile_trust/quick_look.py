"""Quick descriptive look at runs: reversal-aligned P(follow currently-better), phase means, label bias."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run):
    rows = []
    for p in sorted(Path(run).glob('results_shard*.jsonl')):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def main(runs):
    for run in runs:
        rows = load(run)
        if not rows:
            continue
        print(f'\n== {Path(run).name}: {len(rows)} rows, episodes={len({r["episode_id"] for r in rows})}')
        groups = defaultdict(list)
        for r in rows:
            groups[(r['readout'], r['domain'], r['format'])].append(r)
        for key, rs in sorted(groups.items()):
            # follow-better: y if A currently better else 1-y
            fb = np.array([r['y'] if r['p_a'] > 0.5 else 1 - r['y'] for r in rs])
            ph = np.array([r['phase'] for r in rs])
            t = np.array([r['t'] for r in rs])
            # trials since last reversal (or since start)
            since = []
            for r in rs:
                ep_rev = [41, 51, 61, 71] if r['order'] == 'SV' else [11, 21, 31, 41]
                prev = max([x for x in ep_rev if x <= r['t']], default=1)
                since.append(r['t'] - prev)
            since = np.array(since)
            # Label/greedy bias: fraction of greedy equal to first label
            curve = [fb[(since == k) & (t > 1)].mean() for k in range(10)]
            print(f'{key}: n={len(rs)} followBetter stable={fb[ph=="stable"].mean():.3f} '
                  f'volatile={fb[ph=="volatile"].mean():.3f} | since-reversal 0..9: ' +
                  ' '.join(f'{c:.2f}' for c in curve) +
                  f' | mass={np.mean([r["candidate_mass"] for r in rs]):.3f}')


if __name__ == '__main__':
    main(sys.argv[1:])
