"""R06: normative check using EXACTLY the rules given to the VLM (controller.history_window + majority count),
on the actual v3 episodes (stratified) and on robust v3 (IID), fixed h=1/15 and dev-selectable K.
Ties (equal counts) score 0.5. Per-episode accuracies with bootstrap CIs over episodes.
Also reports sum_oracle semantics: segment starts at the last reversal <= t-1 (one-trial-delayed knowledge)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from controller import history_window

RULES = ['sum_all', 'sum_window:5', 'sum_window:10', 'sum_window:20', 'sum_cp', 'sum_oracle']


def start_for(rule, ep, o, t):
    if rule == 'sum_all':
        return 1
    if rule == 'sum_oracle':
        return max([r for r in ep['reversals'] if r <= t - 1], default=1)
    if rule == 'sum_cp':
        return history_window(o, t, 'cp_hard', h=1 / 15)[0]
    return history_window(o, t, rule.replace('sum_window', 'window'), h=1 / 15)[0]


def episode_acc(ep, rule):
    o = [x['a_correct'] for x in ep['trials']]
    sc = []
    for t in range(2, 81):
        s = start_for(rule, ep, o, t)
        seg = o[s - 1:t - 1]
        a = sum(seg); b = len(seg) - a
        sc.append(0.5 if a == b else float((a > b) == bool(o[t - 1])))
    return float(np.mean(sc))


def summarize(path, split=None):
    eps = [json.loads(l) for l in open(path)]
    eps = [e for e in eps if not e['episode_id'].endswith('t') and (split is None or e['split'] == split)]
    rng = np.random.default_rng(0)
    out = {}
    for rule in RULES:
        v = np.array([episode_acc(e, rule) for e in eps])
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        out[rule] = dict(mean=round(float(v.mean()), 4), ci=[round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)], n=len(v))
    return out


if __name__ == '__main__':
    res = {'v3_dev': summarize('episodes_v3/episodes.jsonl', 'development'),
           'v3_heldout': summarize('episodes_v3/episodes.jsonl', 'held_out'),
           'robust_v3': summarize('episodes_robust_v3/episodes.jsonl')}
    json.dump(res, open('runs/design_sim_runner_rule.json', 'w'), indent=1)
    for k, v in res.items():
        print(k, {r: (d['mean'], d['ci']) for r, d in v.items()})
