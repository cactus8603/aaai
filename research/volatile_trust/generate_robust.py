"""Robustness episodes (Astra round-1 2D/5): IID Bernoulli correctness, unbalanced truth labels,
irregular volatile dwell times (6-16 trials). Held-out only; same schema as episodes_v1 (+twins)."""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_episodes import NAMES, load_pools  # noqa: E402

T, HIGH, LOW = 80, 0.8, 0.2


def irregular_volatile(first, n, rng):
    out, p = [], first
    while len(out) < n:
        d = rng.randint(6, 16)
        out += [(p, 'volatile')] * d
        p = HIGH + LOW - p
    return out[:n]


def episode(eid, order, rng, pool):
    if order == 'SV':
        sched = [(HIGH, 'stable')] * 40 + irregular_volatile(LOW, 40, rng)
    else:
        v = irregular_volatile(HIGH, 40, rng)
        last = v[-1][0]
        sched = v + [(HIGH + LOW - last, 'stable')] * 40   # reversal entering stable phase
    trials = []
    for t, (p, ph) in enumerate(sched, 1):
        ac = int(rng.random() < p)
        truth = rng.randrange(2)
        a_rep = truth if ac else 1 - truth
        trials.append(dict(t=t, p_a=p, phase=ph, a_correct=ac, truth_bit=truth, a_report_bit=a_rep,
                           b_report_bit=1 - a_rep, a_listed_first=rng.random() < 0.5))
    a, b = rng.sample(NAMES, 2)
    imgs = {t: {lab: rng.choice(pool[lab])['id'] for lab in ('cat', 'dog')} for t in range(1, T + 1)}
    rev = [t + 1 for t in range(1, T) if sched[t][0] != sched[t - 1][0]]
    return dict(episode_id=eid, split='held_out', order=order, a_name=a, b_name=b,
                label_polarity=rng.randrange(2), reversals=rev, trials=trials, pet_image_ids=imgs,
                generator='robust_iid_irregular')


def main(n=24, seed=20261112):
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'episodes_robust'
    out.mkdir(exist_ok=True)
    pools = load_pools()
    rows = []
    for i in range(n):
        rng = random.Random(seed * 31 + i)
        rows.append(episode(f'rob{i:03d}', 'SV' if i % 2 == 0 else 'VS', rng, pools['held_out']))
    twins = []
    for e in rows:
        t = copy.deepcopy(e)
        t.update(episode_id=e['episode_id'] + 't', a_name=e['b_name'], b_name=e['a_name'],
                 label_polarity=1 - e['label_polarity'], twin_of=e['episode_id'])
        twins.append(t)
    with open(out / 'episodes.jsonl', 'w') as f:
        for r in rows + twins:
            f.write(json.dumps(r) + '\n')
    src = (out.parent / 'episodes_v1' / 'image_index.json')
    if not src.exists():
        src = Path(__file__).parent / 'episodes_v1' / 'image_index.json'
    (out / 'image_index.json').write_text(src.read_text())
    (out / 'protocol.json').write_text(json.dumps(dict(
        n=n, seed=seed, T=T, dwell='uniform int 6-16 in volatile phase', correctness='IID Bernoulli(p_A)',
        truth='IID uniform', reversals_per_episode=[len(r['reversals']) for r in rows],
        sha256=hashlib.sha256((out / 'episodes.jsonl').read_bytes()).hexdigest()), indent=1))
    print(len(rows) + len(twins), [len(r['reversals']) for r in rows])


if __name__ == '__main__':
    main()
