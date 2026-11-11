"""Generate volatile-trust episode skeletons (Behrens-style stable/volatile advisor fidelity).

Each episode is a fixed binary sequence shared by all rendering conditions (domain x format),
so condition effects are paired within episode. Reporter A is, by definition, the reporter
favoured at trial 1. Aliases are drawn from single-token names (verified for all used tokenizers).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['Tom', 'Ben', 'Sam', 'Leo', 'Dan', 'Joe', 'Ted', 'Jim',
         'Tim', 'Ken', 'Ian', 'Jon', 'Ron', 'Hal', 'Lou', 'Roy']
DOMAINS = {'color': ('red', 'blue'), 'pet': ('cat', 'dog')}
T, WINDOW = 80, 10
LEGACY_V1 = False   # --legacy-v1 reproduces the 3090 pilot data (no header_a_first, unrounded p_A)
HIGH, LOW = 0.8, 0.2


def schedule(order):
    """Return per-trial (p_A, phase). A is favoured at trial 1 in both orders."""
    stable = lambda p: [(p, 'stable')] * 40
    def volatile(first):
        out, p = [], first
        for _ in range(4):
            out += [(p, 'volatile')] * WINDOW
            p = HIGH + LOW - p
        return out
    if order == 'SV':          # stable 0.8 -> volatile 0.2,0.8,0.2,0.8
        return stable(HIGH) + volatile(LOW)
    if order == 'VS':          # volatile 0.8,0.2,0.8,0.2 -> stable 0.8
        return volatile(HIGH) + stable(HIGH)
    raise ValueError(order)


def episode(eid, order, split, rng, image_pools):
    sched = schedule(order)
    a_correct, truth_bit = [], []
    for w in range(T // WINDOW):
        p = sched[w * WINDOW][0]
        k = round(WINDOW * p)
        block = [1] * k + [0] * (WINDOW - k)
        rng.shuffle(block)
        truths = [0] * (WINDOW // 2) + [1] * (WINDOW // 2)   # balanced truth label per window
        rng.shuffle(truths)
        a_correct += block
        truth_bit += truths
    a_name, b_name = rng.sample(NAMES, 2)
    first_listed = [rng.random() < 0.5 for _ in range(T)]      # True -> A listed first in that trial
    polarity = rng.randrange(2)                                  # maps truth bit to label index
    trials = []
    for t in range(T):
        truth = truth_bit[t]
        a_rep = truth if a_correct[t] else 1 - truth
        trials.append(dict(t=t + 1, p_a=sched[t][0] if LEGACY_V1 else round(sched[t][0], 2), phase=sched[t][1], a_correct=a_correct[t],
                           truth_bit=truth, a_report_bit=a_rep, b_report_bit=1 - a_rep,
                           a_listed_first=first_listed[t]))
    images = {}
    pool = image_pools[split]
    for t in range(T):
        images[t + 1] = {lab: rng.choice(pool[lab])['id'] for lab in ('cat', 'dog')}
    reversals = [t + 1 for t in range(1, T) if sched[t][0] != sched[t - 1][0]]
    return dict(episode_id=eid, split=split, order=order, a_name=a_name, b_name=b_name,
                label_polarity=polarity, reversals=reversals, trials=trials, pet_image_ids=images)


def load_pools():
    pools = {}
    for split, fname in (('development', 'dev128.jsonl'), ('held_out', 'heldout256.jsonl')):
        rows = [json.loads(l) for l in open(ROOT / 'research/source_trust/natural_images_v1' / fname)]
        pools[split] = {lab: [r for r in rows if r['label'] == lab] for lab in ('cat', 'dog')}
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'episodes_v3')
    ap.add_argument('--dev', type=int, default=16)
    ap.add_argument('--heldout', type=int, default=48)
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--legacy-v1', action='store_true', help='reproduce the pilot episodes_v1 exactly')
    args = ap.parse_args()
    global LEGACY_V1
    LEGACY_V1 = args.legacy_v1
    pools = load_pools()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, n, offset in (('development', args.dev, 0), ('held_out', args.heldout, 1000)):
        for i in range(n):
            rng = random.Random(args.seed * 7919 + offset + i)
            order = 'SV' if i % 2 == 0 else 'VS'
            ep = episode(f'{split[:3]}{i:03d}', order, split, rng, pools)
            # Which name the instructions introduce first; balanced within each order, flipped in twins.
            if not LEGACY_V1:
                ep['header_a_first'] = (i // 2) % 2 == 0
                # cover story: blocks of 4 episodes (both orders x both header orders) share a scenario,
                # so scenario is crossed with order and header order rather than confounded with them
                from scenarios import TEXT_SCENARIOS
                ep['scenario'] = TEXT_SCENARIOS[(i // 4) % len(TEXT_SCENARIOS)]
            rows.append(ep)
    # Image-record lookup (paths and hashes) so the runner never touches the pools by index.
    image_index = {r['id']: dict(label=r['label'], path224=r['images']['224']['path'],
                                 sha224=r['images']['224']['sha256'],
                                 path448=r['images']['448']['path'],
                                 sha448=r['images']['448']['sha256'], split=r['official_split'])
                   for split in pools.values() for lab in split.values() for r in lab}
    with open(args.out / 'episodes.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    orig_digest = hashlib.sha256((args.out / 'episodes.jsonl').read_bytes()).hexdigest()
    # Twins: names swapped (A keeps its role, takes the other name) and label polarity flipped;
    # identical schedule/outcomes/images. Averaging logits over a pair cancels additive name/label priors.
    import copy
    with open(args.out / 'episodes.jsonl', 'a') as f:
        for e in rows:
            t = copy.deepcopy(e)
            t['episode_id'] = e['episode_id'] + 't'
            t['a_name'], t['b_name'] = e['b_name'], e['a_name']
            t['label_polarity'] = 1 - e['label_polarity']
            if not LEGACY_V1:
                t['header_a_first'] = not e['header_a_first']
            t['twin_of'] = e['episode_id']
            f.write(json.dumps(t) + '\n')
    (args.out / 'image_index.json').write_text(json.dumps(image_index, indent=1))
    # Validation
    for r in rows:   # validate originals (twins share trials)
        tr = r['trials']
        for w in range(0, T, WINDOW):
            blk = tr[w:w + WINDOW]
            assert sum(x['a_correct'] for x in blk) == round(WINDOW * blk[0]['p_a'])
            assert sum(x['truth_bit'] for x in blk) == WINDOW // 2
        assert len(r['reversals']) == 4
    dev_ids = {i for r in rows if r['split'] == 'development' for v in r['pet_image_ids'].values() for i in v.values()}
    ho_ids = {i for r in rows if r['split'] == 'held_out' for v in r['pet_image_ids'].values() for i in v.values()}
    assert not dev_ids & ho_ids
    digest = orig_digest
    protocol = dict(T=T, window=WINDOW, high=HIGH, low=LOW, orders=['SV', 'VS'], names=NAMES,
                    domains=DOMAINS, seed=args.seed, n_development=args.dev, n_held_out=args.heldout,
                    episodes_sha256=digest,
                    episodes_sha256_with_twins=hashlib.sha256((args.out / 'episodes.jsonl').read_bytes()).hexdigest(),
                    twins='Each episode X has twin Xt: names swapped and label polarity flipped; same schedule/outcomes/images.',
                    notes=['Every trial is a conflict trial; truth == A report with prob p_A(t).',
                           'Realized A accuracy stratified per 10-trial window; truth balanced per window.',
                           'Reporter A = favoured at trial 1. Within-trial listing order randomized.',
                           'Pet images: development from trainval pool, held_out from test pool.',
                           'header_a_first: which name the instructions mention first; balanced, flipped in twins.',
                           'scenario (v3): cover story, blocks of 4 episodes per scenario; twins share it.'])
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=1))
    print(json.dumps(dict(episodes=len(rows), sha256=digest)))


if __name__ == '__main__':
    main()
