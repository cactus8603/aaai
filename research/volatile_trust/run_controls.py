"""Positive controls (Astra round-1, section 2E). Choice readout = P(predict A's current report).

C1 perception      : one image, name its color/animal.
C2 explicit_rel    : "A is correct 80% of the time; B 20%." + current reports.
C3 explicit_counts : "In the last 10 trials, A was correct 8 times and B 2 times."
C4 easy_short      : 10-trial history, A correct on every trial.
C5 easy_long       : 70 trials at 50/50 then 10 trials with A always correct (80-trial prompt).
Each episode (incl. twins) contributes 4 query trials (t in {20, 40, 60, 80}) per control/domain/format.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'research/source_trust'))
from run_volatile import DOMAINS, NOUN, Renderer, header, label  # noqa: E402
from model_backends import FrozenVLM  # noqa: E402

QUERY_T = (20, 40, 60, 80)


def synth_history(ep, domain, pattern, rng):
    """pattern: list of a_correct bits. Returns text history lines with ep names/polarity."""
    pol = ep['label_polarity']
    lines = ['Past trials:']
    for i, ac in enumerate(pattern, 1):
        truth = rng.randrange(2)
        a_rep = truth if ac else 1 - truth
        a = f"{ep['a_name']} reports {label(domain, a_rep, pol)}"
        b = f"{ep['b_name']} reports {label(domain, 1 - a_rep, pol)}"
        first, second = (a, b) if rng.random() < 0.5 else (b, a)
        lines.append(f"Trial {i}: {first}; {second}. True {NOUN[domain]}: {label(domain, truth, pol)}.")
    return '\n'.join(lines) + '\n'


def build(renderer, ep, domain, fmt, t, control, rng):
    """Return rendered, labels, target for the choice question with a control-specific context."""
    names = (ep['a_name'], ep['b_name']) if rng.random() < 0.5 else (ep['b_name'], ep['a_name'])
    if control == 'explicit_rel':
        rel = {ep['a_name']: 80, ep['b_name']: 20}
        ctx = (f"Summary of past performance: {names[0]} was correct on {rel[names[0]]}% of past trials; "
               f"{names[1]} was correct on {rel[names[1]]}% of past trials.\n")
    elif control == 'explicit_counts':
        cnt = {ep['a_name']: 8, ep['b_name']: 2}
        ctx = (f"Summary of the last 10 trials: {names[0]} was correct {cnt[names[0]]} times; "
               f"{names[1]} was correct {cnt[names[1]]} times.\n")
    elif control == 'easy_short':
        ctx = synth_history(ep, domain, [1] * 10, rng)
    elif control == 'easy_long':
        noise = [1] * 35 + [0] * 35
        rng.shuffle(noise)
        ctx = synth_history(ep, domain, noise + [1] * 10, rng)
    else:
        raise ValueError(control)
    rendered, labels, target = renderer.choice(ep, domain, fmt, t)
    # Replace the header+history text (first content block) with header + control context.
    first = rendered['content'][0]['text']
    tail = first[first.index('Current trial'):]
    n_hist = {'easy_short': 10, 'easy_long': 80}.get(control)
    tail = tail.replace(f'Current trial {t} ', 'Current trial ' if n_hist is None else f'Current trial {n_hist + 1} ', 1)
    rendered['content'][0]['text'] = header(ep, domain) + ctx + tail
    return rendered, labels, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'episodes_v1')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pixels', type=int, choices=(224, 448), default=224)
    ap.add_argument('--sharded', action='store_true')
    ap.add_argument('--formats', nargs='+', default=['text', 'image'])
    ap.add_argument('--controls', nargs='+', default=['perception', 'explicit_rel', 'explicit_counts', 'easy_short', 'easy_long'])
    args = ap.parse_args()
    eps = [json.loads(l) for l in open(args.data_dir / 'episodes.jsonl')]
    eps = [e for e in eps if e['split'] == 'development']
    renderer = Renderer(json.loads((args.data_dir / 'image_index.json').read_text()), args.pixels)
    if args.sharded:
        from backend_multi import ShardedVLM
        backend = ShardedVLM(args.model_dir, args.model_id, pixels=args.pixels)
    else:
        backend = FrozenVLM(args.model_dir, args.model_id, pixels=args.pixels)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run_metadata.json').write_text(json.dumps(backend.metadata, indent=1, default=str))
    out = open(args.output / 'controls.jsonl', 'w')
    # C1 perception
    seen = set()
    for ep in (eps if 'perception' in args.controls else []):
        for t in QUERY_T:
            for domain in DOMAINS:
                for lab in DOMAINS[domain]:
                    key = (domain, lab) if domain == 'color' else ep['pet_image_ids'][str(t)][lab]
                    if key in seen:
                        continue
                    seen.add(key)
                    img = renderer.image(domain, lab, ep, t)
                    labels = list(DOMAINS[domain])
                    rendered = dict(content=[{'type': 'image'}, {'type': 'text', 'text':
                                    f"\nWhat {NOUN[domain]} is shown? Answer with exactly one lowercase word: "
                                    f"{labels[0]} or {labels[1]}."}], images=[img])
                    row = backend.score(rendered, labels, max_new_tokens=1)
                    out.write(json.dumps(dict(control='perception', domain=domain, label=lab,
                                              y=row['probabilities'][lab], greedy=row['greedy_answer'])) + '\n')
    for ep in eps:
        for t in QUERY_T:
            for control in [c for c in args.controls if c != 'perception']:
                for domain in DOMAINS:
                    for fmt in args.formats:
                        r2 = random.Random(f"{ep['episode_id']}-{t}-{control}-{domain}-{fmt}")
                        rendered, labels, target = build(renderer, ep, domain, fmt, t, control, r2)
                        row = backend.score(rendered, labels, max_new_tokens=1)
                        out.write(json.dumps(dict(control=control, episode_id=ep['episode_id'], t=t,
                                                  domain=domain, format=fmt, y=row['probabilities'][target],
                                                  greedy=row['greedy_answer'], target=target,
                                                  input_tokens=row['input_tokens'])) + '\n')
            out.flush()
    out.close()
    print('done')


if __name__ == '__main__':
    main()
