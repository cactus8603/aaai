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
    meta = dict(backend.metadata, started=__import__('time').strftime('%Y-%m-%dT%H:%M:%S'),
                args={k: str(v) for k, v in vars(args).items()})
    with open(args.output / 'run_metadata.jsonl', 'a') as mf:
        mf.write(json.dumps(meta, default=str) + '\n')
    out_path = args.output / 'controls.jsonl'
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add(r['key'])
            except (json.JSONDecodeError, KeyError):
                pass
    out = open(out_path, 'a')

    expected = set()

    def emit(key, rendered, labels, target, **fields):
        expected.add(key)
        if key in done:
            return
        row = backend.score(rendered, labels, max_new_tokens=1)
        out.write(json.dumps(dict(key=key, y=row['probabilities'][target], target=target,
                                  argmax=row['argmax_label'], greedy=row['greedy_answer'],
                                  input_tokens=row['input_tokens'], pixels=args.pixels, **fields)) + '\n')
        done.add(key)

    # C1 perception: single images
    seen = set()
    for ep in (eps if 'perception' in args.controls else []):
        for t in QUERY_T:
            for domain in DOMAINS:
                for lab in DOMAINS[domain]:
                    iid = (domain, lab) if domain == 'color' else ep['pet_image_ids'][str(t)][lab]
                    if iid in seen:
                        continue
                    seen.add(iid)
                    labels = list(DOMAINS[domain])
                    rendered = dict(content=[{'type': 'image'}, {'type': 'text', 'text':
                                    f"\nWhat {NOUN[domain]} is shown? Answer with exactly one lowercase word: "
                                    f"{labels[0]} or {labels[1]}."}], images=[renderer.image(domain, lab, ep, t)])
                    emit(f'perception|{iid}', rendered, labels, lab, control='perception', domain=domain, label=lab)
    for ep in eps:
        base = ep['episode_id'].rstrip('t')      # twins share every random draw -> exact pairs
        for t in QUERY_T:
            # C6 chart reading: on trial k, which reporter was correct? (answer = a name)
            if 'chart_read' in args.controls:
                for j, k in enumerate((3, t // 2, t - 1)):
                    tr = ep['trials'][k - 1]
                    correct = ep['a_name'] if tr['a_correct'] else ep['b_name']
                    names = [ep['a_name'], ep['b_name']]
                    if not ep['trials'][t - 1]['a_listed_first']:
                        names.reverse()
                    rendered = dict(content=[{'type': 'text', 'text':
                                    f'This chart shows trials 1-{t - 1} (four lines of 20 trials; green = that reporter '
                                    'was correct on that trial, gray = wrong, empty = not yet played):\n'},
                                    {'type': 'image'}, {'type': 'text', 'text':
                                    f'\nAccording to the chart, which reporter was correct on trial {k}? '
                                    f'Answer with exactly one name: {names[0]} or {names[1]}.'}],
                                    images=[renderer.chart(ep, t)])
                    emit(f'chart_read|{ep["episode_id"]}|{t}|{k}', rendered, names, correct,
                         control='chart_read', episode_id=ep['episode_id'], t=t, trial_asked=k,
                         a_was_correct=tr['a_correct'])
            # C7 composite panel reading: which animal is in <name>'s panel? (tests name->panel binding)
            if 'composite_read' in args.controls:
                tr = ep['trials'][t - 1]
                pol = ep['label_polarity']
                reps = [(ep['a_name'], label('pet', tr['a_report_bit'], pol)),
                        (ep['b_name'], label('pet', tr['b_report_bit'], pol))]
                if not tr['a_listed_first']:
                    reps.reverse()
                for name, lab in reps:
                    labels = list(DOMAINS['pet'])
                    rendered = dict(content=[{'type': 'image'}, {'type': 'text', 'text':
                                    f'\nThe image has two panels, each with a name above it. What animal is in '
                                    f"{name}'s panel? Answer with exactly one lowercase word: {labels[0]} or {labels[1]}."}],
                                    images=[renderer.composite('pet', reps, ep, t)])
                    emit(f'composite_read|{ep["episode_id"]}|{t}|{name}', rendered, labels, lab,
                         control='composite_read', episode_id=ep['episode_id'], t=t,
                         asked_first_panel=(name == reps[0][0]))
            for control in [c for c in args.controls if c not in ('perception', 'chart_read', 'composite_read')]:
                for domain in DOMAINS:
                    for fmt in args.formats:
                        if fmt == 'chart':
                            continue      # history-replacing controls are undefined for the chart format
                        r2 = random.Random(f"{base}-{t}-{control}-{domain}-{fmt}")
                        rendered, labels, target = build(renderer, ep, domain, fmt, t, control, r2)
                        emit(f'{control}|{ep["episode_id"]}|{t}|{domain}|{fmt}', rendered, labels, target,
                             control=control, episode_id=ep['episode_id'], t=t, domain=domain, format=fmt)
            out.flush()
    out.close()
    if expected and expected <= done:
        (args.output / '_COMPLETE').write_text(json.dumps(dict(rows=len(expected), at=__import__('time').strftime('%F %T'))))
        print(f'COMPLETE {len(expected)} rows')
    else:
        print(f'INCOMPLETE: {len(expected - done)} missing of {len(expected)}')
        sys.exit(3)
    print('done')


if __name__ == '__main__':
    main()
