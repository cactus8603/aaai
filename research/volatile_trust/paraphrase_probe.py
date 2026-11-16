"""Exploratory robustness check: re-run the main task with reworded prompts (text reports, choice readout).

The frozen runner (run_volatile.py) is not modified. This wrapper replaces its prompt-building functions
(header, history, summary, and the text of Renderer.choice) with one of two alternative wordings, then calls
run_volatile.main() unchanged. Episodes, names, answer words, list order and readout are identical; only the
wording and line layout change.

  p1: reworded sentences, same layout (one sentence per trial).
  orig: the runner's own prompts (baseline on the same machine).
  p2: terse record layout ("#t | name=label | ... | truth=label").
  H<x>-L<x>-Q<x>: mix of components (x = o|1|2), e.g. Ho-L2-Qo = original prompt with terse history lines.

Usage: python paraphrase_probe.py --wording orig|p1|p2|H<x>-L<x>-Q<x> [--show] -- <run_volatile.py arguments, --formats text only>
  --show prints example prompts for one development episode (no model needed) and exits.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'source_trust'))
import run_volatile as rv  # noqa: E402
from scenarios import SCENARIOS, plural  # noqa: E402

WORDINGS = ('orig', 'p1', 'p2')   # orig = unchanged runner prompts (same-platform baseline)
ORIG = dict(header=rv.header, history=rv.history, summary=rv.summary, choice=rv.Renderer.choice)


def parse(w):
    """'orig' | 'p1' | 'p2' | component mix 'H<x>-L<x>-Q<x>' with x in o,1,2 (o = original).
    H = task description (header); L = history lines and count summary; Q = current-trial lines and question."""
    if w in WORDINGS:
        return w, w, w
    m = {'o': 'orig', '1': 'p1', '2': 'p2'}
    parts = w.split('-')
    assert [p[0] for p in parts] == ['H', 'L', 'Q'] and all(p[1:] in m for p in parts), w
    return tuple(m[p[1:]] for p in parts)


def _pair(tr, a, b):
    return (a, b) if tr['a_listed_first'] else (b, a)


def make(w):
    def header(ep, domain, hint=True):
        S = SCENARIOS[domain]
        first, second = ((ep['a_name'], ep['b_name']) if ep.get('header_a_first', True)
                         else (ep['b_name'], ep['a_name']))
        U, Us = S['unit'], plural(S['unit'])
        if w == 'p1':
            s = (f"Two {S['roles']}, {first} and {second}, take part in a series of {Us}. In each {U}, they "
                 f"{S['setup']}. Once each {U} is over, {S['reveal']}.")
            if hint:
                s += f" How accurate each {S['role']} is can shift over time."
        else:
            s = (f"Task setup. Unit: {U}. {S['roles'].capitalize()}: {first}, {second}. Each {U}, they "
                 f"{S['setup']}; afterwards, {S['reveal']}.")
            if hint:
                s += f" Note: {S['role']} accuracy is not fixed."
        return s + '\n'

    def summary(ep, t, start, domain='color'):
        Us = plural(SCENARIOS[domain]['unit'])
        if t == 1 or start >= t:
            return (f'No previous {Us} yet.\n' if w == 'p1' else f'Record: empty (no {Us} played).\n')
        seg = ep['trials'][start - 1:t - 1]
        a, n = sum(x['a_correct'] for x in seg), len(seg)
        names = [(ep['a_name'], a), (ep['b_name'], n - a)]
        if not ep['trials'][t - 1]['a_listed_first']:
            names.reverse()
        assert start == 1, 'paraphrase probe supports the all-trials summary only'
        if w == 'p1':
            return (f'Tally over the {n} previous {Us}: {names[0][0]} was right {names[0][1]} times and '
                    f'{names[1][0]} was right {names[1][1]} times.\n')
        return (f'Correct counts so far ({n} {Us}): {names[0][0]}={names[0][1]}, {names[1][0]}={names[1][1]}.\n')

    def history(ep, domain, t, start=1, marker=None, as_summary=False, fmt='raw'):
        assert start == 1 and marker is None, 'paraphrase probe supports full history only'
        S = SCENARIOS[domain]
        U, Us = S['unit'], plural(S['unit'])
        if as_summary:
            return summary(ep, t, start, domain)
        if t == 1:
            return (f'No previous {Us} yet.\n' if w == 'p1' else f'Record: empty (no {Us} played).\n')
        lines = [f'Previous {Us}:' if w == 'p1' else 'Record:']
        prep = 'On' if U == 'day' else 'In'
        pol = ep['label_polarity']
        A, B = ep['a_name'], ep['b_name']
        for tr in ep['trials'][:t - 1]:
            la, lb = rv.label(domain, tr['a_report_bit'], pol), rv.label(domain, tr['b_report_bit'], pol)
            truth = rv.label(domain, tr['truth_bit'], pol)
            ca = tr['a_correct']
            if w == 'p1':
                ok = _pair(tr, f"{A} was {'right' if ca else 'wrong'}", f"{B} was {'wrong' if ca else 'right'}")
                if fmt == 'coded':
                    lines.append(f"{prep} {U} {tr['t']}, {ok[0]} and {ok[1]}.")
                    continue
                rep = _pair(tr, f'{A} said {la}', f'{B} said {lb}')
                line = f"{prep} {U} {tr['t']}, {rep[0]} and {rep[1]}; it turned out {truth}."
                if fmt == 'raw_coded':
                    line += f' So {ok[0]} and {ok[1]}.'
            else:
                ok = _pair(tr, f"{A}={'correct' if ca else 'wrong'}", f"{B}={'wrong' if ca else 'correct'}")
                if fmt == 'coded':
                    lines.append(f"#{tr['t']} | {ok[0]} | {ok[1]}")
                    continue
                rep = _pair(tr, f'{A}={la}', f'{B}={lb}')
                line = f"#{tr['t']} | {rep[0]} | {rep[1]} | truth={truth}"
                if fmt == 'raw_coded':
                    line += f' | {ok[0]} | {ok[1]}'
            lines.append(line)
        return '\n'.join(lines) + '\n'

    def choice(self, ep, domain, fmt, t, hint=True, window=(1, None, False), hfmt='raw'):
        assert fmt == 'text', 'paraphrase probe is text reports only'
        S = SCENARIOS[domain]
        tr = ep['trials'][t - 1]
        pol = ep['label_polarity']
        labels = rv.DOMAINS[domain]
        reps = [(ep['a_name'], rv.label(domain, tr['a_report_bit'], pol)),
                (ep['b_name'], rv.label(domain, tr['b_report_bit'], pol))]
        if not tr['a_listed_first']:
            reps.reverse()
        text = rv.header(ep, domain, hint) + rv.history(ep, domain, t, *window, fmt=hfmt)
        if w == 'p1':
            text += f"Now, {S['unit']} {t} ({S['pending']}).\n"
            text += ''.join(f'{n} says {lab}.\n' for n, lab in reps)
            text += (f"What is your prediction for the {S['noun']} in {S['unit']} {t}? "
                     f"Reply with one lowercase word, either {labels[0]} or {labels[1]}.")
        else:
            text += f"#{t} (current) | " + ' | '.join(f'{n}={lab}' for n, lab in reps) + ' | truth=?\n'
            text += f"Predict truth for #{t}. Output only {labels[0]} or {labels[1]}."
        return dict(content=[{'type': 'text', 'text': text}], images=[]), list(labels), \
            rv.label(domain, tr['a_report_bit'], pol)

    return header, history, summary, choice


def install(w):
    hw, lw, qw = parse(w)
    rv.header = make(hw)[0] if hw != 'orig' else ORIG['header']
    if lw != 'orig':
        _, rv.history, rv.summary, _ = make(lw)
    else:
        rv.history, rv.summary = ORIG['history'], ORIG['summary']
    rv.Renderer.choice = make(qw)[3] if qw != 'orig' else ORIG['choice']


def show(w):
    install(w)
    eps = [json.loads(l) for l in open(HERE / 'episodes_v3/episodes.jsonl')]
    ep = next(e for e in eps if e['split'] == 'development')
    r = rv.Renderer({}, 224)
    for hf, win in (('raw', (1, None, False)), ('raw_coded', (1, None, False)), ('coded', (1, None, False)),
                    ('raw', (1, None, True))):
        for dom in ('pet', ep['scenario']):
            c, labs, tgt = r.choice(ep, dom, 'text', 4, True, win, hf)
            txt = ''.join(x['text'] for x in c['content'] if x['type'] == 'text')
            print(f'===== {w} {hf} summary={win[2]} {dom} target={tgt}\n{txt}\n')


def split_argv(argv):
    """Take --wording W and --show from anywhere in argv; everything else (minus a bare '--') goes to the runner.
    Accepts both 'paraphrase_probe.py --wording W -- <runner args>' and run_jobs.py's
    'paraphrase_probe.py --model-dir ... --output ... --wording W <runner args>'."""
    rest, w, show, it = [], None, False, iter(argv)
    for a in it:
        if a == '--wording':
            w = next(it)
        elif a == '--show':
            show = True
        elif a != '--':
            rest.append(a)
    assert w is not None, '--wording is required'
    parse(w)
    return w, show, rest


if __name__ == '__main__':
    w, show_only, rest = split_argv(sys.argv[1:])
    if show_only:
        show(w)
        sys.exit()
    sys.argv = [sys.argv[0]] + rest
    assert '--formats' in rest and rest[rest.index('--formats') + 1] == 'text' and \
        (len(rest) == rest.index('--formats') + 2 or rest[rest.index('--formats') + 2].startswith('--')), \
        'paraphrase probe: --formats text only'
    install(w)
    out = Path(rest[rest.index('--output') + 1])
    out.mkdir(parents=True, exist_ok=True)
    wf = out / '_WORDING'
    if wf.exists() and wf.read_text().strip() != w:
        sys.exit(f'Refusing to resume: {out} was made with wording {wf.read_text().strip()!r}, not {w!r}')
    wf.write_text(w + '\n')
    print(f'wording: {w} (H, L, Q = {parse(w)})', flush=True)
    rv.main()
