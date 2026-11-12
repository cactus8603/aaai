"""Exploratory robustness check: re-run the main task with reworded prompts (text reports, choice readout).

The frozen runner (run_volatile.py) is not modified. This wrapper replaces its prompt-building functions
(header, history, summary, and the text of Renderer.choice) with one of two alternative wordings, then calls
run_volatile.main() unchanged. Episodes, names, answer words, list order and readout are identical; only the
wording and line layout change.

  p1: reworded sentences, same layout (one sentence per trial).
  orig: the runner's own prompts (baseline on the same machine).
  p2: terse record layout ("#t | name=label | ... | truth=label").

Usage: python paraphrase_probe.py --wording p1|p2 [--show] -- <run_volatile.py arguments, --formats text only>
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
        text = header(ep, domain, hint) + history(ep, domain, t, *window, fmt=hfmt)
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
    header, history, summary, choice = make(w)
    rv.header, rv.history, rv.summary = header, history, summary
    rv.Renderer.choice = choice


def show(w):
    install(w)
    eps = [json.loads(l) for l in open(HERE / 'episodes_v3/episodes.jsonl')]
    ep = next(e for e in eps if e['split'] == 'development')
    r = rv.Renderer({}, 224)
    for hf, win in (('raw', (1, None, False)), ('raw_coded', (1, None, False)), ('coded', (1, None, False)),
                    ('raw', (1, None, True))):
        for dom in ('pet', ep['scenario']):
            c, labs, tgt = r.choice(ep, dom, 'text', 4, True, win, hf)
            print(f'===== {w} {hf} summary={win[2]} {dom} target={tgt}\n{c["content"][0]["text"]}\n')


if __name__ == '__main__':
    pre = sys.argv[1:sys.argv.index('--')] if '--' in sys.argv else sys.argv[1:]
    w = pre[pre.index('--wording') + 1]
    assert w in WORDINGS, w
    if '--show' in pre:
        show(w)
        sys.exit()
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index('--') + 1:]
    assert '--formats' in sys.argv and sys.argv[sys.argv.index('--formats') + 1:][:2][0] == 'text'
    if w != 'orig':
        install(w)
    out = Path(sys.argv[sys.argv.index('--output') + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / '_WORDING').write_text(w + '\n')
    print(f'wording: {w}', flush=True)
    rv.main()
