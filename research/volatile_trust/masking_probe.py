"""Exploratory follow-up: which part of the raw record hurts when correctness is also stated?

The raw-plus-correctness history recovers only part of the gap to the correctness-coded history. This probe keeps
that history's layout and length and masks parts of each past-trial line with '---':

  raw_coded (existing arm): Day 3: Ken said dog; Sam said cat. Outcome: dog. (Ken was correct; Sam was wrong.)
  rep_ok :                  Day 3: Ken said dog; Sam said cat. Outcome: ---. (Ken was correct; Sam was wrong.)
  out_ok :                  Day 3: Ken said ---; Sam said ---. Outcome: dog. (Ken was correct; Sam was wrong.)
  mask_ok:                  Day 3: Ken said ---; Sam said ---. Outcome: ---. (Ken was correct; Sam was wrong.)

mask_ok carries exactly the correctness-coded information at (nearly) the raw-plus-correctness length, so
mask_ok vs coded tests length/layout, and the 2x2 (reports shown/masked x outcome shown/masked) tests which labels
hurt. Header, current-trial lines and question are the runner's own. The frozen runner is not modified: this
wrapper replaces run_volatile.history and registers the new history formats, then calls run_volatile.main().

Usage: python masking_probe.py <run_volatile.py arguments> --history-format rep_ok|out_ok|mask_ok --formats text
       python masking_probe.py --show     (prints example lines, no model needed)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_volatile as rv  # noqa: E402
from run_volatile import SCENARIOS, plural, label  # noqa: E402

MASK = '---'
VARIANTS = {'rep_ok': (True, False), 'out_ok': (False, True), 'mask_ok': (False, False)}  # (reports, outcome) shown
ORIG_HISTORY = rv.history


def history(ep, domain, t, start=1, marker=None, as_summary=False, fmt='raw'):
    if fmt not in VARIANTS:
        return ORIG_HISTORY(ep, domain, t, start, marker, as_summary, fmt)
    assert start == 1 and marker is None and not as_summary, 'masking probe supports the full history only'
    show_rep, show_out = VARIANTS[fmt]
    S = SCENARIOS[domain]
    U, Us = S['unit'], plural(S['unit'])
    if t == 1:
        return f'Past {Us}: none yet.\n'
    lines = [f'Past {Us}:']
    pol = ep['label_polarity']
    for tr in ep['trials'][:t - 1]:
        la = label(domain, tr['a_report_bit'], pol) if show_rep else MASK
        lb = label(domain, tr['b_report_bit'], pol) if show_rep else MASK
        out = label(domain, tr['truth_bit'], pol) if show_out else MASK
        a, b = f"{ep['a_name']} {S['verb']} {la}", f"{ep['b_name']} {S['verb']} {lb}"
        first, second = (a, b) if tr['a_listed_first'] else (b, a)
        ca = f"{ep['a_name']} was {'correct' if tr['a_correct'] else 'wrong'}"
        cb = f"{ep['b_name']} was {'wrong' if tr['a_correct'] else 'correct'}"
        c1, c2 = (ca, cb) if tr['a_listed_first'] else (cb, ca)
        lines.append(f"{U.capitalize()} {tr['t']}: {first}; {second}. {S['outcome']}: {out}. ({c1}; {c2}.)")
    return '\n'.join(lines) + '\n'


def install():
    rv.history = history
    rv.HISTORY_FORMATS = tuple(rv.HISTORY_FORMATS) + tuple(VARIANTS)


def show():
    install()
    eps = [json.loads(x) for x in open(HERE / 'episodes_v3/episodes.jsonl')]
    ep = next(e for e in eps if e['split'] == 'development')
    r = rv.Renderer({}, 224)
    for hf in ('raw_coded',) + tuple(VARIANTS) + ('coded',):
        for dom in ('pet', ep['scenario']):
            c, _, tgt = r.choice(ep, dom, 'text', 4, True, (1, None, False), hf)
            print(f'===== {hf} {dom} target={tgt}\n' + ''.join(x['text'] for x in c['content'] if x['type'] == 'text'))


if __name__ == '__main__':
    if '--show' in sys.argv:
        show()
        sys.exit()
    a = sys.argv[1:]
    assert a[a.index('--formats') + 1] == 'text' and a[a.index('--history-format') + 1] in VARIANTS, \
        'masking probe: --formats text and --history-format rep_ok|out_ok|mask_ok'
    install()
    rv.main()
