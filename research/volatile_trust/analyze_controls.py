"""Positive controls (E1, E4): twin-processed hard accuracy per control with explicit item keys.

Every control row gives y = P(target) where target is the correct answer, plus (new runs) the raw
candidate logits; the target-minus-other logit (``logit_diff`` or ``logit_a``/``logit_target`` minus
``logit_other``) is used when present, otherwise logit(clip(y)) and the stratum is flagged "approx".

Item keys (one item = one question; the twin of an item shares the key):
  perception        (domain, label, image key)                          no twins
  chart_read        (episode pair, t, trial_asked)                      3 items per query t
  composite_read    (episode pair, t, asked source role A/B, panel 1/2) 2 items per query t
  single_trial_who  (episode pair, t, domain, format)                   "who was correct on that trial?"
  explicit_rel, explicit_counts, easy_short[_coded], easy_long[_coded], other controls:
                    (control, episode pair, t, domain, format)
Twins are averaged in logit space; an item is correct when the twin-averaged logit > 0 (0.5 at a tie).
Eligibility uses this twin-processed hard accuracy, never mean P:
  chart_read >= --chart-gate (0.8)  -> chart (B') results interpretable for the model
  explicit_rel >= --explicit-gate (0.8) -> model can use stated reliability (learning-rate analysis gate)
Rows lacking a field needed for their item key are not guessed at: they are reported with a clear
diagnostic (and the analysis exits non-zero with --strict).

CLI:
  python analyze_controls.py RUN_DIR [RUN_DIR ...] [--out-dir DIR] [--episodes PATH] [--strict]
      [--chart-gate 0.8] [--explicit-gate 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_main import (REPO, AnalysisError, _json_default, load_episodes, row_logit,  # noqa: E402
                          split_twin)

GENERIC = ('episode_id', 't', 'domain', 'format')
REQUIRED = {
    'perception': ('domain', 'label'),
    'chart_read': ('episode_id', 't', 'trial_asked'),
    'composite_read': ('episode_id', 't', 'asked_first_panel'),
    'single_trial_who': GENERIC + ('target',),
}


def required_fields(control):
    return REQUIRED.get(control, GENERIC)


class Diagnostics:
    def __init__(self):
        self.missing = defaultdict(lambda: defaultdict(int))   # control -> field -> n rows
        self.other = []

    def add_missing(self, control, fields):
        for f in fields:
            self.missing[control][f] += 1

    def messages(self):
        msgs = []
        for c, fs in sorted(self.missing.items()):
            for f, n in sorted(fs.items()):
                msgs.append(f"{c}: {n} rows lack field '{f}' needed for the item key; these rows were skipped "
                            '(re-run with the current run_controls.py, which records it)')
        return msgs + self.other


def _ep_lookup(run: Path, override=None):
    if override:
        p = Path(override)
        return load_episodes(p if p.suffix == '.jsonl' else p / 'episodes.jsonl')
    for mp in sorted(run.glob('run_metadata*')):
        try:
            recs = ([json.loads(l) for l in mp.read_text().splitlines() if l.strip()] if mp.suffix == '.jsonl'
                    else [json.loads(mp.read_text())])
        except json.JSONDecodeError:
            continue
        for m in recs:
            d = (m.get('args') or {}).get('data_dir')
            if d:
                for cand in (Path(d), REPO / d, HERE / d, HERE / Path(d).name):
                    if (cand / 'episodes.jsonl').exists():
                        return load_episodes(cand / 'episodes.jsonl')
    return None


def composite_role(r, eps):
    """('A'|'B', 1|2) for a composite_read row, or (None, reason)."""
    src = r.get('asked_source')
    name = r.get('asked_name')
    if src in ('A', 'B'):
        role = src
    else:
        if name is None and isinstance(r.get('key'), str) and r['key'].count('|') >= 3:
            name = r['key'].split('|')[3]
        name = name if name is not None else src
        if name is None:
            return None, 'asked_source'
        if eps is None or r['episode_id'] not in eps:
            return None, 'episodes (data_dir) to map the asked name to its source role'
        ep = eps[r['episode_id']]
        role = 'A' if name == ep['a_name'] else ('B' if name == ep['b_name'] else None)
        if role is None:
            return None, 'asked_source (asked name matches neither reporter)'
    panel = r.get('asked_panel')
    if panel is None:
        panel = 1 if r['asked_first_panel'] else 2
    return (role, int(panel)), None


def item_of(r, eps, diag):
    """(control, stratum, item key, twin flag) or None (diagnostic recorded)."""
    c = r.get('control')
    if c is None:
        diag.add_missing('?', ['control'])
        return None
    miss = [f for f in required_fields(c) if r.get(f) is None]
    if miss:
        diag.add_missing(c, miss)
        return None
    if c == 'perception':   # legacy rows without 'key': every row is its own image item
        return c, r['domain'], (r['domain'], r['label'], r.get('key') or r.get('image_id') or f"row{r.get('_idx')}"), False
    base, twin = split_twin(r['episode_id'])
    t = int(r['t'])
    if c == 'chart_read':
        return c, 'all', (base, t, int(r['trial_asked'])), twin
    if c == 'composite_read':
        rp, why = composite_role(r, eps)
        if rp is None:
            diag.add_missing(c, [why])
            return None
        return c, f'panel{rp[1]}', (base, t, rp[0], rp[1]), twin
    return c, f"{r['domain']}/{r['format']}", (base, t, r['domain'], r['format']), twin


def load_control_rows(run: Path):
    files = sorted(run.glob('controls*.jsonl'))
    rows, bad = [], 0
    for p in files:
        for line in open(p):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return files, rows, bad


def summarize_run(run, episodes=None, chart_gate=0.8, explicit_gate=0.8):
    run = Path(run)
    diag = Diagnostics()
    files, rows, bad = load_control_rows(run)
    if not files:
        raise AnalysisError(f'{run}: no controls*.jsonl')
    if bad:
        diag.other.append(f'{bad} unparsable lines skipped')
    eps = _ep_lookup(run, episodes)
    for i, r in enumerate(rows):
        r['_idx'] = i
    items = defaultdict(lambda: defaultdict(list))      # (control, stratum, key) -> twin -> [(l, exact, row)]
    for r in rows:
        it = item_of(r, eps, diag)
        if it is None:
            continue
        c, stratum, key, twin = it
        try:
            l, exact = row_logit(r)
        except AnalysisError as e:
            diag.other.append(f'{c}: {e}')
            continue
        items[(c, stratum, key)][twin].append((l, exact, r))
    per = defaultdict(lambda: dict(credit=[], p=[], approx=0, untwinned=0, conflicts=0, n_rows=0, keys=set()))
    for (c, stratum, key), tw in sorted(items.items(), key=lambda kv: str(kv[0])):
        s = per[(c, stratum)]
        ls, exact = [], True
        for twin, recs in tw.items():
            vals = [x[0] for x in recs]
            s['n_rows'] += len(recs)
            if max(vals) - min(vals) > 1e-6:
                s['conflicts'] += 1
            ls.append(float(np.mean(vals)))
            exact &= all(x[1] for x in recs)
        if c != 'perception' and len(ls) < 2:
            s['untwinned'] += 1
            continue
        l = float(np.mean(ls))
        s['credit'].append(1.0 if l > 0 else (0.0 if l < 0 else 0.5))
        s['p'].append(float(1 / (1 + np.exp(-l))))
        s['approx'] += int(not exact)
        s['keys'].add(key)
    table = []
    for (c, stratum), s in sorted(per.items()):
        n = len(s['credit'])
        table.append(dict(control=c, stratum=stratum, n_items=n, n_rows=s['n_rows'],
                          hard_acc=float(np.mean(s['credit'])) if n else float('nan'),
                          mean_p_twin=float(np.mean(s['p'])) if n else float('nan'),
                          untwinned_items=s['untwinned'], conflicting_items=s['conflicts'],
                          approx_items=s['approx']))
    # pooled per control
    pooled = {}
    for c in sorted({t['control'] for t in table}):
        cr = [x for (cc, _), s in per.items() if cc == c for x in s['credit']]
        pooled[c] = dict(n_items=len(cr), hard_acc=float(np.mean(cr)) if cr else float('nan'),
                         untwinned_items=sum(s['untwinned'] for (cc, _), s in per.items() if cc == c))
    # item structure checks
    if 'chart_read' in pooled:
        q = defaultdict(set)
        for (c, _), s in per.items():
            if c == 'chart_read':
                for (b, t, k) in s['keys']:
                    q[(b, t)].add(k)
        bad_q = sum(len(v) != 3 for v in q.values())
        if bad_q:
            diag.other.append(f'chart_read: {bad_q} of {len(q)} queries do not have exactly 3 asked trials')
    if 'composite_read' in pooled:
        q = defaultdict(set)
        for (c, _), s in per.items():
            if c == 'composite_read':
                for (b, t, role, panel) in s['keys']:
                    q[(b, t)].add(role)
        bad_q = sum(len(v) != 2 for v in q.values())
        if bad_q:
            diag.other.append(f'composite_read: {bad_q} of {len(q)} queries do not have both sources asked')

    def gate(c, thr):
        if c not in pooled:
            return dict(threshold=thr, status='not run')
        p = pooled[c]
        if not p['n_items']:
            return dict(threshold=thr, status='not evaluable (no twin-complete items)')
        st = 'pass' if p['hard_acc'] >= thr else 'fail'
        if p['untwinned_items']:
            st += f" (note: {p['untwinned_items']} un-twinned items excluded)"
        return dict(threshold=thr, hard_acc=p['hard_acc'], n_items=p['n_items'], status=st)
    gates = dict(chart_read=gate('chart_read', chart_gate), explicit_rel=gate('explicit_rel', explicit_gate))
    gates['explicit_rel_by_stratum'] = {t['stratum']: ('pass' if t['hard_acc'] >= explicit_gate else 'fail')
                                        for t in table if t['control'] == 'explicit_rel' and t['n_items']}
    return dict(run=str(run), n_rows=len(rows), table=table, pooled=pooled, gates=gates,
                diagnostics=diag.messages())


def to_markdown(results):
    L = ['# Positive controls (analyze_controls.py)', '',
         'hard acc = share of items whose twin-averaged target-minus-other logit is > 0 (eligibility metric). '
         'mean P is shown for reference only.', '']
    for res in results:
        L += [f"## {Path(res['run']).name} ({res['n_rows']} rows)", '',
              '| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |',
              '|---|---|---|---|---|---|---|---|']
        for t in res['table']:
            L.append(f"| {t['control']} | {t['stratum']} | {t['n_items']} | {t['hard_acc']:.3f} | {t['mean_p_twin']:.3f} | "
                     f"{t['untwinned_items']} | {t['conflicting_items']} | {t['approx_items']} |")
        L += ['', '| gate | threshold | hard acc | status |', '|---|---|---|---|']
        for g in ('chart_read', 'explicit_rel'):
            v = res['gates'][g]
            L.append(f"| {g} | {v['threshold']} | {v.get('hard_acc', float('nan')):.3f} | {v['status']} |")
        by = res['gates']['explicit_rel_by_stratum']
        if by:
            L.append('| explicit_rel by domain/format | | | ' + ', '.join(f'{k}: {v}' for k, v in sorted(by.items())) + ' |')
        if res['diagnostics']:
            L += ['', '**Diagnostics**', ''] + [f'- {m}' for m in res['diagnostics']]
        L.append('')
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--tag', default='controls')
    ap.add_argument('--episodes', default=None, help='episodes.jsonl (default: run metadata data_dir)')
    ap.add_argument('--chart-gate', type=float, default=0.8)
    ap.add_argument('--explicit-gate', type=float, default=0.8)
    ap.add_argument('--strict', action='store_true', help='exit non-zero when any diagnostic is raised')
    a = ap.parse_args(argv)
    try:
        results = [summarize_run(r, a.episodes, a.chart_gate, a.explicit_gate) for r in a.runs]
    except AnalysisError as e:
        sys.exit(f'ERROR: {e}')
    md = to_markdown(results)
    print(md)
    if a.out_dir:
        od = Path(a.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / f'{a.tag}.json').write_text(json.dumps(results, indent=1, default=_json_default))
        (od / f'{a.tag}.md').write_text(md)
    if a.strict and any(r['diagnostics'] for r in results):
        sys.exit(2)


if __name__ == '__main__':
    main()
