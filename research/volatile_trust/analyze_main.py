"""Main analysis (plan v3): raw vs. summarized history gap for source reliability.

History-format arms (``history_class``):
  A  = raw records (who reported what + revealed truth)       history_mode=full, history_format raw/None
  B  = per-trial correctness-coded text                       history_format coded/correctness/...
  B' = correctness chart image                                format=chart (or history_format chart)
  C  = count summary (controller)                             history_mode=sum_*
An *arm* is ``<class>|<history_mode>|<present format>[|nohint]``, e.g. ``A|full|text``, ``C|sum_cp|text``.

Units and cells
  * A *group* is (model@revision, dataset#sha256, split, readout). Datasets are identified by the SHA-256 of
    their episodes.jsonl, so ``dev000`` of episodes_v1 and of episodes_v3 are never merged, and dev and
    held-out episodes are never pooled. A *cell* is group x arm x domain slot ('pet' | 'scenario').
  * Every row carries a run fingerprint (model_id, model_revision, data hash, history_mode, history_format,
    pixels, hazard, readout, hint). A cell must contain exactly one fingerprint; mixing (e.g. 224 and 448
    pixels, two hazards, two coded formats) is refused. Conflicting duplicate rows are refused (formal) or
    averaged with a listing (partial); results never depend on input order.
  * Twins are averaged in *logit* space: the target-minus-other candidate logit difference
    (``logit_diff`` or ``logit_a - logit_other``). Legacy pilot rows without logits fall back to
    logit(clip(y, 1e-6)), which saturates at +-13.8; such cells are flagged "approx" (I04).

Modes (A01)
  * formal (default): every expected episode of the split, every trial t=1..T, every domain slot of the
    group and both twins must be present, and paired conditions must have identical observation masks.
    Otherwise the analysis is refused with a list of what is missing. Only formal held-out results can be
    labelled primary/secondary.
  * ``--partial``: exploratory. Missing keys are listed, un-twinned trials are dropped, paired comparisons
    use the intersection of observed trials, and nothing is labelled primary or secondary.

Tests (episode pair = unit; metrics are twin-averaged, t >= 2)
  primary   : gap = acc(C: --primary-c, ref format) - acc(A), domains pooled (equal-weight mean of the pet
              slot and the scenario slot), Wilcoxon + bootstrap CI, Holm across models. Held-out, formal only.
  secondary : TOST (VLM + summary vs the window-count rule alone, margin --tost-margin, alpha --alpha) for
              the family {sum_all, sum_cp, sum_window:K*}; K* from --k-star or --dev-selection (produced by
              --write-dev-selection on development runs). Uncorrected and Holm-corrected decisions are
              reported separately; a missing family member counts as p = 1.
  exploratory: all other arms/slots, B - A and C - B absolute differences, the normalised position
              (B-A)/(C-A) (only when C - A > --position-threshold and the bootstrap denominator stays > 0),
              per-scenario gaps with a sign test over the text scenarios (domain_slot == 'scenario' only;
              pet is reported separately). Held-out has 8 episode pairs per scenario and the sign test has
              6 units (6 scenarios; minimum two-sided p = 0.03125).
  also      : trial-level choice agreement VLM vs rule on non-tied trials + disagreement table (R03);
              candidate mass and greedy-output format following per cell (R10); history baselines (R05),
              with those that use generator knowledge the VLM does not have flagged.

CLI:
  python analyze_main.py RUN_DIR [RUN_DIR ...] --out-dir DIR [--partial] [--primary-c sum_all]
      [--ref-format text] [--tost-margin 0.02] [--alpha 0.05] [--k-star K | --dev-selection JSON]
      [--write-dev-selection JSON] [--position-threshold 0.05] [--n-boot 10000] [--episodes PATH]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from controller import bocpd, history_window  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

CODED_TEXT = {'coded', 'correctness', 'correctness_text', 'coded_text', 'checks', 'checkmarks',
              'marks', 'B'}
CODED_CHART = {'chart', 'coded_chart', 'correctness_chart', "B'", 'Bp'}
RAW = {None, '', 'raw', 'records', 'raw_records', 'A', 'None'}
DEFAULT_HAZARD = 1 / 15
N_CURVE = 10
CLIP = 1e-6
POSITION_THRESHOLD = 0.05
DEV_WINDOW_CANDIDATES = (5, 10, 20)
FP_FIELDS = ('model_id', 'model_revision', 'data_sha256', 'history_mode', 'history_format', 'pixels',
             'hazard', 'readout', 'hint')
_TWIN_RE = re.compile(r'^(.*\d)t$')
SCENARIO_NOTE = ('Each held-out text scenario has 8 episode pairs; the cross-scenario sign test has 6 units '
                 '(one per text scenario; minimum two-sided p = 0.03125). Pet is reported separately and '
                 'is not part of the cross-scenario test.')


class AnalysisError(RuntimeError):
    """Raised when the data cannot support the requested (formal) analysis."""


# ----------------------------------------------------------------------------- utilities

def logit(p):
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def row_logit(r: dict):
    """(target-minus-other logit, exact). Raw candidate logits when saved; else clipped y (approximate)."""
    v = r.get('logit_diff')
    if v is not None and np.isfinite(float(v)):
        return float(v), True
    a = r.get('logit_a', r.get('logit_target'))
    o = r.get('logit_other')
    if a is not None and o is not None and np.isfinite(float(a)) and np.isfinite(float(o)):
        return float(a) - float(o), True
    if r.get('y') is None:
        raise AnalysisError(f'row has neither logit fields nor y: {sorted(r)[:12]}')
    return float(logit(r['y'])), False


def split_twin(eid: str):
    """'dev003t' -> ('dev003', True); 'dev003' -> ('dev003', False)."""
    m = _TWIN_RE.match(eid)
    return (m.group(1), True) if m else (eid, False)


def norm_hf(hf):
    return 'raw' if hf in RAW else str(hf)


def history_class(row: dict) -> str:
    hm = row.get('history_mode') or 'full'
    hf = row.get('history_format')
    if hm.startswith('sum_'):
        return 'C'
    if row.get('format') == 'chart' or hf in CODED_CHART:
        return "B'"
    if hf == 'raw_coded':          # A+: raw records plus the correctness statement (its own arm)
        return 'A+'
    if hf in RAW:
        return 'A'
    if hf in CODED_TEXT or 'cod' in str(hf) or 'correct' in str(hf):
        return 'B'
    return str(hf)


def arm_of(row: dict) -> str:
    arm = f"{history_class(row)}|{row.get('history_mode') or 'full'}|{row['format']}"
    if row.get('hint', True) is False:
        arm += '|nohint'
    return arm


def arm_parts(arm):
    p = arm.split('|')
    return p[0], p[1], p[2]


def short_model(mid: str) -> str:
    return str(mid).rstrip('/').split('/')[-1]


def _parse_arg(v):
    if isinstance(v, str):
        if v in ('None', ''):
            return None
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _read_json_records(p: Path):
    txt = p.read_text()
    if p.suffix == '.jsonl':
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    return [json.loads(txt)]


def _read_metas(run: Path):
    metas = []
    for p in sorted(run.glob('run_metadata*')):
        try:
            metas += _read_json_records(p)
        except (json.JSONDecodeError, OSError) as e:
            raise AnalysisError(f'{p}: unreadable run metadata ({e})')
    return metas


def _read_fingerprints(run: Path):
    out = []
    for p in sorted(run.glob('_FINGERPRINT*.json')):
        out.append(json.loads(p.read_text()))
    return out


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _resolve_episode_path(metas: list, override: str | None, run: Path) -> Path:
    """No silent fallback: the dataset must be named by the run metadata or by --episodes."""
    if override:
        p = Path(override)
        p = p if p.suffix == '.jsonl' else p / 'episodes.jsonl'
        if not p.exists():
            raise AnalysisError(f'--episodes {override}: {p} not found')
        return p
    dirs = {(m.get('args') or {}).get('data_dir') for m in metas} - {None}
    if not dirs:
        raise AnalysisError(f'{run}: run metadata has no data_dir and --episodes was not given '
                            '(the episodes_v1 fallback was removed)')
    if len(dirs) > 1:
        raise AnalysisError(f'{run}: run metadata names several data_dirs {sorted(dirs)}')
    d = dirs.pop()
    for cand in (Path(d), REPO / d, HERE / d, HERE / Path(d).name):
        if (cand / 'episodes.jsonl').exists():
            return cand / 'episodes.jsonl'
    raise AnalysisError(f'{run}: data_dir {d} not found (pass --episodes)')


_EP_CACHE: dict = {}


def load_episodes(path) -> dict:
    path = Path(path).resolve()
    if path not in _EP_CACHE:
        _EP_CACHE[path] = {e['episode_id']: e for e in map(json.loads, open(path)) if e}
    return _EP_CACHE[path]


def scenario_of(ep: dict, row: dict | None = None, key: str = 'scenario') -> str:
    v = ep.get(key)
    if v is None and row is not None:
        v = row.get('domain')
    return 'na' if v is None else str(v)


def slot_of(row: dict, ep: dict) -> str:
    s = row.get('domain_slot')
    if s:
        if s == 'pet':
            return 'pet'
        # legacy datasets without a per-episode scenario: the single literal text domain is the scenario slot
        return 'scenario' if s in ('scenario', ep.get('scenario')) or ep.get('scenario') is None else str(s)
    d = row.get('domain')
    if d == 'pet':
        return 'pet'
    if ep.get('scenario') is None or d == ep.get('scenario'):
        return 'scenario'      # legacy v1/v2 rows: the single text domain (color) is the scenario slot
    return str(d)


# ----------------------------------------------------------------------------- loading

def _unique(vals, what, run):
    vals = {v for v in vals if v is not None}
    if len(vals) > 1:
        raise AnalysisError(f'{run}: run directory mixes several {what}: {sorted(map(str, vals))}')
    return vals.pop() if vals else None


def read_run(rd: Path, episodes_override=None) -> dict:
    metas = _read_metas(rd)
    fps = _read_fingerprints(rd)
    ep_path = _resolve_episode_path(metas, episodes_override, rd)
    sha = sha256_file(ep_path)
    fp_sha = _unique([f.get('data_sha256') for f in fps], 'data hashes', rd)
    if fp_sha and fp_sha != sha and not episodes_override:
        raise AnalysisError(f'{rd}: episodes file {ep_path} changed since the run (sha mismatch)')
    args = [m.get('args') or {} for m in metas]
    rev = _unique([m.get('model_revision') for m in metas] or [f.get('model_revision') for f in fps],
                  'model revisions', rd)
    hz = _unique([_parse_arg(a.get('hazard')) for a in args] + [f.get('hazard') for f in fps], 'hazards', rd)
    px = _unique([_parse_arg(a.get('pixels')) for a in args] + [f.get('pixels') for f in fps], 'pixel settings', rd)
    restr = sorted({f'{k}={a.get(k)}' for a in args for k in ('max_episodes', 'trials')
                    if _parse_arg(a.get(k)) is not None})
    return dict(path=str(rd), ep_path=str(ep_path), data_sha256=sha, data_name=ep_path.parent.name,
                model_revision=rev, hazard=float(hz) if hz is not None else DEFAULT_HAZARD,
                pixels=int(px) if px is not None else None, restricted=restr, n_rows=0, bad_lines=0)


def load_rows(run_dirs, episodes_override=None, strict=True):
    """Return (rows, info). Rows gain private fields (_cell, _l, _exact, _fp, ...). Order independent."""
    rows, info = [], {'runs': {}, 'bad_lines': [], 'restricted_runs': []}
    for rd in sorted({str(Path(r).resolve()) for r in run_dirs}):
        rd = Path(rd)
        run = read_run(rd, episodes_override)
        eps = load_episodes(run['ep_path'])
        files = sorted(rd.glob('results_shard*.jsonl'))
        if not files:
            raise AnalysisError(f'{rd}: no results_shard*.jsonl')
        for p in files:
            for i, line in enumerate(open(p), 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    run['bad_lines'] += 1
                    info['bad_lines'].append(f'{p}:{i}')
                    continue
                ep = eps.get(r['episode_id'])
                if ep is None:
                    raise AnalysisError(f'{p}:{i}: episode {r["episode_id"]} not in {run["ep_path"]}')
                r['_run'] = str(rd)
                r['_ep_path'] = run['ep_path']
                r['_hazard'] = run['hazard']
                r['_pixels'] = r.get('pixels', run['pixels'])
                r['_hf'] = norm_hf(r.get('history_format'))
                r['_arm'] = arm_of(r)
                r['_slot'] = slot_of(r, ep)
                r['_split'] = ep.get('split', 'na')
                rev = r.get('model_revision') or run['model_revision']
                r['_model'] = f"{short_model(r.get('model_id') or rd.name)}@{(rev or 'unknown')[:8]}"
                r['_data'] = f"{run['data_name']}#{run['data_sha256'][:8]}"
                r['_fp'] = (r.get('model_id'), rev, run['data_sha256'], r.get('history_mode') or 'full', r['_hf'],
                            r['_pixels'], round(run['hazard'], 12), r['readout'], r.get('hint', True))
                r['_group'] = (r['_model'], r['_data'], r['_split'], r['readout'])
                r['_cell'] = r['_group'] + (r['_arm'], r['_slot'])
                r['_l'], r['_exact'] = row_logit(r)
                rows.append(r)
                run['n_rows'] += 1
        if run['restricted']:
            info['restricted_runs'].append(f"{rd}: {', '.join(run['restricted'])}")
        info['runs'][str(rd)] = run
    if strict and info['bad_lines']:
        raise AnalysisError(f"{len(info['bad_lines'])} unparsable result lines (formal mode refuses): "
                            f"{info['bad_lines'][:5]}")
    if strict and info['restricted_runs']:
        raise AnalysisError('runs restricted by --max-episodes/--trials cannot enter a formal analysis: '
                            + '; '.join(info['restricted_runs']))
    rows.sort(key=lambda r: (r['_cell'], r['episode_id'], r['t'], r['_run'], r['_l']))
    return rows, info


# ----------------------------------------------------------------------------- twins + completeness

def _fp_diff(fps):
    fps = sorted(fps, key=str)
    return {f: sorted({str(fp[i]) for fp in fps}) for i, f in enumerate(FP_FIELDS)
            if len({fp[i] for fp in fps}) > 1}


def pair_trials(rows, scenario_key='scenario', strict=True, report=None):
    """Twin-average aligned logits. Returns {cell: {base: [trial dicts sorted by t]}}.

    cell = (model, data, split, readout, arm, slot). Trial dict: t, l (mean twin logit), py, exact,
    a_correct, p_a, phase, window_start, ep, scenario, hazard, history_mode. ``report`` (dict) collects
    duplicates, conflicts, dropped un-twinned trials and missing keys. Formal mode (strict) raises
    AnalysisError on any fingerprint mix, conflicting duplicate, missing twin/trial/episode/slot.
    """
    report = {} if report is None else report
    report.update(duplicates=0, conflicts=[], untwinned_dropped=defaultdict(int), missing={})
    problems = []
    fps = defaultdict(set)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        fps[r['_cell']].add(r['_fp'])
        base, twin = split_twin(r['episode_id'])
        groups[r['_cell']][(base, r['t'])][twin].append(r)
    mixed = {c: _fp_diff(f) for c, f in fps.items() if len(f) > 1}
    if mixed:   # refused in every mode
        raise AnalysisError('cells mix run fingerprints (refused): ' +
                            '; '.join(f"{'/'.join(c)}: {d}" for c, d in sorted(mixed.items())))
    out = {}
    for cell in sorted(groups):
        per = defaultdict(list)
        for (base, t), tw in sorted(groups[cell].items()):
            ls, exact, r0 = [], True, None
            for twin in (False, True):
                rs = tw.get(twin)
                if not rs:
                    continue
                vals = sorted(x['_l'] for x in rs)
                if len(rs) > 1:
                    report['duplicates'] += len(rs) - 1
                    if vals[-1] - vals[0] > 1e-6:
                        msg = f"{'/'.join(cell)} {base}{'t' if twin else ''} t={t}: {len(rs)} conflicting rows"
                        report['conflicts'].append(msg)
                        problems.append(msg)
                ls.append(float(np.mean(vals)))
                exact &= all(x['_exact'] for x in rs)
                r0 = r0 or rs[0]
            if len(ls) < 2:
                report['untwinned_dropped']['/'.join(cell)] += 1
                continue
            eps = load_episodes(r0['_ep_path'])
            ep = eps.get(base)
            if ep is None:
                raise AnalysisError(f'base episode {base} missing from {r0["_ep_path"]}')
            tr = ep['trials'][t - 1]
            ac = int(tr['a_correct'])
            if any(int(x['a_correct']) != ac for rs in tw.values() for x in rs):
                raise AnalysisError(f"a_correct mismatch for {'/'.join(cell)} {base} t={t}: wrong episode file?")
            l = float(np.mean(ls))
            per[base].append(dict(t=t, l=l, py=float(sigmoid(l)), exact=exact, a_correct=ac,
                                  p_a=float(tr['p_a']), phase=tr['phase'], twinned=True,
                                  window_start=r0.get('window_start'), ep=ep, hazard=r0['_hazard'],
                                  history_mode=r0.get('history_mode') or 'full',
                                  scenario='pet' if cell[5] == 'pet' else scenario_of(ep, r0, scenario_key)))
        for b in per:
            per[b].sort(key=lambda x: x['t'])
        out[cell] = dict(sorted(per.items()))
    # completeness against the expected keys of each cell
    missing = completeness(out, groups)
    report['missing'] = missing
    if report['untwinned_dropped']:
        n = sum(report['untwinned_dropped'].values())
        problems.append(f'{n} trials lack a twin')
    if missing:
        problems += [f'{c}: {m["summary"]}' for c, m in sorted(missing.items())]
    if strict and problems:
        raise AnalysisError('formal analysis refused (use --partial for an exploratory look):\n  '
                            + '\n  '.join(problems[:40]) + (f'\n  ... {len(problems) - 40} more' if len(problems) > 40 else ''))
    report['untwinned_dropped'] = dict(report['untwinned_dropped'])
    return out


def completeness(paired, raw_groups):
    """Expected keys per cell: every episode of the split x t = 1..T x both twins; every slot of the group."""
    missing = {}
    slots_by_group = defaultdict(set)
    arms_by_group = defaultdict(set)
    for cell in raw_groups:
        slots_by_group[cell[:4]].add(cell[5])
        arms_by_group[cell[:4]].add(cell[4])
    for cell, keys in raw_groups.items():
        r0 = next(iter(next(iter(keys.values())).values()))[0]
        eps = load_episodes(r0['_ep_path'])
        split = cell[2]
        bases = sorted({split_twin(e)[0] for e, ep in eps.items() if ep.get('split', 'na') == split})
        exp = {(b, t) for b in bases for t in range(1, len(eps[b]['trials']) + 1)}
        got = {(b, x['t']) for b, tr in paired.get(cell, {}).items() for x in tr}
        miss = sorted(exp - got)
        if miss:
            m_eps = sorted({b for b, _ in miss})
            missing['/'.join(cell)] = dict(
                n_missing_trials=len(miss), n_expected=len(exp), episodes_affected=m_eps[:20],
                examples=[f'{b} t={t}' for b, t in miss[:10]],
                summary=f'{len(miss)}/{len(exp)} expected twin-complete trials missing '
                        f'({len(m_eps)} episode pairs affected, e.g. {", ".join(f"{b} t={t}" for b, t in miss[:3])})')
    for g, arms in arms_by_group.items():
        for arm in sorted(arms):
            have = {c[5] for c in raw_groups if c[:4] == g and c[4] == arm}
            lack = sorted(slots_by_group[g] - have)
            if lack:
                key = '/'.join(g + (arm, '*'))
                missing[key] = dict(missing_slots=lack, summary=f'domain slot(s) {lack} missing for this arm')
    return missing


# ----------------------------------------------------------------------------- metrics

def true_reversals(ep):
    return sorted(r for r in ep['reversals'] if r > 1)


def choice_credit(py):
    """1 if A chosen, 0 if B chosen, 0.5 at an exact tie."""
    py = np.asarray(py, float)
    return np.where(py > 0.5, 1.0, np.where(py < 0.5, 0.0, 0.5))


def episode_metrics(trials):
    """Episode-level metrics from twin-averaged trials (t >= 2 only)."""
    tr = [x for x in trials if x['t'] >= 2]
    if not tr:
        return None
    py = np.array([x['py'] for x in tr])
    ac = np.array([x['a_correct'] for x in tr])
    chooseA = choice_credit(py)
    acc = chooseA * ac + (1 - chooseA) * (1 - ac)
    ptruth = np.where(ac == 1, py, 1 - py)
    better_a = np.array([x['p_a'] > 0.5 for x in tr])
    defined = np.array([x['p_a'] != 0.5 for x in tr])
    fb = np.where(better_a, py, 1 - py)
    ph = np.array([x['phase'] for x in tr])
    rev = true_reversals(tr[0]['ep'])
    curve_sum, curve_n = np.zeros(N_CURVE), np.zeros(N_CURVE)
    for x, f, d in zip(tr, fb, defined):
        prior = [r for r in rev if r <= x['t']]
        if prior and d:
            k = x['t'] - prior[-1]
            if k < N_CURVE:
                curve_sum[k] += f
                curve_n[k] += 1
    curve = np.where(curve_n > 0, curve_sum / np.maximum(curve_n, 1), np.nan)

    def m(mask):
        mask = mask & defined
        return float(fb[mask].mean()) if mask.any() else float('nan')
    # log score: computed from the logit directly (no clipping of the exact logits)
    lt = np.array([x.get('l', float(logit(x['py']))) for x in tr])
    lt = np.where(ac == 1, lt, -lt)
    with np.errstate(over='ignore'):
        ls = -np.logaddexp(0.0, -lt)
    return dict(acc=float(acc.mean()), fb=m(np.ones_like(defined)), fb_stable=m(ph == 'stable'),
                fb_volatile=m(ph == 'volatile'), log_score=float(ls.mean()),
                curve=curve.tolist(), auc10=float(np.nanmean(curve)) if np.isfinite(curve).any() else float('nan'),
                n_trials=len(tr), twinned=float(np.mean([x.get('twinned', True) for x in tr])),
                approx=float(np.mean([not x.get('exact', True) for x in tr])),
                scenario=tr[0].get('scenario', 'na'))


SCALAR = ('acc', 'fb', 'fb_stable', 'fb_volatile', 'log_score', 'auc10')


_BOCPD_CACHE: dict = {}


def bocpd_states(o, h):
    """controller.bocpd on the full sequence, cached. states[t-1] == bocpd(o[:t-1], h)[-1] (online filter)."""
    key = (tuple(int(x) for x in o), float(h))
    if key not in _BOCPD_CACHE:
        _BOCPD_CACHE[key] = bocpd(list(key[0]), h=h)
    return _BOCPD_CACHE[key]


def window_start(o, t, mode, h, min_len=3):
    """Same as controller.history_window(o, t, mode, h)[0], using the cached BOCPD pass for cp_hard."""
    if mode == 'cp_hard' and t > 1:
        rl = max(bocpd_states(o, h)[t - 1]['run_length'], min_len)
        return max(1, t - rl)
    return history_window(o, t, mode, h=h)[0]


def rule_choices(trials, rule='window'):
    """Rule P(choose A) in {0, 0.5, 1} on the same trials (t >= 2). Returns ({t: cA}, n_window_mismatch).

    rule='window': majority of the counts the VLM was shown (trials window_start..t-1); for sum_cp the
    window is the BOCPD MAP run length (controller.history_window 'cp_hard'), for sum_all all past trials,
    for sum_window:K the last K. rule='bocpd': posterior-predictive p_A of controller.bocpd (any arm).
    """
    out, mism = {}, 0
    if not trials:
        return out, mism
    ep = trials[0]['ep']
    o = [x['a_correct'] for x in ep['trials']]
    for x in trials:
        t = x['t']
        if t < 2:
            continue
        if rule == 'bocpd':
            p = bocpd_states(o, x['hazard'])[t - 1]['p_a']
            cA = 1.0 if p > 0.5 else (0.0 if p < 0.5 else 0.5)
        else:
            hm = x['history_mode']
            base = {'sum_all': 'full', 'sum_cp': 'cp_hard', 'sum_oracle': 'oracle'}.get(
                hm, hm.replace('sum_window', 'window'))
            if base == 'oracle':
                ws = max([r for r in ep['reversals'] if r <= t - 1], default=1)
            else:
                ws = window_start(o, t, base, x['hazard'])
            if x.get('window_start') is not None and int(x['window_start']) != ws:
                mism += 1
                ws = int(x['window_start'])    # trust what the VLM was actually shown
            seg = o[ws - 1:t - 1]
            s, n = sum(seg), len(seg)
            cA = 0.5 if n == 0 or 2 * s == n else float(2 * s > n)
        out[t] = cA
    return out, mism


def rule_alone_trials(trials, rule='window'):
    """Accuracy credit of the rule alone on the same trials. Returns ({t: credit}, n_window_mismatch)."""
    ch, mism = rule_choices(trials, rule)
    ac = {x['t']: x['a_correct'] for x in trials}
    return {t: c * ac[t] + (1 - c) * (1 - ac[t]) for t, c in ch.items()}, mism


def vlm_rule_metrics(trials):
    """VLM + summary vs the rule alone on identical trials: accuracies + trial-level choice agreement (R03)."""
    tr = [x for x in trials if x['t'] >= 2]
    if not tr:
        return None
    cw, mism = rule_choices(tr, 'window')
    cb, _ = rule_choices(tr, 'bocpd')
    v = choice_credit([x['py'] for x in tr])
    w = np.array([cw[x['t']] for x in tr])
    b = np.array([cb[x['t']] for x in tr])
    ac = np.array([x['a_correct'] for x in tr])
    ph = np.array([x['phase'] for x in tr])

    def cred(c):
        return c * ac + (1 - c) * (1 - ac)
    nt = (v != 0.5) & (w != 0.5)
    dis = nt & (v != w)
    vc = cred(v) == 1
    return dict(acc_vlm=float(cred(v).mean()), acc_rule=float(cred(w).mean()), acc_bocpd=float(cred(b).mean()),
                agree=float((v[nt] == w[nt]).mean()) if nt.any() else float('nan'),
                n_nontied=int(nt.sum()), n_agree=int((nt & (v == w)).sum()), n_disagree=int(dis.sum()),
                dis_vlm_correct=int((dis & vc).sum()), dis_rule_correct=int((dis & ~vc).sum()),
                dis_stable=int((dis & (ph == 'stable')).sum()), dis_volatile=int((dis & (ph == 'volatile')).sum()),
                dis_vlm_chose_A=int((dis & (v == 1)).sum()), dis_vlm_chose_B=int((dis & (v == 0)).sum()),
                n_window_mismatch=mism, n_trials=len(tr), scenario=tr[0].get('scenario', 'na'))


# ----------------------------------------------------------------------------- baselines (R05)

BASELINES = {   # name: (uses information the VLM does not have, description)
    'truth_quota': (True, "knows the generator's 5/5 truth quota per 10-trial block; predicts the label with "
                          'more remaining quota and follows the reporter who reports it (no reliability tracking)'),
    'always_A': (True, 'diagnostic only: always follows latent role A (role identity is hidden from the VLM)'),
    'known_pA': (True, 'knows the true current p_A and follows the more reliable reporter (reference ceiling)'),
    'bocpd_posterior': (False, 'BOCPD posterior predictive p_A from the outcome history'),
    'bocpd_map_window': (False, 'majority within the BOCPD MAP run-length window (= the sum_cp summary rule)'),
    'count_all': (False, 'majority of all past outcomes (= the sum_all summary rule)'),
}


def quota_holds(ep, block=10):
    tb = [x['truth_bit'] for x in ep['trials']]
    return len(tb) % block == 0 and all(2 * sum(tb[i:i + block]) == block for i in range(0, len(tb), block))


def baseline_choices(ep, hazard=DEFAULT_HAZARD, block=10):
    """{name: array (index t-1) of P(choose A) in {0, .5, 1}; nan at t=1 or when undefined}."""
    tr = ep['trials']
    T = len(tr)
    o = [x['a_correct'] for x in tr]
    st = bocpd_states(o, hazard)
    out = {k: np.full(T, np.nan) for k in BASELINES}
    q_ok = quota_holds(ep, block)
    for i in range(1, T):
        t, x = i + 1, tr[i]
        if q_ok:
            b0 = i // block * block
            seen = [tr[j]['truth_bit'] for j in range(b0, i)]
            r1, r0 = block // 2 - sum(seen), block // 2 - (len(seen) - sum(seen))
            out['truth_quota'][i] = 0.5 if r1 == r0 else float(x['a_report_bit'] == int(r1 > r0))
        out['always_A'][i] = 1.0
        out['known_pA'][i] = 0.5 if x['p_a'] == 0.5 else float(x['p_a'] > 0.5)
        p = st[i]['p_a']
        out['bocpd_posterior'][i] = 0.5 if p == 0.5 else float(p > 0.5)
        for name, mode in (('bocpd_map_window', 'cp_hard'), ('count_all', 'full')):
            ws = window_start(o, t, mode, hazard)
            seg = o[ws - 1:t - 1]
            s, n = sum(seg), len(seg)
            out[name][i] = 0.5 if 2 * s == n else float(2 * s > n)
    return out


def baseline_table(keys_by_base, eps_by_base, hazard, n_boot, seed):
    """Baseline accuracies on the given trial mask (episode = unit)."""
    per = defaultdict(dict)
    for b in sorted(keys_by_base):
        ep = eps_by_base[b]
        ch = baseline_choices(ep, hazard)
        ts = np.array(sorted(t for t in keys_by_base[b] if t >= 2))
        if not len(ts):
            continue
        ac = np.array([ep['trials'][t - 1]['a_correct'] for t in ts])
        for name, c in ch.items():
            c = c[ts - 1]
            if np.isnan(c).all():
                continue
            per[name][b] = float(np.nanmean(c * ac + (1 - c) * (1 - ac)))
    out = {}
    for name, (uses_gen, desc) in BASELINES.items():
        v = np.array(list(per.get(name, {}).values()), float)
        out[name] = dict(acc=float(v.mean()) if len(v) else float('nan'), ci95=boot_mean_ci(v, n_boot, seed),
                         n_episodes=len(v), uses_generator_info=uses_gen, description=desc,
                         note='' if len(v) else 'not applicable (e.g. no 5/5 truth quota in this dataset)')
    return out


# ----------------------------------------------------------------------------- statistics

def boot_mean_ci(d, n_boot=10000, seed=0, level=0.95):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(bm, a)), float(np.quantile(bm, 1 - a))]


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float('nan')
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)


def paired_test(x, y, n_boot=10000, seed=0):
    """x, y: dicts base -> value. Returns diff = y - x summary over common episodes."""
    common = sorted(set(x) & set(y))
    d = np.array([y[b] - x[b] for b in common], float)
    ok = np.isfinite(d)
    d, common = d[ok], [b for b, k in zip(common, ok) if k]
    return dict(n=len(d), mean=float(d.mean()) if len(d) else float('nan'),
                sd=float(d.std(ddof=1)) if len(d) > 1 else float('nan'),
                ci95=boot_mean_ci(d, n_boot, seed), p_wilcoxon=wilcoxon_p(d), episodes=common,
                diffs=d.tolist())


def position_ci(a, b, c, n_boot=10000, seed=0, threshold=POSITION_THRESHOLD):
    """Absolute B-A, C-B, C-A (paired, bootstrap CI) and, only when interpretable, (B-A)/(C-A).

    The ratio is computed only if C - A > threshold and the 2.5% bootstrap quantile of C - A stays > 0;
    otherwise ``interpretable`` is False with a ``reason``. The ratio need not lie in [0, 1]. Never raises.
    """
    common = sorted(set(a) & set(b) & set(c))
    nan2 = [float('nan')] * 2
    res = dict(n=len(common), threshold=threshold, position=float('nan'), ci95=nan2, interpretable=False,
               reason='', B_minus_A=dict(mean=float('nan'), ci95=nan2), C_minus_B=dict(mean=float('nan'), ci95=nan2),
               C_minus_A=dict(mean=float('nan'), ci95=nan2))
    if not common:
        res['reason'] = 'no episode present in all of A, B and C'
        return res
    A, B, C = (np.array([v[k] for k in common], float) for v in (a, b, c))
    for k, d in (('B_minus_A', B - A), ('C_minus_B', C - B), ('C_minus_A', C - A)):
        res[k] = dict(mean=float(d.mean()), ci95=boot_mean_ci(d, n_boot, seed))
    res['gap_CA'], res['gap_BA'] = res['C_minus_A']['mean'], res['B_minus_A']['mean']
    den = float((C - A).mean())
    if not np.isfinite(den) or den <= threshold:
        res['reason'] = (f'not interpretable: C-A = {den:.3f} does not exceed the prespecified threshold '
                         f'{threshold}')
        return res
    if len(common) < 2:
        res['reason'] = 'not interpretable: fewer than 2 episodes'
        return res
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    dens = C[idx].mean(1) - A[idx].mean(1)
    if np.quantile(dens, 0.025) <= 0:
        res['reason'] = (f'not interpretable: bootstrap C-A crosses 0 '
                         f'({float(np.mean(dens <= 0)):.1%} of resamples <= 0)')
        return res
    r = (B[idx].mean(1) - A[idx].mean(1)) / dens
    res.update(position=float((B.mean() - A.mean()) / den), interpretable=True,
               ci95=[float(np.quantile(r, .025)), float(np.quantile(r, .975))],
               reason='C-A exceeds threshold; ratio need not lie in [0, 1]')
    return res


def tost(d, margin=0.02, alpha=0.05, n_boot=10000, seed=0):
    """Paired TOST on episode differences d against [-margin, +margin] (uncorrected decision only)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    base = dict(n=n, margin=margin, alpha=alpha, p_holm=None, equivalent_holm=None, holm_family=None)
    if n < 2:
        return dict(base, mean=float(d.mean()) if n else float('nan'), p_lower=float('nan'), p_upper=float('nan'),
                    p_tost=float('nan'), ci90_boot=[float('nan')] * 2, equivalent_uncorrected=False,
                    equivalent_boot_uncorrected=False)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        p_lo = p_hi = 0.0 if abs(m) < margin else 1.0
    else:
        p_lo = float(stats.t.sf((m + margin) / se, n - 1))   # H0: mean <= -margin
        p_hi = float(stats.t.cdf((m - margin) / se, n - 1))  # H0: mean >= +margin
    p = max(p_lo, p_hi)
    ci90 = boot_mean_ci(d, n_boot, seed, level=1 - 2 * alpha)
    return dict(base, mean=float(m), p_lower=p_lo, p_upper=p_hi, p_tost=p, ci90_boot=ci90,
                equivalent_uncorrected=bool(p < alpha),
                equivalent_boot_uncorrected=bool(ci90[0] > -margin and ci90[1] < margin))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(pvals, float)
    adj = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok], kind='stable')]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj.tolist()


def sign_test(values):
    v = np.asarray([x for x in values if np.isfinite(x) and x != 0])
    if len(v) == 0:
        return dict(n=0, n_pos=0, p=float('nan'))
    k = int((v > 0).sum())
    return dict(n=len(v), n_pos=k, p=float(stats.binomtest(k, len(v), 0.5).pvalue))


# ----------------------------------------------------------------------------- masked, pooled metrics

def keyset(paired, cell):
    return {(b, x['t']) for b, tr in paired.get(cell, {}).items() for x in tr}


def pool_metrics(ms):
    """Equal-weight pooling of per-slot episode metrics (counts summed, rates averaged)."""
    out = {}
    for k, v in ms[0].items():
        if k == 'scenario':
            out[k] = v if len(ms) == 1 else 'pooled'
        elif k == 'curve':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                out[k] = np.nanmean(np.array([m[k] for m in ms], float), 0).tolist()
        elif k.startswith(('n_', 'dis_')):
            out[k] = int(sum(m[k] for m in ms))
        else:
            vals = np.array([m[k] for m in ms], float)
            out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return out


def arm_metrics(paired, group, arms, dom, strict=True, metric_fn=episode_metrics, notes=None):
    """Episode metrics for several arms on one common trial mask.

    dom: a slot name or 'all' (equal-weight mean over the slots, episode required in every slot).
    Formal mode refuses unequal observation masks or slot sets; partial mode uses the intersection.
    Returns {arm: {base: metrics}}.
    """
    notes = [] if notes is None else notes
    slot_sets = [{c[5] for c in paired if c[:4] == group and c[4] == a} for a in arms]
    if dom == 'all':
        slots = sorted(set.intersection(*slot_sets)) if slot_sets else []
        if strict and any(s != slot_sets[0] for s in slot_sets):
            raise AnalysisError(f'{group}: arms {arms} have different domain slots {slot_sets}')
        if any(s != set(slots) for s in slot_sets):
            notes.append(f'{arms}: pooled over common slots {slots} only')
    else:
        slots = [dom] if all(dom in s for s in slot_sets) else []
    per_slot = []
    for s in slots:
        cells = [group + (a, s) for a in arms]
        ks = [keyset(paired, c) for c in cells]
        common = set.intersection(*ks)
        if any(k != common for k in ks):
            if strict:
                raise AnalysisError(f'{group} slot {s}: observation masks differ between {arms} '
                                    f'({[len(k) for k in ks]} trials)')
            notes.append(f'{arms} slot {s}: {sum(len(k - common) for k in ks)} trials outside the common mask dropped')
        res = {}
        for a, c in zip(arms, cells):
            res[a] = {}
            for b, tr in paired[c].items():
                m = metric_fn([x for x in tr if (b, x['t']) in common])
                if m is not None:
                    res[a][b] = m
        per_slot.append(res)
    out = {}
    for a in arms:
        if not per_slot:
            out[a] = {}
            continue
        bases = set.intersection(*[set(r[a]) for r in per_slot])
        out[a] = {b: (per_slot[0][a][b] if len(per_slot) == 1 else pool_metrics([r[a][b] for r in per_slot]))
                  for b in sorted(bases)}
    return out


def summarize_cells(paired, groups, n_boot, seed, strict):
    res = {}
    for g in groups:
        arms = sorted({c[4] for c in paired if c[:4] == g})
        slots = sorted({c[5] for c in paired if c[:4] == g})
        for arm in arms:
            for dom in slots + ['all']:
                per = arm_metrics(paired, g, [arm], dom, strict=False)[arm]
                if not per:
                    continue
                s = dict(n_episodes=len(per))
                for k in SCALAR:
                    v = np.array([m[k] for m in per.values()], float)
                    v = v[np.isfinite(v)]
                    s[k] = float(v.mean()) if len(v) else float('nan')
                    if k in ('acc', 'auc10'):
                        s[k + '_ci95'] = boot_mean_ci(v, n_boot, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    s['curve'] = np.nanmean(np.array([m['curve'] for m in per.values()], float), 0).round(4).tolist()
                s['approx_frac'] = float(np.mean([m['approx'] for m in per.values()]))
                res['/'.join(g + (arm, dom))] = s
    return res


def readout_quality(rows, eps_cache=None):
    """R10: per cell, candidate mass and the rate at which the greedy output is one of the candidates."""
    acc = defaultdict(lambda: dict(mass=[], follow=[], argmax_match=[]))
    for r in rows:
        d = acc[r['_cell']]
        if r.get('candidate_mass') is not None:
            d['mass'].append(float(r['candidate_mass']))
        g = r.get('greedy')
        if g is not None:
            if r['readout'] == 'judgment':
                ep = load_episodes(r['_ep_path'])[r['episode_id']]
                cands = {ep['a_name'], ep['b_name']}
            else:
                cands = set(SCENARIOS.get(r.get('domain'), {}).get('labels', ()))
            cands = {str(c).strip().lower() for c in cands | {r.get('target')} if c is not None}
            d['follow'].append(str(g).strip().strip('.').lower() in cands)
    out = []
    for cell, d in sorted(acc.items()):
        m = np.array(d['mass'], float)
        out.append(dict(cell='/'.join(cell), model=cell[0], data=cell[1], split=cell[2], readout=cell[3],
                        arm=cell[4], format=arm_parts(cell[4])[2], slot=cell[5], n_mass=len(m),
                        mass_mean=float(m.mean()) if len(m) else float('nan'),
                        mass_min=float(m.min()) if len(m) else float('nan'),
                        frac_mass_lt_05=float((m < 0.5).mean()) if len(m) else float('nan'),
                        n_greedy=len(d['follow']),
                        greedy_follow_rate=float(np.mean(d['follow'])) if d['follow'] else float('nan')))
    return out


# ----------------------------------------------------------------------------- dev selection of K*

def select_k_star(paired, ref_format='text'):
    """Pick K* = argmax dev accuracy over sum_window:K arms (domains pooled, ref format); ties -> smaller K."""
    sel = {}
    for g in sorted({c[:4] for c in paired}):
        if g[2] != 'development':
            continue
        cand = {}
        for arm in sorted({c[4] for c in paired if c[:4] == g}):
            cls, hm, fmt = arm_parts(arm)
            if cls == 'C' and hm.startswith('sum_window:') and fmt == ref_format and '|nohint' not in arm:
                slots = {c[5] for c in paired if c[:4] == g and c[4] == arm}
                if slots != {'pet', 'scenario'}:
                    raise AnalysisError(f'{g[0]}: K* selection requires both pet and scenario slots for {arm}')
                per = arm_metrics(paired, g, [arm], 'all', strict=True)[arm]
                if per:
                    cand[int(hm.split(':')[1])] = float(np.mean([m['acc'] for m in per.values()]))
        if set(cand) != set(DEV_WINDOW_CANDIDATES):
            raise AnalysisError(f'{g[0]}: K* selection requires complete candidates {DEV_WINDOW_CANDIDATES}; '
                                f'found {sorted(cand)}')
        k = min(cand, key=lambda K: (-cand[K], K))
        model = g[0].split('@')[0]
        if model in sel:
            raise AnalysisError(f'{model}: K* selection mixes development datasets or model revisions')
        sel[model] = dict(k_star=k, acc_by_k={str(K): v for K, v in sorted(cand.items())}, model_label=g[0],
                          data=g[1], split=g[2], readout=g[3], ref_format=ref_format)
    return sel


def load_k_star(path, model_label, ref_format='text', readout='choice'):
    d = json.loads(Path(path).read_text())
    if d.get('mode') != 'formal':
        raise AnalysisError(f'{path}: K* selection must be formal, never partial')
    ent = d.get('models', {}).get(model_label.split('@')[0])
    if ent is None:
        return None
    if ent.get('split') != 'development':
        raise AnalysisError(f'{path}: K* for {model_label} was not selected on development data')
    if ent.get('model_label') and ent['model_label'] != model_label:
        raise AnalysisError(f"{path}: K* selected on {ent['model_label']}, analysing {model_label}")
    if ent.get('readout') != readout or ent.get('ref_format', 'text') != ref_format:
        raise AnalysisError(f'{path}: K* selection uses a different readout or reference format')
    try:
        scores = {int(k): float(v) for k, v in ent['acc_by_k'].items()}
        K = int(ent['k_star'])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f'{path}: invalid K* candidate scores') from exc
    if set(scores) != set(DEV_WINDOW_CANDIDATES) or not all(np.isfinite(v) for v in scores.values()):
        raise AnalysisError(f'{path}: K* selection requires complete candidates {DEV_WINDOW_CANDIDATES}')
    if K != min(scores, key=lambda k: (-scores[k], k)):
        raise AnalysisError(f'{path}: K* does not match the recorded development winner')
    return K


def assign_secondary_family(tests, members, alpha=0.05, label=''):
    """Holm over a prespecified TOST family. ``tests`` aligns with ``members``; None = member not run
    (counted as p = 1, conservative). Sets p_holm / equivalent_holm; equivalent_uncorrected is untouched.
    Returns the list of missing members."""
    ps = [t['p_tost'] if t is not None and np.isfinite(t['p_tost']) else 1.0 for t in tests]
    for hm, t, a in zip(members, tests, holm(ps)):
        if t is not None:
            t.update(p_holm=a, equivalent_holm=bool(a < alpha), role='secondary',
                     holm_family=f'{label}: TOST {{{", ".join(members)}}}')
    return [hm for hm, t in zip(members, tests) if t is None]


# ----------------------------------------------------------------------------- pipeline

def analyze(run_dirs, primary_c='sum_all', ref_format='text', tost_margin=0.02, n_boot=10000, seed=0,
            episodes=None, scenario_key='scenario', readout='choice', partial=False, alpha=0.05,
            k_star=None, dev_selection=None, position_threshold=POSITION_THRESHOLD):
    strict = not partial
    rows, info = load_rows(run_dirs, episodes, strict=strict)
    rows = [r for r in rows if r['readout'] == readout]
    if not rows:
        raise AnalysisError(f'no rows with readout={readout}')
    rep = {}
    paired = pair_trials(rows, scenario_key, strict=strict, report=rep)
    info.update(duplicates=rep['duplicates'], conflicts=rep['conflicts'], missing=rep['missing'],
                untwinned_dropped=rep['untwinned_dropped'],
                approx_rows=int(sum(not r['_exact'] for r in rows)), n_rows=len(rows))
    groups = sorted({c[:4] for c in paired})
    robust_groups = {c[:4] for c, per in paired.items() for tr in per.values()
                     if tr and tr[0]['ep'].get('generator') == 'robust_iid_irregular'}
    notes = []
    out = dict(mode='partial' if partial else 'formal', info=info, notes=notes,
               settings=dict(primary_c=primary_c, ref_format=ref_format, tost_margin=tost_margin, alpha=alpha,
                             n_boot=n_boot, seed=seed, readout=readout, scenario_key=scenario_key,
                             position_threshold=position_threshold, k_star=k_star, dev_selection=dev_selection),
               cells=summarize_cells(paired, groups, n_boot, seed, strict),
               readout_quality=readout_quality(rows), gaps=[], positions=[], b_vs_a=[], tost=[],
               agreement=[], scenarios=[], baselines=[], families={})
    mism_total = 0
    for g in groups:
        model, data, split, ro = g
        arms = sorted({c[4] for c in paired if c[:4] == g})
        slots = sorted({c[5] for c in paired if c[:4] == g})
        formal_ho = strict and split == 'held_out'
        # baselines on the trial mask of the reference A arm (all A trials if absent)
        ref_cells = [c for c in paired if c[:4] == g and c[4] == f'A|full|{ref_format}'] or \
                    [c for c in paired if c[:4] == g]
        kb = defaultdict(set)
        epb = {}
        for c in ref_cells:
            for b, tr in paired[c].items():
                kb[b] |= {x['t'] for x in tr}
                epb[b] = tr[0]['ep']
        hz = next((tr[0]['hazard'] for c in ref_cells for tr in paired[c].values() if tr), DEFAULT_HAZARD)
        if kb:
            out['baselines'].append(dict(model=model, data=data, split=split, mask=ref_cells[0][4],
                                         table=baseline_table(kb, epb, hz, n_boot, seed)))

        def a_ref(fmt):
            for f in (fmt, ref_format):
                if f'A|full|{f}' in arms:
                    return f'A|full|{f}'
            return None
        for dom in slots + ['all']:
            for arm in arms:
                cls, hm, fmt = arm_parts(arm)
                if cls != 'C':
                    continue
                aa = a_ref(fmt)
                if aa is not None:
                    mm = arm_metrics(paired, g, [aa, arm], dom, strict, notes=notes)
                    gt = paired_test({b: m['acc'] for b, m in mm[aa].items()},
                                     {b: m['acc'] for b, m in mm[arm].items()}, n_boot, seed)
                    confirmatory = (formal_ho and hm == primary_c and fmt == ref_format and
                                  arm_parts(aa)[2] == ref_format and dom == 'all' and '|nohint' not in arm)
                    is_robustness = confirmatory and g in robust_groups
                    is_primary = confirmatory and not is_robustness
                    gt.update(model=model, data=data, split=split, domain=dom, A=aa, C=arm, primary=is_primary,
                              robustness=is_robustness,
                              role='primary' if is_primary else ('secondary' if is_robustness else 'exploratory'),
                              approx=bool(any(m['approx'] > 0 for v in mm.values() for m in v.values())))
                    for k in ('log_score', 'auc10', 'fb_stable', 'fb_volatile'):
                        gk = paired_test({b: m[k] for b, m in mm[aa].items()},
                                         {b: m[k] for b, m in mm[arm].items()}, min(n_boot, 2000), seed)
                        gt[f'd_{k}'] = dict(mean=gk['mean'], ci95=gk['ci95'])
                    if dom == 'scenario':   # A04: text scenarios only, pet never enters
                        sc = defaultdict(list)
                        for b, dv in zip(gt['episodes'], gt['diffs']):
                            sc[mm[arm][b]['scenario']].append(dv)
                        per = {s: dict(n=len(v), mean=float(np.mean(v)), ci95=boot_mean_ci(v, n_boot, seed))
                               for s, v in sorted(sc.items())}
                        out['scenarios'].append(dict(model=model, data=data, split=split, A=aa, C=arm,
                                                     domain_slot='scenario', per_scenario=per,
                                                     sign_test=sign_test([v['mean'] for v in per.values()]),
                                                     note=SCENARIO_NOTE))
                    out['gaps'].append(gt)
                # TOST + trial-level agreement: VLM + summary vs the rule alone on identical trials
                vr = arm_metrics(paired, g, [arm], dom, strict, metric_fn=vlm_rule_metrics, notes=notes)[arm]
                if not vr:
                    continue
                mism_total += sum(m['n_window_mismatch'] for m in vr.values()) if dom == 'all' else 0
                for rname, key in (('window_counts', 'acc_rule'), ('bocpd_posterior', 'acc_bocpd')):
                    common = sorted(vr)
                    d = [vr[b]['acc_vlm'] - vr[b][key] for b in common]
                    t = tost(d, tost_margin, alpha, n_boot=n_boot, seed=seed)
                    t.update(model=model, data=data, split=split, domain=dom, C=arm, rule=rname,
                             acc_vlm=float(np.mean([vr[b]['acc_vlm'] for b in common])),
                             acc_rule=float(np.mean([vr[b][key] for b in common])),
                             p_wilcoxon=wilcoxon_p(d), ci95=boot_mean_ci(d, n_boot, seed), role='exploratory')
                    out['tost'].append(t)
                tot = {k: int(sum(m[k] for m in vr.values())) for k in vr[next(iter(vr))]
                       if k.startswith(('n_', 'dis_'))}
                ag = np.array([m['agree'] for m in vr.values()], float)
                out['agreement'].append(dict(model=model, data=data, split=split, domain=dom, C=arm,
                                             rule='window_counts', n_episodes=len(vr),
                                             agree_trial_pooled=tot['n_agree'] / tot['n_nontied'] if tot['n_nontied'] else float('nan'),
                                             agree_episode_mean=float(np.nanmean(ag)) if np.isfinite(ag).any() else float('nan'),
                                             agree_ci95=boot_mean_ci(ag, n_boot, seed), **tot))
            # B / B' vs A (absolute), and the normalised position when interpretable
            for arm in arms:
                cls, hm, fmt = arm_parts(arm)
                if cls not in ('A+', 'B', "B'"):     # A+ (raw + correctness) vs A isolates "adding correctness"
                    continue
                aa = a_ref(fmt if cls in ('A+', 'B') else ref_format)
                if aa is None:
                    continue
                mm = arm_metrics(paired, g, [aa, arm], dom, strict, notes=notes)
                gb = paired_test({b: m['acc'] for b, m in mm[aa].items()},
                                 {b: m['acc'] for b, m in mm[arm].items()}, n_boot, seed)
                gb.update(model=model, data=data, split=split, domain=dom, A=aa, B=arm, role='exploratory')
                out['b_vs_a'].append(gb)
                for carm in arms:
                    if arm_parts(carm)[0] == 'C' and arm_parts(carm)[2] == arm_parts(aa)[2]:
                        m3 = arm_metrics(paired, g, [aa, arm, carm], dom, strict, notes=notes)
                        p = position_ci(*({b: m['acc'] for b, m in m3[x].items()} for x in (aa, arm, carm)),
                                        n_boot=n_boot, seed=seed, threshold=position_threshold)
                        p.update(model=model, data=data, split=split, domain=dom, A=aa, B=arm, C=carm)
                        out['positions'].append(p)
    info['rule_window_mismatches'] = mism_total
    # ---- primary family: Holm across models (formal, held-out only)
    fam = [x for x in out['gaps'] if x['primary']]
    for x, a in zip(fam, holm([x['p_wilcoxon'] for x in fam])):
        x['p_holm'] = a
    out['families']['primary'] = dict(members=[f"{x['model']}: {x['C']} - {x['A']} ({x['domain']})" for x in fam],
                                      test='Wilcoxon signed-rank (two-sided)', correction='Holm', alpha=alpha)
    # E3(d): repeat the gap test on robustness episodes as a separate secondary family.
    robust_fam = [x for x in out['gaps'] if x['robustness']]
    if robust_fam:
        for x, a in zip(robust_fam, holm([x['p_wilcoxon'] for x in robust_fam])):
            x['p_holm'] = a
        out['families']['robustness'] = dict(
            members=[f"{x['model']}: {x['C']} - {x['A']} ({x['domain']})" for x in robust_fam],
            role='secondary', test='Wilcoxon signed-rank (two-sided)', correction='Holm', alpha=alpha)
    # ---- secondary family per model: TOST {sum_all, sum_cp, sum_window:K*} (formal, held-out only)
    for g in groups:
        if not strict or g[2] != 'held_out' or g in robust_groups:
            continue
        model = g[0]
        if k_star is not None:
            K = int(k_star)
        elif dev_selection:
            K = load_k_star(dev_selection, model, ref_format, readout)
        else:
            K = None
        members = ['sum_all', 'sum_cp', f'sum_window:{K}' if K is not None else 'sum_window:K*']
        rows_f = []
        for hm in members:
            hit = [t for t in out['tost'] if (t['model'], t['data'], t['split']) == g[:3] and t['domain'] == 'all'
                   and t['rule'] == 'window_counts' and t['C'] == f'C|{hm}|{ref_format}']
            rows_f.append(hit[0] if hit else None)
        missing = assign_secondary_family(rows_f, members, alpha, label=model)
        out['families'][f'secondary/{model}/{g[1]}'] = dict(
            members=[f'C|{hm}|{ref_format} vs window-count rule (domains pooled)' for hm in members],
            k_star=K, k_star_source='--k-star' if k_star is not None else (dev_selection or 'MISSING'),
            margin=tost_margin, alpha=alpha, correction='Holm (missing members counted as p = 1)',
            missing=missing)
    for x in out['gaps'] + out['b_vs_a']:
        x.pop('diffs', None)
    return out


# ----------------------------------------------------------------------------- reporting

def _f(x, d=3):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def _ci(c, d=3):
    return f'[{_f(c[0], d)}, {_f(c[1], d)}]' if c else '[nan, nan]'


def _p(p):
    return 'nan' if p is None or not np.isfinite(p) else (f'{p:.2g}' if p >= 1e-4 else f'{p:.1e}')


def _yn(b):
    return '-' if b is None else ('yes' if b else 'no')


def to_markdown(res):
    partial = res['mode'] == 'partial'
    L = ['# Raw vs. summarized history gap (analyze_main.py)', '']
    if partial:
        L += ['> **EXPLORATORY (--partial): not a formal analysis. Nothing below is primary or secondary.**', '']
    L += [f"mode: {res['mode']}; runs: {', '.join(Path(r).name for r in res['info']['runs'])}", '']
    if res['info'].get('missing') or res['info'].get('untwinned_dropped') or res['info'].get('bad_lines'):
        L += ['## Missing / dropped data', '']
        for c, m in sorted(res['info'].get('missing', {}).items()):
            L.append(f"- {c}: {m['summary']}")
        for c, n in sorted(res['info'].get('untwinned_dropped', {}).items()):
            L.append(f'- {c}: {n} un-twinned trials dropped')
        if res['info'].get('bad_lines'):
            L.append(f"- {len(res['info']['bad_lines'])} unparsable lines skipped")
        L.append('')
    L += ['## Episode-level metrics (twin-averaged in logit space, t >= 2)', '',
          '"all" = equal-weight mean of the pet and scenario slots. approx = share of trials whose logit was '
          'reconstructed from clipped y (legacy rows; saturates at +-13.8).', '',
          '| model | data | split | arm | slot | n ep | acc [95% CI] | FB stable | FB volatile | log score | AUC10 | approx |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, s in res['cells'].items():
        model, data, split, ro, arm, dom = k.split('/')
        L.append(f"| {model} | {data} | {split} | {arm} | {dom} | {s['n_episodes']} | {_f(s['acc'])} {_ci(s['acc_ci95'])} | "
                 f"{_f(s['fb_stable'])} | {_f(s['fb_volatile'])} | {_f(s['log_score'])} | {_f(s['auc10'])} | "
                 f"{_f(s['approx_frac'], 2)}{' (approx)' if s['approx_frac'] > 0 else ''} |")
    L += ['', '## Gap: acc(C) - acc(A), paired by episode', '',
          '| model | split | slot | C arm | A arm | n | gap [95% CI] | Wilcoxon p | Holm p | role |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for g in res['gaps']:
        L.append(f"| {g['model']} | {g['split']} | {g['domain']} | {g['C']} | {g['A']} | {g['n']} | "
                 f"{_f(g['mean'])} {_ci(g['ci95'])} | {_p(g['p_wilcoxon'])} | {_p(g.get('p_holm'))} | "
                 f"{g['role']}{' (approx)' if g.get('approx') else ''} |")
    fam = res['families'].get('primary')
    if fam:
        L += ['', f"Primary family ({fam['test']}, {fam['correction']}, alpha={fam['alpha']}): "
              + ('; '.join(fam['members']) if fam['members'] else 'none (requires formal mode on held-out data)')]
    fam = res['families'].get('robustness')
    if fam:
        L += ['', f"Secondary robustness family ({fam['test']}, {fam['correction']}, alpha={fam['alpha']}): "
              + '; '.join(fam['members'])]
    L += ['', "## B / B' vs A and C vs B (absolute accuracy differences)", '',
          '| model | split | slot | B arm | A arm | n | B-A [95% CI] | Wilcoxon p |', '|---|---|---|---|---|---|---|---|']
    for g in res['b_vs_a']:
        L.append(f"| {g['model']} | {g['split']} | {g['domain']} | {g['B']} | {g['A']} | {g['n']} | "
                 f"{_f(g['mean'])} {_ci(g['ci95'])} | {_p(g['p_wilcoxon'])} |")
    if res['positions']:
        L += ['', f"Normalised position (B-A)/(C-A) is shown only when C-A > {res['settings']['position_threshold']} "
              'and the bootstrap C-A stays > 0; it need not lie in [0, 1].', '',
              '| model | split | slot | B arm | C arm | n | B-A [95% CI] | C-B [95% CI] | C-A | position [95% CI] |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for p in res['positions']:
            pos = f"{_f(p['position'], 2)} {_ci(p['ci95'], 2)}" if p['interpretable'] else p['reason']
            L.append(f"| {p['model']} | {p['split']} | {p['domain']} | {p['B']} | {p['C']} | {p['n']} | "
                     f"{_f(p['B_minus_A']['mean'])} {_ci(p['B_minus_A']['ci95'])} | "
                     f"{_f(p['C_minus_B']['mean'])} {_ci(p['C_minus_B']['ci95'])} | {_f(p['C_minus_A']['mean'])} | {pos} |")
    m, al = res['settings']['tost_margin'], res['settings']['alpha']
    L += ['', f'## TOST: VLM + summary vs rule alone (margin +-{m}, alpha {al})', '',
          'Equivalence here means only that the accuracy difference lies within the margin; it does not imply '
          'trial-by-trial rule following or identical internal computation (see agreement below).', '']
    for k, f in sorted(res['families'].items()):
        if k.startswith('secondary/'):
            L.append(f"- Secondary family {k.split('/', 1)[1]}: members {f['members']}; K* = {f['k_star']} "
                     f"(source {f['k_star_source']}); margin {f['margin']}; alpha {f['alpha']}; {f['correction']}"
                     + (f"; MISSING {f['missing']}" if f['missing'] else ''))
    L += ['', '| model | split | slot | C arm | rule | n | acc VLM | acc rule | diff [90% CI] | p TOST | equiv. (uncorr.) | Holm p | equiv. (Holm) | role |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t in res['tost']:
        L.append(f"| {t['model']} | {t['split']} | {t['domain']} | {t['C']} | {t['rule']} | {t['n']} | {_f(t['acc_vlm'])} | "
                 f"{_f(t['acc_rule'])} | {_f(t['mean'])} {_ci(t['ci90_boot'])} | {_p(t['p_tost'])} | "
                 f"{_yn(t['equivalent_uncorrected'])} | {_p(t['p_holm'])} | {_yn(t['equivalent_holm'])} | {t['role']} |")
    L += ['', '## Trial-level choice agreement: VLM + summary vs window-count rule (non-tied trials)', '',
          '| model | split | slot | C arm | n ep | non-tied | agree (pooled) | agree ep-mean [95% CI] | disagree | VLM right | rule right | stable | volatile | VLM->A | VLM->B |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in res['agreement']:
        L.append(f"| {a['model']} | {a['split']} | {a['domain']} | {a['C']} | {a['n_episodes']} | {a['n_nontied']} | "
                 f"{_f(a['agree_trial_pooled'])} | {_f(a['agree_episode_mean'])} {_ci(a['agree_ci95'])} | {a['n_disagree']} | "
                 f"{a['dis_vlm_correct']} | {a['dis_rule_correct']} | {a['dis_stable']} | {a['dis_volatile']} | "
                 f"{a['dis_vlm_chose_A']} | {a['dis_vlm_chose_B']} |")
    L += ['', '## Per-scenario gap (domain_slot = scenario only; pet reported separately above)', '',
          SCENARIO_NOTE, '', '| model | split | C arm | scenario | n pairs | gap [95% CI] |', '|---|---|---|---|---|---|']
    for s in res['scenarios']:
        for sc, v in s['per_scenario'].items():
            L.append(f"| {s['model']} | {s['split']} | {s['C']} | {sc} | {v['n']} | {_f(v['mean'])} {_ci(v['ci95'])} |")
        st = s['sign_test']
        L.append(f"| {s['model']} | {s['split']} | {s['C']} | sign test | {st['n']} scen. | {st['n_pos']} positive, p={_p(st['p'])} |")
    L += ['', '## History baselines (same trial mask as the reference A arm)', '',
          'Baselines marked "yes" use generator information the VLM does not have and are not fair competitors.', '',
          '| model | split | baseline | acc [95% CI] | n ep | uses generator info | description |', '|---|---|---|---|---|---|---|']
    for b in res['baselines']:
        for name, v in b['table'].items():
            L.append(f"| {b['model']} | {b['split']} | {name} | {_f(v['acc'])} {_ci(v['ci95'])} | {v['n_episodes']} | "
                     f"{'yes' if v['uses_generator_info'] else 'no'} | {v['description']}{(' ' + v['note']) if v['note'] else ''} |")
    L += ['', '## Readout quality (candidate mass, greedy output follows candidate format)', '',
          '| model | split | arm | slot | format | n | mass mean | mass min | frac mass<0.5 | greedy follows |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for q in res['readout_quality']:
        L.append(f"| {q['model']} | {q['split']} | {q['arm']} | {q['slot']} | {q['format']} | {q['n_mass']} | "
                 f"{_f(q['mass_mean'])} | {_f(q['mass_min'])} | {_f(q['frac_mass_lt_05'])} | {_f(q['greedy_follow_rate'])} |")
    info = res['info']
    L += ['', f"Rule-window mismatches (recomputed vs logged window_start): {info['rule_window_mismatches']}; "
          f"identical duplicate rows merged: {info['duplicates']}; legacy y-only rows (approx logits): {info['approx_rows']}."]
    if res['notes']:
        L += ['', '### Notes', ''] + [f'- {n}' for n in sorted(set(res['notes']))]
    return '\n'.join(L) + '\n'


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_dev_selection(path, run_dirs, ref_format='text', episodes=None, readout='choice', partial=False):
    if partial:
        raise AnalysisError('K* selection cannot be written from partial data')
    rows, _ = load_rows(run_dirs, episodes, strict=not partial)
    rows = [r for r in rows if r['readout'] == readout]
    if not rows or any(r['_split'] != 'development' for r in rows):
        raise AnalysisError('K* selection requires development runs only')
    paired = pair_trials(rows, strict=not partial)
    sel = select_k_star(paired, ref_format)
    doc = dict(created_by='analyze_main.py --write-dev-selection', criterion=(
        f'K* = argmax development accuracy of C|sum_window:K|{ref_format} (domains pooled; ties -> smaller K)'),
        mode='partial' if partial else 'formal', models=sel)
    Path(path).write_text(json.dumps(doc, indent=1))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out-dir', default=str(HERE / 'analysis'))
    ap.add_argument('--tag', default='main')
    ap.add_argument('--partial', action='store_true',
                    help='exploratory: allow incomplete data (lists what is missing; nothing labelled primary)')
    ap.add_argument('--primary-c', default='sum_all', help='history_mode of the primary C arm')
    ap.add_argument('--ref-format', default='text')
    ap.add_argument('--tost-margin', type=float, default=0.02)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--k-star', type=int, default=None, help='dev-selected K for the sum_window:K* family member')
    ap.add_argument('--dev-selection', default=None, help='JSON written by --write-dev-selection (K* per model)')
    ap.add_argument('--write-dev-selection', default=None,
                    help='select K* on the given development runs, write this JSON and exit')
    ap.add_argument('--position-threshold', type=float, default=POSITION_THRESHOLD)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', default=None, help='override episodes.jsonl (default: run metadata data_dir)')
    ap.add_argument('--scenario-key', default='scenario')
    ap.add_argument('--readout', default='choice')
    a = ap.parse_args(argv)
    try:
        if a.write_dev_selection:
            doc = write_dev_selection(a.write_dev_selection, a.runs, a.ref_format, a.episodes, a.readout, a.partial)
            print(json.dumps(doc, indent=1))
            return
        res = analyze(a.runs, a.primary_c, a.ref_format, a.tost_margin, a.n_boot, a.seed, a.episodes,
                      a.scenario_key, a.readout, a.partial, a.alpha, a.k_star, a.dev_selection, a.position_threshold)
    except AnalysisError as e:
        sys.exit(f'ERROR: {e}')
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tag = a.tag + ('_partial' if a.partial else '')
    (od / f'{tag}.json').write_text(json.dumps(res, indent=1, default=_json_default))
    md = to_markdown(res)
    (od / f'{tag}.md').write_text(md)
    print(md)
    print(f'wrote {od / tag}.json / .md')


if __name__ == '__main__':
    main()
