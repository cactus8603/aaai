"""E9 analysis: reasoning-mode readout (run_thinking.py outputs), R07.

Inputs
  E9 runs  : directories (searched recursively) holding thinking_shard{i}of{n}.jsonl plus their
             run_metadata_shard{i}of{n}.jsonl and _COMPLETE_shard{i}of{n} markers. Shards of one job are merged.
  --direct : direct-answer runs of the same checkpoint (run_volatile results_shard*.jsonl, e.g. dev_full_text,
             dev_sum_all; run_controls controls*.jsonl, e.g. dev_controls) for the matched comparison.
  --cross-checkpoint E9_MODEL_ID DIRECT_MODEL_ID : explicitly request a separate matched comparison
             across checkpoints (e.g. Qwen/Qwen3-VL-8B-Thinking and Qwen/Qwen3-VL-8B-Instruct).
             This does not relax the same-checkpoint gate and does not isolate a reasoning-only effect.

Units
  group     = (model_id@revision, reasoning_mode). Every row of a group must share one fingerprint apart from
              history_mode / history_format (revision, data hash, hazard, max_new_tokens, pixels); a mix is refused
              in every mode. Rows of the direct-answer side must come from the same model_id AND revision and the
              same episodes file, otherwise the same-checkpoint comparison is refused. An explicitly
              requested cross-checkpoint comparison permits different IDs/revisions but requires a single
              identified revision on each side and the same episode/trial/domain matching and completeness.
  condition = control name ('control|easy_short', ...) or main arm '<A|B|C>|<history_mode>|<history_format>|<readout>'.

Per group x condition (rows, both twins counted separately; CIs: bootstrap over episode pairs)
  valid_rate         share of rows with a valid, naturally generated answer (== natural readout; checked)
  acc_valid          accuracy of natural answers, denominator = valid answers (primary)
  acc_invalid_wrong  sensitivity: invalid answers count as wrong (denominator = all rows)
  acc_invalid_half   sensitivity: invalid answers count 0.5
  truncation / forced-readout rates, answer_status counts
  reasoning tokens mean/median, generated tokens mean/median/total, generated tokens per valid answer, seconds
  p_natural          mean P(correct label) at the natural answer position, natural rows only, plus candidate mass
  FORCED_p           mean P(correct) at the forced read-out position -- flagged, never part of any number above.
  Accuracy is never derived from a probability or from a forced read-out: only the parsed natural answer counts.
  Main task: correct iff (answer == A's report) == a_correct; controls: correct iff answer == target.
  Note: P after reasoning is conditional on the generated reasoning; it is not the direct-answer P.

Matched comparison (any group whose model_id/revision has --direct rows; in E9 that is qwen3vl32b cot)
  item = (condition, episode pair, t, domain). E9 item credit = mean over available twins of the natural
  answer credit (treatments: valid_only [invalid twin dropped; item dropped if no valid twin], invalid_wrong,
  invalid_half). Direct item credit = analyze_main/analyze_controls credit of the twin-averaged logit
  (twins averaged in logit space where both exist). Per condition: per-pair mean over matched items,
  diff = E9 - direct, bootstrap CI + Wilcoxon over episode pairs. Gap: per pair, over (t, domain) where
  E9 A, E9 C, direct A, direct C all exist, gap = mean(C - A); delta_gap = gap_E9 - gap_direct.

Completeness (formal mode, default; --partial only lists problems)
  per job (fingerprint x n shards): all shards 0..n-1 present, each shard holds exactly the key set implied by
  its run metadata (and the episodes file), no duplicates, a matching _COMPLETE marker, no --max-episodes; job
  arguments match the E9 spec of portable/make_jobs.py; every group has both the raw (A, with controls) and
  the summary (C, sum_all, no controls) job; every E9 item has a twin-complete direct counterpart.

CLI
  python analyze_thinking.py E9_DIR [E9_DIR ...] [--direct DIR ...] [--out-dir DIR] [--tag e9] [--partial]
      [--n-boot 10000] [--seed 0] [--episodes PATH]
Writes <out-dir>/<tag>[_partial].json and .md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402
from analyze_main import AnalysisError  # noqa: E402

DEFAULT_CONTROLS = ['explicit_rel', 'easy_short', 'easy_short_coded', 'easy_long', 'easy_long_coded']
# E9 jobs as written by portable/make_jobs.py
E9_SPEC = dict(split='development', domain='scenario', control_t=[40, 80], main_trials=[20, 40, 60, 80],
               readouts=['choice'])
E9_JOBS = {('full', 'raw'): DEFAULT_CONTROLS, ('sum_all', 'raw'): []}
# The H200 job plan also includes a correctness-only arm; older E9 plans omit it.
E9_OPTIONAL_JOBS = {('full', 'coded'): []}
GROUP_FP_KEYS = ('model_id', 'model_revision', 'data_sha256', 'reasoning_mode', 'hazard', 'max_new_tokens', 'pixels')
TREATMENTS = ('valid_only', 'invalid_wrong', 'invalid_half')
_SHARD_RE = re.compile(r'thinking_shard(\d+)of(\d+)\.jsonl$')


# ----------------------------------------------------------------------------- helpers

def combined_data_hash(data_dir: Path):
    """run_thinking.data_hash (episodes.jsonl + image_index.json); None if either file is absent."""
    h = hashlib.sha256()
    for name in ('episodes.jsonl', 'image_index.json'):
        p = Path(data_dir) / name
        if not p.exists():
            return None
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def fp_json(fp: dict) -> str:
    return json.dumps(fp, sort_keys=True)


def read_jsonl(path: Path, problems: list):
    """Rows of a JSONL file. An unterminated last line is dropped (the writer always terminates lines);
    a corrupt line is skipped. Both are recorded in `problems` (formal mode refuses on any)."""
    data = Path(path).read_bytes()
    lines = data.split(b'\n')
    tail = lines.pop()
    if tail.strip():
        problems.append(f'{path}: unterminated last line dropped ({len(tail)} bytes)')
    rows = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            problems.append(f'{path}:{n}: corrupt JSON line skipped')
    return rows


def resolve_domain(domain, ep):
    """run_volatile.resolve_domain (copied: importing run_volatile pulls in the model backend)."""
    return ep['scenario'] if domain == 'scenario' else domain


def condition_of(r: dict) -> str:
    if r.get('task') == 'control' or (r.get('control') and r.get('task') != 'main'):
        return f"control|{r['control']}"
    row = dict(r, format=r.get('format', 'text'))
    return f"{am.history_class(row)}|{r.get('history_mode') or 'full'}|{am.norm_hf(r.get('history_format'))}|{r['readout']}"


def expected_keys(args: dict, eps_list: list, i: int, n: int):
    """Exactly run_thinking.enumerate_jobs' key set for shard i/n under these (string-valued) args."""
    split = args.get('split', 'development')
    dev = [e for e in eps_list if e['split'] == 'development'][i::n]
    main = [e for e in eps_list if e['split'] == split][i::n]
    me = am._parse_arg(args.get('max_episodes'))
    if me is not None:
        dev, main = dev[:me], main[:me]
    dom_arg = args.get('domain')
    keys = set()
    for ep in dev:
        dom = resolve_domain(dom_arg, ep)
        for t in am._parse_arg(args.get('control_t')) or []:
            for c in am._parse_arg(args.get('controls')) or []:
                keys.add(f'{c}|{ep["episode_id"]}|{t}|{dom}|text')
    for ep in main:
        for t in am._parse_arg(args.get('main_trials')) or []:
            for ro in am._parse_arg(args.get('readouts')) or []:
                keys.add(f'main|{ep["episode_id"]}|{t}|{ro}|{dom_arg}|text|'
                         f'{args.get("history_mode")}|{args.get("history_format")}')
    return keys, [e['episode_id'] for e in dev], [e['episode_id'] for e in main]


def check_invariants(r: dict, where: str):
    """Natural/forced separation as written by run_thinking.make_record. Broken rows are refused in every mode."""
    natural = r.get('answer_readout') == 'natural'
    if r.get('answer_readout') not in ('natural', 'forced'):
        raise AnalysisError(f'{where}: answer_readout={r.get("answer_readout")!r}')
    bad = []
    if bool(r.get('answer_valid')) != natural:
        bad.append('answer_valid != natural')
    if (r.get('answer') is not None) != natural:
        bad.append('answer presence != natural')
    if bool(r.get('answer_forced')) == natural:
        bad.append('answer_forced == natural')
    if (r.get('y_natural') is not None) != natural or (r.get('y_forced') is not None) == natural:
        bad.append('y_natural/y_forced do not match the readout kind')
    if natural and r.get('answer_is_target') is None:
        bad.append('natural row without answer_is_target')
    if bad:
        raise AnalysisError(f'{where}: natural/forced invariant broken ({"; ".join(bad)})')


# ----------------------------------------------------------------------------- loading E9

def find_shard_files(paths):
    files = set()
    for p in paths:
        p = Path(p)
        if p.is_file() and _SHARD_RE.search(p.name):
            files.add(p.resolve())
        elif p.is_dir():
            files |= {f.resolve() for f in p.rglob('thinking_shard*.jsonl') if _SHARD_RE.search(f.name)}
        else:
            raise AnalysisError(f'{p}: not a directory or thinking_shard*.jsonl file')
    if not files:
        raise AnalysisError(f'no thinking_shard*.jsonl under {[str(p) for p in paths]}')
    return sorted(files)


def load_e9(paths, episodes_override=None, strict=True):
    """Return (rows, info). Rows gain _group, _cond, _base, _twin, _correct (None if invalid), _p (P(correct)
    of whichever readout the row has), _ep_sha. info lists jobs and problems (completeness)."""
    problems, notes = [], []
    rows, jobs = [], defaultdict(dict)
    for f in find_shard_files(paths):
        i, n = map(int, _SHARD_RE.search(f.name).groups())
        rd = f.parent
        mp = rd / f'run_metadata_shard{i}of{n}.jsonl'
        metas = am._read_json_records(mp) if mp.exists() else []
        if not metas:
            if not episodes_override:
                raise AnalysisError(f'{f}: no {mp.name} and no --episodes (cannot resolve the dataset)')
            problems.append(f'{f}: no run metadata (expected keys unknown)')
        mfps = {fp_json(m['fingerprint']) for m in metas if 'fingerprint' in m}
        if len(mfps) > 1:
            raise AnalysisError(f'{mp}: metadata holds several fingerprints (mixed runs refused): {sorted(mfps)}')
        meta_fp = json.loads(mfps.pop()) if mfps else None
        ep_path = am._resolve_episode_path(metas, episodes_override, rd)
        eps = am.load_episodes(ep_path)
        ep_sha = am.sha256_file(ep_path)
        comb = combined_data_hash(ep_path.parent)
        file_rows = read_jsonl(f, problems)
        fps = {fp_json(r.get('fingerprint') or {}) for r in file_rows}
        if meta_fp is not None:
            fps.add(fp_json(meta_fp))
        if len(fps) > 1:
            raise AnalysisError(f'{f}: rows/metadata from several fingerprints (refused): {sorted(fps)}')
        fp = json.loads(fps.pop()) if fps else (meta_fp or {})
        if fp.get('data_sha256') and comb and fp['data_sha256'] != comb:
            msg = f'{f}: dataset {ep_path.parent} differs from the one the run used (data_sha256 mismatch)'
            if not episodes_override:
                raise AnalysisError(msg)
            notes.append(msg + ' [--episodes override]')
        job = (fp_json(fp), n)
        if i in jobs[job]:
            problems.append(f'shard {i}/{n} of one job given twice: {jobs[job][i]["file"]} and {f}')
        sh = dict(file=str(f), n_rows=len(file_rows), ep_path=str(ep_path))
        jobs[job][i] = sh
        args = metas[-1].get('args', {}) if metas else {}
        sh['args'] = args
        if metas:
            exp, dev_ids, main_ids = expected_keys(args, list(eps.values()), i, n)
            for name, ids in (('control_episodes', dev_ids), ('main_episodes', main_ids)):
                if name in metas[-1] and metas[-1][name] != ids:
                    problems.append(f'{f}: metadata {name} differ from the episodes file split')
            got = Counter(r.get('key') for r in file_rows)
            dup = sum(c - 1 for c in got.values())
            missing, extra = exp - set(got), set(got) - exp
            sh.update(n_expected=len(exp), missing=len(missing), extra=len(extra), duplicated=dup)
            if missing or extra or dup:
                problems.append(f'{f}: {len(set(got) & exp)}/{len(exp)} expected keys, {len(missing)} missing '
                                f'(e.g. {sorted(missing)[:2]}), {len(extra)} extra, {dup} duplicated')
            if am._parse_arg(args.get('max_episodes')) is not None:
                problems.append(f'{f}: run restricted by --max-episodes={args.get("max_episodes")}')
            dev_spec = spec_deviation(args)
            if dev_spec:
                problems.append(f'{f}: job deviates from the E9 spec: {dev_spec}')
        marker = rd / f'_COMPLETE_shard{i}of{n}'
        mk = json.loads(marker.read_text()) if marker.exists() else None
        sh['marker'] = bool(mk) and mk.get('fingerprint') == fp
        if not sh['marker']:
            problems.append(f'{f}: no _COMPLETE marker for this fingerprint')
        for k, r in enumerate(file_rows, 1):
            where = f'{f}:{k}'
            check_invariants(r, where)
            ep = eps.get(r['episode_id'])
            if ep is None:
                raise AnalysisError(f'{where}: episode {r["episode_id"]} not in {ep_path}')
            r['_file'], r['_ep_sha'], r['_fpd'] = str(f), ep_sha, fp
            r['_group'] = (fp.get('model_id'), fp.get('model_revision'), fp.get('reasoning_mode'))
            r['_cond'] = condition_of(r)
            r['_base'], r['_twin'] = am.split_twin(r['episode_id'])
            y = r['y_natural'] if r['answer_readout'] == 'natural' else r['y_forced']
            if r.get('task') == 'main':
                ac = int(ep['trials'][r['t'] - 1]['a_correct'])
                if int(r['a_correct']) != ac:
                    raise AnalysisError(f'{where}: a_correct differs from {ep_path}')
                r['_correct'] = None if r['answer_is_target'] is None else float(bool(r['answer_is_target']) == bool(ac))
                r['_p'] = None if y is None else (y if ac else 1 - y)
            else:
                r['_correct'] = None if r['answer_is_target'] is None else float(bool(r['answer_is_target']))
                r['_p'] = y
            rows.append(r)
    # merge: identical keys under one fingerprint may appear once
    seen, merged, dups = {}, [], []
    for r in sorted(rows, key=lambda r: (r['_file'], r['key'])):
        k = (fp_json(r['_fpd']), r['key'])
        if k in seen:
            dups.append(f"{r['key']} in {seen[k]} and {r['_file']}")
            continue
        seen[k] = r['_file']
        merged.append(r)
    if dups:
        problems.append(f'{len(dups)} duplicate keys across files (kept the first): {dups[:3]}')
    rows = merged
    # fingerprint mixing within a group (refused in every mode)
    by_model = defaultdict(set)
    for r in rows:
        by_model[(r['_fpd'].get('model_id'), r['_fpd'].get('reasoning_mode'))].add(
            tuple((k, r['_fpd'].get(k)) for k in GROUP_FP_KEYS))
    mixed = {k: v for k, v in by_model.items() if len(v) > 1}
    if mixed:
        raise AnalysisError('E9 rows of one model x mode mix fingerprints (refused): ' + '; '.join(
            f'{k}: {_fp_diff(v)}' for k, v in sorted(mixed.items(), key=str)))
    # job-level completeness
    for (fpj, n), shards in sorted(jobs.items()):
        lack = sorted(set(range(n)) - set(shards))
        if lack:
            problems.append(f'job {_job_label(json.loads(fpj))} (n={n}): shards {lack} missing')
    have = defaultdict(set)
    for (fpj, n) in jobs:
        fp = json.loads(fpj)
        have[(fp.get('model_id'), fp.get('model_revision'), fp.get('reasoning_mode'))].add(
            (fp.get('history_mode'), fp.get('history_format')))
    for g, hm in sorted(have.items(), key=str):
        for need in sorted(set(E9_JOBS) - hm):
            problems.append(f'{_group_label(g)}: E9 job with history {need[0]}/{need[1]} missing')
    info = dict(jobs={f'{_job_label(json.loads(fpj))} n={n}': shards for (fpj, n), shards in sorted(jobs.items())},
                problems=problems, notes=notes, n_rows=len(rows))
    if strict and problems:
        raise AnalysisError('formal E9 analysis refused (use --partial for an exploratory look):\n  '
                            + '\n  '.join(problems[:40]) + (f'\n  ... {len(problems) - 40} more' if len(problems) > 40 else ''))
    return rows, info


def spec_deviation(args: dict):
    out = []
    for k, v in E9_SPEC.items():
        got = am._parse_arg(args.get(k))
        if got != v:
            out.append(f'{k}={got!r} (spec {v!r})')
    hk = (args.get('history_mode'), args.get('history_format'))
    jobs = E9_JOBS | E9_OPTIONAL_JOBS
    if hk not in jobs:
        out.append(f'history {hk} not an E9 job')
    elif (am._parse_arg(args.get('controls')) or []) != jobs[hk]:
        out.append(f'controls={am._parse_arg(args.get("controls"))!r} (spec {jobs[hk]!r})')
    return '; '.join(out)


def _fp_diff(fps):
    fps = [dict(f) for f in fps]
    return {k: sorted({str(f.get(k)) for f in fps}) for k in GROUP_FP_KEYS if len({str(f.get(k)) for f in fps}) > 1}


def _group_label(g):
    mid, rev, mode = g
    return f'{am.short_model(mid)}@{(rev or "unknown")[:8]} [{mode}]'


def _job_label(fp):
    return (f"{_group_label((fp.get('model_id'), fp.get('model_revision'), fp.get('reasoning_mode')))} "
            f"{fp.get('history_mode')}/{fp.get('history_format')}")


# ----------------------------------------------------------------------------- per-cell summary

def boot_ratio_ci(num_by_unit, den_by_unit, n_boot=10000, seed=0, level=0.95):
    """Cluster bootstrap of sum(num)/sum(den) over units (episode pairs)."""
    units = sorted(u for u in den_by_unit if den_by_unit[u] > 0)
    if len(units) < 2:
        return [float('nan'), float('nan')]
    num = np.array([num_by_unit[u] for u in units], float)
    den = np.array([den_by_unit[u] for u in units], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n_boot, len(units)))
    r = num[idx].sum(1) / den[idx].sum(1)
    a = (1 - level) / 2
    return [float(np.quantile(r, a)), float(np.quantile(r, 1 - a))]


def _mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else float('nan')


def _median(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else float('nan')


def cell_summary(rows, n_boot=10000, seed=0, with_t=True):
    n = len(rows)
    nat = [r for r in rows if r['answer_readout'] == 'natural']
    forced = [r for r in rows if r['answer_readout'] == 'forced']
    nv = len(nat)
    k = sum(r['_correct'] for r in nat)
    num, den = defaultdict(float), defaultdict(float)
    for r in rows:
        den[r['_base']] += r['answer_readout'] == 'natural'
        num[r['_base']] += r['_correct'] or 0.0
    gen = [r.get('generated_tokens') for r in rows]
    out = dict(
        n_rows=n, n_episode_pairs=len({r['_base'] for r in rows}), n_valid=nv,
        valid_rate=nv / n if n else float('nan'),
        acc_valid=k / nv if nv else float('nan'),
        acc_valid_ci95=boot_ratio_ci(num, den, n_boot, seed),
        acc_invalid_wrong=k / n if n else float('nan'),
        acc_invalid_half=(k + 0.5 * (n - nv)) / n if n else float('nan'),
        truncation_rate=_mean([float(bool(r.get('truncated'))) for r in rows]),
        forced_rate=len(forced) / n if n else float('nan'),
        reasoning_closed_rate=_mean([float(bool(r.get('reasoning_closed'))) for r in rows]),
        answer_status=dict(Counter(str(r.get('answer_status')) for r in rows)),
        reasoning_tokens_mean=_mean([r.get('reasoning_tokens') for r in rows]),
        reasoning_tokens_median=_median([r.get('reasoning_tokens') for r in rows]),
        generated_tokens_mean=_mean(gen), generated_tokens_median=_median(gen),
        generated_tokens_total=int(sum(g or 0 for g in gen)),
        generated_tokens_per_valid_answer=(sum(g or 0 for g in gen) / nv) if nv else float('nan'),
        input_tokens_mean=_mean([r.get('input_tokens') for r in rows]),
        seconds_total=float(sum(r.get('elapsed_seconds') or 0 for r in rows)),
        p_natural=_mean([r['_p'] for r in nat]),
        mass_natural=_mean([r.get('candidate_mass') for r in nat]),
        FORCED_n=len(forced), FORCED_p=_mean([r['_p'] for r in forced]),
        FORCED_mass=_mean([r.get('candidate_mass') for r in forced]),
        FORCED_note='forced read-out after a truncated/unparsable generation; excluded from every other number')
    if with_t:
        by_t = defaultdict(list)
        for r in rows:
            by_t[r['t']].append(r)
        out['by_t'] = {t: {k2: v for k2, v in cell_summary(rs, n_boot, seed, False).items()
                           if k2 in ('n_rows', 'valid_rate', 'acc_valid', 'acc_invalid_wrong', 'truncation_rate',
                                     'reasoning_tokens_mean', 'p_natural')}
                       for t, rs in sorted(by_t.items())}
    return out


# ----------------------------------------------------------------------------- direct-answer rows

def load_direct(paths, episodes_override=None, strict=True):
    """Direct-answer rows of run_volatile (results_shard*) and run_controls (controls*). Each row gains
    _cond, _base, _twin, _l (target-minus-other logit), _model_id, _rev, _ep_sha, _ep_path, _slot."""
    vol, ctl, problems = [], [], []
    for p in sorted({str(Path(x).resolve()) for x in paths}):
        p = Path(p)
        has_v, has_c = any(p.glob('results_shard*.jsonl')), any(p.glob('controls*.jsonl'))
        if not (has_v or has_c):
            raise AnalysisError(f'--direct {p}: no results_shard*.jsonl or controls*.jsonl')
        if has_v:
            vol.append(p)
        if has_c:
            ctl.append(p)
    rows = []
    if vol:
        vrows, info = am.load_rows(vol, episodes_override, strict=strict)
        problems += [f'unparsable line {b}' for b in info['bad_lines']]
        for r in vrows:
            run = info['runs'][r['_run']]
            if r.get('format') != 'text' or r.get('hint', True) is not True:
                continue
            r.update(_cond=condition_of(dict(r, task='main')), _model_id=r.get('model_id'), _rev=r['_fp'][1],
                     _ep_sha=run['data_sha256'], _src=r['_run'], _fpx=r['_fp'])
            r['_base'], r['_twin'] = am.split_twin(r['episode_id'])
            rows.append(r)
    for rd in ctl:
        metas = am._read_metas(rd)
        ep_path = am._resolve_episode_path(metas, episodes_override, rd)
        eps = am.load_episodes(ep_path)
        sha = am.sha256_file(ep_path)
        fps = am._read_fingerprints(rd)
        if any(f.get('data_sha256') not in (None, sha) for f in fps) and not episodes_override:
            raise AnalysisError(f'{rd}: episodes file {ep_path} changed since the run (sha mismatch)')
        mid = am._unique([m.get('model_id') for m in metas] + [f.get('model_id') for f in fps], 'model ids', rd)
        rev = am._unique([m.get('model_revision') for m in metas], 'model revisions', rd)
        if rev is None:
            rev = am._unique([f.get('model_revision') for f in fps], 'model revisions', rd)
        for f in sorted(rd.glob('controls*.jsonl')):
            for r in read_jsonl(f, problems):
                if r.get('format') != 'text':
                    continue
                if r['episode_id'] not in eps:
                    raise AnalysisError(f'{f}: episode {r["episode_id"]} not in {ep_path}')
                l, exact = am.row_logit(r)
                r.update(_cond=f"control|{r['control']}", _model_id=mid, _rev=rev, _ep_sha=sha, _l=l, _exact=exact,
                         _src=str(rd), _slot=None, _ep_path=str(ep_path), _fpx=(mid, rev, sha, r.get('pixels')))
                r['_base'], r['_twin'] = am.split_twin(r['episode_id'])
                rows.append(r)
    if strict and problems:
        raise AnalysisError('direct rows: ' + '; '.join(problems[:10]))
    return rows, problems


def direct_items(rows, strict=True):
    """{(cond, base, t, domain): credit} from twin-averaged logits; plus a list of problems."""
    problems = []
    fps = defaultdict(set)
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        fps[r['_cond']].add(r['_fpx'])
        g[(r['_cond'], r['_base'], r['t'], r['domain'])][r['_twin']].append(r)
    mixed = {c: sorted(map(str, f)) for c, f in fps.items() if len(f) > 1}
    if mixed:
        raise AnalysisError(f'direct rows of one condition mix fingerprints (refused): {mixed}')
    out = {}
    for key, tw in sorted(g.items(), key=lambda kv: str(kv[0])):
        ls = []
        for twin, rs in sorted(tw.items()):
            vals = [x['_l'] for x in rs]
            if len(rs) > 1 and max(vals) - min(vals) > 1e-6:
                problems.append(f'direct {key} twin={twin}: {len(rs)} conflicting rows')
            ls.append(float(np.mean(vals)))
        if len(ls) < 2:
            problems.append(f'direct {key}: only one twin')
        l = float(np.mean(ls))
        if key[0].startswith('control|'):
            out[key] = 1.0 if l > 0 else (0.0 if l < 0 else 0.5)
        else:
            r0 = next(iter(tw.values()))[0]
            ac = int(r0['a_correct'])
            ca = float(am.choice_credit(am.sigmoid(l)))
            out[key] = ca * ac + (1 - ca) * (1 - ac)
    return out, problems


# ----------------------------------------------------------------------------- matched comparison

def e9_items(rows, treatment):
    """{(cond, base, t, domain): credit}: mean over available twins (both when both are available)."""
    g = defaultdict(list)
    for r in rows:
        if r['answer_readout'] == 'natural':
            c = r['_correct']
        else:
            c = {'valid_only': None, 'invalid_wrong': 0.0, 'invalid_half': 0.5}[treatment]
        key = (r['_cond'], r['_base'], r['t'], r['domain'])
        g[key]                     # register the item even when it has no valid twin
        if c is not None:
            g[key].append(c)
    return {k: float(np.mean(v)) for k, v in g.items() if v}, set(g)


def _per_pair(items, keys):
    by = defaultdict(list)
    for k in keys:
        by[k[1]].append(items[k])
    return {b: float(np.mean(v)) for b, v in by.items()}


def gap_pairs(items, a_cond, c_cond):
    """{base: mean over (t, domain) of C - A}, restricted to (base, t, domain) present in `items` for both arms."""
    a = {k[1:]: v for k, v in items.items() if k[0] == a_cond}
    c = {k[1:]: v for k, v in items.items() if k[0] == c_cond}
    return a, c


def matched_comparison(e9_rows, d_items, n_boot=10000, seed=0):
    conds = sorted({r['_cond'] for r in e9_rows})
    res = dict(conditions={}, gap={})
    _, all_keys = e9_items(e9_rows, 'invalid_wrong')
    res['unmatched_e9_items'] = sorted(map(list, (k for k in all_keys if k not in d_items)), key=str)
    for tr in TREATMENTS:
        e_items, _ = e9_items(e9_rows, tr)
        for cond in conds:
            ek = {k for k in e_items if k[0] == cond}
            dk = {k for k in d_items if k[0] == cond}
            common = sorted(ek & dk, key=str)
            entry = res['conditions'].setdefault(cond, dict(n_items_e9=len({k for k in all_keys if k[0] == cond}),
                                                            n_items_direct=len(dk)))
            pt = am.paired_test(_per_pair(d_items, common), _per_pair(e_items, common), n_boot, seed)
            entry[tr] = dict(n_items_matched=len(common), n_pairs=pt['n'],
                             acc_e9=_mean([e_items[k] for k in common]),
                             acc_direct=_mean([d_items[k] for k in common]),
                             diff_e9_minus_direct=pt['mean'], ci95=pt['ci95'], p_wilcoxon=pt['p_wilcoxon'])
        # raw-vs-summarized gap: A|full|raw|<ro> vs C|sum_all|raw|<ro>
        for ro in sorted({c.split('|')[3] for c in conds if not c.startswith('control|')}):
            a_c, c_c = f'A|full|raw|{ro}', f'C|sum_all|raw|{ro}'
            if a_c not in conds or c_c not in conds:
                continue
            ea, ec = gap_pairs(e_items, a_c, c_c)
            da, dc = gap_pairs(d_items, a_c, c_c)
            sub = sorted(set(ea) & set(ec) & set(da) & set(dc))
            ge, gd = defaultdict(list), defaultdict(list)
            for s in sub:
                ge[s[0]].append(ec[s] - ea[s])
                gd[s[0]].append(dc[s] - da[s])
            ge = {b: float(np.mean(v)) for b, v in ge.items()}
            gd = {b: float(np.mean(v)) for b, v in gd.items()}
            pt = am.paired_test(gd, ge, n_boot, seed)
            res['gap'].setdefault(ro, {})[tr] = dict(
                n_items=len(sub), n_pairs=pt['n'],
                gap_e9=_mean(list(ge.values())), gap_e9_ci95=am.boot_mean_ci(list(ge.values()), n_boot, seed),
                gap_direct=_mean(list(gd.values())), gap_direct_ci95=am.boot_mean_ci(list(gd.values()), n_boot, seed),
                delta_gap_e9_minus_direct=pt['mean'], ci95=pt['ci95'], p_wilcoxon=pt['p_wilcoxon'])
    return res


# ----------------------------------------------------------------------------- driver

def compare_direct_rows(g, rs, direct_rows, strict, n_boot, seed):
    """Shared episode/trial/domain matching; model identity is checked by the caller."""
    shas = {r['_ep_sha'] for r in direct_rows} | {r['_ep_sha'] for r in rs}
    if len(shas) > 1:
        raise AnalysisError(f'{_group_label(g)}: E9 and direct rows use different episodes files (refused)')
    slots = {r.get('domain_slot') for r in rs if r.get('task') == 'main'}
    ekeys = {(r['_cond'], r['_base'], r['t'], r['domain']) for r in rs}
    direct_rows = [r for r in direct_rows if (r['_cond'], r['_base'], r['t'], r['domain']) in ekeys and
                   (r['_cond'].startswith('control|') or r.get('domain_slot') in slots)]
    d_items, ip = direct_items(direct_rows, strict)
    matched = matched_comparison(rs, d_items, n_boot, seed)
    matched['direct_problems'] = ip
    matched['direct_runs'] = sorted({r['_src'] for r in direct_rows})
    probs = list(ip)
    if matched['unmatched_e9_items']:
        probs.append(f"{len(matched['unmatched_e9_items'])} E9 items have no direct counterpart "
                     f"(e.g. {matched['unmatched_e9_items'][:2]})")
    if strict and probs:
        raise AnalysisError(f'{_group_label(g)} matched comparison refused (use --partial):\n  '
                            + '\n  '.join(probs[:20]))
    return matched


def analyze(e9_paths, direct_paths=(), episodes=None, partial=False, n_boot=10000, seed=0,
            cross_checkpoint_pairs=()):
    strict = not partial
    rows, info = load_e9(e9_paths, episodes, strict)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r['_group']].append(r)
    res = dict(mode='partial' if partial else 'formal', e9_inputs=info, groups={}, matched={},
               cross_checkpoint_matched={},
               note=('P after reasoning is conditional on the generated reasoning and is not the same quantity as '
                     'the direct-answer P. FORCED_* columns come from forced read-outs and are excluded from all '
                     'accuracy and P numbers. Thinking and Instruct are different checkpoints; the main comparison '
                     'is CoT vs direct answering of the same checkpoint (R07). Explicit cross-checkpoint '
                     'comparisons are reported separately and cannot attribute differences solely to reasoning.'))
    for g, rs in sorted(by_group.items(), key=str):
        by_cond = defaultdict(list)
        for r in rs:
            by_cond[r['_cond']].append(r)
        res['groups'][_group_label(g)] = {c: cell_summary(x, n_boot, seed) for c, x in sorted(by_cond.items())}
    if direct_paths:
        drows, dprob = load_direct(direct_paths, episodes, strict)
        res['direct_problems'] = dprob
        for g, rs in sorted(by_group.items(), key=str):
            mid, rev, mode = g
            same = [r for r in drows if r['_model_id'] == mid]
            if not same:
                continue
            revs = {r['_rev'] for r in same}
            if revs != {rev}:
                raise AnalysisError(f'{_group_label(g)}: direct rows come from revision(s) {sorted(map(str, revs))}, '
                                    f'not {rev} (different checkpoint; refused)')
            res['matched'][_group_label(g)] = compare_direct_rows(g, rs, same, strict, n_boot, seed)
    elif cross_checkpoint_pairs:
        raise AnalysisError('--cross-checkpoint requires --direct inputs')
    for source_id, direct_id in cross_checkpoint_pairs:
        if source_id == direct_id:
            raise AnalysisError('cross-checkpoint pairs must have different model IDs; use the strict same-checkpoint comparison')
        sources = [(g, rs) for g, rs in by_group.items() if g[0] == source_id]
        direct_rows = [r for r in drows if r['_model_id'] == direct_id]
        if not sources or not direct_rows:
            raise AnalysisError(f'cross-checkpoint {source_id} vs {direct_id}: requested model has no input rows')
        revisions = {r['_rev'] for r in direct_rows}
        if len(revisions) != 1 or None in revisions or '' in revisions:
            raise AnalysisError(f'cross-checkpoint {direct_id}: require one known direct revision, got {revisions}')
        direct_rev = next(iter(revisions))
        for g, rs in sources:
            if not g[1]:
                raise AnalysisError(f'cross-checkpoint {source_id}: require a known E9 revision')
            m = compare_direct_rows(g, rs, direct_rows, strict, n_boot, seed)
            m.update(comparison_type='cross_checkpoint', source_model_id=source_id, source_revision=g[1],
                     source_mode=g[2], direct_model_id=direct_id, direct_revision=direct_rev,
                     interpretation='Matched behavioral comparison across checkpoints; training, prompting and readout '
                                    'can differ, so differences cannot be attributed solely to reasoning.')
            label = f'{_group_label(g)} vs {am.short_model(direct_id)}@{direct_rev[:8]}'
            res['cross_checkpoint_matched'][label] = m
    return res


# ----------------------------------------------------------------------------- reporting

def to_markdown(res):
    f, ci, p = am._f, am._ci, am._p

    def esc(x):
        return str(x).replace('|', '\\|')
    L = ['# E9 reasoning-mode readout (analyze_thinking.py)', '']
    if res['mode'] == 'partial':
        L += ['> **EXPLORATORY (--partial): incomplete input; not a formal analysis.**', '']
    probs = res['e9_inputs']['problems'] + res['e9_inputs']['notes']
    if probs:
        L += [f'Input problems ({len(probs)}):', ''] + [f'- {x}' for x in probs[:20]] + ['']
    L += [res['note'], '',
          '## Per model x mode x condition', '',
          'acc|valid = accuracy of natural answers over valid answers (CI: bootstrap over episode pairs); '
          'acc(inv=0) / acc(inv=.5) = sensitivity with invalid answers as wrong / as 0.5. '
          'P_nat = mean P(correct) at the natural answer position (natural rows only). '
          '**FORCED** columns are forced read-outs, flagged and excluded from every other column.', '',
          '| model [mode] | condition | n | valid | acc\\|valid [95% CI] | acc(inv=0) | acc(inv=.5) | trunc | forced '
          '| reas. tok mean / median | gen tok total | tok / valid ans | P_nat | mass_nat | **FORCED** n / P |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g, cells in res['groups'].items():
        for c, s in cells.items():
            L.append(f"| {g} | {esc(c)} | {s['n_rows']} | {f(s['valid_rate'])} | {f(s['acc_valid'])} {ci(s['acc_valid_ci95'])} "
                     f"| {f(s['acc_invalid_wrong'])} | {f(s['acc_invalid_half'])} | {f(s['truncation_rate'])} "
                     f"| {f(s['forced_rate'])} | {f(s['reasoning_tokens_mean'], 0)} / {f(s['reasoning_tokens_median'], 0)} "
                     f"| {s['generated_tokens_total']} | {f(s['generated_tokens_per_valid_answer'], 0)} "
                     f"| {f(s['p_natural'])} | {f(s['mass_natural'])} | {s['FORCED_n']} / {f(s['FORCED_p'])} |")
    comparisons = [(f'Matched comparison: {g} vs direct answering (same checkpoint)', m)
                   for g, m in res.get('matched', {}).items()]
    comparisons += [(f'Cross-checkpoint matched comparison: {g}', m)
                    for g, m in res.get('cross_checkpoint_matched', {}).items()]
    for title, m in comparisons:
        L += ['', f'## {title}', '']
        if m.get('interpretation'):
            L += [m['interpretation'], '']
        L += [
              f"Direct runs: {', '.join(m['direct_runs'])}. Unmatched E9 items: {len(m['unmatched_e9_items'])}. "
              'diff = E9 - direct; unit = episode pair (twins averaged where both are available).', '',
              '| condition | treatment | items | pairs | acc E9 | acc direct | diff [95% CI] | p (Wilcoxon) |',
              '|---|---|---|---|---|---|---|---|']
        for c, e in m['conditions'].items():
            for tr in TREATMENTS:
                x = e[tr]
                L.append(f"| {esc(c)} | {tr} | {x['n_items_matched']} | {x['n_pairs']} | {f(x['acc_e9'])} "
                         f"| {f(x['acc_direct'])} | {f(x['diff_e9_minus_direct'])} {ci(x['ci95'])} | {p(x['p_wilcoxon'])} |")
        if m['gap']:
            L += ['', 'Raw-vs-summarized gap (C - A) and its change:', '',
                  '| readout | treatment | items | pairs | gap E9 [CI] | gap direct [CI] | delta gap E9 - direct [CI] | p |',
                  '|---|---|---|---|---|---|---|---|']
            for ro, d in m['gap'].items():
                for tr in TREATMENTS:
                    if tr not in d:
                        continue
                    x = d[tr]
                    L.append(f"| {ro} | {tr} | {x['n_items']} | {x['n_pairs']} | {f(x['gap_e9'])} {ci(x['gap_e9_ci95'])} "
                             f"| {f(x['gap_direct'])} {ci(x['gap_direct_ci95'])} "
                             f"| {f(x['delta_gap_e9_minus_direct'])} {ci(x['ci95'])} | {p(x['p_wilcoxon'])} |")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='E9 run directories (searched recursively) or thinking_shard files')
    ap.add_argument('--direct', nargs='*', default=[],
                    help='direct-answer run dirs: same checkpoint or explicitly mapped with --cross-checkpoint')
    ap.add_argument('--cross-checkpoint', nargs=2, action='append', default=[],
                    metavar=('E9_MODEL_ID', 'DIRECT_MODEL_ID'),
                    help='explicit separate comparison across model IDs using the supplied --direct runs; repeatable')
    ap.add_argument('--out-dir', default=str(HERE / 'analysis'))
    ap.add_argument('--tag', default='e9')
    ap.add_argument('--partial', action='store_true',
                    help='exploratory: allow incomplete input (problems are listed instead of refused)')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', default=None, help='override episodes.jsonl (default: run metadata data_dir)')
    a = ap.parse_args(argv)
    try:
        res = analyze(a.runs, a.direct, a.episodes, a.partial, a.n_boot, a.seed,
                      cross_checkpoint_pairs=a.cross_checkpoint)
    except AnalysisError as e:
        sys.exit(f'ERROR: {e}')
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tag = a.tag + ('_partial' if a.partial else '')
    (od / f'{tag}.json').write_text(json.dumps(res, indent=1, default=am._json_default))
    md = to_markdown(res)
    (od / f'{tag}.md').write_text(md)
    print(md)
    print(f'wrote {od / tag}.json / .md')
    return res


if __name__ == '__main__':
    main()
