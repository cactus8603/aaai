"""Reasoning (chain-of-thought) readout on a subset of the volatile-trust prompts (E9).

Prompts are rendered by the unmodified renderers in run_volatile.py / run_controls.py (import only), so
the question text and the underlying history are identical to the direct-answer runs. The model then
reasons freely (greedy); the final answer is parsed strictly (model_backends.parse_label) and candidate
logits are read once (FrozenVLM.score_with_reasoning).

Modes
  thinking : native thinking template (Qwen3-VL-*-Thinking); answer = text after </think>.
  cot      : non-thinking model; the prompt gets "Think step by step, then give the final answer as
             'Answer: <word>'." appended; answer needs an 'Answer: <word>' marker.
  auto     : thinking if the rendered generation prompt opens a <think> block, else cot.

Subset
  control rows : E1 positive controls (default explicit_rel, easy_short, easy_short_coded, easy_long,
                 easy_long_coded), text only, one scenario (--domain), query trials --control-t, dev
                 episodes. Seeds are exactly run_controls' (the `_coded` suffix is stripped), so raw/coded
                 and reasoning/direct conditions see identical synthetic histories. `--controls` with no
                 values = main task only.
  main rows    : main-task readout, text format, trials --main-trials, episodes of --split, history mode
                 --history-mode resolved exactly as run_volatile (full | window:K | cp_hard | cp_soft |
                 oracle | sum_all | sum_cp | sum_window:K | sum_oracle; sum_* render the count summary).

Output (thinking_shard{i}of{n}.jsonl), per row, kept apart by kind (never mix them):
  free answer   : answer (None unless valid), answer_valid, answer_status, answer_parse_reason,
                  answer_is_target (None unless valid), truncated, reasoning_closed, hit_eos
  readout       : answer_readout ('natural' | 'forced'), answer_forced  ('readout' = choice/judgment)
                  natural -> y_natural, logit_{target,other,diff}_natural   (forced fields None)
                  forced  -> y_forced,  logit_{target,other,diff}_forced    (natural fields None)
                  candidate_logits = raw logits of both labels for that readout; logit_diff = target - other
  cost          : input_tokens, reasoning_tokens, answer_tokens, generated_tokens, seconds
  fingerprint   : model revision, data hash, mode, history mode/format, hazard, max_new_tokens, pixels
Resume: an incomplete last line is truncated before appending; a corrupt inner line or a row from a
different fingerprint aborts. _COMPLETE_shard{i}of{n} is written only after re-reading the file and
finding exactly the expected key set (no duplicates/missing/extra keys) under this fingerprint (exit 3
otherwise).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/source_trust'))
sys.path.insert(0, str(HERE))
import run_volatile as rv  # noqa: E402
import run_controls as rc  # noqa: E402
from controller import history_window  # noqa: E402
from model_backends import FrozenVLM  # noqa: E402

DEFAULT_CONTROLS = ['explicit_rel', 'easy_short', 'easy_short_coded', 'easy_long', 'easy_long_coded']
REASONING_FIELDS = ('reasoning_mode', 'answer_forced', 'answer', 'answer_valid', 'answer_status',
                    'answer_parse_reason', 'truncated', 'reasoning_closed', 'hit_eos', 'input_tokens',
                    'reasoning_tokens', 'answer_tokens', 'generated_tokens', 'answer_token_index',
                    'answer_leading_space', 'teacher_forced_argmax_is_generated', 'final_text',
                    'generation_seconds', 'elapsed_seconds', 'reasoning')


def resolve_window(ep, t, mode, hazard):
    """(start, marker, summary) for the history shown before trial t. Verbatim copy of run_volatile.main's
    per-trial history-mode resolution (test_parsing.py checks the two agree on every mode)."""
    o_seq = [tr['a_correct'] for tr in ep['trials']]
    summ = mode.startswith('sum_')
    base = {'sum_all': 'full', 'sum_cp': 'cp_hard', 'sum_oracle': 'oracle'}.get(
        mode, mode.replace('sum_window', 'window') if summ else mode)
    if base == 'oracle':   # diagnostic upper bound: true last reversal
        window = (max([r for r in ep['reversals'] if r <= t - 1], default=1), None)
    else:
        window = history_window(o_seq, t, base, h=hazard)
    return (window[0], window[1], summ)


def control_rng(ep, t, control, domain, fmt='text'):
    """Exactly run_controls.main's seeding: twins share draws, `_coded` variants share the raw draws."""
    base = ep['episode_id'].rstrip('t')
    seed_ctrl = control.replace('_coded', '')
    return random.Random(f"{base}-{t}-{seed_ctrl}-{domain}-{fmt}")


def data_hash(data_dir):
    h = hashlib.sha256()
    for name in ('episodes.jsonl', 'image_index.json'):
        h.update(name.encode())
        h.update((Path(data_dir) / name).read_bytes())
    return h.hexdigest()


def enumerate_jobs(args, ctrl_eps, main_eps, renderer):
    """[(key, make, fields)]; make() -> (rendered, labels, target)."""
    jobs = []
    for ep in ctrl_eps:
        dom = rv.resolve_domain(args.domain, ep)
        for t in args.control_t:
            for control in args.controls:
                key = f'{control}|{ep["episode_id"]}|{t}|{dom}|text'

                def make(ep=ep, dom=dom, t=t, control=control):
                    return rc.build(renderer, ep, dom, 'text', t, control, control_rng(ep, t, control, dom))
                jobs.append((key, make, dict(task='control', control=control, episode_id=ep['episode_id'],
                                             t=t, domain=dom, format='text')))
    for ep in main_eps:
        dom = rv.resolve_domain(args.domain, ep)
        for t in args.main_trials:
            window = resolve_window(ep, t, args.history_mode, args.hazard)
            for readout in args.readouts:
                key = (f'main|{ep["episode_id"]}|{t}|{readout}|{args.domain}|text|'
                       f'{args.history_mode}|{args.history_format}')

                def make(ep=ep, dom=dom, t=t, readout=readout, window=window):
                    if readout == 'choice':
                        return renderer.choice(ep, dom, 'text', t, True, window, args.history_format)
                    return renderer.judgment(ep, dom, t, True, window, args.history_format)
                tr = ep['trials'][t - 1]
                jobs.append((key, make, dict(
                    task='main', episode_id=ep['episode_id'], order=ep['order'], t=t, readout=readout,
                    domain=dom, domain_slot=args.domain, format='text', p_a=tr['p_a'], phase=tr['phase'],
                    a_correct=tr['a_correct'], hint=True, history_mode=args.history_mode,
                    history_format=args.history_format, window_start=window[0], marker=window[1],
                    summary=window[2])))
    keys = [k for k, _, _ in jobs]
    assert len(set(keys)) == len(keys), 'duplicate job keys'
    return jobs


def read_rows(path, repair=False):
    """Parse a JSONL output. An incomplete last line (no newline, or unparsable) is dropped -- and cut
    from the file when repair=True -- so later appends start on a clean line. A corrupt inner line raises."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.split(b'\n')
    tail = lines.pop()                    # b'' when the file ends with a newline
    rows, good_end = [], 0
    for n, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(lines) - 1 and not tail:     # last newline-terminated line is broken
                break
            raise RuntimeError(f'{path}: corrupt JSON on line {n + 1}; refusing to resume')
        good_end += len(line) + 1
    if good_end != len(data) and repair:
        with open(path, 'r+b') as f:
            f.truncate(good_end)
        print(f'resume: truncated incomplete tail of {path} ({len(data) - good_end} bytes)', flush=True)
    return rows


def load_done(path, fingerprint):
    """Keys already written under this fingerprint (repairs a torn tail). Foreign rows abort."""
    done = set()
    for r in read_rows(path, repair=True):
        if r.get('fingerprint') != fingerprint:
            raise RuntimeError(f'{path} holds rows from a different configuration '
                               f'({r.get("fingerprint")} != {fingerprint}); use a fresh --output')
        if r['key'] in done:
            raise RuntimeError(f'{path}: duplicate key {r["key"]}')
        done.add(r['key'])
    return done


def verify_complete(path, expected_keys, fingerprint):
    """Re-read the file; return (ok, message). ok iff keys == expected exactly once, all with this fingerprint."""
    try:
        rows = read_rows(path)
    except RuntimeError as e:
        return False, str(e)
    keys = [r.get('key') for r in rows]
    expected = set(expected_keys)
    seen = set(keys)
    dup = len(keys) - len(seen)
    missing = expected - seen
    extra = seen - expected
    foreign = sum(r.get('fingerprint') != fingerprint for r in rows)
    ok = bool(expected) and not (dup or missing or extra or foreign)
    return ok, (f'{len(seen & expected)}/{len(expected)} expected keys, {dup} duplicated, {len(missing)} missing, '
                f'{len(extra)} extra, {foreign} rows with another fingerprint')


def make_record(key, fields, row, labels, target, fingerprint):
    other = next(l for l in labels if l != target)
    natural = row['readout'] == 'natural'
    if natural != (row['answer'] is not None):
        raise RuntimeError('backend broke the natural-answer invariant')
    lt, lo = row['logits'][target], row['logits'][other]
    kind = dict(y=row['probabilities'][target], logit_target=lt, logit_other=lo, logit_diff=lt - lo)
    rec = dict(key=key, **fields, target=target, other=other,
               answer_is_target=(None if row['answer'] is None else row['answer'] == target),
               answer_readout=row['readout'], readout_argmax=row['argmax_label'], candidate_logits=row['logits'],
               candidate_mass=row['canonical_candidate_mass'], prompt_sha256=row['prompt_sha256'],
               **{k: row[k] for k in REASONING_FIELDS})
    for suffix, active in (('natural', natural), ('forced', not natural)):
        for k, v in kind.items():
            rec[f'{k}_{suffix}'] = v if active else None
    rec.update(fingerprint=fingerprint, **fingerprint)
    return rec


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--data-dir', type=Path, default=HERE / 'episodes_v3')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', choices=('development', 'held_out'), default='development',
                    help='episodes for the main-task rows (control rows always use development)')
    ap.add_argument('--shard', default='0/1', help='i/n: episode shard (same convention as run_volatile)')
    ap.add_argument('--pixels', type=int, choices=(224, 448), default=224)
    ap.add_argument('--sharded', action='store_true', help='shard weights over all visible GPUs')
    ap.add_argument('--mode', choices=('auto', 'thinking', 'cot'), default='auto')
    ap.add_argument('--max-new-tokens', type=int, default=4096)
    ap.add_argument('--domain', default='color', help="one scenario, or 'scenario' (episode's own; v3 data)")
    ap.add_argument('--controls', nargs='*', default=list(DEFAULT_CONTROLS),
                    help='positive controls; pass the flag with no values for main task only')
    ap.add_argument('--control-t', type=int, nargs='*', default=list(rc.QUERY_T))
    ap.add_argument('--main-trials', type=int, nargs='*', default=list(range(10, 81, 10)))
    ap.add_argument('--readouts', nargs='+', default=['choice'], choices=('choice', 'judgment'))
    ap.add_argument('--history-mode', default='full',
                    help='full | window:K | cp_hard | cp_soft | oracle | sum_all | sum_cp | sum_window:K | sum_oracle')
    ap.add_argument('--hazard', type=float, default=1 / 15)
    ap.add_argument('--history-format', choices=rv.HISTORY_FORMATS, default='raw')
    ap.add_argument('--max-episodes', type=int, help='cap episodes per task (after sharding)')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    i, n = map(int, args.shard.split('/'))
    all_eps = [json.loads(l) for l in open(args.data_dir / 'episodes.jsonl')]
    ctrl_eps = [e for e in all_eps if e['split'] == 'development'][i::n][:args.max_episodes]
    main_eps = [e for e in all_eps if e['split'] == args.split][i::n][:args.max_episodes]
    renderer = rv.Renderer(json.loads((args.data_dir / 'image_index.json').read_text()), args.pixels)
    jobs = enumerate_jobs(args, ctrl_eps, main_eps, renderer)    # fails fast on a bad history mode
    keys = [k for k, _, _ in jobs]

    if args.sharded:
        from backend_multi import ShardedVLM
        backend = ShardedVLM(args.model_dir, args.model_id, pixels=args.pixels)
    else:
        backend = FrozenVLM(args.model_dir, args.model_id, pixels=args.pixels)
    mode = args.mode
    if mode == 'auto':
        probe = backend.render_prompt([{'type': 'text', 'text': 'x'}], short_answer=False,
                                      template_kwargs=dict(backend.template_kwargs, enable_thinking=True))
        mode = 'thinking' if '<think>' in probe[probe.rindex('x'):] else 'cot'
    fingerprint = dict(model_id=args.model_id, model_revision=backend.metadata.get('model_revision'),
                       data_sha256=data_hash(args.data_dir), reasoning_mode=mode,
                       history_mode=args.history_mode, history_format=args.history_format,
                       hazard=args.hazard, max_new_tokens=args.max_new_tokens, pixels=args.pixels)

    args.output.mkdir(parents=True, exist_ok=True)
    marker_path = args.output / f'_COMPLETE_shard{i}of{n}'
    marker_path.unlink(missing_ok=True)      # re-earned below against the current file + fingerprint
    meta = dict(backend.metadata, fingerprint=fingerprint, args={k: str(v) for k, v in vars(args).items()},
                control_episodes=[e['episode_id'] for e in ctrl_eps],
                main_episodes=[e['episode_id'] for e in main_eps], n_expected=len(keys),
                started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(args.output / f'run_metadata_shard{i}of{n}.jsonl', 'a') as mf:
        mf.write(json.dumps(meta, default=str) + '\n')

    out_path = args.output / f'thinking_shard{i}of{n}.jsonl'
    done = load_done(out_path, fingerprint)
    started, count, tokens = time.time(), 0, 0
    with open(out_path, 'a') as out:
        for key, make, fields in jobs:
            if key in done:
                continue
            rendered, labels, target = make()
            row = backend.score_with_reasoning(rendered, labels, max_new_tokens=args.max_new_tokens, mode=mode)
            rec = make_record(key, fields, row, labels, target, fingerprint)
            out.write(json.dumps(rec) + '\n')
            out.flush()
            done.add(key)
            count += 1
            tokens += row['generated_tokens']
            el = time.time() - started
            print(json.dumps(dict(n=count, key=key, readout=row['readout'], answer=row['answer'],
                                  status=row['answer_status'], p_target=round(row['probabilities'][target], 3),
                                  reasoning_tokens=row['reasoning_tokens'], truncated=row['truncated'],
                                  sec=round(row['elapsed_seconds'], 1), sec_per_row=round(el / count, 1),
                                  tok_per_sec=round(tokens / max(el, 1e-9), 1))), flush=True)

    ok, msg = verify_complete(out_path, keys, fingerprint)
    if ok:
        marker_path.write_text(json.dumps(dict(rows=len(keys), fingerprint=fingerprint,
                                               keys_sha256=hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest(),
                                               at=time.strftime('%F %T'))))
        print(f'COMPLETE {len(keys)} rows', flush=True)
    else:
        print(f'INCOMPLETE: {msg}', flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
