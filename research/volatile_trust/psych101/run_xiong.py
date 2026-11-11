"""Frozen-VLM replay of Psych-101 xiong2023neural/exp1 (changing bandit), mirroring run_volatile.py.

Per episode (participant x 100-trial game) and trial t = 1..100: the model sees the human's trials 1..t-1
and is scored on the two machine letters (raw first-token logits, normalized over the 2 letters).
One --format per process:
  text | image_outcome | chart                     (choice readout; --hazard-info hidden [primary] | told)
  perc_outcome | perc_chart                         (perception controls; hazard text irrelevant)
Image px is chosen from the model type (config.json): InternVL -> 448 for every format (its backend only
accepts one 448 px tile); other models -> 224 for text/image_outcome/perc_outcome, 448 for chart/perc_chart.
--pixels overrides only for non-InternVL models. The effective px is stored in every row and in the tag.

Output (tag = <format>_<hazard_info>_px<px>_n<n_img> for choice, <format>_px<px> for perception):
  results_<tag>_shard<i>of<n>.jsonl       one row per (episode, item); raw logits of both letters are saved
                                          (`logits`, and for choice rows `logit_diff` = slot1 - slot2)
  fingerprint_<tag>_shard<i>of<n>.json    run fingerprint (model revision, data sha256, renderer/runner
                                          version, fonts, px, n_img, hazard_info, format)
  run_metadata_<tag>_shard<i>of<n>.jsonl  one appended line per launch (never overwritten)
  _COMPLETE_<tag>_shard<i>of<n>           written only for a full-design run (30 participants x 3 h x 100
                                          trials = 9,000 keys over all shards for choice formats) whose rows
                                          equal this shard's expected key set exactly, all with this fingerprint.
Resume: a truncated last line is cut off; then the stored fingerprint must equal the current one (and every
row must carry it), otherwise the run refuses to append (use a new --output).
--shard i/n splits episodes across processes/GPUs; --sharded spreads one model over all visible GPUs.

Example:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/volatile_trust/psych101/run_xiong.py \
     --model-dir .cache/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
     --model-id Qwen/Qwen3-VL-8B-Instruct --format text --output research/volatile_trust/psych101/runs/qwen3vl8b_text
  .venv/bin/python research/volatile_trust/psych101/run_xiong.py --selftest      (CPU, fake backend)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'research/source_trust'))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import render_xiong as rx  # noqa: E402

RUNNER_VERSION = 'run_xiong/v3 (raw letter logits saved; fingerprinted resume)'
FORMATS = ('text', 'image_outcome', 'chart', 'perc_outcome', 'perc_chart')
CHOICE_FORMATS = ('text', 'image_outcome', 'chart')
DEFAULT_PX = {'text': 224, 'image_outcome': 224, 'perc_outcome': 224, 'chart': 448, 'perc_chart': 448}
FIXED_PX_MODEL_TYPES = {'internvl': 448}          # backend accepts only this size (one tile)
FULL_HAZARDS = (0.1, 0.2, 0.4)
FULL_PARTICIPANTS = 30
FULL_KEYS = {'text': 9000, 'image_outcome': 9000, 'chart': 9000, 'perc_outcome': 450, 'perc_chart': 540}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_type_of(model_dir, model_id=''):
    cfg = Path(model_dir) / 'config.json'
    mt = json.loads(cfg.read_text()).get('model_type', '') if cfg.exists() else ''
    if not mt and 'internvl' in model_id.lower():
        mt = 'internvl'
    return mt


def resolve_px(model_type, fmt, override=None):
    """Effective image px. InternVL: 448 for every format (override must agree); others: override or default."""
    fixed = FIXED_PX_MODEL_TYPES.get(model_type)
    if fixed is not None:
        if override is not None and override != fixed:
            raise SystemExit(f'{model_type} requires --pixels {fixed} (got {override})')
        return fixed
    return override or DEFAULT_PX[fmt]


def make_tag(fmt, hazard_info, px, n_img):
    return f'{fmt}_px{px}' if fmt.startswith('perc') else f'{fmt}_{hazard_info}_px{px}_n{n_img}'


def run_fingerprint(args, model_type, px):
    perc = args.format.startswith('perc')
    fp = dict(runner_version=RUNNER_VERSION, model_id=args.model_id, model_revision=Path(args.model_dir).name,
              model_type=model_type, data_sha256=sha256_file(args.data),
              renderer_sha256=sha256_file(Path(rx.__file__)), fonts=rx.font_provenance(),
              format=args.format, image_px=px,
              hazard_info=None if perc else args.hazard_info, n_img=None if perc else args.n_img)
    fp['hash'] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    return fp


def expected_keys(eps, fmt, trials=None):
    if fmt.startswith('perc'):
        return {(ep['episode_id'], it['item_id']) for ep in eps for it in rx.perception_items(ep, fmt)}
    return {(ep['episode_id'], t) for ep in eps for t in (trials or range(1, ep['n_trials'] + 1))}


def read_rows(out_path):
    """Cut a truncated last line (crash mid-write) off the file, then return the rows (other corruption: error)."""
    if not out_path.exists():
        return []
    raw = out_path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        out_path.write_bytes(raw)
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def check_fingerprint(fp, fp_path, rows):
    """Refuse to append under a different fingerprint (or to legacy rows without one)."""
    if fp_path.exists():
        old = json.loads(fp_path.read_text())
        if old != fp:
            diff = {k: (old.get(k), fp.get(k)) for k in set(old) | set(fp) if old.get(k) != fp.get(k) and k != 'hash'}
            raise SystemExit(f'refusing to resume {fp_path.parent}: run fingerprint differs {diff}; use a new --output')
    elif rows:
        raise SystemExit(f'refusing to resume {fp_path.parent}: rows exist but no fingerprint file (legacy run)')
    bad = sum(r.get('run_fingerprint') != fp['hash'] for r in rows)
    if bad:
        raise SystemExit(f'refusing to resume: {bad} rows carry a different run_fingerprint')
    fp_path.write_text(json.dumps(fp, indent=1, sort_keys=True))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--model-id')
    ap.add_argument('--format', choices=FORMATS)
    ap.add_argument('--hazard-info', choices=('told', 'hidden'), default='hidden',
                    help='hidden = PRIMARY (volatility inferred from experience); told = secondary contrast')
    ap.add_argument('--data', type=Path, default=HERE / 'data/episodes.jsonl')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--pixels', type=int, choices=(224, 448),
                    help='override for non-InternVL models (default: 224 text/image_outcome/perc_outcome, '
                         '448 chart/perc_chart); InternVL always uses 448')
    ap.add_argument('--n-img', type=int, default=rx.N_IMG, help='image_outcome: pictures for the last N trials')
    ap.add_argument('--max-participants', type=int)
    ap.add_argument('--hazards', type=float, nargs='+', default=list(FULL_HAZARDS))
    ap.add_argument('--trials', type=int, nargs='+', help='subset of trial indices (debug)')
    ap.add_argument('--sharded', action='store_true', help='shard weights over all visible GPUs')
    ap.add_argument('--selftest', action='store_true', help='CPU check with a fake backend')
    return ap


def load_backend(args, px):
    if args.sharded:
        from backend_multi import ShardedVLM
        return ShardedVLM(args.model_dir, args.model_id, pixels=px)
    from model_backends import FrozenVLM
    return FrozenVLM(args.model_dir, args.model_id, pixels=px)


def run(args, backend_factory=load_backend):
    model_type = model_type_of(args.model_dir, args.model_id)
    px = resolve_px(model_type, args.format, args.pixels)
    all_eps = [json.loads(line) for line in open(args.data) if line.strip()]
    pids = sorted({e['participant'] for e in all_eps})[:args.max_participants]
    eps = [e for e in all_eps if e['participant'] in pids and e['hazard'] in args.hazards]
    eps.sort(key=lambda e: (e['participant'], e['game']))
    full_design = (not args.trials and args.max_participants is None
                   and sorted(args.hazards) == list(FULL_HAZARDS) and len(pids) == FULL_PARTICIPANTS)
    design_keys = expected_keys(eps, args.format, args.trials)
    if full_design and len(design_keys) != FULL_KEYS[args.format]:
        raise SystemExit(f'full design has {len(design_keys)} keys, expected {FULL_KEYS[args.format]}')
    i, n = map(int, args.shard.split('/'))
    eps = eps[i::n]
    args.output.mkdir(parents=True, exist_ok=True)
    perc = args.format.startswith('perc')
    tag = make_tag(args.format, args.hazard_info, px, args.n_img)
    stem = f'{tag}_shard{i}of{n}'
    out_path = args.output / f'results_{stem}.jsonl'
    marker = args.output / f'_COMPLETE_{stem}'
    fp = run_fingerprint(args, model_type, px)
    rows = read_rows(out_path)
    check_fingerprint(fp, args.output / f'fingerprint_{stem}.json', rows)
    if marker.exists() and not args.trials:
        marker.unlink()                       # re-validated at the end of this pass (debug --trials never touch it)
    done = {(r['episode_id'], r['item']) for r in rows}
    expected = expected_keys(eps, args.format, args.trials)
    backend = backend_factory(args, px)
    bm = getattr(backend, 'metadata', {})
    if bm.get('model_revision', fp['model_revision']) != fp['model_revision']:
        raise SystemExit(f"backend revision {bm.get('model_revision')} != fingerprint {fp['model_revision']}")
    if bm.get('pixels', px) != px:
        raise SystemExit(f"backend pixels {bm.get('pixels')} != effective px {px}")
    meta = dict(bm, launched=time.strftime('%Y-%m-%dT%H:%M:%S%z'), fingerprint=fp,
                args={k: str(v) for k, v in vars(args).items()}, image_px=px, n_img=args.n_img,
                rows_before=len(rows), full_design=full_design,
                episodes=[e['episode_id'] for e in eps], fonts=rx.font_provenance(),
                protocol='teacher-forced human history; P(letter) from raw first-token logits over 2 letters')
    with open(args.output / f'run_metadata_{stem}.jsonl', 'a') as fh:
        fh.write(json.dumps(meta, default=str) + '\n')
    started, count = time.time(), 0
    with open(out_path, 'a') as out:
        for ep in eps:
            if perc:
                jobs = [(it['item_id'], it) for it in rx.perception_items(ep, args.format)]
            else:
                jobs = [(t, None) for t in (args.trials or range(1, ep['n_trials'] + 1))]
            for item, it in jobs:
                if (ep['episode_id'], item) in done:
                    continue
                if it is None:
                    rendered, labels = rx.choice_query(ep, item, args.format, args.hazard_info,
                                                       n_img=args.n_img, chart_px=px, img_px=px)
                else:
                    rendered, labels = rx.perception_query(ep, it, args.format, chart_px=px, img_px=px)
                row = backend.score(rendered, labels, max_new_tokens=1)
                probs = row['probabilities']
                logits = {k: float(v) for k, v in row['logits'].items()}
                rec = dict(episode_id=ep['episode_id'], participant=ep['participant'], game=ep['game'],
                           hazard=ep['hazard'], item=item, format=args.format, hazard_info=args.hazard_info,
                           model_id=args.model_id, model_revision=fp['model_revision'], image_px=px,
                           n_img=args.n_img, tag=tag, run_fingerprint=fp['hash'],
                           n_images=len(rendered['images']),
                           probabilities=probs, logits=logits, candidate_mass=row['canonical_candidate_mass'],
                           greedy=row['greedy_answer'], input_tokens=row['input_tokens'],
                           prompt_sha256=row['prompt_sha256'], images_sha256=rx.image_digest(rendered['images']))
                if it is None:
                    tr = ep['trials'][item - 1]
                    s1, s2 = ep['labels']
                    rec.update(t=item, p_slot1=probs[s1], logit_diff=logits[s1] - logits[s2],
                               human_choice=tr['choice'], labels=ep['labels'])
                else:
                    rec.update(t=it['t'], answer=it['answer'], subtask=it.get('subtask', 'read'),
                               correct_prob=probs[it['answer']],
                               correct_argmax=max(probs, key=probs.get) == it['answer'])
                out.write(json.dumps(rec) + '\n')
                count += 1
            out.flush()
            print(json.dumps(dict(episode=ep['episode_id'], rows=count,
                                  sec_per_row=(time.time() - started) / max(count, 1))), flush=True)
    rows = read_rows(out_path)
    written = {(r['episode_id'], r['item']) for r in rows}
    missing, extra = expected - written, written - expected
    foreign = sum(r.get('run_fingerprint') != fp['hash'] for r in rows)
    status = dict(tag=tag, shard=args.shard, fingerprint=fp['hash'], full_design=full_design,
                  design_keys_all_shards=len(design_keys), expected=len(expected), written=len(written),
                  rows=len(rows), duplicates=len(rows) - len(written), missing=len(missing),
                  unexpected=len(extra), foreign_fingerprint=foreign)
    if full_design and not missing and not extra and not foreign and len(rows) == len(written):
        marker.write_text(json.dumps(status) + '\n')
    print(json.dumps(dict(status, complete=marker.exists())), flush=True)
    return dict(status, complete=marker.exists())


# ------------------------------------------------------------------ CPU self-test (fake backend)
class FakeBackend:
    def __init__(self, args, px):
        self.metadata = dict(model_id=args.model_id, model_revision=Path(args.model_dir).name, pixels=px)

    def score(self, rendered, labels, max_new_tokens=1):
        h = int(hashlib.sha256(json.dumps([p.get('text', '') for p in rendered['content']]).encode()).hexdigest()[:6], 16)
        lg = {lab: 25.0 * ((h >> (4 * j)) % 16 - 7.5) / 7.5 for j, lab in enumerate(labels)}   # up to +-25: saturates
        m = max(lg.values())
        z = sum(2.718281828459045 ** (v - m) for v in lg.values())
        probs = {k: 2.718281828459045 ** (v - m) / z for k, v in lg.items()}
        return dict(probabilities=probs, logits=lg, canonical_candidate_mass=1.0,
                    greedy_answer=max(lg, key=lg.get), input_tokens=1, prompt_sha256='x')


def selftest():
    import math
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix='xiong_selftest_'))
    try:
        mdirs = {}
        for name, mt in (('InternVL3_5-8B-HF', 'internvl'), ('Qwen3-VL-8B-Instruct', 'qwen3_vl')):
            d = tmp / name / 'rev1'
            d.mkdir(parents=True)
            (d / 'config.json').write_text(json.dumps(dict(model_type=mt)))
            mdirs[mt] = d
        # I02: effective px from the model type
        for f in FORMATS:
            assert resolve_px('internvl', f) == 448
            assert resolve_px('qwen3_vl', f) == DEFAULT_PX[f]
        assert resolve_px('qwen3_vl', 'text', 448) == 448
        try:
            resolve_px('internvl', 'text', 224)
            raise AssertionError('InternVL 224 must be refused')
        except SystemExit:
            pass
        # full-design key counts
        all_eps = [json.loads(line) for line in open(HERE / "data/episodes.jsonl") if line.strip()]
        for f in FORMATS:
            assert len(expected_keys(all_eps, f)) == FULL_KEYS[f], f

        def args_for(mt, out, **kw):
            a = build_parser().parse_args([])
            a.model_dir, a.model_id, a.format, a.output = mdirs[mt], f'fake/{mt}', 'text', out
            for k, v in kw.items():
                setattr(a, k, v)
            return a

        out = tmp / 'run'
        a = args_for('internvl', out, shard='0/30')            # 3 episodes x 100 trials, full design
        st = run(a, FakeBackend)
        assert st['complete'] and st['written'] == 300 and st['design_keys_all_shards'] == 9000, st
        assert st['tag'] == 'text_hidden_px448_n5'
        rows = read_rows(out / f"results_{st['tag']}_shard0of30.jsonl")
        r0 = rows[0]
        assert r0['image_px'] == 448 and r0['tag'].endswith('px448_n5')
        s1, s2 = r0['labels']
        assert abs(r0['logit_diff'] - (r0['logits'][s1] - r0['logits'][s2])) < 1e-9
        assert abs(1 / (1 + math.exp(-r0['logit_diff'])) - r0['p_slot1']) < 1e-9
        # truncated last line is cut and re-scored; metadata appended per launch
        p = out / f"results_{st['tag']}_shard0of30.jsonl"
        raw = p.read_bytes()
        p.write_bytes(raw[:-40])
        st2 = run(a, FakeBackend)
        assert st2['complete'] and st2['rows'] == 300 and st2['duplicates'] == 0, st2
        assert len((out / f"run_metadata_{st['tag']}_shard0of30.jsonl").read_text().splitlines()) == 2
        # A02: a different fingerprint is refused (data file, model revision)
        data2 = tmp / 'episodes_mod.jsonl'
        data2.write_bytes((HERE / 'data/episodes.jsonl').read_bytes() + b'\n')
        for kw in (dict(data=data2),):
            try:
                run(args_for('internvl', out, shard='0/30', **kw), FakeBackend)
                raise AssertionError('changed fingerprint must be refused')
            except SystemExit as e:
                assert 'fingerprint differs' in str(e), e
        rev2 = tmp / 'InternVL3_5-8B-HF' / 'rev2'
        shutil.copytree(mdirs['internvl'], rev2)
        a3 = args_for('internvl', out, shard='0/30')
        a3.model_dir = rev2
        try:
            run(a3, FakeBackend)
            raise AssertionError('changed revision must be refused')
        except SystemExit as e:
            assert 'model_revision' in str(e), e
        # rows with a foreign fingerprint are refused even if the fingerprint file was replaced
        fpf = out / f"fingerprint_{st['tag']}_shard0of30.json"
        fpd = json.loads(fpf.read_text())
        fpd['hash'] = 'deadbeef'
        fpf.write_text(json.dumps(fpd))
        try:
            run(a, FakeBackend)
            raise AssertionError('foreign rows must be refused')
        except SystemExit:
            pass
        # subset runs never write a marker; qwen text -> 224, chart -> 448
        st4 = run(args_for('qwen3_vl', tmp / 'sub', max_participants=1), FakeBackend)
        assert not st4['complete'] and st4['written'] == 300 and st4['tag'] == 'text_hidden_px224_n5', st4
        assert make_tag('chart', 'hidden', resolve_px('qwen3_vl', 'chart'), 5) == 'chart_hidden_px448_n5'
        print(json.dumps(dict(selftest='ok', checks=['px by model type', 'full-design key counts', 'logits saved',
                                                     'truncated line', 'metadata appended', 'fingerprint refusal',
                                                     'marker only for full design'])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    for k in ('model_dir', 'model_id', 'format', 'output'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k.replace("_", "-")} is required')
    run(args)


if __name__ == '__main__':
    main()
