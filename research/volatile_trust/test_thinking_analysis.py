"""Tests for analyze_thinking.py (E9, R07) on synthetic run_thinking / run_volatile / run_controls outputs.

Run: python -m pytest -q research/volatile_trust/test_thinking_analysis.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402
import analyze_thinking as at  # noqa: E402
from analyze_main import AnalysisError  # noqa: E402

V3 = HERE / 'episodes_v3'
NB = 200
EPS = list(am.load_episodes(V3 / 'episodes.jsonl').values())
DEV = [e for e in EPS if e['split'] == 'development']
CT, MT = [40, 80], [20, 40, 60, 80]
FULL = ('full', 'raw')
SUMM = ('sum_all', 'raw')


def fingerprint(hist, mode='cot', rev='rev32b', model='Qwen/Qwen3-VL-32B-Instruct', **kw):
    fp = dict(model_id=model, model_revision=rev, data_sha256=at.combined_data_hash(V3), reasoning_mode=mode,
              history_mode=hist[0], history_format=hist[1], hazard=1 / 15, max_new_tokens=16384, pixels=224)
    fp.update(kw)
    return fp


def correct_policy(f):
    """(natural, answer_is_target, y_target): always a valid, correct answer with P(target) = 0.9 / 0.1."""
    chose = True if f['task'] == 'control' else bool(f['a_correct'])
    return True, chose, 0.9 if chose else 0.1


def e9_row(fp, key, fields, natural, chose_target, y):
    r = dict(key=key, **fields, target='x', other='z', answer_is_target=chose_target if natural else None,
             answer_readout='natural' if natural else 'forced', readout_argmax='x', candidate_logits={'x': 1, 'z': 0},
             candidate_mass=0.9 if natural else 0.01, prompt_sha256='0', reasoning_mode=fp['reasoning_mode'],
             answer_forced=not natural, answer=('x' if natural else None), answer_valid=natural,
             answer_status='answer_marker' if natural else 'parse_failed',
             answer_parse_reason=None if natural else 'truncated', truncated=not natural, reasoning_closed=natural,
             hit_eos=natural, input_tokens=100, reasoning_tokens=50 if natural else 1000, answer_tokens=2,
             generated_tokens=52 if natural else 1000, answer_token_index=0, answer_leading_space=True,
             teacher_forced_argmax_is_generated=None, final_text='', generation_seconds=1.0, elapsed_seconds=1.0,
             reasoning='')
    for suffix, active in (('natural', natural), ('forced', not natural)):
        r[f'y_{suffix}'] = y if active else None
        r[f'logit_target_{suffix}'] = 0.0 if active else None
        r[f'logit_other_{suffix}'] = 0.0 if active else None
        r[f'logit_diff_{suffix}'] = 0.0 if active else None
    r.update(fingerprint=fp, **fp)
    return r


def shard_jobs(hist, controls, i, n):
    """Independent re-implementation of run_thinking.enumerate_jobs keys + fields (no renderer)."""
    eps = DEV[i::n]
    out = []
    for ep in eps:
        dom = ep['scenario']
        for t in CT:
            for c in controls:
                out.append((f'{c}|{ep["episode_id"]}|{t}|{dom}|text',
                            dict(task='control', control=c, episode_id=ep['episode_id'], t=t, domain=dom, format='text')))
    for ep in eps:
        for t in MT:
            tr = ep['trials'][t - 1]
            out.append((f'main|{ep["episode_id"]}|{t}|choice|scenario|text|{hist[0]}|{hist[1]}',
                        dict(task='main', episode_id=ep['episode_id'], order=ep['order'], t=t, readout='choice',
                             domain=ep['scenario'], domain_slot='scenario', format='text', p_a=tr['p_a'],
                             phase=tr['phase'], a_correct=tr['a_correct'], hint=True, history_mode=hist[0],
                             history_format=hist[1], window_start=1, marker=None, summary=hist[0].startswith('sum_'))))
    return out, [e['episode_id'] for e in eps]


def write_job(root, hist, policy=correct_policy, n=4, shards=None, drop_rows=0, marker=True, fp=None, **fpkw):
    controls = at.E9_JOBS[hist]
    fp = fp or fingerprint(hist, **fpkw)
    root = Path(root)
    for i in (range(n) if shards is None else shards):
        d = root / f's{i}'
        d.mkdir(parents=True, exist_ok=True)
        jobs, ids = shard_jobs(hist, controls, i, n)
        rows = [e9_row(fp, k, f, *policy(f)) for k, f in jobs]
        if drop_rows:
            rows = rows[:-drop_rows]
        with open(d / f'thinking_shard{i}of{n}.jsonl', 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        args = dict(data_dir=str(V3), split='development', domain='scenario', controls=str(controls),
                    control_t=str(CT), main_trials=str(MT), readouts="['choice']", history_mode=hist[0],
                    history_format=hist[1], max_episodes='None', shard=f'{i}/{n}')
        meta = dict(model_id=fp['model_id'], model_revision=fp['model_revision'], fingerprint=fp, args=args,
                    control_episodes=ids, main_episodes=ids, n_expected=len(jobs))
        (d / f'run_metadata_shard{i}of{n}.jsonl').write_text(json.dumps(meta) + '\n')
        if marker:
            (d / f'_COMPLETE_shard{i}of{n}').write_text(json.dumps(dict(rows=len(jobs), fingerprint=fp)))
    return root


def write_e9(tmp, policy=correct_policy, **kw):
    a = write_job(Path(tmp) / 'e9_cot_raw', FULL, policy, **kw)
    c = write_job(Path(tmp) / 'e9_cot_summ', SUMM, policy, **kw)
    return [a, c]


def write_direct_main(root, hist, credit_fn, rev='rev32b', model='Qwen/Qwen3-VL-32B-Instruct', twins=(False, True)):
    """run_volatile-style rows; credit_fn(ep_base, t) -> +1 (choose the correct reporter) / -1."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in DEV:
        base, twin = am.split_twin(ep['episode_id'])
        if twin not in twins:
            continue
        for t in range(1, 81):
            tr = ep['trials'][t - 1]
            s = credit_fn(base, t)
            ld = 2.0 * s * (1 if tr['a_correct'] else -1)
            rows.append(dict(episode_id=ep['episode_id'], order=ep['order'], t=t, readout='choice',
                             domain=ep['scenario'], domain_slot='scenario', format='text', p_a=tr['p_a'],
                             phase=tr['phase'], a_correct=tr['a_correct'], y=float(am.sigmoid(ld)), logit_a=ld,
                             logit_other=0.0, logit_diff=ld, candidate_mass=0.99, target='x', model_id=model,
                             hint=True, history_mode=hist[0], pixels=224, history_format=hist[1], window_start=1))
    with open(root / 'results_shard0of1.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    (root / 'run_metadata_shard0of1.jsonl').write_text(json.dumps(dict(
        model_id=model, model_revision=rev, args=dict(data_dir=str(V3), hazard=str(1 / 15), pixels='224'))) + '\n')
    (root / '_FINGERPRINT_shard0of1.json').write_text(json.dumps(dict(
        model_id=model, model_revision=rev, data_sha256=am.sha256_file(V3 / 'episodes.jsonl'), pixels=224,
        hazard=1 / 15, history_mode=hist[0], history_format=hist[1])))
    return root


def write_direct_controls(root, credit_fn, rev='rev32b', model='Qwen/Qwen3-VL-32B-Instruct'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'controls.jsonl', 'w') as fh:
        for ep in DEV:
            base, _ = am.split_twin(ep['episode_id'])
            for t in (20, 40, 60, 80):
                for c in at.DEFAULT_CONTROLS:
                    for dom, fmt in ((ep['scenario'], 'text'), ('pet', 'text'), (ep['scenario'], 'image')):
                        ld = 2.0 * credit_fn(c, base, t)
                        fh.write(json.dumps(dict(key=f'{c}|{ep["episode_id"]}|{t}|{dom}|{fmt}', y=float(am.sigmoid(ld)),
                                                 target='x', logit_a=ld, logit_other=0.0, logit_diff=ld, control=c,
                                                 episode_id=ep['episode_id'], t=t, domain=dom, format=fmt,
                                                 pixels=224)) + '\n')
    (root / 'run_metadata.jsonl').write_text(json.dumps(dict(model_id=model, model_revision=rev,
                                                             args=dict(data_dir=str(V3)))) + '\n')
    (root / '_FINGERPRINT.json').write_text(json.dumps(dict(model_id=model, model_revision=rev, pixels=224,
                                                            data_sha256=am.sha256_file(V3 / 'episodes.jsonl'))))
    return root


# ----------------------------------------------------------------------------- tests

def test_expected_keys_match_run_thinking():
    rt = pytest.importorskip('run_thinking')
    for hist, extra in ((FULL, ['--controls'] + at.DEFAULT_CONTROLS), (SUMM, ['--controls'])):
        argv = ['--model-dir', 'x', '--model-id', 'y', '--output', 'o', '--domain', 'scenario', '--control-t', '40',
                '80', '--main-trials', '20', '40', '60', '80', '--history-mode', hist[0], '--shard', '1/4'] + extra
        args = rt.build_parser().parse_args(argv)
        eps = DEV[1::4]
        keys = {k for k, _, _ in rt.enumerate_jobs(args, eps, eps, None)}
        exp, _, _ = at.expected_keys({k: str(v) for k, v in vars(args).items()}, EPS, 1, 4)
        assert keys == exp
        assert keys == {k for k, _ in shard_jobs(hist, at.E9_JOBS[hist], 1, 4)[0]}


def test_formal_complete_input_and_shard_merging(tmp_path):
    runs = write_e9(tmp_path)
    res = at.analyze(runs, n_boot=NB)
    g = res['groups']['Qwen3-VL-32B-Instruct@rev32b [cot]']
    assert set(g) == {'A|full|raw|choice', 'C|sum_all|raw|choice'} | {f'control|{c}' for c in at.DEFAULT_CONTROLS}
    assert g['A|full|raw|choice']['n_rows'] == len(DEV) * len(MT)            # 4 shards merged
    assert g['control|easy_short']['n_rows'] == len(DEV) * len(CT)
    assert g['A|full|raw|choice']['acc_valid'] == 1.0 and g['A|full|raw|choice']['valid_rate'] == 1.0
    # same answer whether the job root, the shard dirs or the shard files are given
    shard_dirs = sorted(str(p) for r in runs for p in r.iterdir())
    files = sorted(str(f) for r in runs for f in r.rglob('thinking_shard*.jsonl'))
    for alt in (shard_dirs, files, [str(tmp_path)]):
        assert (json.dumps(at.analyze(alt, n_boot=NB)['groups'], default=str)
                == json.dumps(res['groups'], default=str))


def test_natural_forced_separation(tmp_path):
    def policy(f):
        # t = 80 rows: forced read-out whose P strongly favours the CORRECT label; must not touch main numbers
        if f['t'] == 80:
            chose = True if f['task'] == 'control' else bool(f['a_correct'])
            return False, None, 0.99 if chose else 0.01
        chose = True if f['task'] == 'control' else not f['a_correct']      # natural answers all WRONG
        return True, chose, 0.8 if chose else 0.2   # P(target) agrees with the answer
    res = at.analyze(write_e9(tmp_path, policy), n_boot=NB)
    s = res['groups']['Qwen3-VL-32B-Instruct@rev32b [cot]']['A|full|raw|choice']
    assert s['forced_rate'] == pytest.approx(0.25) and s['truncation_rate'] == pytest.approx(0.25)
    assert s['valid_rate'] == pytest.approx(0.75)
    assert s['acc_valid'] == 0.0 and s['acc_invalid_wrong'] == 0.0           # forced argmax never counted
    assert s['acc_invalid_half'] == pytest.approx(0.125)
    assert s['p_natural'] == pytest.approx(0.2)                               # natural rows only
    assert s['FORCED_n'] == len(DEV) and s['FORCED_p'] == pytest.approx(0.99)  # separate, flagged column
    assert s['mass_natural'] == pytest.approx(0.9) and s['FORCED_mass'] == pytest.approx(0.01)
    assert s['by_t'][80]['valid_rate'] == 0.0 and s['by_t'][80]['p_natural'] != s['by_t'][80]['p_natural']  # nan


def test_invalid_answer_denominator():
    rows = []
    # 10 rows over 2 episode pairs: 6 valid (4 correct), 4 invalid
    spec = [(True, 1.0)] * 4 + [(True, 0.0)] * 2 + [(False, None)] * 4
    for k, (nat, c) in enumerate(spec):
        rows.append(dict(answer_readout='natural' if nat else 'forced', _correct=c, _p=0.7 if nat else 0.95,
                         _base=f'dev00{k % 2}', t=20, truncated=not nat, reasoning_closed=nat,
                         answer_status='x', reasoning_tokens=10, generated_tokens=12, candidate_mass=0.5,
                         input_tokens=1, elapsed_seconds=1))
    s = at.cell_summary(rows, NB)
    assert s['valid_rate'] == pytest.approx(0.6)
    assert s['acc_valid'] == pytest.approx(4 / 6)
    assert s['acc_invalid_wrong'] == pytest.approx(0.4)
    assert s['acc_invalid_half'] == pytest.approx(0.6)
    assert s['generated_tokens_per_valid_answer'] == pytest.approx(120 / 6)
    assert s['p_natural'] == pytest.approx(0.7) and s['FORCED_p'] == pytest.approx(0.95)


def test_refuses_incomplete_input(tmp_path):
    # one shard missing
    a = write_job(tmp_path / 'm1' / 'raw', FULL, shards=[0, 1, 2])
    c = write_job(tmp_path / 'm1' / 'summ', SUMM)
    with pytest.raises(AnalysisError, match='shards \\[3\\] missing'):
        at.analyze([a, c], n_boot=NB)
    res = at.analyze([a, c], partial=True, n_boot=NB)
    assert res['mode'] == 'partial' and any('missing' in p for p in res['e9_inputs']['problems'])
    # rows missing from a shard (marker present but stale)
    a = write_job(tmp_path / 'm2' / 'raw', FULL, drop_rows=3)
    with pytest.raises(AnalysisError, match='3 missing'):
        at.analyze([a, write_job(tmp_path / 'm2' / 'summ', SUMM)], n_boot=NB)
    # no completion marker
    a = write_job(tmp_path / 'm3' / 'raw', FULL, marker=False)
    with pytest.raises(AnalysisError, match='_COMPLETE'):
        at.analyze([a, write_job(tmp_path / 'm3' / 'summ', SUMM)], n_boot=NB)
    # summary job absent
    with pytest.raises(AnalysisError, match='sum_all/raw missing'):
        at.analyze([write_job(tmp_path / 'm4' / 'raw', FULL)], n_boot=NB)
    # torn last line
    a = write_job(tmp_path / 'm5' / 'raw', FULL)
    with open(a / 's0' / 'thinking_shard0of4.jsonl', 'a') as fh:
        fh.write('{"key": "torn')
    with pytest.raises(AnalysisError, match='unterminated'):
        at.analyze([a, write_job(tmp_path / 'm5' / 'summ', SUMM)], n_boot=NB)


def test_refuses_duplicates_and_fingerprint_mix(tmp_path):
    runs = write_e9(tmp_path / 'a')
    copy = write_job(tmp_path / 'b' / 'raw', FULL, shards=[0])            # same shard of the same job again
    with pytest.raises(AnalysisError, match='given twice'):
        at.analyze(runs + [copy], n_boot=NB)
    # one model x mode with two max_new_tokens: refused even in --partial
    other = write_job(tmp_path / 'c' / 'summ', SUMM, max_new_tokens=4096)
    with pytest.raises(AnalysisError, match='mix fingerprints'):
        at.analyze([runs[0], other], partial=True, n_boot=NB)
    # rows of another fingerprint inside a shard file
    a = write_job(tmp_path / 'd' / 'raw', FULL)
    f = a / 's0' / 'thinking_shard0of4.jsonl'
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    rows[0]['fingerprint'] = dict(rows[0]['fingerprint'], model_revision='other')
    f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(AnalysisError, match='several fingerprints'):
        at.analyze([a], partial=True, n_boot=NB)


def test_refuses_broken_natural_forced_invariant(tmp_path):
    a = write_job(tmp_path / 'raw', FULL)
    f = a / 's0' / 'thinking_shard0of4.jsonl'
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    rows[0]['y_forced'] = 0.5                       # a natural row that also carries a forced P
    f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(AnalysisError, match='invariant'):
        at.analyze([a], partial=True, n_boot=NB)


def matched_setup(tmp_path):
    bases = sorted({am.split_twin(e['episode_id'])[0] for e in DEV})
    b0 = bases[0]

    def policy(f):
        base, twin = am.split_twin(f['episode_id'])
        if f['task'] == 'control':
            return True, f['control'] != 'easy_long', 0.9      # easy_long always wrong under CoT
        # A item (b0, t=20): original twin invalid, twin valid & correct; everything else correct
        if f['history_mode'] == 'full' and base == b0 and f['t'] == 20 and not twin:
            return False, None, 0.5
        return True, bool(f['a_correct']), 0.9
    runs = write_e9(tmp_path / 'e9', policy)
    # direct: raw history wrong on every trial, summary right on every trial -> gap_direct = 1
    d_a = write_direct_main(tmp_path / 'dev_full_text', FULL, lambda b, t: -1)
    d_c = write_direct_main(tmp_path / 'dev_sum_all', SUMM, lambda b, t: +1)
    d_ctl = write_direct_controls(tmp_path / 'dev_controls', lambda c, b, t: +1)
    return runs, [d_a, d_c, d_ctl], len(bases)


def test_matched_comparison(tmp_path):
    runs, direct, n_pairs = matched_setup(tmp_path)
    res = at.analyze(runs, direct, n_boot=NB)
    m = res['matched']['Qwen3-VL-32B-Instruct@rev32b [cot]']
    assert m['unmatched_e9_items'] == []
    A = m['conditions']['A|full|raw|choice']
    n_items = n_pairs * len(MT)
    for tr in at.TREATMENTS:
        assert A[tr]['n_items_matched'] == n_items and A[tr]['n_pairs'] == n_pairs
        assert A[tr]['acc_direct'] == 0.0
    # one item has one invalid twin: valid_only keeps the valid twin (1), inv=0 -> 0.5, inv=.5 -> 0.75
    assert A['valid_only']['acc_e9'] == 1.0
    assert A['invalid_wrong']['acc_e9'] == pytest.approx(1 - 0.5 / n_items)
    assert A['invalid_half']['acc_e9'] == pytest.approx(1 - 0.25 / n_items)
    assert A['valid_only']['diff_e9_minus_direct'] == pytest.approx(1.0)
    assert A['invalid_wrong']['diff_e9_minus_direct'] == pytest.approx(1 - 0.5 / n_items)
    C = m['conditions']['C|sum_all|raw|choice']['valid_only']
    assert C['acc_e9'] == 1.0 and C['acc_direct'] == 1.0 and C['diff_e9_minus_direct'] == 0.0
    el = m['conditions']['control|easy_long']['valid_only']
    assert el['n_items_matched'] == n_pairs * len(CT) and el['diff_e9_minus_direct'] == pytest.approx(-1.0)
    assert m['conditions']['control|easy_short']['valid_only']['diff_e9_minus_direct'] == 0.0
    g = m['gap']['choice']['valid_only']
    assert g['n_pairs'] == n_pairs and g['gap_direct'] == pytest.approx(1.0) and g['gap_e9'] == pytest.approx(0.0)
    assert g['delta_gap_e9_minus_direct'] == pytest.approx(-1.0)
    assert g['ci95'] == [pytest.approx(-1.0), pytest.approx(-1.0)]
    gw = m['gap']['choice']['invalid_wrong']
    assert gw['gap_e9'] == pytest.approx(0.5 / len(MT) / n_pairs)            # only b0's A item is 0.5
    # the markdown renders both sections
    md = at.to_markdown(res)
    assert 'Matched comparison' in md and 'delta gap' in md


def test_matched_comparison_refusals(tmp_path):
    runs, direct, _ = matched_setup(tmp_path)
    # direct rows of another revision of the same model: different checkpoint, refused
    other = write_direct_main(tmp_path / 'other_rev', FULL, lambda b, t: 1, rev='otherrev')
    with pytest.raises(AnalysisError, match='different checkpoint'):
        at.analyze(runs, [other], partial=True, n_boot=NB)
    # direct controls missing: formal refuses (unmatched E9 items), partial reports them
    with pytest.raises(AnalysisError, match='no direct counterpart'):
        at.analyze(runs, direct[:2], n_boot=NB)
    res = at.analyze(runs, direct[:2], partial=True, n_boot=NB)
    m = res['matched']['Qwen3-VL-32B-Instruct@rev32b [cot]']
    assert len(m['unmatched_e9_items']) == len(DEV) * len(CT) * len(at.DEFAULT_CONTROLS) // 2
    # direct side with only one twin: formal refuses, partial uses the available twin
    one = write_direct_main(tmp_path / 'one_twin', FULL, lambda b, t: -1, twins=(False,))
    with pytest.raises(AnalysisError, match='only one twin'):
        at.analyze(runs, [one, direct[1], direct[2]], n_boot=NB)
    res = at.analyze(runs, [one, direct[1], direct[2]], partial=True, n_boot=NB)
    A = res['matched']['Qwen3-VL-32B-Instruct@rev32b [cot]']['conditions']['A|full|raw|choice']['valid_only']
    assert A['acc_direct'] == 0.0 and A['n_items_matched'] == len(DEV) // 2 * len(MT)


def test_thinking_groups_without_direct_rows(tmp_path):
    runs = write_e9(tmp_path / 'cot')
    think = [write_job(tmp_path / 'th' / 'raw', FULL, mode='thinking', model='Qwen/Qwen3-VL-8B-Thinking', rev='r8'),
             write_job(tmp_path / 'th' / 'summ', SUMM, mode='thinking', model='Qwen/Qwen3-VL-8B-Thinking', rev='r8')]
    direct = matched_setup(tmp_path / 'm')[1]
    res = at.analyze(runs + think, direct, n_boot=NB)
    assert set(res['groups']) == {'Qwen3-VL-32B-Instruct@rev32b [cot]', 'Qwen3-VL-8B-Thinking@r8 [thinking]'}
    assert set(res['matched']) == {'Qwen3-VL-32B-Instruct@rev32b [cot]'}


def test_cli_writes_json_and_markdown(tmp_path):
    runs = write_e9(tmp_path / 'e9')
    out = tmp_path / 'out'
    at.main([str(r) for r in runs] + ['--out-dir', str(out), '--n-boot', str(NB)])
    assert json.loads((out / 'e9.json').read_text())['mode'] == 'formal'
    assert '| model [mode] |' in (out / 'e9.md').read_text()
    with pytest.raises(SystemExit):
        at.main([str(runs[0]), '--out-dir', str(out)])            # summary job missing -> refused
    at.main([str(runs[0]), '--out-dir', str(out), '--partial', '--n-boot', str(NB)])
    assert (out / 'e9_partial.md').exists()
