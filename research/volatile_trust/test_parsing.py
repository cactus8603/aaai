"""Tests for the reasoning-mode answer parser (model_backends.parse_label / parse_generation) and the E9
runner run_thinking.py on CPU with a stub backend (no model weights).

Run: python -m pytest -q research/volatile_trust/test_parsing.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'source_trust'))
import model_backends as mb  # noqa: E402
import run_controls as rc  # noqa: E402
import run_thinking as rt  # noqa: E402
import run_volatile as rv  # noqa: E402

DATA = HERE / 'episodes_v3'
L = ['red', 'blue']
FAILED = mb.PARSE_FAILED


def P(text, cot=False, truncated=False):
    return mb.parse_label(text, L, cot=cot, truncated=truncated)


# ----------------------------------------------------------------------------- parse_label (I07)

@pytest.mark.parametrize('text,label,status', [
    ('red', 'red', 'bare_label'),
    ('  **Blue**.\n', 'blue', 'bare_label'),
    ('Answer: red', 'red', 'answer_marker'),
    ('Some reasoning.\n**Answer:** blue.', 'blue', 'answer_marker'),
    ('Final answer:\nred', 'red', 'answer_marker'),
    ('Answer: red\nBecause blue was wrong on most trials.', 'red', 'answer_marker'),
    ('So \\boxed{blue}', 'blue', 'boxed'),
    ('\\boxed{\\text{red}}', 'red', 'boxed'),
    ('\\boxed{red} ... Answer: red', 'red', 'answer_marker'),
    ('Answer: red. Checking again... Answer: red', 'red', 'answer_marker'),
])
def test_accepted_forms(text, label, status):
    got, st, off, reason = P(text)
    assert (got, st, reason) == (label, status, None)
    assert text[off:off + len(label)].lower() == label


@pytest.mark.parametrize('text,reason', [
    ('The true color is red, not blue.', 'conflicting_mentions'),       # negation must not pick 'blue'
    ('It is either red or blue.', 'conflicting_mentions'),
    ('Answer: red, not blue', 'conflicting_mentions'),
    ('Answer: not blue', 'marker_without_label'),
    ('Answer: red\n...wait.\nAnswer: blue', 'conflicting_answers'),     # repeated different answers
    ('\\boxed{red} \\boxed{blue}', 'conflicting_answers'),
    ('\\boxed{red}\nAnswer: blue', 'conflicting_answers'),
    ('\\boxed{red or blue}', 'boxed_not_single_label'),
    ('', 'no_answer'),
    ('I cannot tell.', 'no_answer'),
    ('Answer:', 'marker_without_label'),
    ('The color is probably red.', 'no_answer_marker'),                  # single mention in prose: not accepted
])
def test_failures(text, reason):
    assert P(text) == (None, FAILED, None, reason)


def test_truncation_always_fails():
    assert P('Answer: red', truncated=True) == (None, FAILED, None, 'truncated')
    assert P('red', truncated=True)[1] == FAILED


def test_cot_needs_marker_box_or_bare():
    assert P('Ken was right more often, so the color is red.', cot=True)[3] == 'no_answer_marker'
    assert P('Reasoning...\nAnswer: red', cot=True)[0] == 'red'
    assert P('red', cot=True)[0] == 'red'


def test_parse_generation_thinking_and_cot():
    g = mb.parse_generation('<think>red or blue? red, not blue.</think>\n\nred', L, 'thinking', False)
    assert g['answer'] == 'red' and g['closed'] and g['answer_status'] == 'bare_label'
    raw = '<think>red or blue? red, not blue.</think>\n\nred'
    assert raw[g['answer_offset']:].startswith('red') and g['answer_offset'] > raw.index('</think>')
    g = mb.parse_generation('<think>The color is red, not blue.</think>\n\nThe true color is red, not blue.',
                            L, 'thinking', False)
    assert g['answer'] is None and g['answer_parse_reason'] == 'conflicting_mentions' and g['closed']
    g = mb.parse_generation('<think>still thinking about red', L, 'thinking', True)
    assert g['answer'] is None and g['answer_parse_reason'] == 'truncated' and not g['closed']
    g = mb.parse_generation('<think>done</think>\n\nAnswer: red, and', L, 'thinking', True)   # truncated after close
    assert g['answer'] is None and g['answer_parse_reason'] == 'truncated' and g['closed']
    g = mb.parse_generation('<think>never closed', L, 'thinking', False)
    assert g['answer_parse_reason'] == 'reasoning_not_closed'
    g = mb.parse_generation('Step 1 ... red looks likely.\nAnswer: red', L, 'cot', False)
    assert g['answer'] == 'red' and g['closed'] and g['close_at'] == g['final'].index('Answer:')
    g = mb.parse_generation('Step 1 ... red looks likely.', L, 'cot', False)
    assert g['answer'] is None and not g['closed']


# ----------------------------------------------------------------------------- stub backends

class Row(dict):
    def __missing__(self, key):
        return 0


class StubVLM:
    """Records the rendered prompts; deterministic rows. score() mimics the direct readout,
    score_with_reasoning() alternates natural / forced readouts."""
    prompts = {}

    def __init__(self, model_dir, model_id, pixels=224):
        self.metadata = dict(model_id=model_id, model_revision=Path(model_dir).name, pixels=pixels)
        self.template_kwargs = {'enable_thinking': False}
        self.calls = 0

    def _text(self, rendered):
        return ''.join(c.get('text', '') for c in rendered['content'])

    def score(self, rendered, labels, max_new_tokens=8):
        return Row(probabilities={l: 0.5 for l in labels}, argmax_label=labels[0],
                   logits={l: 0.0 for l in labels}, canonical_candidate_mass=1.0, greedy_answer=labels[0],
                   input_tokens=1, prompt_sha256=hashlib.sha256(self._text(rendered).encode()).hexdigest(),
                   _text=self._text(rendered))

    def render_prompt(self, content, short_answer=True, template_kwargs=None):
        return 'x<think>'

    def score_with_reasoning(self, rendered, labels, max_new_tokens=4096, mode='thinking'):
        self.calls += 1
        natural = self.calls % 2 == 1
        StubVLM.prompts[self._text(rendered)] = True
        lg = {labels[0]: 20.0, labels[1]: -15.0}
        return dict(probabilities={labels[0]: 1.0, labels[1]: 0.0}, argmax_label=labels[0], logits=lg,
                    canonical_candidate_mass=0.9, prompt_sha256='h', reasoning_mode=mode,
                    readout='natural' if natural else 'forced', answer_forced=not natural,
                    answer=labels[0] if natural else None, answer_valid=natural,
                    answer_status='bare_label' if natural else FAILED,
                    answer_parse_reason=None if natural else 'truncated', truncated=not natural,
                    reasoning_closed=natural, hit_eos=natural, input_tokens=10, reasoning_tokens=5,
                    answer_tokens=1, generated_tokens=7, answer_token_index=6, answer_leading_space=False,
                    teacher_forced_argmax_is_generated=True if natural else None, final_text='',
                    generation_seconds=0.0, elapsed_seconds=0.0, reasoning='')


def run_rows(path):
    return [json.loads(l) for l in open(path)]


def capture(monkeypatch, module, argv):
    """Run module.main() with the stub backend; return the rows it wrote (score() rows keep their prompt)."""
    texts = []
    orig = StubVLM.score

    def score(self, rendered, labels, max_new_tokens=8):
        row = orig(self, rendered, labels, max_new_tokens)
        texts.append(row['_text'])
        return row
    monkeypatch.setattr(StubVLM, 'score', score)
    monkeypatch.setattr(module, 'FrozenVLM', StubVLM)
    monkeypatch.setattr(sys, 'argv', [module.__name__ + '.py'] + argv)
    try:
        module.main()
    except SystemExit as e:
        assert e.code in (None, 0)
    return texts


# ----------------------------------------------------------------------------- I03: history modes

MODES = ['full', 'window:10', 'cp_hard', 'cp_soft', 'oracle', 'sum_all', 'sum_cp', 'sum_window:10', 'sum_oracle']
TRIALS = [1, 2, 20, 41, 45, 60, 80]


class A:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.parametrize('mode', MODES)
def test_history_modes_match_run_volatile(mode, tmp_path, monkeypatch):
    """run_volatile.main (stub backend) and run_thinking render identical windows and prompts."""
    out = tmp_path / 'rv'
    texts = capture(monkeypatch, rv, ['--model-dir', str(tmp_path / 'rev'), '--model-id', 'm', '--data-dir', str(DATA),
                                      '--split', 'development', '--output', str(out), '--max-episodes', '4',
                                      '--readouts', 'choice', '--formats', 'text', '--domains', 'scenario',
                                      '--history-mode', mode, '--trials', *map(str, TRIALS)])
    rv_rows = run_rows(out / 'results_shard0of1.jsonl')
    assert len(rv_rows) == 4 * len(TRIALS) == len(texts)
    eps = {e['episode_id']: e for e in map(json.loads, open(DATA / 'episodes.jsonl'))}
    args = A(domain='scenario', controls=[], control_t=[], main_trials=TRIALS, readouts=['choice'],
             history_mode=mode, history_format='raw', hazard=1 / 15)
    renderer = rv.Renderer(json.loads((DATA / 'image_index.json').read_text()), 224)
    main_eps = [eps[e] for e in dict.fromkeys(r['episode_id'] for r in rv_rows)]
    jobs = {(f['episode_id'], f['t']): (make, f) for _, make, f in rt.enumerate_jobs(args, [], main_eps, renderer)}
    for r, text in zip(rv_rows, texts):
        make, f = jobs[(r['episode_id'], r['t'])]
        assert (f['window_start'], f['marker'], f['summary']) == (r['window_start'], r['marker'], r['summary'])
        assert rt.resolve_window(eps[r['episode_id']], r['t'], mode, 1 / 15) == (r['window_start'], r['marker'], r['summary'])
        rendered, _, target = make()
        assert ''.join(c.get('text', '') for c in rendered['content']) == text
        assert target == r['target']
        if mode.startswith('sum_') and r['t'] > 1:
            assert 'Summary of' in text and 'was correct' in text


# ----------------------------------------------------------------------------- I03: control seeds

def test_control_prompts_match_run_controls(tmp_path, monkeypatch):
    eps = [e for e in map(json.loads, open(DATA / 'episodes.jsonl')) if e['split'] == 'development']
    scen = sorted({e['scenario'] for e in eps})
    out = tmp_path / 'rc'
    texts = capture(monkeypatch, rc, ['--model-dir', str(tmp_path / 'rev'), '--model-id', 'm', '--data-dir', str(DATA),
                                      '--output', str(out), '--formats', 'text', '--domains', *scen,
                                      '--controls', *rt.DEFAULT_CONTROLS])
    rc_rows = run_rows(out / 'controls.jsonl')
    by_key = {r['key']: t for r, t in zip(rc_rows, texts)}
    args = A(domain='scenario', controls=rt.DEFAULT_CONTROLS, control_t=[40, 80], main_trials=[], readouts=['choice'],
             history_mode='full', history_format='raw', hazard=1 / 15)
    renderer = rv.Renderer(json.loads((DATA / 'image_index.json').read_text()), 224)
    jobs = rt.enumerate_jobs(args, eps, [], renderer)
    assert len(jobs) == len(eps) * 2 * 5
    for key, make, _ in jobs:
        rendered, _, _ = make()
        assert ''.join(c.get('text', '') for c in rendered['content']) == by_key[key], key


def test_coded_controls_share_raw_draws():
    ep = next(e for e in map(json.loads, open(DATA / 'episodes.jsonl')) if e['split'] == 'development')
    assert rt.control_rng(ep, 40, 'easy_long_coded', 'color').random() == \
        rt.control_rng(ep, 40, 'easy_long', 'color').random()
    twin = dict(ep, episode_id=ep['episode_id'] + 't')
    assert rt.control_rng(twin, 40, 'easy_short', 'color').random() == \
        rt.control_rng(ep, 40, 'easy_short', 'color').random()


# ----------------------------------------------------------------------------- end-to-end, resume, completion (I04/I05/R07)

def thinking_argv(tmp_path, *extra):
    return ['--model-dir', str(tmp_path / 'rev1'), '--model-id', 'm', '--data-dir', str(DATA),
            '--output', str(tmp_path / 'out'), '--split', 'development', '--domain', 'scenario', '--mode', 'thinking',
            '--max-new-tokens', '16384', '--shard', '0/4', '--control-t', '40', '80', '--main-trials', '20', '40',
            '--pixels', '224', '--max-episodes', '2', *extra]


def run_thinking(argv, monkeypatch):
    monkeypatch.setattr(rt, 'FrozenVLM', StubVLM)
    try:
        rt.main(argv)
        return 0
    except SystemExit as e:
        return e.code


OUT = 'thinking_shard0of4.jsonl'
MARK = '_COMPLETE_shard0of4'


def test_end_to_end_records(tmp_path, monkeypatch):
    argv = thinking_argv(tmp_path, '--history-mode', 'sum_all', '--history-format', 'raw', '--controls')
    assert run_thinking(argv, monkeypatch) == 0
    rows = run_rows(tmp_path / 'out' / OUT)
    assert len(rows) == 2 * 2 and all(r['task'] == 'main' and r['summary'] for r in rows)
    for r in rows:
        nat = r['answer_readout'] == 'natural'
        assert r['answer_forced'] == (not nat) and r['answer_valid'] == nat
        assert (r['y_natural'] is None) != nat and (r['y_forced'] is None) == nat
        assert (r['answer_is_target'] is None) == (not nat)
        sfx = 'natural' if nat else 'forced'
        assert r[f'logit_diff_{sfx}'] == r[f'logit_target_{sfx}'] - r[f'logit_other_{sfx}']
        assert abs(r[f'logit_diff_{sfx}']) == 35.0          # raw logits survive softmax saturation
        assert set(r['candidate_logits']) == {r['target'], r['other']}
        assert r['fingerprint']['history_mode'] == 'sum_all' and r['fingerprint']['max_new_tokens'] == 16384
        assert 'y' not in r
    assert (tmp_path / 'out' / MARK).exists()
    # default controls (5) + main on the job-manifest CLI
    argv = thinking_argv(tmp_path, '--history-mode', 'full', '--history-format', 'raw',
                         '--controls', *rt.DEFAULT_CONTROLS)
    argv[argv.index('--output') + 1] = str(tmp_path / 'out2')
    assert run_thinking(argv, monkeypatch) == 0
    rows = run_rows(tmp_path / 'out2' / OUT)
    assert sum(r['task'] == 'control' for r in rows) == 2 * 2 * 5


def test_defaults():
    a = rt.build_parser().parse_args(['--model-dir', 'x', '--model-id', 'm', '--output', 'o'])
    assert a.data_dir.name == 'episodes_v3' and a.controls == rt.DEFAULT_CONTROLS
    a = rt.build_parser().parse_args(['--model-dir', 'x', '--model-id', 'm', '--output', 'o', '--controls'])
    assert a.controls == []


def _base(tmp_path, monkeypatch):
    argv = thinking_argv(tmp_path, '--history-mode', 'full', '--controls', 'explicit_rel')
    assert run_thinking(argv, monkeypatch) == 0
    path = tmp_path / 'out' / OUT
    return argv, path, path.read_bytes()


def test_resume_truncates_half_line(tmp_path, monkeypatch):
    argv, path, data = _base(tmp_path, monkeypatch)
    lines = data.split(b'\n')[:-1]
    path.write_bytes(b'\n'.join(lines[:-1]) + b'\n' + lines[-1][:40])     # torn last row, no newline
    (tmp_path / 'out' / MARK).unlink()
    assert run_thinking(argv, monkeypatch) == 0
    rows = run_rows(path)
    assert len(rows) == len(lines) and len({r['key'] for r in rows}) == len(lines)
    assert (tmp_path / 'out' / MARK).exists()


def test_corrupt_inner_line_aborts(tmp_path, monkeypatch):
    argv, path, data = _base(tmp_path, monkeypatch)
    lines = data.split(b'\n')
    lines[1] = lines[1][:30]
    path.write_bytes(b'\n'.join(lines))
    with pytest.raises(RuntimeError, match='corrupt'):
        run_thinking(argv, monkeypatch)
    assert not (tmp_path / 'out' / MARK).exists()


def test_verify_rejects_duplicate_missing_extra_foreign(tmp_path, monkeypatch):
    argv, path, data = _base(tmp_path, monkeypatch)
    rows = run_rows(path)
    keys = [r['key'] for r in rows]
    fp = rows[0]['fingerprint']
    assert rt.verify_complete(path, keys, fp)[0]

    def check(rows_, expect_ok=False):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows_))
        assert rt.verify_complete(path, keys, fp)[0] is expect_ok
    check(rows + rows[:1])                                       # duplicate
    check(rows[:-1])                                             # missing
    check(rows + [dict(rows[0], key='explicit_rel|zzz|1|color|text')])   # extra
    check(rows[:-1] + [dict(rows[-1], fingerprint=dict(fp, max_new_tokens=1))])   # foreign fingerprint
    assert not rt.verify_complete(path, [], fp)[0]               # nothing expected is never complete
    # the runner itself refuses a duplicated file and leaves no marker
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows + rows[:1]))
    with pytest.raises(RuntimeError, match='duplicate'):
        run_thinking(argv, monkeypatch)
    assert not (tmp_path / 'out' / MARK).exists()


def test_stale_marker_and_fingerprint_change(tmp_path, monkeypatch):
    argv, path, data = _base(tmp_path, monkeypatch)
    # different max_new_tokens on the same output: refuse, and the old marker does not survive
    argv2 = list(argv)
    argv2[argv2.index('--max-new-tokens') + 1] = '4096'
    with pytest.raises(RuntimeError, match='different configuration'):
        run_thinking(argv2, monkeypatch)
    assert not (tmp_path / 'out' / MARK).exists()
    # other model revision / pixels / history mode are fingerprinted too
    fp = run_rows(path)[0]['fingerprint']
    for k in ('model_revision', 'data_sha256', 'reasoning_mode', 'history_mode', 'history_format',
              'max_new_tokens', 'pixels'):
        assert k in fp
    argv3 = list(argv)
    argv3[argv3.index('--model-dir') + 1] = str(tmp_path / 'rev2')
    with pytest.raises(RuntimeError, match='different configuration'):
        run_thinking(argv3, monkeypatch)
    # resuming the original configuration re-earns the marker without new rows
    assert run_thinking(argv, monkeypatch) == 0
    assert path.read_bytes() == data and (tmp_path / 'out' / MARK).exists()
