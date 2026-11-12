"""Tests for analyze_main.py / analyze_mechanism.py / analyze_controls.py on synthetic data.

Run: python -m pytest -q research/volatile_trust/test_analysis.py

Regression tests for the external review (suggestion.md) reproduce the review's failing example first
and then check the fixed behaviour: I04, A01-A05, I06, R03, R05, R08, R10.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_controls as ac  # noqa: E402
import analyze_main as am  # noqa: E402
import analyze_mechanism as mech  # noqa: E402
from controller import history_window  # noqa: E402

V3 = os.path.join(HERE, 'episodes_v3', 'episodes.jsonl')
V1 = os.path.join(HERE, 'episodes_v1', 'episodes.jsonl')
ROB = os.path.join(HERE, 'episodes_robust_v3', 'episodes.jsonl')
NB = 200


@pytest.fixture(scope='module')
def episodes():
    return am.load_episodes(V3)


def _dev(episodes, n=16):
    return {k: v for k, v in episodes.items() if k.startswith('dev') and not k.endswith('t')}


# ----------------------------------------------------------------------------- fake data helpers

def make_eps(tmp_path, bases, src=V3, name='eps'):
    """Small dataset: the given base episodes and their twins, copied from ``src``."""
    eps = am.load_episodes(src)
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    with open(d / 'episodes.jsonl', 'w') as f:
        for b in bases:
            for e in (b, b + 't'):
                f.write(json.dumps(eps[e]) + '\n')
    return d


HO6 = ['hel000', 'hel004', 'hel008', 'hel012', 'hel016', 'hel020']    # one per text scenario


def _window(o, t, mode):
    if not mode.startswith('sum_'):
        return 1
    if mode == 'sum_oracle':
        return None
    base = {'sum_all': 'full', 'sum_cp': 'cp_hard'}.get(mode, mode.replace('sum_window', 'window'))
    return history_window(o, t, base)[0]


def write_run(tmp_path, name, epdir, logit_fn, mode='full', fmt='text', bias=1.5, hf='raw',
              slots=('pet', 'scenario'), fields='logits', pixels=224, hazard=1 / 15, keep=None,
              revision='rev1', model='fake/M', meta=True, split=None, extra=None):
    """Fake run. Original twin logit = L + bias, twin = L - bias (bias cancels on averaging).

    logit_fn(ep, slot) -> 80 task logits; keep(ep, tr, slot, twin) -> bool drops rows when False.
    fields='logits' writes logit_a/logit_other/logit_diff (+ y); fields='y' writes y only (legacy pilot).
    """
    d = tmp_path / name
    d.mkdir()
    eps = am.load_episodes(epdir / 'episodes.jsonl')
    if meta:
        args = dict(data_dir=str(epdir), hazard=str(hazard), pixels=str(pixels), max_episodes='None', trials='None')
        (d / 'run_metadata_shard0of1.jsonl').write_text(json.dumps(dict(model_id=model, model_revision=revision,
                                                                          args=args)) + '\n')
    with open(d / 'results_shard0of1.jsonl', 'w') as f:
        for eid in sorted(e for e in eps if not e.endswith('t')):
            if split and eps[eid]['split'] != split:
                continue
            o = [tr['a_correct'] for tr in eps[eid]['trials']]
            for slot in slots:
                L = logit_fn(eps[eid], slot)
                for twin, b in (('', bias), ('t', -bias)):
                    ep = eps[eid + twin]
                    dom = 'pet' if slot == 'pet' else ep['scenario']
                    for tr in ep['trials']:
                        t = tr['t']
                        if keep is not None and not keep(ep, tr, slot, twin):
                            continue
                        lg = float(L[t - 1] + b)
                        r = dict(episode_id=eid + twin, t=t, readout='choice', domain=dom, domain_slot=slot, format=fmt,
                                 p_a=tr['p_a'], phase=tr['phase'], a_correct=tr['a_correct'],
                                 y=float(am.sigmoid(lg)), model_id=model, hint=True, history_mode=mode,
                                 history_format=hf, pixels=pixels, window_start=_window(o, t, mode),
                                 candidate_mass=0.9, greedy='cat' if dom == 'pet' else am.SCENARIOS[dom]['labels'][0],
                                 target='cat' if dom == 'pet' else am.SCENARIOS[dom]['labels'][0])
                        if fields == 'logits':
                            r.update(logit_a=lg + 7.0, logit_other=7.0, logit_diff=lg)
                        if extra:
                            r.update(extra(ep, tr, slot, twin))
                        f.write(json.dumps(r) + '\n')
    return str(d)


def oracle_logits(ep, slot=None):
    return np.array([2.0 if tr['p_a'] > 0.5 else -2.0 for tr in ep['trials']])


def sum_all_logits(ep, slot=None):
    o = np.array([tr['a_correct'] for tr in ep['trials']], float)
    cs = np.concatenate([[0], np.cumsum(o)])
    n = np.arange(len(o))
    return np.where(n == 0, 0.0, 4.0 * (cs[:-1] - n / 2) / np.maximum(n, 1))


def window_logits(K):
    def f(ep, slot=None):
        o = [tr['a_correct'] for tr in ep['trials']]
        out = np.zeros(len(o))
        for t in range(2, len(o) + 1):
            ws = history_window(o, t, f'window:{K}')[0]
            seg = o[ws - 1:t - 1]
            out[t - 1] = 0.0 if 2 * sum(seg) == len(seg) else (3.0 if 2 * sum(seg) > len(seg) else -3.0)
        return out
    return f


def cp_logits(ep, slot=None):
    o = [tr['a_correct'] for tr in ep['trials']]
    out = np.zeros(len(o))
    for t in range(2, len(o) + 1):
        ws = history_window(o, t, 'cp_hard')[0]
        seg = o[ws - 1:t - 1]
        out[t - 1] = 0.0 if 2 * sum(seg) == len(seg) else (3.0 if 2 * sum(seg) > len(seg) else -3.0)
    return out


def noise_logits(ep, slot=None):
    return np.random.default_rng(int(ep['episode_id'][-3:]) + (7 if slot == 'pet' else 0)).normal(0, 1, 80)


# ----------------------------------------------------------------------------- primary test (unchanged API)

def test_paired_gap_detects_effect_and_null():
    rng = np.random.default_rng(1)
    A = {f'e{i}': 0.5 + 0.02 * rng.standard_normal() for i in range(48)}
    C = {k: v + 0.10 + 0.03 * rng.standard_normal() for k, v in A.items()}
    g = am.paired_test(A, C, n_boot=2000)
    assert g['n'] == 48 and abs(g['mean'] - 0.10) < 0.02
    assert g['ci95'][0] < g['mean'] < g['ci95'][1] and g['ci95'][0] > 0.05
    assert g['p_wilcoxon'] < 1e-6
    N = {k: v + 0.03 * rng.standard_normal() for k, v in A.items()}
    g0 = am.paired_test(A, N, n_boot=2000)
    assert g0['ci95'][0] < 0 < g0['ci95'][1] and g0['p_wilcoxon'] > 0.01


def test_paired_gap_uses_only_common_episodes():
    g = am.paired_test({'a': 0.5, 'b': 0.5, 'c': 0.4}, {'a': 0.6, 'b': 0.7}, n_boot=100)
    assert g['n'] == 2 and g['mean'] == pytest.approx(0.15)


def test_holm():
    adj = am.holm([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])


def test_position_ci():
    rng = np.random.default_rng(2)
    A = {i: 0.5 + 0.01 * rng.standard_normal() for i in range(40)}
    C = {i: A[i] + 0.2 for i in A}
    B = {i: A[i] + 0.05 + 0.01 * rng.standard_normal() for i in A}
    p = am.position_ci(A, B, C, n_boot=2000)
    assert p['interpretable'] and abs(p['position'] - 0.25) < 0.03 and p['ci95'][0] < 0.25 < p['ci95'][1]
    assert p['B_minus_A']['mean'] == pytest.approx(0.05, abs=0.01) and p['C_minus_B']['mean'] == pytest.approx(0.15, abs=0.01)


def test_tost_equivalent_and_not():
    rng = np.random.default_rng(3)
    eq = am.tost(0.005 * rng.standard_normal(48), margin=0.02, n_boot=2000)
    assert eq['equivalent_uncorrected'] and eq['equivalent_boot_uncorrected'] and eq['p_tost'] < 0.001
    ne = am.tost(0.03 + 0.005 * rng.standard_normal(48), margin=0.02, n_boot=2000)
    assert not ne['equivalent_uncorrected'] and ne['p_tost'] > 0.5
    noisy = am.tost(0.05 * rng.standard_normal(10), margin=0.02, n_boot=2000)   # underpowered: not equivalent
    assert not noisy['equivalent_uncorrected']
    assert 'equivalent' not in eq and eq['equivalent_holm'] is None      # no ambiguous flag


# ----------------------------------------------------------------------------- end to end

def test_twin_average_cancels_bias_and_end_to_end(tmp_path):
    epd = make_eps(tmp_path, HO6)
    ra = write_run(tmp_path, 'A', epd, noise_logits)
    rc = write_run(tmp_path, 'C', epd, sum_all_logits, mode='sum_all')
    ro = write_run(tmp_path, 'O', epd, oracle_logits, mode='sum_oracle')
    res = am.analyze([ra, rc, ro], n_boot=NB, primary_c='sum_all')
    assert res['mode'] == 'formal'
    o = [v for k, v in res['cells'].items() if k.endswith('C|sum_oracle|text/all')][0]
    assert o['fb_stable'] == pytest.approx(float(am.sigmoid(2.0)), abs=1e-6)
    t = [x for x in res['tost'] if x['C'] == 'C|sum_all|text' and x['domain'] == 'all' and x['rule'] == 'window_counts'][0]
    assert t['mean'] == pytest.approx(0.0, abs=1e-12) and t['equivalent_uncorrected']
    assert t['role'] == 'secondary' and t['p_holm'] is not None
    g = [x for x in res['gaps'] if x['primary']]
    assert len(g) == 1 and g[0]['C'] == 'C|sum_all|text' and g[0]['mean'] > 0.05 and 'p_holm' in g[0]
    assert res['info']['rule_window_mismatches'] == 0
    md = am.to_markdown(res)
    assert 'Primary family' in md and 'equiv. (Holm)' in md


def test_reversal_curve_only_true_reversals(episodes):
    ep = dict(next(iter(_dev(episodes).values())))
    ep['reversals'] = [1] + list(ep['reversals'])      # a spurious reversal at t=1 must be ignored
    trials = [dict(t=tr['t'], py=0.9, a_correct=tr['a_correct'], p_a=tr['p_a'], phase=tr['phase'], twinned=True,
                   ep=ep, scenario='na') for tr in ep['trials']]
    m = am.episode_metrics(trials)
    rev = am.true_reversals(ep)
    assert 1 not in rev
    exp0 = np.mean([0.9 if ep['trials'][r - 1]['p_a'] > 0.5 else 0.1 for r in rev])
    assert m['curve'][0] == pytest.approx(exp0)


def test_rule_alone_matches_counts(episodes):
    ep = next(iter(_dev(episodes).values()))
    trials = [dict(t=tr['t'], a_correct=tr['a_correct'], window_start=None, ep=ep, hazard=1 / 15,
                   history_mode='sum_window:5') for tr in ep['trials']]
    cr, mm = am.rule_alone_trials(trials)
    o = [tr['a_correct'] for tr in ep['trials']]
    t = 30
    seg = o[t - 6:t - 1]
    cA = float(sum(seg) > 2.5)
    assert cr[t] == cA * o[t - 1] + (1 - cA) * (1 - o[t - 1]) and mm == 0


# ----------------------------------------------------------------------------- I04 logits

@pytest.mark.parametrize('bias', [15.0, 30.0])
def test_i04_raw_logits_recover_task_logit(tmp_path, bias):
    epd = make_eps(tmp_path, HO6[:1])
    L3 = lambda ep, slot: np.full(80, 3.0)  # noqa: E731
    # review's failing example: only clipped y saved -> twin average is biased toward 0.5 and flagged approx
    rl = write_run(tmp_path, 'legacy', epd, L3, bias=bias, fields='y', slots=('pet',))
    rows, _ = am.load_rows([rl])
    p = am.pair_trials(rows)
    tr = next(iter(next(iter(p.values())).values()))
    assert not tr[5]['exact']
    if bias == 15.0:
        assert tr[5]['py'] == pytest.approx(0.7125, abs=1e-3)      # the review's number
    assert tr[5]['py'] < 0.9
    # with raw candidate logits the task logit is recovered exactly
    rx = write_run(tmp_path, 'exact', epd, L3, bias=bias, fields='logits', slots=('pet',))
    rows, _ = am.load_rows([rx])
    p = am.pair_trials(rows)
    tr = next(iter(next(iter(p.values())).values()))
    assert all(x['exact'] for x in tr)
    assert tr[5]['l'] == pytest.approx(3.0, abs=1e-9) and tr[5]['py'] == pytest.approx(0.9526, abs=1e-4)


def test_i04_logit_fields_priority():
    assert am.row_logit(dict(logit_diff=40.0, y=1.0)) == (40.0, True)
    assert am.row_logit(dict(logit_a=12.0, logit_other=-20.0, y=1.0)) == (32.0, True)
    l, exact = am.row_logit(dict(y=1.0))
    assert not exact and l == pytest.approx(np.log((1 - 1e-6) / 1e-6))


def test_i04_legacy_cells_flagged_approx(tmp_path):
    epd = make_eps(tmp_path, HO6)
    ra = write_run(tmp_path, 'A', epd, noise_logits, fields='y')
    rc = write_run(tmp_path, 'C', epd, sum_all_logits, mode='sum_all')
    res = am.analyze([ra, rc], n_boot=NB)
    a = [v for k, v in res['cells'].items() if k.endswith('A|full|text/all')][0]
    c = [v for k, v in res['cells'].items() if k.endswith('C|sum_all|text/all')][0]
    assert a['approx_frac'] == 1.0 and c['approx_frac'] == 0.0
    assert all(g['approx'] for g in res['gaps'])
    assert '(approx)' in am.to_markdown(res)


# ----------------------------------------------------------------------------- A01 completeness

def _a01_runs(tmp_path):
    """Review example: A and C use the same y = 0.8 strategy; A keeps only wrong trials, C only right ones."""
    epd = make_eps(tmp_path, HO6)
    L = lambda ep, slot: np.full(80, float(np.log(4)))  # noqa: E731  (y = 0.8 -> always choose A)
    ra = write_run(tmp_path, 'A', epd, L, bias=0.0, keep=lambda ep, tr, s, tw: tr['a_correct'] == 0)
    rc = write_run(tmp_path, 'C', epd, L, bias=0.0, mode='sum_all', keep=lambda ep, tr, s, tw: tr['a_correct'] == 1)
    return ra, rc


def test_a01_false_gap_refused_in_formal_mode(tmp_path):
    ra, rc = _a01_runs(tmp_path)
    with pytest.raises(am.AnalysisError, match='missing'):
        am.analyze([ra, rc], n_boot=NB)


def test_a01_partial_mode_lists_missing_and_has_no_false_gap(tmp_path):
    ra, rc = _a01_runs(tmp_path)
    res = am.analyze([ra, rc], n_boot=NB, partial=True)
    # the per-arm accuracies reproduce the review's setting (A = 0, C = 1 on their own masks) ...
    a = [v for k, v in res['cells'].items() if k.endswith('A|full|text/all')][0]
    c = [v for k, v in res['cells'].items() if k.endswith('C|sum_all|text/all')][0]
    assert a['acc'] == 0.0 and c['acc'] == 1.0
    # ... but the paired gap uses the common observation mask (empty), so no gap = 1.0 / p = 0.0005
    g = [x for x in res['gaps'] if x['domain'] == 'all'][0]
    assert g['n'] == 0 and not np.isfinite(g['mean'])
    assert not any(x['primary'] for x in res['gaps']) and all(x['role'] == 'exploratory' for x in res['gaps'])
    assert all(t['role'] == 'exploratory' for t in res['tost'])
    assert res['info']['missing'] and res['mode'] == 'partial'
    md = am.to_markdown(res)
    assert 'EXPLORATORY' in md and 'Missing / dropped data' in md


def test_a01_missing_twin_trial_or_slot_refused(tmp_path):
    epd = make_eps(tmp_path, HO6)
    full = write_run(tmp_path, 'C', epd, sum_all_logits, mode='sum_all')
    no_twin = write_run(tmp_path, 'A1', epd, noise_logits,
                        keep=lambda ep, tr, s, tw: not (s == 'pet' and tw == 't' and ep['episode_id'] == 'hel000t'
                                                       and tr['t'] == 40))
    with pytest.raises(am.AnalysisError, match='twin'):
        am.analyze([no_twin, full], n_boot=NB)
    no_slot = write_run(tmp_path, 'A2', epd, noise_logits, slots=('scenario',))
    with pytest.raises(am.AnalysisError, match='slot'):
        am.analyze([no_slot, full], n_boot=NB)
    res = am.analyze([no_twin, full], n_boot=NB, partial=True)
    assert sum(res['info']['untwinned_dropped'].values()) == 1


def test_a01_bad_json_refused_in_formal(tmp_path):
    epd = make_eps(tmp_path, HO6)
    ra = write_run(tmp_path, 'A', epd, noise_logits)
    with open(os.path.join(ra, 'results_shard0of1.jsonl'), 'a') as f:
        f.write('{"episode_id": "hel0')
    with pytest.raises(am.AnalysisError, match='unparsable'):
        am.load_rows([ra])


# ----------------------------------------------------------------------------- A02 fingerprints / identity

def test_a02_mixed_pixels_in_one_cell_refused(tmp_path):
    """Review: 224 / 448 complete twin pairs with y = .2 / .8 collapsed last-write-wins. Now refused."""
    epd = make_eps(tmp_path, HO6)
    r224 = write_run(tmp_path, 'p224', epd, lambda ep, s: np.full(80, float(am.logit(0.2))), fmt='image', pixels=224)
    r448 = write_run(tmp_path, 'p448', epd, lambda ep, s: np.full(80, float(am.logit(0.8))), fmt='image', pixels=448)
    with pytest.raises(am.AnalysisError, match='pixels'):
        am.analyze([r224, r448], n_boot=NB, partial=True)
    with pytest.raises(am.AnalysisError, match='pixels'):
        am.analyze([r448, r224], n_boot=NB, partial=True)


def test_a02_mixed_hazard_refused(tmp_path):
    epd = make_eps(tmp_path, HO6)
    r1 = write_run(tmp_path, 'h1', epd, cp_logits, mode='sum_cp', hazard=0.5, slots=('pet',))
    r2 = write_run(tmp_path, 'h2', epd, cp_logits, mode='sum_cp', hazard=1 / 15, slots=('scenario',))
    r3 = write_run(tmp_path, 'h3', epd, cp_logits, mode='sum_cp', hazard=1 / 15, slots=('pet',))
    with pytest.raises(am.AnalysisError, match='hazard'):
        am.analyze([r1, r3], n_boot=NB, partial=True)
    am.analyze([r2, r3], n_boot=NB, partial=True)   # one hazard: fine


def test_a02_input_order_independent(tmp_path):
    epd = make_eps(tmp_path, HO6)
    ra = write_run(tmp_path, 'A', epd, noise_logits)
    rc = write_run(tmp_path, 'C', epd, sum_all_logits, mode='sum_all')
    rb = write_run(tmp_path, 'B', epd, oracle_logits, hf='coded')
    j = [json.dumps(am.analyze(order, n_boot=NB), default=am._json_default, sort_keys=True)
         for order in ([ra, rc, rb], [rb, rc, ra], [rc, ra, rb])]
    assert j[0] == j[1] == j[2]


def test_a02_conflicting_duplicates_refused(tmp_path):
    epd = make_eps(tmp_path, HO6)
    ra = write_run(tmp_path, 'A', epd, noise_logits)
    rows = open(os.path.join(ra, 'results_shard0of1.jsonl')).read().splitlines()
    r = json.loads(rows[10])
    r['logit_diff'] += 1.0
    d = tmp_path / 'A_resume'
    d.mkdir()
    (d / 'run_metadata_shard0of1.jsonl').write_text(open(os.path.join(ra, 'run_metadata_shard0of1.jsonl')).read())
    (d / 'results_shard0of1.jsonl').write_text(json.dumps(r) + '\n')
    with pytest.raises(am.AnalysisError, match='conflicting'):
        am.analyze([ra, str(d)], n_boot=NB)
    (d / 'results_shard0of1.jsonl').write_text(rows[10] + '\n')       # identical resume duplicate: merged
    am.analyze([ra, str(d)], n_boot=NB, partial=True)


def test_a02_dataset_identity_v1_v3_never_merged(tmp_path):
    """Review: v1 and v3 'dev000' were put into one episode unit. Now separate groups by data hash."""
    e1 = make_eps(tmp_path, ['dev000', 'dev001'], src=V1, name='v1')
    e3 = make_eps(tmp_path, ['dev000', 'dev001'], src=V3, name='v3')
    r1 = write_run(tmp_path, 'r1', e1, noise_logits, slots=('pet',))
    r3 = write_run(tmp_path, 'r3', e3, noise_logits, slots=('pet',))
    rows, _ = am.load_rows([r1, r3])
    p = am.pair_trials(rows)
    assert len(p) == 2
    datas = sorted(c[1] for c in p)
    assert datas[0] != datas[1] and datas[0].startswith('v1#') and datas[1].startswith('v3#')
    for per in p.values():
        assert all(len(tr) == 80 for tr in per.values())       # no doubled t


def test_a02_no_episodes_v1_fallback(tmp_path):
    epd = make_eps(tmp_path, HO6)
    r = write_run(tmp_path, 'nometa', epd, noise_logits, meta=False)
    with pytest.raises(am.AnalysisError, match='fallback'):
        am.load_rows([r])
    rows, _ = am.load_rows([r], episodes_override=str(epd))
    assert rows


# ----------------------------------------------------------------------------- A03 TOST + Holm family

def test_a03_holm_does_not_keep_uncorrected_equivalence():
    """Review: p = 0.0449 -> Holm (3 tests) 0.1347 still printed equivalent=True."""
    rng = np.random.default_rng(0)
    d = rng.standard_normal(48)
    d = (d - d.mean()) / d.std(ddof=1)
    # choose sd so that p_tost = 0.0449 with mean 0, margin 0.02, n = 48
    from scipy import stats
    se = 0.02 / stats.t.isf(0.0449, 47)
    t = am.tost(d * se * np.sqrt(48), margin=0.02, alpha=0.05, n_boot=NB)
    assert t['p_tost'] == pytest.approx(0.0449, abs=1e-6) and t['equivalent_uncorrected']
    others = [dict(p_tost=0.5), dict(p_tost=0.6)]
    fam = [t] + others
    missing = am.assign_secondary_family(fam, ['sum_all', 'sum_cp', 'sum_window:10'], alpha=0.05, label='M')
    assert missing == []
    assert t['p_holm'] == pytest.approx(0.1347, abs=1e-4)
    assert t['equivalent_uncorrected'] and t['equivalent_holm'] is False and t['role'] == 'secondary'


def test_a03_missing_member_counts_as_p1():
    t = dict(p_tost=0.01)
    missing = am.assign_secondary_family([t, None, None], ['sum_all', 'sum_cp', 'sum_window:K*'])
    assert missing == ['sum_cp', 'sum_window:K*'] and t['p_holm'] == pytest.approx(0.03)


def test_a03_family_members_and_dev_selection(tmp_path):
    epd = make_eps(tmp_path, ['dev000', 'dev001', 'dev002', 'dev003'] + HO6)
    runs_dev = [write_run(tmp_path, f'dev_w{K}', epd, window_logits(K), mode=f'sum_window:{K}', split='development')
                for K in (5, 10, 20)]
    # make K=10 the dev winner: K=5 run follows the rule badly (negated)
    runs_dev[0] = write_run(tmp_path, 'dev_w5_bad', epd, lambda ep, s: -window_logits(5)(ep), mode='sum_window:5',
                            split='development')
    runs_dev[2] = write_run(tmp_path, 'dev_w20_bad', epd, lambda ep, s: -window_logits(20)(ep), mode='sum_window:20',
                            split='development')
    sel_path = tmp_path / 'dev_sel.json'
    doc = am.write_dev_selection(str(sel_path), runs_dev)
    assert doc['models']['M']['k_star'] == 10 and doc['models']['M']['split'] == 'development'
    ho = dict(A=write_run(tmp_path, 'A', epd, noise_logits, split='held_out'),
              all=write_run(tmp_path, 'Call', epd, sum_all_logits, mode='sum_all', split='held_out'),
              cp=write_run(tmp_path, 'Ccp', epd, cp_logits, mode='sum_cp', split='held_out'),
              w5=write_run(tmp_path, 'Cw5', epd, window_logits(5), mode='sum_window:5', split='held_out'),
              w10=write_run(tmp_path, 'Cw10', epd, window_logits(10), mode='sum_window:10', split='held_out'),
              orc=write_run(tmp_path, 'Corc', epd, oracle_logits, mode='sum_oracle', split='held_out'))
    res = am.analyze(list(ho.values()), n_boot=NB, dev_selection=str(sel_path))
    fam = [v for k, v in res['families'].items() if k.startswith('secondary/')]
    assert len(fam) == 1 and fam[0]['k_star'] == 10 and fam[0]['margin'] == 0.02 and fam[0]['alpha'] == 0.05
    assert fam[0]['members'] == ['C|sum_all|text vs window-count rule (domains pooled)',
                                 'C|sum_cp|text vs window-count rule (domains pooled)',
                                 'C|sum_window:10|text vs window-count rule (domains pooled)']
    sec = {t['C'] for t in res['tost'] if t['role'] == 'secondary'}
    assert sec == {'C|sum_all|text', 'C|sum_cp|text', 'C|sum_window:10|text'}
    for t in res['tost']:
        if t['role'] != 'secondary':
            assert t['p_holm'] is None and t['equivalent_holm'] is None
    # --k-star overrides
    res2 = am.analyze(list(ho.values()), n_boot=NB, k_star=5)
    assert {t['C'] for t in res2['tost'] if t['role'] == 'secondary'} == {'C|sum_all|text', 'C|sum_cp|text',
                                                                           'C|sum_window:5|text'}
    md = am.to_markdown(res)
    assert 'Secondary family' in md and 'margin 0.02' in md and 'alpha 0.05' in md


def test_robust_results_do_not_expand_confirmatory_family(tmp_path):
    main_eps = make_eps(tmp_path, ['hel000', 'hel001'], name='main')
    robust_eps = make_eps(tmp_path, ['rob000', 'rob001'], src=ROB, name='robust')
    main_runs = [write_run(tmp_path, 'main_A', main_eps, noise_logits),
                 write_run(tmp_path, 'main_C', main_eps, sum_all_logits, mode='sum_all')]
    robust_runs = [write_run(tmp_path, 'rob_A', robust_eps, noise_logits),
                   write_run(tmp_path, 'rob_C', robust_eps, sum_all_logits, mode='sum_all')]
    main = am.analyze(main_runs, n_boot=NB)
    combined = am.analyze(main_runs + robust_runs, n_boot=NB)
    assert combined['families']['primary'] == main['families']['primary']
    primary = [g for g in combined['gaps'] if g['primary']]
    assert len(primary) == 1
    assert primary[0]['p_holm'] == main['gaps'][-1]['p_holm']
    robust_only = am.analyze(robust_runs, n_boot=NB)
    robust_gaps = [g for g in combined['gaps'] if g['data'].startswith('robust#')]
    assert robust_gaps == robust_only['gaps']
    assert [g['role'] for g in robust_gaps if g['domain'] == 'all'] == ['secondary']
    assert combined['families']['robustness'] == robust_only['families']['robustness']
    assert not any('/robust#' in k for k in combined['families'])


def test_dev_selection_refuses_partial_missing_candidate_and_missing_slot(tmp_path):
    epd = make_eps(tmp_path, ['dev000', 'dev001'])
    runs = [write_run(tmp_path, f'w{K}', epd, window_logits(K), mode=f'sum_window:{K}')
            for K in (5, 10, 20)]
    dest = tmp_path / 'selection.json'
    with pytest.raises(am.AnalysisError, match='partial'):
        am.write_dev_selection(dest, runs, partial=True)
    with pytest.raises(am.AnalysisError, match='5.*10.*20'):
        am.write_dev_selection(dest, runs[:2])
    assert not dest.exists()
    missing_slot = [write_run(tmp_path, f'scenario_w{K}', epd, window_logits(K), mode=f'sum_window:{K}',
                              slots=('scenario',)) for K in (5, 10, 20)]
    with pytest.raises(am.AnalysisError, match='pet.*scenario'):
        am.write_dev_selection(dest, missing_slot)
    doc = am.write_dev_selection(dest, runs)
    assert am.load_k_star(dest, 'M@rev1') in (5, 10, 20)
    for change in ('partial', 'missing_candidate', 'wrong_winner'):
        changed = json.loads(json.dumps(doc))
        if change == 'partial':
            changed['mode'] = 'partial'
        elif change == 'missing_candidate':
            changed['models']['M']['acc_by_k'].pop('20')
        else:
            changed['models']['M']['k_star'] = 999
        dest.write_text(json.dumps(changed))
        with pytest.raises(am.AnalysisError):
            am.load_k_star(dest, 'M@rev1')


# ----------------------------------------------------------------------------- A04 scenarios

def test_a04_pet_only_effect_does_not_pass_scenario_test(tmp_path):
    """Review: text gaps all 0, pet gap .405 -> six scenarios '+.2025', sign test p = .03125. Now pet-free."""
    epd = make_eps(tmp_path, HO6)
    a = write_run(tmp_path, 'A', epd, lambda ep, s: -oracle_logits(ep) if s == 'pet' else oracle_logits(ep))
    c = write_run(tmp_path, 'C', epd, oracle_logits, mode='sum_all')
    res = am.analyze([a, c], n_boot=NB)
    g = {x['domain']: x for x in res['gaps']}
    assert g['pet']['mean'] > 0.3 and g['scenario']['mean'] == pytest.approx(0.0)
    assert g['all']['mean'] == pytest.approx(g['pet']['mean'] / 2)
    sc = [s for s in res['scenarios'] if s['C'] == 'C|sum_all|text']
    assert len(sc) == 1 and sc[0]['domain_slot'] == 'scenario'
    assert set(sc[0]['per_scenario']) == {'color', 'weather', 'stock', 'sports', 'medical', 'news'}
    assert all(v['mean'] == 0.0 for v in sc[0]['per_scenario'].values())
    assert sc[0]['sign_test']['n'] == 0 and not (sc[0]['sign_test']['p'] < 0.05)
    assert '8 episode pairs' in sc[0]['note'] and '6 units' in sc[0]['note']


# ----------------------------------------------------------------------------- A05 position

def test_a05_position_not_interpretable_never_crashes():
    # review: two episodes A = C = .5, B = .6 -> IndexError
    p = am.position_ci({0: .5, 1: .5}, {0: .6, 1: .6}, {0: .5, 1: .5}, n_boot=500)
    assert not p['interpretable'] and 'threshold' in p['reason'] and not np.isfinite(p['position'])
    assert p['B_minus_A']['mean'] == pytest.approx(0.1) and p['C_minus_B']['mean'] == pytest.approx(-0.1)
    p = am.position_ci({0: .6, 1: .6}, {0: .6, 1: .7}, {0: .5, 1: .55}, n_boot=500)       # C < A
    assert not p['interpretable']
    rng = np.random.default_rng(0)       # mean C - A above threshold but bootstrap crosses 0
    A = {i: 0.5 for i in range(6)}
    C = {i: 0.5 + v for i, v in enumerate([0.6, -0.2, -0.2, 0.2, -0.05, 0.05])}
    B = {i: 0.5 + 0.1 * rng.random() for i in range(6)}
    p = am.position_ci(A, B, C, n_boot=2000)
    assert not p['interpretable'] and 'crosses 0' in p['reason']
    p = am.position_ci({}, {}, {}, n_boot=10)
    assert p['n'] == 0 and not p['interpretable']


def test_a05_end_to_end_with_zero_gap(tmp_path):
    epd = make_eps(tmp_path, HO6)
    a = write_run(tmp_path, 'A', epd, sum_all_logits)
    b = write_run(tmp_path, 'B', epd, oracle_logits, hf='coded')
    c = write_run(tmp_path, 'C', epd, sum_all_logits, mode='sum_all')     # C == A exactly
    res = am.analyze([a, b, c], n_boot=NB)
    pos = [p for p in res['positions'] if p['domain'] == 'all'][0]
    assert not pos['interpretable'] and pos['B_minus_A']['mean'] > 0
    assert 'not interpretable' in am.to_markdown(res)


# ----------------------------------------------------------------------------- R03 agreement

def test_r03_trial_level_agreement_and_disagreement_table(tmp_path):
    epd = make_eps(tmp_path, HO6)

    def flipped(ep, s):
        L = sum_all_logits(ep)
        L = np.where(L == 0, 0.0, np.sign(L) * 2.0)
        L[[9, 19, 29]] *= -1           # flip three trials
        return L
    c = write_run(tmp_path, 'C', epd, flipped, mode='sum_all')
    a = write_run(tmp_path, 'A', epd, noise_logits)
    res = am.analyze([a, c], n_boot=NB)
    ag = [x for x in res['agreement'] if x['domain'] == 'pet'][0]
    nd = ag['n_disagree']
    assert 0 < nd <= 3 * 6 and ag['dis_vlm_correct'] + ag['dis_rule_correct'] == nd
    assert ag['dis_stable'] + ag['dis_volatile'] == nd and ag['dis_vlm_chose_A'] + ag['dis_vlm_chose_B'] == nd
    assert ag['agree_trial_pooled'] == pytest.approx(1 - nd / ag['n_nontied'])
    assert 'Trial-level choice agreement' in am.to_markdown(res)


# ----------------------------------------------------------------------------- R05 baselines

def test_r05_baselines_reproduce_review_dev_numbers(episodes):
    dev = _dev(episodes)
    kb = {b: set(range(1, 81)) for b in dev}
    tab = am.baseline_table(kb, dev, 1 / 15, 200, 0)
    exp = dict(truth_quota=0.6606, always_A=0.6464, bocpd_posterior=0.6748, bocpd_map_window=0.6780, known_pA=0.7983)
    for k, v in exp.items():
        assert tab[k]['acc'] == pytest.approx(v, abs=5e-5), k
    assert tab['truth_quota']['uses_generator_info'] and tab['always_A']['uses_generator_info']
    assert tab['known_pA']['uses_generator_info']
    assert not tab['bocpd_posterior']['uses_generator_info'] and not tab['bocpd_map_window']['uses_generator_info']


def test_r05_quota_baseline_not_applicable_on_iid():
    rob = {k: v for k, v in am.load_episodes(ROB).items() if not k.endswith('t')}
    tab = am.baseline_table({b: set(range(1, 81)) for b in rob}, rob, 1 / 15, 50, 0)
    if not any(am.quota_holds(e) for e in rob.values()):
        assert tab['truth_quota']['n_episodes'] == 0 and 'not applicable' in tab['truth_quota']['note']
    assert np.isfinite(tab['bocpd_posterior']['acc'])


# ----------------------------------------------------------------------------- R10 readout quality

def test_r10_candidate_mass_and_format_following(tmp_path):
    epd = make_eps(tmp_path, HO6)
    a = write_run(tmp_path, 'A', epd, noise_logits,
                  extra=lambda ep, tr, s, tw: dict(candidate_mass=0.3 if tr['t'] <= 20 else 0.9,
                                                   greedy='The' if tr['t'] <= 8 else ('cat' if s == 'pet' else None)))
    c = write_run(tmp_path, 'C', epd, sum_all_logits, mode='sum_all')
    res = am.analyze([a, c], n_boot=NB)
    q = [x for x in res['readout_quality'] if x['arm'] == 'A|full|text' and x['slot'] == 'pet'][0]
    assert q['mass_mean'] == pytest.approx(0.25 * 0.3 + 0.75 * 0.9) and q['frac_mass_lt_05'] == pytest.approx(0.25)
    assert q['greedy_follow_rate'] == pytest.approx(72 / 80) and q['format'] == 'text'
    assert 'Readout quality' in am.to_markdown(res)


# ----------------------------------------------------------------------------- R08 mechanism

def test_kernel_recovers_delta_rule_exponential(episodes):
    per = mech.synthetic_paired(_dev(episodes), 'delta', alpha=0.3, beta=8.0)
    k = mech.kernel_cell({b: [v] for b, v in per.items()}, K=15, n_boot=20)
    assert k['exp_fit']['lam'] == pytest.approx(0.7, abs=0.03)
    w = np.array(k['kernel'])
    assert np.all(np.diff(w[:8]) < 0)
    assert abs(k['coef']['copy_prev_truth']['w']) < 0.05 and abs(k['coef']['label_prior']['w']) < 0.05


def test_kernel_copy_agent_loads_on_label_only(episodes):
    per = mech.synthetic_paired(_dev(episodes), 'copy', w_copy=2.5, noise=0.3, seed=1)
    k = mech.kernel_cell({b: [v] for b, v in per.items()}, K=15, n_boot=20)
    assert k['coef']['copy_prev_truth']['w'] == pytest.approx(2.5, abs=0.15)
    assert np.max(np.abs(k['kernel'])) < 0.15
    lo, hi = k['coef']['copy_prev_truth']['ci95']
    assert lo > 2.0


def test_kappa_from_counts_known_values():
    po, pe, k = mech.kappa_from_counts([40, 10, 10, 40])
    assert po == pytest.approx(0.8) and pe == pytest.approx(0.5) and k == pytest.approx(0.6)


def test_kappa_is_chance_corrected():
    rng = np.random.default_rng(4)
    m = (rng.random(20000) < 0.9).astype(float)
    s = (rng.random(20000) < 0.9).astype(float)
    po, pe, k = mech.kappa_from_counts(mech.counts(m, s))
    assert po > 0.8 and abs(k) < 0.03


def test_heuristic_identifies_copy_and_delta_agents(episodes):
    dev = _dev(episodes)
    per = mech.synthetic_paired(dev, 'copy', w_copy=3.0)
    h = mech.heuristic_cell({b: [v] for b, v in per.items()}, n_boot=50)
    assert h['strategies']['copy_prev_truth']['kappa'] == pytest.approx(1.0)
    assert h['best_matching'] == {'copy_prev_truth': len(dev)}
    assert abs(h['strategies']['oracle']['kappa']) < 0.15
    per = mech.synthetic_paired(dev, 'delta', alpha=0.9, beta=8.0)
    h = mech.heuristic_cell({b: [v] for b, v in per.items()}, n_boot=50)
    assert h['strategies']['follow_prev_correct']['kappa'] > 0.95


def test_strategy_predictions_definitions(episodes):
    ep = next(iter(_dev(episodes).values()))
    sp = mech.strategy_predictions(ep)
    tr = ep['trials']
    for t in range(2, 81):
        assert sp['copy_prev_truth'][t - 1] == float(tr[t - 1]['a_report_bit'] == tr[t - 2]['truth_bit'])
        assert sp['follow_prev_correct'][t - 1] == tr[t - 2]['a_correct']
    assert all(np.isnan(sp[k][0]) for k in mech.STRATEGIES if k != 'oracle')


@pytest.mark.parametrize('src', [V3, ROB])
def test_r08_new_synthetic_agents_are_distinguished(src):
    eps = {k: v for k, v in am.load_episodes(src).items() if not k.endswith('t')}
    exp = dict(window='last5', count_all='count_all', label_prior='label_prior', copy='copy_prev_truth')
    for agent, strat in exp.items():
        per = mech.synthetic_paired(eps, agent)
        h = mech.heuristic_cell({b: [v] for b, v in per.items()}, n_boot=20)
        assert h['strategies'][strat]['kappa'] == pytest.approx(1.0), agent
    # label prior shows up as the label coefficient of the kernel, not as history weight
    k = mech.kernel_cell({b: [v] for b, v in mech.synthetic_paired(eps, 'label_prior').items()}, n_boot=10)
    assert k['coef']['label_prior']['w'] == pytest.approx(3.0, abs=0.05) and np.max(np.abs(k['kernel'])) < 0.05
    # window agent: flat weight on lags 1..5, ~0 beyond
    k = mech.kernel_cell({b: [v] for b, v in mech.synthetic_paired(eps, 'window', K=5).items()}, n_boot=10)
    w = np.array(k['kernel'])
    assert w[:5].min() > 3 * np.abs(w[6:]).max()
    # mixture: intermediate kappa with copy, and both copy weight and history weight in the kernel
    per = mech.synthetic_paired(eps, 'mixture')
    h = mech.heuristic_cell({b: [v] for b, v in per.items()}, n_boot=20)
    assert 0.2 < h['strategies']['copy_prev_truth']['kappa'] < 0.8
    k = mech.kernel_cell({b: [v] for b, v in per.items()}, n_boot=10)
    assert k['coef']['copy_prev_truth']['w'] > 0.5 and k['kernel'][0] > 0.2


def test_r08_strategy_matrix_on_common_mask(episodes):
    per = mech.synthetic_paired(_dev(episodes), 'delta', alpha=0.3)
    h = mech.heuristic_cell({b: [v] for b, v in per.items()}, n_boot=20)
    M = h['strategy_matrix']
    for a in mech.STRATEGIES:
        assert M[a][a]['kappa'] == pytest.approx(1.0) or not np.isfinite(M[a][a]['kappa'])
        for b in mech.STRATEGIES:
            assert M[a][b]['agreement'] == pytest.approx(M[b][a]['agreement'])
    assert 0 < h['n_common_mask'] < h['n_trials']
    assert all('kappa_common' in v for v in h['strategies'].values())


def test_r08_mechanism_end_to_end_wording_and_robust_cells(tmp_path):
    epd = make_eps(tmp_path, HO6)
    rob = make_eps(tmp_path, ['rob000', 'rob001'], src=ROB, name='rob')
    a = write_run(tmp_path, 'A', epd, noise_logits)
    r = write_run(tmp_path, 'Arob', rob, noise_logits)
    res = mech.analyze([a, r], n_boot=10)
    keys = list(res['kernels'])
    assert len(keys) == 2 and any(k.split('/')[1].startswith('rob#') for k in keys)
    md = mech.to_markdown(res)
    assert 'consistent with' in md and 'Effective weight' in md


# ----------------------------------------------------------------------------- I06 controls

def _write_controls(tmp_path, rows, epdir=None, name='ctrl'):
    d = tmp_path / name
    d.mkdir()
    if epdir is not None:
        (d / 'run_metadata.json').write_text(json.dumps(dict(args=dict(data_dir=str(epdir)))))
    with open(d / 'controls.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return d


def test_i06_chart_and_composite_item_keys(tmp_path, episodes):
    epd = make_eps(tmp_path, ['dev000', 'dev001'])
    eps = am.load_episodes(epd / 'episodes.jsonl')
    rows = []
    for eid in ('dev000', 'dev000t', 'dev001', 'dev001t'):
        ep = eps[eid]
        for t in (20, 40):
            for k in (3, t // 2, t - 1):     # 3 asked trials per query; asked trial 3 is wrong for the model
                ld = -1.0 if k == 3 else 2.0
                rows.append(dict(key=f'chart_read|{eid}|{t}|{k}', control='chart_read', episode_id=eid, t=t,
                                 trial_asked=k, y=float(am.sigmoid(ld)), logit_diff=ld, target=ep['a_name']))
            tr = ep['trials'][t - 1]
            reps = [ep['a_name'], ep['b_name']] if tr['a_listed_first'] else [ep['b_name'], ep['a_name']]
            for name in reps:
                first = name == reps[0]
                ld = 3.0 if first else -0.5   # the second panel is misread
                rows.append(dict(key=f'composite_read|{eid}|{t}|{name}', control='composite_read', episode_id=eid,
                                 t=t, asked_first_panel=first, y=float(am.sigmoid(ld)), logit_diff=ld, target='cat'))
    d = _write_controls(tmp_path, rows, epd)
    res = ac.summarize_run(d)
    tab = {(t['control'], t['stratum']): t for t in res['table']}
    assert tab[('chart_read', 'all')]['n_items'] == 2 * 2 * 3          # 3 items per query kept
    assert tab[('chart_read', 'all')]['hard_acc'] == pytest.approx(2 / 3)
    assert tab[('composite_read', 'panel1')]['n_items'] == 4 and tab[('composite_read', 'panel1')]['hard_acc'] == 1.0
    assert tab[('composite_read', 'panel2')]['n_items'] == 4 and tab[('composite_read', 'panel2')]['hard_acc'] == 0.0
    assert res['gates']['chart_read']['status'].startswith('fail')
    assert not res['diagnostics']


def test_i06_missing_fields_give_diagnostics_not_keyerror(tmp_path):
    # review: chart_read / composite_read rows without domain/format raised KeyError: 'domain'
    rows = [dict(control='chart_read', episode_id='dev000', t=20, y=0.9),            # no trial_asked
            dict(control='explicit_rel', episode_id='dev000', t=20, y=0.9, format='text'),   # no domain
            dict(control='composite_read', episode_id='dev000', t=20, asked_first_panel=True, y=0.9)]
    d = _write_controls(tmp_path, rows)
    res = ac.summarize_run(d)
    msg = ' '.join(res['diagnostics'])
    assert "'trial_asked'" in msg and "'domain'" in msg and 'composite_read' in msg
    with pytest.raises(SystemExit):
        ac.main([str(d), '--strict'])


def test_i06_gate_uses_twin_processed_hard_accuracy(tmp_path):
    # mean P = 0.7 but every twin-averaged item correct -> pass; and the converse
    rows = []
    for i in range(10):
        for tw in ('', 't'):
            rows.append(dict(control='chart_read', episode_id=f'dev00{i}{tw}', t=20, trial_asked=3,
                             y=0.7, logit_diff=float(am.logit(0.7))))
            rows.append(dict(control='explicit_rel', episode_id=f'dev00{i}{tw}', t=20, domain='color', format='text',
                             y=0.95 if i < 7 else 0.2, target='Ken'))
    res = ac.summarize_run(_write_controls(tmp_path, rows))
    assert res['gates']['chart_read']['status'] == 'pass' and res['gates']['chart_read']['hard_acc'] == 1.0
    ex = res['gates']['explicit_rel']
    assert ex['status'] == 'fail' and ex['hard_acc'] == pytest.approx(0.7)


def test_i06_single_trial_who_and_coded_controls(tmp_path):
    rows = []
    for i in range(4):
        for tw in ('', 't'):
            for c in ('single_trial_who', 'easy_short_coded', 'easy_long_coded'):
                ld = 2.0 if (c != 'single_trial_who' or i < 3) else -2.0
                rows.append(dict(control=c, episode_id=f'dev00{i}{tw}', t=40, domain='weather', format='text',
                                 y=float(am.sigmoid(ld)), logit_a=ld + 1, logit_other=1.0, target='Ken'))
    rows.append(dict(control='single_trial_who', episode_id='dev009', t=40, domain='weather', format='text', y=0.9,
                     target='Ken'))
    res = ac.summarize_run(_write_controls(tmp_path, rows))
    tab = {(t['control'], t['stratum']): t for t in res['table']}
    s = tab[('single_trial_who', 'weather/text')]
    assert s['hard_acc'] == pytest.approx(0.75) and s['n_items'] == 4 and s['untwinned_items'] == 1
    assert s['approx_items'] == 0
    assert tab[('easy_short_coded', 'weather/text')]['hard_acc'] == 1.0
    assert tab[('easy_long_coded', 'weather/text')]['n_items'] == 4
    bad = _write_controls(tmp_path, [dict(control='single_trial_who', episode_id='dev000', t=40, domain='weather',
                                          format='text', y=0.9)], name='bad')
    assert "'target'" in ' '.join(ac.summarize_run(bad)['diagnostics'])


def test_a04_legacy_literal_domain_slot_maps_to_scenario():
    ep_v1 = am.load_episodes(V1)['dev000']
    ep_v3 = am.load_episodes(V3)['dev000']
    assert am.slot_of(dict(domain='color', domain_slot='color'), ep_v1) == 'scenario'     # legacy pilot rows
    assert am.slot_of(dict(domain='color'), ep_v1) == 'scenario'
    assert am.slot_of(dict(domain='pet', domain_slot='pet'), ep_v3) == 'pet'
    assert am.slot_of(dict(domain=ep_v3['scenario'], domain_slot='scenario'), ep_v3) == 'scenario'
    other = 'news' if ep_v3['scenario'] != 'news' else 'stock'
    assert am.slot_of(dict(domain=other, domain_slot=other), ep_v3) == other               # E1-style literal
