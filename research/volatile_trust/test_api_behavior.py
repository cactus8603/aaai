"""Regression tests for genuine-reversal windows and distinct behavioral scores."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_api_behavior as api


def test_reversal_trial_is_zero_and_initial_block_is_not_a_reversal():
    ep = dict(reversals=[1, 6, 16], trials=[{}] * 20)
    offsets = api.since_reversal(ep)
    assert offsets[:5] == [None] * 5
    assert offsets[5:15] == list(range(10))
    assert offsets[15:] == list(range(5))
    assert api.since_reversal(dict(reversals=[1], trials=[{}] * 10)) == [None] * 10


def test_reversal_windows_are_disjoint_and_accuracy_is_not_follow_better():
    # The scheduled better advisor can be wrong: following it is follow-better=1,
    # accuracy=0. Also include excluded initial, stable and beyond-window records.
    rows = [dict(base='e0', phase='volatile', sr=k, correct=0., follow_better=1.) for k in range(10)]
    rows += [dict(base='e0', phase=phase, sr=k, correct=1., follow_better=0.)
             for phase, k in [('volatile', None), ('volatile', 10), ('stable', 0)]]
    result = api.reversal_summary(rows)
    assert [v['n_items'] for v in result.values()] == [3, 4, 3]
    for v in result.values():
        assert v['accuracy']['estimate_ci95'] == [0., 0., 0.]
        assert v['follow_better']['estimate_ci95'] == [1., 1., 1.]


def test_reversal_summary_weights_pairs_equally_and_excludes_undefined_better():
    rows = [dict(base='e0', phase='volatile', sr=0, correct=1., follow_better=None)]
    rows += [dict(base='e1', phase='volatile', sr=k, correct=0., follow_better=1.) for k in range(3)]
    ent = api.reversal_summary(rows)['1_3']
    assert ent['accuracy']['estimate_ci95'][0] == pytest.approx(.5)
    assert ent['accuracy']['n_items'] == 4
    assert ent['follow_better']['n_items'] == 3
    assert ent['follow_better']['n_pairs'] == 1
    assert api.reversal_summary([])['1_3']['accuracy']['estimate_ci95'] is None


def test_analysis_scores_actual_outcome_separately_from_scheduled_reliability(monkeypatch):
    # Following the higher-probability advisor is not guaranteed to be correct.
    # Mock only the unrelated regression optimizer to keep this integration test small.
    monkeypatch.setattr(api, 'logit_fit', lambda X, y: np.zeros(X.shape[1]))
    ep = dict(reversals=[1, 10], trials=[dict(a_correct=0, p_a=.8, phase='stable')] * 8
              + [dict(a_correct=0, p_a=.8, phase='volatile')] * 4)
    eps = {'e0': ep, 'e0t': ep}
    rules = {eid: {k: np.ones(12) for k in api.RULES} for eid in eps}
    offsets = {eid: api.since_reversal(e) for eid, e in eps.items()}
    out = api.analyse({(eid, t): 1. for eid in eps for t in (4, 12)}, eps, rules, offsets)
    assert out['acc'][0] == 0.
    window = out['volatile_reversal_windows']['1_3']
    assert window['accuracy']['estimate_ci95'][0] == 0.
    assert window['follow_better']['estimate_ci95'][0] == 1.
