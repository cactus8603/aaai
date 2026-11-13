"""Readout-order regression tests using the main analysis synthetic-run fixtures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_main as am
import analyze_readout_sensitivity as rs
from test_analysis import make_eps, write_run


def test_all_trials_exposes_twin_averaging_effect_without_mass_selection(tmp_path):
    epdir = make_eps(tmp_path, ['hel000', 'hel001'])
    # Both formats have perfect task logits. Raw's opposite position biases make
    # one copy wrong, while cancelling before a decision produces perfect scores.
    def correct_logits(ep, slot):
        return np.array([1. if tr['a_correct'] else -1. for tr in ep['trials']])
    low_mass = lambda ep, tr, slot, twin: dict(candidate_mass=.1)
    runs = [write_run(tmp_path, 'raw', epdir, correct_logits, bias=2., extra=low_mass),
            write_run(tmp_path, 'coded', epdir, correct_logits, hf='coded_text', bias=0.),
            write_run(tmp_path, 'summary', epdir, correct_logits, mode='sum_all', bias=0.)]
    out = rs.analyze_runs(runs, n_boot=100, expected_pairs=2)
    primary, per_copy = (out['methods'][m] for m in rs.METHODS)
    assert primary['conditions']['raw']['mean'] == 1.
    assert per_copy['conditions']['raw']['mean'] == .5
    assert primary['gaps']['summary_minus_raw']['mean'] == 0.
    assert per_copy['gaps']['summary_minus_raw']['mean'] == .5
    assert out['gap_shift_per_copy_minus_primary']['summary_minus_raw']['mean'] == .5
    assert out['readout_quality']['raw']['frac_mass_lt_05'] == 1.
    assert out['readout_quality']['raw']['n_rows'] == 2 * 2 * 2 * 79
    assert per_copy['conditions']['raw']['n_paired_trials'] == 2 * 2 * 79


def test_exact_ties_get_half_credit_before_or_after_pairing(tmp_path):
    epdir = make_eps(tmp_path, ['hel000', 'hel001'])
    ties = lambda ep, slot: np.zeros(len(ep['trials']))
    runs = [write_run(tmp_path, 'raw', epdir, ties, bias=0.),
            write_run(tmp_path, 'coded', epdir, ties, hf='coded_text', bias=0.),
            write_run(tmp_path, 'summary', epdir, ties, mode='sum_all', bias=0.)]
    out = rs.analyze_runs(runs, n_boot=100, expected_pairs=2)
    for method in out['methods'].values():
        assert all(x['mean'] == .5 for x in method['conditions'].values())
        assert all(x['mean'] == 0 for x in method['gaps'].values())


def test_missing_copy_cannot_be_silently_selected_out(tmp_path):
    epdir = make_eps(tmp_path, ['hel000', 'hel001'])
    one = lambda ep, slot: np.ones(len(ep['trials']))
    runs = [write_run(tmp_path, 'raw', epdir, one, keep=lambda ep, tr, slot, twin: not (tr['t'] == 2 and twin)),
            write_run(tmp_path, 'coded', epdir, one, hf='coded_text'),
            write_run(tmp_path, 'summary', epdir, one, mode='sum_all')]
    with pytest.raises(am.AnalysisError, match='lack a twin'):
        rs.analyze_runs(runs, n_boot=100, expected_pairs=2)
