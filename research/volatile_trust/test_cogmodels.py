"""Tests for cogmodels.py. Run with: python -m pytest -q test_cogmodels.py"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cogmodels as cm  # noqa: E402


def _ep(model, theta, seed=0, order="SV", noise=0.0, g=None):
    rng = np.random.default_rng(seed)
    s = cm.make_schedule(order, 0.8, rng=rng)
    y = cm.simulate(model, theta, s, noise, rng, grid_value=g)
    return cm.Episode.from_dict({**s, "y": y.tolist(), "episode_id": f"e{seed}"})


@pytest.mark.parametrize("order", ["SV", "VS"])
def test_schedule_follows_design(order):
    rng = np.random.default_rng(3)
    for stable_p in (0.8, 0.2):
        s = cm.make_schedule(order, stable_p, rng=rng)
        a, pa, ph = np.array(s["a_correct"]), np.array(s["p_a"]), np.array(s["phase"])
        assert len(a) == 80
        for w in range(8):
            sl = slice(10 * w, 10 * w + 10)
            assert a[sl].sum() == round(10 * pa[sl][0])
            assert len(set(ph[sl])) == 1
        st = pa[ph == "stable"]
        vol = pa[ph == "volatile"].reshape(4, 10)[:, 0]
        assert np.allclose(st, stable_p)
        assert np.all(np.diff(vol) != 0) and set(vol) == {0.2, 0.8}
        if order == "SV":
            assert ph[0] == "stable" and vol[0] == pytest.approx(1 - stable_p)
        else:
            assert ph[0] == "volatile"


def test_delta_rule_manual():
    o = np.array([1, 0, 1, 1.0])
    q = cm.delta_rule(o, np.full(4, 0.5))
    assert np.allclose(q, [0.5, 0.75, 0.375, 0.6875])


def test_m2_m3_reduce_to_m1():
    ep = _ep("M1_delta", [0.3, 8, 0.2])
    p1 = cm.MODELS["M1_delta"].predict(np.array([0.3, 8, 0.2]), ep)
    p2 = cm.MODELS["M2_phase"].predict(np.array([0.3, 0.3, 8, 0.2]), ep)
    p3 = cm.MODELS["M3_confirm"].predict(np.array([0.3, 0.3, 8, 0.2]), ep)
    assert np.allclose(p1, p2) and np.allclose(p1, p3)


def test_m3_confirm_semantics():
    # q starts at 0.5 (tie: mean alpha), then q > 0.5. A wrong (o=0) is disconfirming.
    q = cm.confirm_rule(np.array([1, 0, 0.0]), a_conf=0.0, a_disc=1.0)
    assert q[1] == pytest.approx(0.75)  # tie -> alpha 0.5
    assert q[2] == pytest.approx(0.0)   # disconfirming, alpha 1
    q = cm.confirm_rule(np.array([1, 1, 0.0]), a_conf=0.0, a_disc=0.2)
    assert q[1] == pytest.approx(0.55)  # tie -> alpha 0.1
    assert q[2] == pytest.approx(0.55)  # confirming outcome, alpha 0 -> no change


def test_count_and_window():
    o = np.array([1, 1, 0, 1, 0.0])
    assert np.allclose(cm.count_belief(o), [1 / 2, 2 / 3, 3 / 4, 3 / 5, 4 / 6])
    assert np.allclose(cm.count_belief(o, 2), [1 / 2, 2 / 3, 3 / 4, 2 / 4, 2 / 4])


def test_changepoint_tracks_reversal():
    o = np.array([1] * 30 + [0] * 30, float)
    q = cm.changepoint_belief(o, 0.05)
    assert q[29] > 0.85 and q[-1] < 0.15
    qn = cm.changepoint_belief(o, cm.H_GRID)
    assert np.all((qn > 0) & (qn < 1)) and qn[29] > 0.8 and qn[-1] < 0.2


def test_volatile_learner_behrens_signature():
    """The volatile learner's implied LR, averaged over schedules, is higher in the volatile phase."""
    rng = np.random.default_rng(0)
    diffs = []
    for i in range(8):
        s = cm.make_schedule(["SV", "VS"][i % 2], 0.8, rng=rng)
        ep = cm.Episode.from_dict({**s, "y": None})
        r = cm.volatile_belief(ep.o)
        lr = cm.implied_lr_by_phase(r["q"], ep.o, ep.vol)
        diffs.append(lr["volatile"] - lr["stable"])
    assert np.mean(diffs) > 0


@pytest.mark.parametrize("model,theta", [
    ("M1_delta", [0.3, 10, 0.3]),
    ("M2_phase", [0.1, 0.6, 12, -0.2]),
    ("M3_confirm", [0.1, 0.5, 12, 0.0]),
    ("WSLS", [2.0, 0.5]),
])
def test_noiseless_fit_recovers(model, theta):
    ep = _ep(model, theta, seed=1)
    r = cm.fit_model(model, ep, n_restarts=6)
    assert r["excess_nll"] < 0.05
    fitted = [r["params"][p] for p in cm.MODELS[model].params]
    assert np.allclose(fitted, theta, rtol=0.1, atol=0.05)


def test_fit_outputs_and_ic():
    ep = _ep("M2_phase", [0.1, 0.6, 12, 0.0], noise=0.3)
    out = cm.fit_episode(ep)
    for m, f in out["fits"].items():
        assert f["k"] == cm.MODELS[m].k and f["n"] == 80
        assert f["aic_heuristic"] == pytest.approx(2 * f["k"] + 2 * f["nll"])
        assert f["bic_heuristic"] == pytest.approx(f["k"] * np.log(80) + 2 * f["nll"])
        assert f["excess_nll"] >= -1e-6
    assert out["fits"]["M2_phase"]["nll"] <= out["fits"]["M1_delta"]["nll"] + 1e-4
    assert out["fits"]["M2_phase"]["bic_heuristic"] < out["fits"]["M0_bias"]["bic_heuristic"]
    assert out["episode_id"] == "e0"


def test_pooled_fit():
    eps = [_ep("M1_delta", [0.3, 10, 0.0], seed=s, order=o) for s, o in [(0, "SV"), (1, "VS")]]
    r = cm.fit_model("M1_delta", eps)
    assert r["n"] == 160 and r["params"]["alpha"] == pytest.approx(0.3, abs=0.03)


def test_effective_lr_detects_phase_difference():
    d = []
    for s in range(6):
        ep = _ep("M2_phase", [0.05, 0.6, 10, 0.0], seed=s, order=["SV", "VS"][s % 2])
        d.append(cm.effective_learning_rate(ep.y, ep.o, ep.vol)["diff_signed"])
    assert np.mean(d) > 0
    ep = _ep("M2_phase", [0.05, 0.6, 10, 0.0])
    r = cm.effective_learning_rate(ep.y, ep.o, ["stable"] * 40 + ["volatile"] * 40)
    assert r["stable"]["n"] == 40 and r["volatile"]["n"] == 39


def test_bms_pxp():
    rng = np.random.default_rng(0)
    L = rng.normal(0, 1, (20, 3))
    L[:, 1] += 6
    r = cm.bms_random_effects(L)
    assert np.argmax(r["pxp"]) == 1 and r["pxp"][1] > 0.9 and r["bor"] < 0.05
    r0 = cm.bms_random_effects(np.zeros((20, 3)))
    assert r0["bor"] > 0.5 and np.allclose(r0["pxp"], 1 / 3, atol=0.15)


def test_cli_roundtrip(tmp_path):
    inp, outp = tmp_path / "res.jsonl", tmp_path / "fits.json"
    with open(inp, "w") as fh:
        for s in range(3):
            ep = _ep("M1_delta", [0.3, 10, 0.0], seed=s, noise=0.2)
            sched = cm.make_schedule("SV", 0.8, rng=np.random.default_rng(s))
            d = {**sched, "y": ep.y.tolist(), "episode_id": f"e{s}", "model": "toyVLM",
                 "domain": "color", "format": "text"}
            fh.write(json.dumps(d) + "\n")
    cm.main(["fit", "--input", str(inp), "--output", str(outp), "--jobs", "1", "--restarts", "2",
             "--models", "M0_bias", "M1_delta", "M2_phase", "M4_cp_norm"])
    out = json.loads(outp.read_text())
    assert len(out["episodes"]) == 3 and out["episodes"][0]["domain"] == "color"
    g = "model=toyVLM|domain=color|format=text"
    assert g in out["comparison_heuristic"] and "pxp" in out["comparison_heuristic"][g]["bms_heuristic"]
    assert out["comparison_heuristic"][g]["best_by_sum_bic_heuristic"] in ("M1_delta", "M2_phase")
    assert out["pooled"][g]["M1_delta"]["n"] == 240
    cv = out["cv"][g]
    assert cv["M1_delta"]["n_folds"] == 3 and cv["M1_delta"]["n_trials"] == 240
    assert cv["M1_delta"]["heldout_nll"] < cv["M0_bias"]["heldout_nll"]
    assert cv["M4_cp_norm"]["heldout_nll"] == pytest.approx(
        sum(e["fits"]["M4_cp_norm"]["nll"] for e in out["episodes"]))
    assert np.isfinite(out["eff_lr_calibrated_summary"][g]["diff_signed"]["mean"])


def test_parameter_recovery_small():
    rep = cm.parameter_recovery(n_per_model=12, noise_sd=0.3, seed=1)
    assert rep["param_recovery"]["M1_delta"]["alpha"]["pearson"] > 0.7
    assert rep["param_recovery"]["M2_phase"]["alpha_volatile"]["pearson"] > 0.6


def test_raw_elr_is_confounded_and_calibrated_is_null():
    """M1 data: raw diff is biased negative; calibrated diff is about 0. M2 (alpha_v > alpha_s): positive."""
    raw, cal, cal2 = [], [], []
    for s in range(8):
        order = ["SV", "VS"][s % 2]
        ep = _ep("M1_delta", [0.4, 10, 0.0], seed=s, order=order, noise=0.3)
        r = cm.calibrated_effective_learning_rate(ep, n_restarts=3, n_surrogates=100, seed=s)
        raw.append(r["raw"]["diff_signed"])
        cal.append(r["calibrated"]["diff_signed"])
        ep2 = _ep("M2_phase", [0.15, 0.6, 10, 0.0], seed=s, order=order, noise=0.3)
        cal2.append(cm.calibrated_effective_learning_rate(ep2, n_restarts=3, n_surrogates=100,
                                                          seed=s)["calibrated"]["diff_signed"])
    assert np.mean(raw) < -0.2
    assert abs(np.mean(cal)) < 0.1
    assert np.mean(cal2) > 0.3 and np.mean(cal2) > np.mean(cal) + 0.3


def test_calibrated_noiseless_m1_exact_zero():
    ep = _ep("M1_delta", [0.3, 10, 0.2], seed=2)
    r = cm.calibrated_effective_learning_rate(ep, n_surrogates=0)
    assert abs(r["calibrated"]["diff_signed"]) < 0.02


def test_loeo_cv_prefers_generating_model():
    eps = [_ep("M2_phase", [0.05, 0.7, 12, 0.0], seed=s, order=["SV", "VS"][s % 2], noise=0.3)
           for s in range(6)]
    r1 = cm.loeo_cv("M1_delta", eps)
    r2 = cm.loeo_cv("M2_phase", eps)
    assert r1["n_folds"] == 6 and r2["heldout_nll"] < r1["heldout_nll"]


EPISODE_KEYS = ("data", "split", "episode_id")


def _repeated_rows():
    # The two views of an episode have identical targets. Leaving just one view
    # out gives the bias fit access to its held-out episode's target distribution.
    return [dict(data="toy", split="held_out", episode_id=f"e{i}", domain=slot,
                 model="toyVLM", format="raw", a_correct=[1, 0, 1, 0],
                 phase=["stable", "stable", "volatile", "volatile"], y=[p] * 4)
            for i, p in enumerate((0.1, 0.9)) for slot in ("pet", "scenario")]


def test_grouped_cv_excludes_other_view_and_scores_both():
    rows = _repeated_rows()
    grouped = cm.loeo_cv("M0_bias", rows, n_restarts=1, episode_group_by=EPISODE_KEYS)
    # Each fold must train only on the opposite episode: predict .9 for .1 and
    # .1 for .9. This analytic loss fails if either held-out view leaks into train.
    expected = -(0.1 * np.log(0.9) + 0.9 * np.log(0.1))
    assert grouped["heldout_nll_per_trial"] == pytest.approx(expected, abs=1e-5)
    assert grouped["n_folds"] == grouped["n_independent_episodes"] == 2
    assert grouped["n_rows"] == 4 and grouped["n_trials"] == 16
    assert [p["n_rows"] for p in grouped["per_episode"]] == [2, 2]
    assert [p["n"] for p in grouped["per_episode"]] == [8, 8]
    assert grouped["episode_group_by"] == list(EPISODE_KEYS)
    assert grouped["per_episode"][0]["episode_group"]["episode_id"] == "e0"

    # Omission preserves the generic API's former row-level behavior.
    legacy = cm.loeo_cv("M0_bias", rows, n_restarts=1)
    assert legacy["n_folds"] == 4
    assert legacy["heldout_nll_per_trial"] < grouped["heldout_nll_per_trial"]

    reordered = cm.loeo_cv("M0_bias", [rows[i] for i in (3, 0, 2, 1)], n_restarts=1,
                            episode_group_by=EPISODE_KEYS)
    assert reordered["heldout_nll"] == pytest.approx(grouped["heldout_nll"])
    one = cm.loeo_cv("M0_bias", rows, n_restarts=1, folds=[1], episode_group_by=EPISODE_KEYS)
    assert one["n_folds"] == 1 and one["n_rows"] == 2 and one["n_trials"] == 8
    assert one["per_episode"][0]["episode_id"] == "e1"


def test_grouped_cv_keeps_dataset_and_split_in_identity():
    rows = _repeated_rows()[:2]
    rows += [dict(rows[0], data="other"), dict(rows[0], split="development")]
    result = cm.loeo_cv("M0_bias", rows, n_restarts=1, episode_group_by=EPISODE_KEYS)
    assert result["n_folds"] == 3
    assert [p["n_rows"] for p in result["per_episode"]] == [2, 1, 1]
    with pytest.raises(ValueError, match="at least two independent episodes"):
        cm.loeo_cv("M0_bias", rows[:2], episode_group_by=EPISODE_KEYS)


def _elr_row(eid, value, **meta):
    return dict(data="toy", split="held_out", episode_id=eid,
                eff_lr_calibrated={"calibrated": {"diff_signed": value, "diff_pe": value}}, **meta)


def test_elr_groups_views_before_uncertainty_and_is_duplicate_invariant():
    rows = [_elr_row("e0", 1.0), _elr_row("e0", 3.0), _elr_row("e1", 10.0)]
    grouped = cm.summarize_calibrated_elr(rows, EPISODE_KEYS)
    for k, r in grouped.items():
        assert r["n"] == 2 and r["n_rows"] == r["n_valid_rows"] == 3
        assert r["mean"] == pytest.approx(6.0)  # equal weight for means 2 and 10
        assert r["sd"] == pytest.approx(np.std([2, 10], ddof=1))
        assert r["t"] == pytest.approx(1.5)
        doubled = cm.summarize_calibrated_elr(rows + rows, EPISODE_KEYS)[k]
        for statistic in ("n", "mean", "sd", "t"):
            assert doubled[statistic] == pytest.approx(r[statistic])
    legacy = cm.summarize_calibrated_elr(rows)["diff_signed"]
    assert legacy["n"] == 3 and legacy["mean"] == pytest.approx(14 / 3)


def test_elr_missing_values_and_dataset_identity():
    rows = [_elr_row("e0", 2), _elr_row("e0", float("nan")), _elr_row("e1", float("nan"))]
    rows += [dict(rows[0], data="other", eff_lr_calibrated={"calibrated": {"diff_signed": 6, "diff_pe": 6}})]
    r = cm.summarize_calibrated_elr(rows, EPISODE_KEYS)["diff_signed"]
    assert r["n"] == 2 and r["n_rows"] == 4 and r["n_valid_rows"] == 2
    assert r["mean"] == pytest.approx(4)


def test_fit_file_validates_group_metadata_before_fitting(tmp_path, monkeypatch):
    inp = tmp_path / "missing_id.jsonl"
    inp.write_text(json.dumps({"data": "toy", "split": "held_out"}) + "\n")

    def should_not_fit(*args):
        pytest.fail("expensive fitting started before validating episode identity")

    monkeypatch.setattr(cm, "_fit_task", should_not_fit)
    with pytest.raises(ValueError, match="missing grouping metadata.*episode_id"):
        cm.fit_file(str(inp), episode_group_by=EPISODE_KEYS)


def test_fit_file_propagates_episode_grouping(tmp_path):
    inp = tmp_path / "grouped.jsonl"
    rows = []
    for i in range(2):
        sched = cm.make_schedule(rng=np.random.default_rng(i))
        for slot, beta in (("pet", 8), ("scenario", 12)):
            y = cm.simulate("M1_delta", [0.3, beta, 0.0], sched)
            rows.append(dict(sched, y=y.tolist(), model="toyVLM", format="raw", data="toy",
                             split="held_out", episode_id=f"e{i}", domain=slot))
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    outp = tmp_path / "fits.json"
    cm.main(["fit", "--input", str(inp), "--output", str(outp), "--models", "M0_bias", "M1_delta",
             "--group-by", "model", "format", "--episode-group-by", *EPISODE_KEYS,
             "--restarts", "1", "--cv-restarts", "1", "--jobs", "1", "--no-pooled"])
    out = json.loads(outp.read_text())
    g = "model=toyVLM|format=raw"
    assert out["episode_group_by"] == list(EPISODE_KEYS)
    assert len(out["episodes"]) == 4  # preserve the separate cover-story fits
    assert out["cv"][g]["M0_bias"]["n_folds"] == 2
    assert out["cv"][g]["M0_bias"]["n_rows"] == 4
    assert out["eff_lr_calibrated_summary"][g]["diff_signed"]["n"] == 2
    assert out["comparison_heuristic"][g]["n_rows"] == 4
    assert out["comparison_heuristic"][g]["n_independent_episodes"] == 2
    assert "descriptive only" in out["comparison_heuristic"][g]["unit"]


def test_volatile_adapter_requests_episode_grouping(tmp_path, monkeypatch):
    import analyze_cogmodels as ac

    def fake_fit_file(path, **kwargs):
        assert kwargs["episode_group_by"] == EPISODE_KEYS
        assert len(open(path).readlines()) == 4
        return {"cv": {}}

    monkeypatch.setattr(ac, "export", lambda *args: [dict(r, p_a=[0.8] * 4) for r in _repeated_rows()])
    monkeypatch.setattr(cm, "fit_file", fake_fit_file)
    monkeypatch.setattr(sys, "argv", ["analyze_cogmodels.py", "unused", "--out", str(tmp_path / "fits.json")])
    ac.main()
