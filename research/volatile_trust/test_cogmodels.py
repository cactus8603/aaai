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
        assert f["aic"] == pytest.approx(2 * f["k"] + 2 * f["nll"])
        assert f["bic"] == pytest.approx(f["k"] * np.log(80) + 2 * f["nll"])
        assert f["excess_nll"] >= -1e-6
    assert out["fits"]["M2_phase"]["nll"] <= out["fits"]["M1_delta"]["nll"] + 1e-4
    assert out["fits"]["M2_phase"]["bic"] < out["fits"]["M0_bias"]["bic"]
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
    assert g in out["comparison"] and "pxp" in out["comparison"][g]["bms"]
    assert out["comparison"][g]["best_by_sum_bic"] in ("M1_delta", "M2_phase")
    assert out["pooled"][g]["M1_delta"]["n"] == 240


def test_parameter_recovery_small():
    rep = cm.parameter_recovery(n_per_model=12, noise_sd=0.3, seed=1)
    assert rep["param_recovery"]["M1_delta"]["alpha"]["pearson"] > 0.7
    assert rep["param_recovery"]["M2_phase"]["alpha_volatile"]["pearson"] > 0.6
