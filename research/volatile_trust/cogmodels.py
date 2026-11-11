"""Cognitive models of in-context source-trust learning (volatile-trust task).

See DESIGN.md. Each episode has T conflict trials between reporters A and B. After
trial t the truth is revealed, giving the outcome ``o_t = a_correct[t]`` (1 if A was
correct). The VLM's soft choice ``y_t = P(follow A)`` at trial t is made from outcomes
1..t-1 only. Each cognitive model maps outcomes 1..t-1 to a predicted P_t, and is fit by
minimising the soft cross-entropy

    NLL = -sum_t [ y_t log P_t + (1 - y_t) log(1 - P_t) ].

With soft targets, NLL is a pseudo-likelihood (y_t is treated as the expected value of
a Bernoulli choice). AIC = 2k + 2 NLL and BIC = k ln(n) + 2 NLL, with n = number of trials.
These AIC/BIC (and BMS built on them) are HEURISTIC: their output keys carry a
``_heuristic`` suffix. The recommended comparison is ``loeo_cv``, the held-out soft
cross-entropy under leave-one-episode-out pooled fits.
We also report ``excess_nll = NLL - sum_t H(y_t)``, which is the summed KL divergence
and equals 0 for a perfect fit.

Models (``MODELS`` registry):

==================  ======================================================  ========
name                description                                             free k
==================  ======================================================  ========
M0_bias             P = sigmoid(b)                                          1
WSLS                P = sigmoid(b + w * (2 o_{t-1} - 1)); P_1 = sigmoid(b)  2
count_beta          q = (s+1)/(n+2) over all past trials                    2+1
window_k            q = (s+1)/(n+2) over the last k trials (k from a grid)  2+1+1
M1_delta            q <- q + alpha (o - q), q_1 = 0.5                       3
M2_phase            alpha_stable / alpha_volatile by phase of updated trial 4
M3_confirm          alpha_conf / alpha_disc (confirming the trusted source) 4
M4_cp               Bayesian change-point observer, hazard h from a grid    2+1
M4_cp_norm          change-point observer marginalised over h, P = E[p]     0
M4_vol              Behrens-style volatile learner, readout beta, b          2
M4_vol_norm         Behrens-style volatile learner, P = predictive          0
==================  ======================================================  ========

Unless noted, the readout is P = sigmoid(beta * (q - 0.5) + b), with q the belief
that A is correct.

WSLS note: every trial is a conflict trial, so a deterministic win-stay/lose-shift on
the agent's own previous choice is the same as "follow whoever was right last".
The soft version therefore has two A-relative probabilities,
p_A_after_A_correct = sigmoid(b + w) and p_A_after_A_wrong = sigmoid(b - w).
The fit reports these as ``p_stay_after_win`` and ``p_shift_after_loss``, which are
equivalent when b = 0.

M3 convention: the "trusted" reporter at trial t is A if q_t > 0.5 and B if q_t < 0.5.
An outcome is *confirming* if the trusted reporter was correct, i.e. the update moves q
away from 0.5. It is *disconfirming* otherwise, i.e. the update moves q toward or across
0.5. At an exact tie (q_t = 0.5, e.g. trial 1) the mean of the two rates is used.
We do not use an A-relative asymmetry (o=1 vs o=0) because the identity of A is
counterbalanced, so that asymmetry has no psychological meaning.

M2 convention: the update after outcome t uses the alpha of ``phase[t]``.

CLI
---
    python cogmodels.py fit --input results.jsonl --output fits.json [--jobs 8]
    python cogmodels.py recover --output runs/param_recovery.json
"""
from __future__ import annotations

import os

if __name__ == "__main__":  # one BLAS thread per worker process; set before numpy import
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logit, logsumexp

EPS = 1e-6
DATA_KEYS = ("a_correct", "phase", "p_a", "y")

# ----------------------------------------------------------------------------- utils


def _clip(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def soft_nll(y: np.ndarray, p: np.ndarray) -> float:
    """Summed soft cross-entropy between targets y and predictions p."""
    p = _clip(np.asarray(p, float))
    y = np.asarray(y, float)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def entropy_sum(y: np.ndarray) -> float:
    """Summed binary entropy of y (the lower bound of soft_nll)."""
    return soft_nll(y, y)


def _readout(q: np.ndarray, beta: float, b: float) -> np.ndarray:
    return expit(beta * (np.asarray(q) - 0.5) + b)


def _phase_is_volatile(phase: Sequence[str]) -> np.ndarray:
    return np.array([str(s).lower().startswith("v") for s in phase], dtype=bool)


@dataclass
class Episode:
    """Data for one episode. ``o`` = a_correct (0/1), ``vol`` = volatile-phase mask."""

    o: np.ndarray
    vol: np.ndarray
    y: np.ndarray | None = None
    p_a: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        y = None if d.get("y") is None else np.asarray(d["y"], float)
        p_a = None if d.get("p_a") is None else np.asarray(d["p_a"], float)
        meta = {k: v for k, v in d.items() if k not in DATA_KEYS}
        return cls(np.asarray(d["a_correct"], float), _phase_is_volatile(d["phase"]), y, p_a, meta)

    @property
    def T(self) -> int:
        return len(self.o)


# --------------------------------------------------------------- latent belief models


def delta_rule(o: np.ndarray, alphas: np.ndarray, q1: float = 0.5) -> np.ndarray:
    """q_t (belief before trial t) for q_{t+1} = q_t + alpha_t (o_t - q_t).

    ``alphas`` is a per-trial learning rate (length T). This is a plain Python loop over
    floats, which is faster than numpy for T = 80.
    """
    q = q1
    out = [0.0] * len(o)
    for t, (ot, a) in enumerate(zip(o.tolist(), alphas.tolist())):
        out[t] = q
        q += a * (ot - q)
    return np.asarray(out)


def confirm_rule(o: np.ndarray, a_conf: float, a_disc: float, q1: float = 0.5) -> np.ndarray:
    """Delta rule with separate rates for confirming and disconfirming outcomes (M3)."""
    q = q1
    a_tie = 0.5 * (a_conf + a_disc)
    out = [0.0] * len(o)
    for t, ot in enumerate(o.tolist()):
        out[t] = q
        if q == 0.5:
            a = a_tie
        else:
            confirming = (q > 0.5) == (ot > 0.5)
            a = a_conf if confirming else a_disc
        q += a * (ot - q)
    return np.asarray(out)


def count_belief(o: np.ndarray, k: int | None = None) -> np.ndarray:
    """q_t = (s+1)/(n+2) over all past outcomes (k=None) or the last k outcomes."""
    c = np.concatenate([[0.0], np.cumsum(o)])  # c[t] = sum of o[0:t]
    t = np.arange(len(o))
    if k is None:
        s, n = c[t], t
    else:
        lo = np.maximum(t - k, 0)
        s, n = c[t] - c[lo], t - lo
    return (s + 1.0) / (n + 2.0)


P_GRID = np.linspace(0.005, 0.995, 100)
H_GRID = np.geomspace(1e-3, 0.5, 25)


def changepoint_belief(o: np.ndarray, h: float | np.ndarray, p_grid: np.ndarray = P_GRID,
                       h_prior: np.ndarray | None = None) -> np.ndarray:
    """Posterior-predictive E[p_A] before each trial under a change-point model.

    p_A is constant except at change points (prob. h per trial), where it is redrawn
    from Uniform(p_grid). If ``h`` is an array, the observer is Bayesian over h as well
    (h is static and unknown, with prior ``h_prior``, uniform by default). Returns q_t
    for t = 1..T.
    """
    hs = np.atleast_1d(np.asarray(h, float))[:, None]  # (H,1)
    G = len(p_grid)
    prior_h = np.full(len(hs), 1.0 / len(hs)) if h_prior is None else np.asarray(h_prior, float)
    post = prior_h[:, None] * np.full((1, G), 1.0 / G)  # joint (H,G), sums to 1
    out = np.empty(len(o))
    for t, ot in enumerate(o):
        # predict: transition with hazard, keeping each h-slice's mass
        mass = post.sum(1, keepdims=True)
        pred = (1.0 - hs) * post + hs * mass / G
        out[t] = float((pred * p_grid).sum())
        lik = p_grid if ot > 0.5 else 1.0 - p_grid
        post = pred * lik
        post /= post.sum()
    return out


def _gauss_kernel(x: np.ndarray, sd: float) -> np.ndarray:
    """Row-stochastic transition matrix on grid x with Gaussian steps of s.d. ``sd``."""
    K = np.exp(-0.5 * ((x[None, :] - x[:, None]) / sd) ** 2)
    return K / K.sum(1, keepdims=True)


V_GRID = np.linspace(np.log(0.02), np.log(3.0), 20)  # log s.d. of the logit-p step
K_GRID = np.array([0.05, 0.2, 0.5, 1.0])               # s.d. of the step in v (meta-volatility)
_VOL_P = np.linspace(0.01, 0.99, 60)


def volatile_belief(o: np.ndarray, p_grid: np.ndarray = _VOL_P, v_grid: np.ndarray = V_GRID,
                    k_grid: np.ndarray = K_GRID) -> dict[str, np.ndarray]:
    """Behrens et al. (2007)-style hierarchical volatile learner on a grid.

    Generative model: logit p_{t+1} ~ N(logit p_t, exp(v_t)^2), v_{t+1} ~ N(v_t, k^2),
    and k is static and unknown. Priors are uniform on the (p, v, k) grids.
    Returns ``q`` (E[p] before trial t), ``v`` (E[v] before trial t) and
    ``alpha_implied`` = (q_{t+1} - q_t) / (o_t - q_t).
    """
    lp = logit(p_grid)
    Tp = np.stack([_gauss_kernel(lp, np.exp(v)) for v in v_grid])  # (V,P,P) rows: from p
    Tv = np.stack([_gauss_kernel(v_grid, k) for k in k_grid])       # (K,V,V)
    K, V, P = len(k_grid), len(v_grid), len(p_grid)
    post = np.full((K, V, P), 1.0 / (K * V * P))
    q = np.empty(len(o))
    vmean = np.empty(len(o))
    for t, ot in enumerate(o):
        pred = np.einsum("kvp,kvw->kwp", post, Tv)       # v transition
        pred = np.einsum("kwp,wpq->kwq", pred, Tp)       # p transition given new v
        pp = pred.sum((0, 1))
        q[t] = float(pp @ p_grid)
        vmean[t] = float(pred.sum((0, 2)) @ v_grid)
        post = pred * (p_grid if ot > 0.5 else 1.0 - p_grid)
        post /= post.sum()
    return {"q": q, "v": vmean, "alpha_implied": implied_learning_rate(q, o)}


def implied_learning_rate(q: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Per-trial alpha_t = (q_{t+1} - q_t) / (o_t - q_t). The last trial is NaN."""
    q = np.asarray(q, float)
    a = np.full(len(q), np.nan)
    pe = o[:-1] - q[:-1]
    ok = np.abs(pe) > 1e-9
    a[:-1][ok] = (q[1:] - q[:-1])[ok] / pe[ok]
    return a


# ------------------------------------------------------------------- model registry


@dataclass
class CogModel:
    """A cognitive model.

    ``prepare(ep, g)`` returns a context for the grid value ``g``; this is where the
    parameter-free latent quantities are precomputed. ``predict(theta, ctx)`` returns
    P(follow A) for every trial. ``bounds`` gives the continuous parameters.
    """

    name: str
    params: list[str]
    bounds: list[tuple[float, float]]
    prepare: Callable[[Episode, Any], Any]
    predict: Callable[[np.ndarray, Any], np.ndarray]
    grid_name: str | None = None
    grid: Sequence[Any] | None = None
    description: str = ""

    @property
    def k(self) -> int:
        return len(self.params) + (1 if self.grid_name else 0)


BETA_B = [(0.0, 40.0), (-6.0, 6.0)]
UNIT = (0.0, 1.0)


def _ctx_o(ep: Episode, g=None):
    return ep


def _pred_delta(th, ep):
    a, beta, b = th
    return _readout(delta_rule(ep.o, np.full(ep.T, a)), beta, b)


def _pred_phase(th, ep):
    a_s, a_v, beta, b = th
    return _readout(delta_rule(ep.o, np.where(ep.vol, a_v, a_s)), beta, b)


def _pred_confirm(th, ep):
    a_c, a_d, beta, b = th
    return _readout(confirm_rule(ep.o, a_c, a_d), beta, b)


def _pred_wsls(th, ep):
    w, b = th
    prev = np.concatenate([[0.0], 2.0 * ep.o[:-1] - 1.0])
    return expit(b + w * prev)


def _pred_bias(th, ep):
    return np.full(ep.T, expit(th[0]))


def _pred_q(th, q):
    return _readout(q, th[0], th[1])


_VOL_CACHE: dict[bytes, dict] = {}


def _vol_ctx(ep: Episode, g=None):
    key = ep.o.astype(np.int8).tobytes()
    if key not in _VOL_CACHE:
        _VOL_CACHE[key] = volatile_belief(ep.o)
    return _VOL_CACHE[key]["q"]


def _cp_norm_ctx(ep: Episode, g=None):
    return changepoint_belief(ep.o, H_GRID)


MODELS: dict[str, CogModel] = {m.name: m for m in [
    CogModel("M0_bias", ["b"], [(-6.0, 6.0)], _ctx_o, _pred_bias, description="constant bias"),
    CogModel("WSLS", ["w", "b"], [(-10.0, 10.0), (-6.0, 6.0)], _ctx_o, _pred_wsls,
             description="soft win-stay/lose-shift (follow last winner)"),
    CogModel("count_beta", ["beta", "b"], BETA_B, lambda ep, g: count_belief(ep.o), _pred_q,
             description="Beta count over all past trials"),
    CogModel("window_k", ["beta", "b"], BETA_B, lambda ep, k: count_belief(ep.o, k), _pred_q,
             grid_name="k", grid=[3, 5, 8, 10, 15, 20, 30], description="Beta count, last k"),
    CogModel("M1_delta", ["alpha", "beta", "b"], [UNIT] + BETA_B, _ctx_o, _pred_delta,
             description="delta rule, single alpha"),
    CogModel("M2_phase", ["alpha_stable", "alpha_volatile", "beta", "b"], [UNIT, UNIT] + BETA_B,
             _ctx_o, _pred_phase, description="delta rule, alpha by phase"),
    CogModel("M3_confirm", ["alpha_conf", "alpha_disc", "beta", "b"], [UNIT, UNIT] + BETA_B,
             _ctx_o, _pred_confirm, description="delta rule, confirm/disconfirm alphas"),
    CogModel("M4_cp", ["beta", "b"], BETA_B, lambda ep, h: changepoint_belief(ep.o, h), _pred_q,
             grid_name="h", grid=list(H_GRID), description="change-point observer, fitted h"),
    CogModel("M4_cp_norm", [], [], _cp_norm_ctx, lambda th, q: q,
             description="change-point observer marginalised over h, P = E[p]"),
    CogModel("M4_vol", ["beta", "b"], BETA_B, _vol_ctx, _pred_q,
             description="Behrens volatile learner, sigmoid readout"),
    CogModel("M4_vol_norm", [], [], _vol_ctx, lambda th, q: q,
             description="Behrens volatile learner, P = predictive"),
]}
DEFAULT_MODELS = list(MODELS)

# --------------------------------------------------------------------------- fitting


def _optimize(f: Callable[[np.ndarray], float], bounds, n_restarts: int, rng: np.random.Generator,
              n_probe: int = 64) -> tuple[np.ndarray, float]:
    """Bounded L-BFGS-B from the best ``n_restarts`` of ``n_probe`` random points."""
    if not bounds:
        return np.zeros(0), f(np.zeros(0))
    lo, hi = np.array(bounds, float).T
    probes = lo + (hi - lo) * rng.random((n_probe, len(bounds)))
    vals = np.array([f(x) for x in probes])
    best_x, best_f = None, np.inf
    for x0 in probes[np.argsort(vals)[:n_restarts]]:
        r = minimize(f, x0, method="L-BFGS-B", bounds=bounds)
        if r.fun < best_f:
            best_x, best_f = np.asarray(r.x), float(r.fun)
    return best_x, best_f


def fit_model(model: CogModel | str, episodes: Episode | Sequence[Episode], n_restarts: int = 6,
              seed: int = 0) -> dict:
    """Fit one model to one episode, or pooled over several episodes with shared params.

    Returns nll, excess_nll, k, n, aic_heuristic, bic_heuristic, params and the fitted
    per-trial predictions (``p_hat``, concatenated across episodes when pooled).

    AIC/BIC here are HEURISTIC. With soft targets the soft cross-entropy is a
    pseudo-likelihood, and the effective sample size is not n trials. Prefer
    ``loeo_cv`` (held-out soft cross-entropy) for model comparison.
    """
    model = MODELS[model] if isinstance(model, str) else model
    eps = [episodes] if isinstance(episodes, Episode) else list(episodes)
    rng = np.random.default_rng(seed)
    best = None
    for g in (model.grid if model.grid_name else [None]):
        ctxs = [model.prepare(ep, g) for ep in eps]

        def f(th):
            return sum(soft_nll(ep.y, model.predict(th, c)) for ep, c in zip(eps, ctxs))

        x, val = _optimize(f, model.bounds, n_restarts, rng)
        if best is None or val < best[1]:
            best = (x, val, g, ctxs)
    x, nll, g, ctxs = best
    n = sum(ep.T for ep in eps)
    params = {p: float(v) for p, v in zip(model.params, x)}
    if model.grid_name:
        params[model.grid_name] = float(g)
    if model.name == "WSLS":
        params["p_stay_after_win"] = float(expit(x[1] + x[0]))   # P(A | A just correct)
        params["p_shift_after_loss"] = float(expit(-(x[1] - x[0])))  # P(B | A just wrong)
    p_hat = np.concatenate([model.predict(x, c) for c in ctxs])
    k = model.k
    return {
        "model": model.name, "nll": nll, "excess_nll": nll - sum(entropy_sum(ep.y) for ep in eps),
        "k": k, "n": n, "aic_heuristic": 2 * k + 2 * nll, "bic_heuristic": k * math.log(n) + 2 * nll,
        "params": params, "p_hat": p_hat.tolist(),
    }


def fit_episode(ep: Episode | dict, models: Sequence[str] = DEFAULT_MODELS, n_restarts: int = 6,
                seed: int = 0, keep_pred: bool = False, n_surrogates: int = 200) -> dict:
    """Fit every model in ``models`` to one episode.

    Adds the raw effective LR (``eff_lr_uncalibrated``) and the M1-calibrated one
    (``eff_lr_calibrated``). Use the calibrated one for the stable-vs-volatile claim.
    """
    ep = Episode.from_dict(ep) if isinstance(ep, dict) else ep
    fits = {}
    for m in models:
        fits[m] = fit_model(m, ep, n_restarts, seed)
    cal = calibrated_effective_learning_rate(ep, m1_fit=fits.get("M1_delta"), n_restarts=n_restarts,
                                             seed=seed, n_surrogates=n_surrogates)
    if not keep_pred:
        for r in fits.values():
            r.pop("p_hat")
    return {**ep.meta, "fits": fits, "eff_lr_uncalibrated": cal["raw"], "eff_lr_calibrated": cal}


# ------------------------------------------------------- model-agnostic learning rate


def effective_learning_rate(y: np.ndarray, o: np.ndarray, phase, clip: float = 1e-4,
                            exclude: np.ndarray | None = None) -> dict:
    """UNCALIBRATED effective learning rate: regress the trial-to-trial update in the implied
    belief logit on the outcome, per phase.

    Warning: this statistic is not null-calibrated. Even a constant-alpha delta-rule (M1)
    agent gives a nonzero, typically negative, volatile-minus-stable difference. Beliefs
    are more extreme in the stable phase, and the logit transform stretches updates there.
    For example, alpha=.3, beta=10 gives diff_signed of about -0.26. Use
    ``calibrated_effective_learning_rate`` for inference.

    d_t = logit(y_{t+1}) - logit(y_t) for t = 1..T-1, regressed (OLS with intercept) on
      * the signed outcome s_t = 2 o_t - 1 (``slope_signed``), and
      * the prediction error pe_t = o_t - y_t (``slope_pe``).
    Rows are split by the phase of trial t, the trial whose outcome drives the update.
    ``exclude`` is an optional boolean mask over t that drops trials, e.g. trials at
    phase boundaries. Returns per-phase slopes and the volatile-minus-stable difference.
    """
    y = np.clip(np.asarray(y, float), clip, 1 - clip)
    o = np.asarray(o, float)
    vol = phase if isinstance(phase, np.ndarray) and phase.dtype == bool else _phase_is_volatile(phase)
    d = np.diff(logit(y))
    s = 2 * o[:-1] - 1
    pe = o[:-1] - y[:-1]
    keep = np.ones(len(d), bool) if exclude is None else ~np.asarray(exclude, bool)[:-1]
    out = {}
    for name, mask in (("stable", ~vol[:-1]), ("volatile", vol[:-1]), ("all", np.ones(len(d), bool))):
        m = mask & keep
        r = {"n": int(m.sum())}
        for key, x in (("signed", s), ("pe", pe)):
            if m.sum() >= 3 and np.var(x[m]) > 0:
                X = np.column_stack([np.ones(m.sum()), x[m]])
                coef = np.linalg.lstsq(X, d[m], rcond=None)[0]
                r[f"slope_{key}"], r[f"intercept_{key}"] = float(coef[1]), float(coef[0])
            else:
                r[f"slope_{key}"] = r[f"intercept_{key}"] = float("nan")
        out[name] = r
    for key in ("signed", "pe"):
        out[f"diff_{key}"] = out["volatile"][f"slope_{key}"] - out["stable"][f"slope_{key}"]
    return out


_ELR_KEYS = ("slope_signed", "slope_pe")


def _elr_diff(a: dict, b: dict) -> dict:
    """a - b for the per-phase slopes and the volatile-minus-stable differences."""
    out = {ph: {k: a[ph][k] - b[ph][k] for k in _ELR_KEYS} for ph in ("stable", "volatile", "all")}
    out.update({k: a[k] - b[k] for k in ("diff_signed", "diff_pe")})
    return out


def calibrated_effective_learning_rate(ep: Episode | dict, m1_fit: dict | None = None, n_restarts: int = 6,
                                       seed: int = 0, n_surrogates: int = 200, exclude=None) -> dict:
    """Effective LR calibrated against a constant-alpha (M1) null.

    1. Fit M1 to the episode, or reuse ``m1_fit``, which must contain ``p_hat``.
    2. Compute the raw ELR statistic on the data (``raw``).
    3. Compute the same statistic on M1-implied data. With ``n_surrogates > 0``, these are
       surrogates y* = sigmoid(logit p_hat + e), with e ~ N(0, s^2) iid and s equal to the
       s.d. of the residual logit(y) - logit(p_hat); ``null`` is their mean and ``null_sd``
       their s.d. Noise matters for slope_pe, because y appears in the regressor. With
       ``n_surrogates = 0``, ``null`` is the statistic on the noiseless p_hat.
    4. ``calibrated = raw - null``. A calibrated ``diff_signed > 0`` means a larger
       volatile-phase update than a single-alpha learner fitted to the same episode
       would show. ``z_diff_*`` = calibrated diff / null_sd (when surrogates are used).
    """
    ep = Episode.from_dict(ep) if isinstance(ep, dict) else ep
    if m1_fit is None or "p_hat" not in m1_fit:
        m1_fit = fit_model("M1_delta", ep, n_restarts, seed)
    p_hat = _clip(np.asarray(m1_fit["p_hat"], float), 1e-4)
    raw = effective_learning_rate(ep.y, ep.o, ep.vol, exclude=exclude)
    out = {"raw": raw, "m1_params": m1_fit["params"]}
    if n_surrogates > 0:
        yl = logit(np.clip(ep.y, 1e-4, 1 - 1e-4))
        sd = float(np.std(yl - logit(p_hat)))
        rng = np.random.default_rng(seed)
        sims = [effective_learning_rate(expit(logit(p_hat) + sd * rng.standard_normal(ep.T)), ep.o, ep.vol,
                                        exclude=exclude) for _ in range(n_surrogates)]

        def agg(fn):
            v = np.array([fn(r) for r in sims])
            return float(np.nanmean(v)), float(np.nanstd(v))

        null = {ph: {k: agg(lambda r: r[ph][k])[0] for k in _ELR_KEYS} for ph in ("stable", "volatile", "all")}
        null.update({k: agg(lambda r: r[k])[0] for k in ("diff_signed", "diff_pe")})
        out["null_sd"] = {k: agg(lambda r: r[k])[1] for k in ("diff_signed", "diff_pe")}
        out["residual_logit_sd"] = sd
    else:
        null = effective_learning_rate(p_hat, ep.o, ep.vol, exclude=exclude)
    out["null"] = null
    out["calibrated"] = _elr_diff(raw, null)
    if "null_sd" in out:
        for k in ("diff_signed", "diff_pe"):
            sdk = out["null_sd"][k]
            out[f"z_{k}"] = out["calibrated"][k] / sdk if sdk > 0 else float("nan")
    return out


def implied_lr_by_phase(q: np.ndarray, o: np.ndarray, vol: np.ndarray) -> dict:
    """Mean implied alpha (see ``implied_learning_rate``) per phase, for any latent q."""
    a = implied_learning_rate(q, o)
    return {"stable": float(np.nanmean(a[~vol])), "volatile": float(np.nanmean(a[vol]))}


# --------------------------------------------------------------- model comparison


def bms_random_effects(log_evidence: np.ndarray, alpha0: float = 1.0, n_samples: int = 200_000,
                       seed: int = 0, max_iter: int = 500) -> dict:
    """Variational random-effects BMS (Stephan 2009) with protected exceedance probabilities
    (Rigoux 2014). ``log_evidence`` has shape (n_subjects, n_models). Pass -BIC/2 for it.
    """
    L = np.asarray(log_evidence, float)
    N, K = L.shape
    a0 = np.full(K, alpha0)
    alpha = a0.copy()
    for _ in range(max_iter):
        u = L + digamma(alpha) - digamma(alpha.sum())
        g = np.exp(u - logsumexp(u, axis=1, keepdims=True))
        new = a0 + g.sum(0)
        if np.max(np.abs(new - alpha)) < 1e-8:
            alpha = new
            break
        alpha = new
    # exceedance probabilities by Dirichlet sampling
    r = np.random.default_rng(seed).dirichlet(alpha, n_samples)
    xp = np.bincount(r.argmax(1), minlength=K) / n_samples
    # free energies of the full model (F1) and of the null model with equal frequencies (F0)
    Elogr = digamma(alpha) - digamma(alpha.sum())
    gs = np.clip(g, 1e-300, None)
    kl_dir = (gammaln(alpha.sum()) - gammaln(alpha).sum() - gammaln(a0.sum()) + gammaln(a0).sum()
              + ((alpha - a0) * Elogr).sum())
    F1 = float((g * (L + Elogr - np.log(gs))).sum() - kl_dir)
    F0 = float(logsumexp(L - np.log(K), axis=1).sum())
    bor = float(expit(F0 - F1))  # = 1 / (1 + exp(F1 - F0))
    pxp = (1 - bor) * xp + bor / K
    return {"alpha": alpha.tolist(), "expected_freq": (alpha / alpha.sum()).tolist(),
            "xp": xp.tolist(), "bor": bor, "pxp": pxp.tolist()}


def compare_models(episode_fits: Sequence[dict], models: Sequence[str] | None = None) -> dict:
    """HEURISTIC comparison: summed pseudo-BIC/AIC/NLL across episodes, plus random-effects BMS
    on -BIC/2. With soft targets these are pseudo-likelihood scores, so treat them as
    descriptive only and use ``loeo_cv`` for inference."""
    models = list(models or episode_fits[0]["fits"].keys())
    bic = np.array([[e["fits"][m]["bic_heuristic"] for m in models] for e in episode_fits])
    out = {"models": models, "n_episodes": len(episode_fits),
           "sum_bic_heuristic": dict(zip(models, bic.sum(0).tolist())),
           "sum_aic_heuristic": dict(zip(models, [float(sum(e["fits"][m]["aic_heuristic"] for e in episode_fits)) for m in models])),
           "sum_nll": dict(zip(models, [float(sum(e["fits"][m]["nll"] for e in episode_fits)) for m in models])),
           "best_by_sum_bic_heuristic": models[int(bic.sum(0).argmin())],
           "n_best_bic_heuristic": dict(zip(models, np.bincount(bic.argmin(1), minlength=len(models)).tolist()))}
    if len(episode_fits) >= 2 and len(models) >= 2:
        out["bms_heuristic"] = bms_random_effects(-bic / 2)
    return out


def loeo_cv(model: str, episodes: Sequence[Episode | dict], n_restarts: int = 3, seed: int = 0,
            folds: Sequence[int] | None = None) -> dict:
    """Leave-one-episode-out cross-validation of a pooled (shared-parameter) fit.

    For each held-out episode, fit the model pooled over the remaining episodes and
    score the held-out soft cross-entropy. This is the recommended model-comparison
    score; unlike the heuristic AIC/BIC it does not rely on treating soft targets as
    Bernoulli observations. ``folds`` restricts the run to a subset (used for
    parallelism); the totals then cover only those folds.
    """
    eps = [Episode.from_dict(e) if isinstance(e, dict) else e for e in episodes]
    m = MODELS[model]
    idx = range(len(eps)) if folds is None else folds
    per = []
    for i in idx:
        train = [e for j, e in enumerate(eps) if j != i]
        if m.bounds or m.grid_name:
            r = fit_model(m, train, n_restarts, seed)
            theta = np.array([r["params"][p] for p in m.params])
            g = r["params"].get(m.grid_name) if m.grid_name else None
            if m.grid_name == "k":
                g = int(g)
        else:
            theta, g = np.zeros(0), None
        p = m.predict(theta, m.prepare(eps[i], g))
        nll = soft_nll(eps[i].y, p)
        per.append({"fold": int(i), "heldout_nll": nll, "heldout_excess_nll": nll - entropy_sum(eps[i].y),
                    "n": eps[i].T, "episode_id": eps[i].meta.get("episode_id")})
    return _cv_summary(model, per)


def _cv_summary(model: str, per: list[dict]) -> dict:
    n = sum(r["n"] for r in per)
    tot = sum(r["heldout_nll"] for r in per)
    exc = sum(r["heldout_excess_nll"] for r in per)
    return {"model": model, "n_folds": len(per), "n_trials": n, "heldout_nll": tot,
            "heldout_nll_per_trial": tot / n, "heldout_excess_nll_per_trial": exc / n,
            "per_episode": sorted(per, key=lambda r: r["fold"])}


# --------------------------------------------------------- schedules and simulation


def make_schedule(order: str = "SV", stable_p: float = 0.8, vol_start: float | None = None,
                  rng: np.random.Generator | None = None, T: int = 80, window: int = 10) -> dict:
    """Minimal DESIGN.md schedule for tests and recovery. This is not the main data generator.

    The stable phase has T/2 trials at ``stable_p``. The volatile phase has T/2 trials
    alternating 0.8/0.2 every ``window`` trials. For SV it starts opposite to stable_p;
    for VS ``vol_start`` is counterbalanced (random if None). Each window has exactly
    round(window * p) A-correct trials, in shuffled positions.
    """
    rng = rng or np.random.default_rng()
    half = T // 2
    if vol_start is None:
        vol_start = round(1.0 - stable_p, 10) if order == "SV" else float(rng.choice([0.2, 0.8]))
    nwin = half // window
    vol_ps = [round(vol_start if i % 2 == 0 else 1.0 - vol_start, 10) for i in range(nwin)]
    stable = [("stable", stable_p)] * (half // window)
    volatile = [("volatile", p) for p in vol_ps]
    blocks = stable + volatile if order == "SV" else volatile + stable
    a, ph, pa = [], [], []
    for name, p in blocks:
        w = np.zeros(window, int)
        w[: int(round(window * p))] = 1
        rng.shuffle(w)
        a += w.tolist()
        ph += [name] * window
        pa += [p] * window
    return {"a_correct": a, "phase": ph, "p_a": pa, "order": order}


def random_schedule(rng: np.random.Generator) -> dict:
    return make_schedule(order=str(rng.choice(["SV", "VS"])), stable_p=float(rng.choice([0.8, 0.2])),
                         rng=rng)


def simulate(model: str, theta: Sequence[float], sched: dict, noise_sd: float = 0.0,
             rng: np.random.Generator | None = None, grid_value=None) -> np.ndarray:
    """Soft choices y from a model. Optional Gaussian noise is added in logit space."""
    ep = Episode.from_dict({**sched, "y": None})
    m = MODELS[model]
    p = m.predict(np.asarray(theta, float), m.prepare(ep, grid_value))
    if noise_sd > 0:
        rng = rng or np.random.default_rng()
        p = expit(logit(_clip(p)) + noise_sd * rng.standard_normal(len(p)))
    return _clip(p)


RECOVERY_PRIORS = {
    "M1_delta": {"alpha": (0.05, 0.8), "beta": (3.0, 20.0), "b": (-1.0, 1.0)},
    "M2_phase": {"alpha_stable": (0.05, 0.8), "alpha_volatile": (0.05, 0.8), "beta": (3.0, 20.0), "b": (-1.0, 1.0)},
    "M3_confirm": {"alpha_conf": (0.05, 0.8), "alpha_disc": (0.05, 0.8), "beta": (3.0, 20.0), "b": (-1.0, 1.0)},
}


def _recover_one(args):
    gen, i, noise_sd, seed, fit_models = args
    rng = np.random.default_rng([seed, i, list(RECOVERY_PRIORS).index(gen)])
    pri = RECOVERY_PRIORS[gen]
    theta = {k: float(rng.uniform(*v)) for k, v in pri.items()}
    sched = random_schedule(rng)
    y = simulate(gen, list(theta.values()), sched, noise_sd, rng)
    ep = Episode.from_dict({**sched, "y": y.tolist()})
    fits = {m: fit_model(m, ep, n_restarts=4, seed=i) for m in fit_models}
    return {"gen": gen, "true": theta, "order": sched["order"],
            "fit": {m: {"params": f["params"], "bic": f["bic_heuristic"], "aic": f["aic_heuristic"]} for m, f in fits.items()}}


def parameter_recovery(n_per_model: int = 60, noise_sd: float = 0.3, seed: int = 0, jobs: int = 1) -> dict:
    """Simulate M1/M2/M3 agents on random design schedules, refit them, and report recovery.

    Reports the Pearson and Spearman r between true and fitted params (each generator refit
    with itself), and a BIC model-recovery confusion matrix over {M1, M2, M3}.
    """
    from scipy.stats import pearsonr, spearmanr

    gens = list(RECOVERY_PRIORS)
    tasks = [(g, i, noise_sd, seed, gens) for g in gens for i in range(n_per_model)]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_recover_one, tasks, chunksize=4))
    else:
        res = [_recover_one(t) for t in tasks]
    report = {"n_per_model": n_per_model, "noise_sd_logit": noise_sd, "seed": seed,
              "priors": RECOVERY_PRIORS, "param_recovery": {}, "confusion_bic": {}}
    for g in gens:
        rows = [r for r in res if r["gen"] == g]
        pr = {}
        for p in RECOVERY_PRIORS[g]:
            t = np.array([r["true"][p] for r in rows])
            f = np.array([r["fit"][g]["params"][p] for r in rows])
            pr[p] = {"pearson": float(pearsonr(t, f)[0]), "spearman": float(spearmanr(t, f)[0]),
                     "median_abs_err": float(np.median(np.abs(t - f)))}
        report["param_recovery"][g] = pr
        for ic in ("bic", "aic"):
            wins = [min(gens, key=lambda m: r["fit"][m][ic]) for r in rows]
            report.setdefault(f"confusion_{ic}", {})[g] = {m: wins.count(m) / len(rows) for m in gens}
        # Confusion restricted to agents whose two alphas differ by > 0.2 (M2/M3 only)
        pa = [p for p in RECOVERY_PRIORS[g] if p.startswith("alpha")]
        if len(pa) == 2:
            sub = [r for r in rows if abs(r["true"][pa[0]] - r["true"][pa[1]]) > 0.2]
            wins = [min(gens, key=lambda m: r["fit"][m]["bic"]) for r in sub]
            report.setdefault("confusion_bic_alpha_gap_gt_0.2", {})[g] = {
                "n": len(sub), **{m: wins.count(m) / max(len(sub), 1) for m in gens}}
    # The Behrens signature in M2 recovery: is the sign of (alpha_v - alpha_s) recovered?
    rows = [r for r in res if r["gen"] == "M2_phase"]
    big = [r for r in rows if abs(r["true"]["alpha_volatile"] - r["true"]["alpha_stable"]) > 0.1]
    report["M2_sign_recovery"] = float(np.mean([
        np.sign(r["true"]["alpha_volatile"] - r["true"]["alpha_stable"])
        == np.sign(r["fit"]["M2_phase"]["params"]["alpha_volatile"] - r["fit"]["M2_phase"]["params"]["alpha_stable"])
        for r in big])) if big else float("nan")
    report["M2_sign_recovery_n"] = len(big)
    report["raw"] = res
    return report


# --------------------------------------------------------------------------------- CLI


def _fit_task(args):
    d, models, n_restarts, seed = args
    return fit_episode(d, models, n_restarts, seed)


def _cv_task(args):
    rows, model, fold, n_restarts, seed = args
    return model, loeo_cv(model, rows, n_restarts, seed, folds=[fold])["per_episode"][0]


def _group_key(meta: dict, keys: Sequence[str]) -> str:
    return "|".join(f"{k}={meta.get(k)}" for k in keys)


def _pooled_task(args):
    rows, model, n_restarts, seed = args
    r = fit_model(model, [Episode.from_dict(d) for d in rows], n_restarts, seed)
    r.pop("p_hat")
    return r


def _map(fn, tasks, jobs):
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(jobs) as ex:
            return list(ex.map(fn, tasks))
    return [fn(t) for t in tasks]


def summarize_calibrated_elr(episode_fits: Sequence[dict]) -> dict:
    """Across-episode summary of the M1-calibrated volatile-minus-stable ELR difference."""
    out = {}
    for k in ("diff_signed", "diff_pe"):
        v = np.array([e["eff_lr_calibrated"]["calibrated"][k] for e in episode_fits], float)
        v = v[np.isfinite(v)]
        n = len(v)
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        out[k] = {"mean": float(v.mean()) if n else float("nan"), "sd": sd, "n": n,
                  "t": float(v.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")}
    return out


def fit_file(input_path: str, models: Sequence[str] = DEFAULT_MODELS, group_by: Sequence[str] = ("model", "domain", "format"),
             n_restarts: int = 6, seed: int = 0, jobs: int = 1, pooled: bool = True, cv: bool = True,
             cv_restarts: int = 3) -> dict:
    """Fit all episodes in a JSONL file.

    Returns per-episode fits, per-group heuristic comparison (summed pseudo-BIC, BMS),
    pooled fits, the leave-one-episode-out CV score per group and model (``cv``, the
    recommended comparison), and the across-episode calibrated-ELR summary.
    """
    with open(input_path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    eps = _map(_fit_task, [(d, list(models), n_restarts, seed) for d in rows], jobs)
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(eps):
        groups.setdefault(_group_key(e, group_by), []).append(i)
    comparison = {g: compare_models([eps[i] for i in idx], models) for g, idx in groups.items()}
    elr = {g: summarize_calibrated_elr([eps[i] for i in idx]) for g, idx in groups.items()}
    pooled_out: dict = {}
    if pooled:
        keys = [(g, m) for g in groups for m in models]
        res = _map(_pooled_task, [([rows[i] for i in groups[g]], m, n_restarts, seed) for g, m in keys], jobs)
        for (g, m), r in zip(keys, res):
            pooled_out.setdefault(g, {})[m] = r
    cv_out: dict = {}
    if cv:
        keys = [(g, m, f) for g, idx in groups.items() if len(idx) >= 2 for m in models for f in range(len(idx))]
        res = _map(_cv_task, [([rows[i] for i in groups[g]], m, f, cv_restarts, seed) for g, m, f in keys], jobs)
        per: dict = {}
        for (g, m, _), (_, r) in zip(keys, res):
            per.setdefault(g, {}).setdefault(m, []).append(r)
        for g, d in per.items():
            cv_out[g] = {m: _cv_summary(m, v) for m, v in d.items()}
            cv_out[g]["best_by_cv"] = min(d, key=lambda m: cv_out[g][m]["heldout_nll"])
    return {"models": {m: {"params": MODELS[m].params, "grid": MODELS[m].grid_name, "k": MODELS[m].k,
                           "description": MODELS[m].description} for m in models},
            "group_by": list(group_by), "episodes": eps, "comparison_heuristic": comparison,
            "cv": cv_out, "eff_lr_calibrated_summary": elr, "pooled": pooled_out}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit cognitive models to episodes in a JSONL file")
    f.add_argument("--input", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    f.add_argument("--group-by", nargs="+", default=["model", "domain", "format"])
    f.add_argument("--restarts", type=int, default=6)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    f.add_argument("--no-pooled", action="store_true")
    f.add_argument("--no-cv", action="store_true", help="skip leave-one-episode-out CV")
    f.add_argument("--cv-restarts", type=int, default=3)
    r = sub.add_parser("recover", help="parameter/model recovery for M1/M2/M3")
    r.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "param_recovery.json"))
    r.add_argument("--n", type=int, default=60)
    r.add_argument("--noise", type=float, default=0.3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        out = fit_file(a.input, a.models, a.group_by, a.restarts, a.seed, a.jobs, not a.no_pooled,
                       not a.no_cv, a.cv_restarts)
    else:
        out = parameter_recovery(a.n, a.noise, a.seed, a.jobs)
        summary = {g: {p: round(v["pearson"], 3) for p, v in pr.items()} for g, pr in out["param_recovery"].items()}
        print(json.dumps({"pearson": summary, "confusion_bic": out["confusion_bic"],
                          "confusion_aic": out["confusion_aic"],
                          "confusion_bic_gap": out["confusion_bic_alpha_gap_gt_0.2"],
                          "M2_sign_recovery": out["M2_sign_recovery"]}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
