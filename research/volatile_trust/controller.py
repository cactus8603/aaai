"""Surprise-gated history controller (training-free, black-box).

Bayesian online change-point detection (Adams & MacKay 2007) on the observed feedback stream
o_t = 1[reporter A was correct], Beta-Bernoulli likelihood, constant hazard h. This is the
normative counterpart of Nassar-style change-point-probability learning-rate control (an
LC-NE "unexpected uncertainty" analogue). Output at trial t (using o_1..o_{t-1} only):
  run_length : MAP number of trials since the last inferred change
  cpp        : posterior probability that a change occurred at the latest trial
"""
import numpy as np


def bocpd(o, h=1 / 15, a0=1.0, b0=1.0):
    """Return list of dicts for t=1..len(o)+1 (state *before* each trial)."""
    o = np.asarray(o, dtype=float)
    T = len(o)
    R = np.zeros(T + 1)          # run-length posterior over r = 0..t
    R[0] = 1.0
    a = np.array([a0])
    b = np.array([b0])
    states = [dict(run_length=0, cpp=0.0, p_a=0.5)]
    for t in range(T):
        x = o[t]
        pred = np.where(x == 1, a / (a + b), b / (a + b))       # predictive prob per run length
        growth = R[:t + 1] * pred * (1 - h)
        cp = (R[:t + 1] * pred * h).sum()
        newR = np.concatenate([[cp], growth])
        newR /= newR.sum()
        a = np.concatenate([[a0], a + x])
        b = np.concatenate([[b0], b + (1 - x)])
        R = np.zeros(T + 1)
        R[:t + 2] = newR
        rl = int(np.argmax(newR))
        p_a = float((newR * a / (a + b)).sum())
        states.append(dict(run_length=rl, cpp=float(newR[0] + (newR[1] if t > 0 else 0)), p_a=p_a))
    return states


def history_window(o, t, mode, h=1 / 15, min_len=3):
    """Return (start_index_1based, marker_trial or None) for the history shown before trial t."""
    if mode == 'full' or t == 1:
        return 1, None
    if mode.startswith('window:'):
        k = int(mode.split(':')[1])
        return max(1, t - k), None
    st = bocpd(o[:t - 1], h=h)[-1]
    rl = max(st['run_length'], min_len)
    start = max(1, t - rl)
    if mode == 'cp_hard':
        return start, None
    if mode == 'cp_soft':
        return 1, (start if start > 1 else None)
    raise ValueError(mode)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    p = [0.8] * 40 + [0.2] * 10 + [0.8] * 10 + [0.2] * 10 + [0.8] * 10
    o = (rng.random(80) < np.array(p)).astype(int)
    for t in (10, 41, 45, 50, 55, 62, 75, 80):
        print(t, history_window(o, t, 'cp_hard'), round(bocpd(o[:t - 1])[-1]['p_a'], 2))
