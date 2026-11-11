"""Normative design simulation (no VLM): when does adaptive (BOCPD) integration beat the best fixed window?
IID Bernoulli outcomes; schedule = stable phase (len S, p_high) + volatile phase (dwell D, alternating)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from controller import bocpd

def schedule(p, S, V, D, order):
    st = [p] * S
    vol, q = [], 1 - p
    while len(vol) < V:
        vol += [q] * D; q = 1 - q
    vol = vol[:V]
    if order == 'SV':
        return st + vol
    vol = [1 - x for x in vol]                     # VS: volatile first (starts at p_high), then a reversal
    return vol + [1 - vol[-1]] * S                 # into the stable phase

def run(p, S, V, D, n=300, seed=0, hs=(1/8, 1/15, 1/30), Ks=(3, 5, 7, 10, 15, 20, 30)):
    rng = np.random.default_rng(seed)
    acc = {f'bocpd_h{round(1/h)}': [] for h in hs}
    acc.update({f'last{K}': [] for K in Ks}); acc['count_all'] = []; acc['oracle_seg'] = []
    for i in range(n):
        sch = schedule(p, S, V, D, 'SV' if i % 2 == 0 else 'VS')
        o = (rng.random(len(sch)) < np.array(sch)).astype(int)
        rev = [t for t in range(1, len(sch)) if sch[t] != sch[t-1]]
        sts = {h: bocpd(o, h=h) for h in hs}
        for t in range(1, len(o)):
            h_ = o[:t]
            def score(pa): return 0.5 if pa == 0.5 else float((pa > 0.5) == bool(o[t]))
            for h in hs: acc[f'bocpd_h{round(1/h)}'].append(score(sts[h][t]['p_a']))
            for K in Ks: acc[f'last{K}'].append(score(h_[-K:].mean()))
            acc['count_all'].append(score(h_.mean()))
            start = max([r for r in rev if r <= t - 1], default=0)
            acc['oracle_seg'].append(score(o[start:t].mean() if t > start else 0.5))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    bw = max((m[f'last{K}'], K) for K in Ks)
    bb = max((m[f'bocpd_h{round(1/h)}'], round(1/h)) for h in hs)
    return dict(p=p, S=S, V=V, D=D, bocpd_best=round(bb[0], 3), h_best=bb[1], bocpd_h15=round(m['bocpd_h15'], 3),
                best_window=round(bw[0], 3), K_best=bw[1], count_all=round(m['count_all'], 3),
                oracle=round(m['oracle_seg'], 3), adv_vs_window=round(bb[0] - bw[0], 3))

if __name__ == '__main__':
    out = []
    for p in (0.85, 0.8, 0.75, 0.7, 0.65):
        for S, V, D in ((40, 40, 10), (60, 60, 10), (60, 60, 15), (80, 40, 10)):
            r = run(p, S, V, D); out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open('runs/design_sim.json', 'w'), indent=1)
