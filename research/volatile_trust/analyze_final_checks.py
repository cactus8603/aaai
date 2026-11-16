"""Final numeric checks for the manuscript from EXISTING outputs (CPU only, no model or API calls).

F1  Following the new better advisor after A-became-worse reversals, split by episode order
    (volatile-first: reversals at t=11, 31, A's accumulated lead 6; stable-first: t=41, 61, lead 24).
F2  Cohen's kappa / choice agreement of CorrectnessOnly and Summary with the all-trial and last-five count
    rules (read from the existing held-out mechanism outputs; copies combined).
F3  Masking P1-P4 (read from the existing held-out masking output, unrounded, 10,000 resamples).
F4  Per-copy recovery ratio and CorrectnessOnly - Summary per copy (readout-sensitivity per-pair values).
F5  API output tokens (GPT-6-Astra) and visible reasoning (Claude Fable 5.2).

Usage (from research/volatile_trust): ../../.venv/bin/python analyze_final_checks.py
Writes analysis/final_checks/final_checks.json and summary.md (relative paths only).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402
import analyze_mechanism as mech  # noqa: E402
import cogmodels as cm  # noqa: E402

OUT = HERE / 'analysis' / 'final_checks'
MODELS = {'qwen3vl8b': 'Qwen3-VL-8B', 'internvl8b': 'InternVL3.5-8B', 'gemma3_27b': 'Gemma-3-27B',
          'qwen3vl32b': 'Qwen3-VL-32B', 'internvl38b': 'InternVL3.5-38B'}
RUNS = {'Raw': ('ho_full_text', 'A|full|text'), 'RawCorrectness': ('ho_full_rawcoded', 'A+|full|text'),
        'CorrectnessOnly': ('ho_full_coded', 'B|full|text'), 'Summary': ('ho_sum_all', 'C|sum_all|text')}
N = 10
N_BOOT = 10000


def heldout_episodes():
    eps = mech.load_episodes(HERE / 'episodes_v3' / 'episodes.jsonl')
    return {e: ep for e, ep in sorted(eps.items()) if not e.endswith('t') and ep['split'] == 'held_out'}


def a_worse_reversals(ep):
    return [r for r in am.true_reversals(ep) if ep['trials'][r - 1]['p_a'] < 0.5]


def lead_at(ep, r):
    return int(sum(2 * x['a_correct'] - 1 for x in ep['trials'][:r - 1]))


def follow_by_reversal(ep, choice_a):
    """choice_a: {t: credit for choosing A (1/0/0.5)}. Returns {rev: array(N) of following new better (B)}."""
    out = {}
    for r in a_worse_reversals(ep):
        v = np.full(N, np.nan)
        for k in range(N):
            t = r + k
            if t in choice_a and ep['trials'][t - 1]['p_a'] < 0.5:
                v[k] = 1 - choice_a[t]
        out[r] = v
    return out


def model_per_episode(m, cond):
    run, arm = RUNS[cond]
    rows, _ = am.load_rows([HERE / 'runs_h200' / m / run], strict=False)
    rows = [r for r in rows if r['readout'] == 'choice']
    paired = am.pair_trials(rows, strict=False)
    cells = [c for c in paired if c[2] == 'held_out' and c[3] == 'choice' and c[4] == arm]
    assert {c[5] for c in cells} == {'pet', 'scenario'}, cells
    by_base = defaultdict(list)
    for c in cells:
        for b, tr in paired[c].items():
            ch = {x['t']: float(am.choice_credit(x['py'])) for x in tr if x['t'] >= 2}
            by_base[b].append(follow_by_reversal(tr[0]['ep'], ch))
    out = {}
    for b, lst in by_base.items():
        if len(lst) != 2:
            continue
        out[b] = {r: np.nanmean(np.array([d[r] for d in lst]), 0) for r in lst[0]}
    return out


def agent_per_episode(eps, kind):
    out = {}
    for b, ep in eps.items():
        o = np.array([x['a_correct'] for x in ep['trials']], float)
        if kind == 'learner':
            q = cm.volatile_belief(o)['q']
            ch = {i + 1: float(am.choice_credit(q[i])) for i in range(len(o)) if i + 1 >= 2}
        else:
            sp = mech.strategy_predictions(ep)[kind]
            ch = {i + 1: (0.5 if np.isnan(sp[i]) else float(sp[i])) for i in range(len(o)) if i + 1 >= 2}
        out[b] = follow_by_reversal(ep, ch)
    return out


def summarize_split(per_model, eps):
    """per_model: {name: {base: {rev: curve}}}. Returns split summaries (k=9 and mean of k=7..9)."""
    order = {b: eps[b]['order'] for b in eps}
    res = {}
    for stat, sl in (('trial10', slice(9, 10)), ('trials8_10', slice(7, 10))):
        res[stat] = {}
        for grp, cond in (('all', lambda b, r: True), ('volatile_first_lead6', lambda b, r: order[b] == 'VS'),
                          ('stable_first_lead24', lambda b, r: order[b] == 'SV'),
                          ('rev_t11', lambda b, r: r == 11), ('rev_t31', lambda b, r: r == 31),
                          ('rev_t41', lambda b, r: r == 41), ('rev_t61', lambda b, r: r == 61)):
            per_m, per_ep = {}, {}
            for name, d in per_model.items():
                ep_vals = {b: float(np.nanmean([np.nanmean(c[sl]) for r, c in revs.items() if cond(b, r)]))
                           for b, revs in d.items() if any(cond(b, r) for r in revs)}
                per_m[name] = float(np.mean(list(ep_vals.values())))
                per_ep[name] = ep_vals
            res[stat][grp] = dict(mean=float(np.mean(list(per_m.values()))),
                                  range=[min(per_m.values()), max(per_m.values())], per_model=per_m,
                                  n_pairs=len(next(iter(per_ep.values()))))
            res[stat][grp]['_per_ep'] = per_ep
        # difference stable-first minus volatile-first: bootstrap over episode pairs within order group
        rng = np.random.default_rng(0)
        names = list(per_model)
        A = res[stat]['volatile_first_lead6']['_per_ep']
        B = res[stat]['stable_first_lead24']['_per_ep']
        ba, bb = sorted(A[names[0]]), sorted(B[names[0]])
        Ma = np.array([[A[n][b] for b in ba] for n in names]).mean(0)   # model-mean per episode
        Mb = np.array([[B[n][b] for b in bb] for n in names]).mean(0)
        boots = np.array([Mb[rng.integers(0, len(Mb), len(Mb))].mean() - Ma[rng.integers(0, len(Ma), len(Ma))].mean()
                          for _ in range(N_BOOT)])
        per_model_diff = {n: float(np.mean(list(B[n].values())) - np.mean(list(A[n].values()))) for n in names}
        res[stat]['stable_minus_volatile_first'] = dict(
            mean=float(Mb.mean() - Ma.mean()), ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            per_model=per_model_diff,
            slope_per_lead_point=float((Mb.mean() - Ma.mean()) / 18),
            note='lead at onset is 6 for every volatile-first reversal and 24 for every stable-first reversal, '
                 'so a regression on lead is identical to this two-group contrast (slope = difference / 18)')
        for g in res[stat].values():
            g.pop('_per_ep', None)
    return res


def f1():
    eps = heldout_episodes()
    leads = defaultdict(set)
    for b, ep in eps.items():
        for r in a_worse_reversals(ep):
            leads[(ep['order'], r)].add(lead_at(ep, r))
    out = dict(lead_at_onset={f'{o} t={r}': sorted(v) for (o, r), v in sorted(leads.items())},
               n_pairs=dict(VS=sum(e['order'] == 'VS' for e in eps.values()),
                            SV=sum(e['order'] == 'SV' for e in eps.values())), conditions={})
    for cond in RUNS:
        per_model = {MODELS[m]: model_per_episode(m, cond) for m in MODELS}
        out['conditions'][cond] = summarize_split(per_model, eps)
        print('F1', cond, 'done', file=sys.stderr)
    for kind, label in (('learner', 'Bayesian volatility learner'), ('count_all', 'All-trial count rule'),
                        ('last5', 'Last-five count rule')):
        out['conditions'][label] = summarize_split({label: agent_per_episode(eps, kind)}, eps)
    return out


def f2():
    out = {}
    for m, name in MODELS.items():
        r = json.loads((HERE / 'analysis/h200_ho' / f'mech_{m}.json').read_text())
        for cond in ('CorrectnessOnly', 'Summary', 'Raw'):
            arm = RUNS[cond][1]
            key = next(k for k in r['heuristics'] if '/held_out/choice/' in k and k.endswith('/' + arm))
            s = r['heuristics'][key]['all']['strategies']
            out.setdefault(cond, {})[name] = {st: dict(kappa=s[st]['kappa'], kappa_ci95=s[st]['kappa_ci95'],
                                                       agreement_nontied=s[st]['agreement'], n=s[st]['n'])
                                              for st in ('count_all', 'last5')}
    for cond, d in out.items():
        mods = [d[n] for n in MODELS.values()]
        for st in ('count_all', 'last5'):
            k = [v[st]['kappa'] for v in mods]
            a = [v[st]['agreement_nontied'] for v in mods]
            d[f'range_{st}'] = dict(kappa=[min(k), max(k)], agreement=[min(a), max(a)])
    return out


def f3():
    r = json.loads((HERE / 'analysis/h200_ho/masking/masking.json').read_text())
    out = {}
    for name, v in r.items():
        c = v['contrasts']
        g = lambda k: dict(mean_pp=100 * c[k]['mean'], ci95_pp=[100 * x for x in c[k]['ci95']], p=c[k]['p'], n=c[k]['n'])  # noqa: E731
        w = v['copy_w']
        out[name] = dict(
            P1_hide_outcome_main=g('hide outcome (main)'),
            P2_hide_both_vs_rawcorrectness=g('mask_ok - A+  (hide both)'),
            P3_outcome_main_minus_reports_main=g('outcome minus reports effect'),
            hide_reports_main=g('hide reports (main)'), interaction=g('interaction'),
            hide_outcome_simple=g('rep_ok - A+   (hide outcome)'), hide_reports_simple=g('out_ok - A+   (hide reports)'),
            both_masked_minus_correctness_only=g('mask_ok - B   (length/layout)'),
            P4_repeat_weight=dict(outcome_shown=[w['A+'], w['out_ok']], outcome_masked=[w['rep_ok'], w['mask_ok']],
                                  holds=bool(min(w['A+'], w['out_ok']) > max(w['rep_ok'], w['mask_ok']))))
        out[name]['holds'] = dict(P1=out[name]['P1_hide_outcome_main']['ci95_pp'][0] > 0,
                                  P2=out[name]['P2_hide_both_vs_rawcorrectness']['ci95_pp'][0] > 0,
                                  P3=out[name]['P3_outcome_main_minus_reports_main']['ci95_pp'][0] > 0,
                                  P4=out[name]['P4_repeat_weight']['holds'])
    return out


def f4():
    r = json.loads((HERE / 'analysis/h200_ho/readout_sensitivity/readout_sensitivity.json').read_text())
    out = {}
    for name, e in r['models'].items():
        out[name] = {}
        for meth, v in e['methods'].items():
            pp = {k: x['per_pair'] for k, x in v['conditions'].items()}
            pos = am.position_ci(pp['raw'], pp['correct_only'], pp['summary'], N_BOOT, 0)
            g = v['gaps']['summary_minus_correct_only']
            out[name][meth] = dict(acc={k: 100 * x['mean'] for k, x in v['conditions'].items()},
                                   recovery_ratio=pos['position'], recovery_ci95=pos['ci95'],
                                   interpretable=pos['interpretable'],
                                   correct_only_minus_summary_pp=-100 * g['mean'],
                                   correct_only_minus_summary_ci95_pp=[-100 * g['ci95'][1], -100 * g['ci95'][0]],
                                   p_wilcoxon=g['p_wilcoxon'], p_holm=g.get('p_holm'))
    return out


def f5():
    out = {}
    rows = [json.loads(l) for l in (HERE / 'runs_api/planA_gpt6astra.jsonl').open()]
    for cond in ('raw', 'correct_only', 'summary', 'all'):
        rr = [r for r in rows if cond == 'all' or r['condition'] == cond]
        tok = np.array([r['usage']['output_tokens'] for r in rr])
        rea = np.array([r['usage'].get('output_tokens_details', {}).get('reasoning_tokens', 0) or 0 for r in rr])
        out.setdefault('GPT-6-Astra', {})[cond] = dict(
            n=len(rr), median=float(np.median(tok)), p90=float(np.quantile(tok, .9)), p95=float(np.quantile(tok, .95)),
            p99=float(np.quantile(tok, .99)), max=int(tok.max()), share_nonzero_reasoning=float(np.mean(rea > 0)),
            max_reasoning=int(rea.max()), share_over_5_tokens=float(np.mean(tok > 5)),
            effort=sorted({r['effort'] for r in rr}))
    rows = [json.loads(l) for l in (HERE / 'runs_api/planA_fable52.jsonl').open()]
    for cond in ('raw', 'correct_only', 'summary', 'all'):
        rr = [r for r in rows if cond == 'all' or r['condition'] == cond]
        vis = np.array([r['parse'] == 'final_line' for r in rr])
        tok = np.array([r['usage']['output_tokens'] for r in rr])
        out.setdefault('Claude Fable 5.2', {})[cond] = dict(
            n=len(rr), share_visible_reasoning=float(vis.mean()),
            parse_counts={p: int(sum(r['parse'] == p for r in rr)) for p in sorted({r['parse'] for r in rr})},
            median_output_tokens=float(np.median(tok)), p95_output_tokens=float(np.quantile(tok, .95)))
    out['definition'] = ('visible reasoning: parse == final_line (text before a final one-word answer line); '
                         'Astra output_tokens include reasoning tokens')
    return out


def f6():
    h = json.loads((HERE / 'psych101/analysis_human.json').read_text())['sources']['human/short100']['tests']['xiong_mf.phi']
    d = json.loads((HERE / 'analysis/h200_ho/psych101.json').read_text())['human_model_comparisons']['by_model']
    out = dict(human=dict(median_slope=h['median_slope_per_unit_h'], frac_positive=h['frac_positive_slope'],
                          wilcoxon_p=h['slope_wilcoxon_p'], position_adjusted=h['fe_position_adjusted'],
                          median_by_h_position=h['median_by_h_position']), model_minus_human={})
    for cond, key in (('hidden', 'model_hidden_vs_human_told'), ('stated', 'model_told_vs_human_told')):
        vals = [(k.split('/')[1], d[k][key]) for k in sorted(d) if '/text/' in k and key in d[k]]
        for (name, v), ph in zip(vals, am.holm([v['wilcoxon_p'] for _, v in vals])):
            out['model_minus_human'].setdefault(cond, {})[name] = dict(
                median_diff=v['median_diff'], ci95=v['median_diff_ci95'], p=v['wilcoxon_p'], p_holm_within_5=ph)
    return out


def fmt_r(x, d=1, s=100):
    return f'{s * x:.{d}f}'


def render(res):
    L = ['# Final numeric checks (existing outputs only)', '']
    L.append('## F1 Following the new better advisor after A became worse (share, ties half)')
    L.append(f"A's lead at onset: {res['F1']['lead_at_onset']}")
    for stat in ('trial10', 'trials8_10'):
        L.append(f'\n### {stat}')
        L.append('| Condition | all | volatile-first (lead 6) | stable-first (lead 24) | t=11 | t=31 | t=41 | t=61 | stable-volatile [95% CI] |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for cond, v in res['F1']['conditions'].items():
            s = v[stat]
            cell = lambda g: (f"{fmt_r(s[g]['mean'], 0)}% ({fmt_r(s[g]['range'][0], 0)}-{fmt_r(s[g]['range'][1], 0)})"  # noqa: E731
                              if s[g]['range'][0] != s[g]['range'][1] else f"{fmt_r(s[g]['mean'], 0)}%")
            d = s['stable_minus_volatile_first']
            L.append(f"| {cond} | " + ' | '.join(cell(g) for g in ('all', 'volatile_first_lead6', 'stable_first_lead24',
                                                                     'rev_t11', 'rev_t31', 'rev_t41', 'rev_t61'))
                     + f" | {fmt_r(d['mean'], 1)} [{fmt_r(d['ci95'][0], 1)}, {fmt_r(d['ci95'][1], 1)}] pp |")
    L.append('\n## F2 Agreement with count rules (held-out, copies combined, non-tied trials)')
    for cond in ('CorrectnessOnly', 'Summary', 'Raw'):
        d = res['F2'][cond]
        for st in ('count_all', 'last5'):
            r = d[f'range_{st}']
            L.append(f"- {cond} vs {st}: kappa {r['kappa'][0]:.2f}-{r['kappa'][1]:.2f}; agreement "
                     f"{fmt_r(r['agreement'][0])}-{fmt_r(r['agreement'][1])}%")
    L.append('\n## F3 Masking (held-out), pp [95% CI]')
    for name, v in res['F3'].items():
        p1 = v['P1_hide_outcome_main']
        L.append(f"- {name}: P1 main {p1['mean_pp']:+.2f} [{p1['ci95_pp'][0]:+.2f}, {p1['ci95_pp'][1]:+.2f}]; "
                 f"holds {v['holds']}")
    L.append('\n## F4 Recovery ratio')
    for name, v in res['F4'].items():
        a, b = v['twin_averaged_logit'], v['per_copy_hard_choice']
        L.append(f"- {name}: combined {a['recovery_ratio']:.3f} [{a['recovery_ci95'][0]:.2f}, {a['recovery_ci95'][1]:.2f}]; "
                 f"per copy {b['recovery_ratio']:.3f} [{b['recovery_ci95'][0]:.2f}, {b['recovery_ci95'][1]:.2f}]; "
                 f"per-copy CorrectnessOnly-Summary {b['correct_only_minus_summary_pp']:+.2f} "
                 f"[{b['correct_only_minus_summary_ci95_pp'][0]:+.2f}, {b['correct_only_minus_summary_ci95_pp'][1]:+.2f}] pp")
    L.append('\n## F5 API outputs')
    for cond, v in res['F5']['GPT-6-Astra'].items():
        L.append(f"- Astra {cond}: n={v['n']} median {v['median']:.0f}, p90 {v['p90']:.0f}, p95 {v['p95']:.0f}, "
                 f"p99 {v['p99']:.0f}, max {v['max']}; nonzero reasoning {fmt_r(v['share_nonzero_reasoning'])}%")
    for cond, v in res['F5']['Claude Fable 5.2'].items():
        L.append(f"- Fable {cond}: visible reasoning {fmt_r(v['share_visible_reasoning'])}% (n={v['n']})")
    L.append('\n## F6 Psych-101 forgetting-rate slope')
    hh = res['F6']['human']
    L.append(f"- Humans: median slope {hh['median_slope']:.2f}, positive {fmt_r(hh['frac_positive'])}%, Wilcoxon p={hh['wilcoxon_p']:.4f}; "
             f"position-adjusted b_h {hh['position_adjusted']['b_h']:.2f} {[round(x, 2) for x in hh['position_adjusted']['b_h_ci95']]}")
    for cond, dd in res['F6']['model_minus_human'].items():
        for name, v in dd.items():
            L.append(f"- {cond} {name}: {v['median_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                     f"p={v['p']:.4f} Holm(5)={v['p_holm_within_5']:.3f}")
    return '\n'.join(L) + '\n'


def main():
    res = dict(F2=f2(), F3=f3(), F4=f4(), F5=f5(), F6=f6(), F1=f1())
    OUT.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(res, indent=1, default=am._json_default)
    assert '": "/' not in txt and str(HERE) not in txt, 'absolute path in output'
    (OUT / 'final_checks.json').write_text(txt + '\n')
    (OUT / 'summary.md').write_text(render(res))
    print(render(res))


if __name__ == '__main__':
    main()
