"""All-trial readout sensitivity from existing complete held-out logits (CPU only).

Compare the primary estimator (average counterbalanced logits, then hard choice)
with hard choices scored separately for each copy, then averaged within the pair.
Both use t >= 2, equal pet/scenario weights and the same episode-pair bootstrap.
Candidate mass is descriptive only: no item is selected or dropped by mass.

Usage: .venv/bin/python research/volatile_trust/analyze_readout_sensitivity.py
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_main as am  # noqa: E402

MODELS = {'Qwen3-VL-8B': 'qwen3vl8b', 'InternVL3.5-8B': 'internvl8b',
          'Gemma-3-27B': 'gemma3_27b', 'Qwen3-VL-32B': 'qwen3vl32b',
          'InternVL3.5-38B': 'internvl38b'}
RUNS = {'raw': 'ho_full_text', 'correct_only': 'ho_full_coded', 'summary': 'ho_sum_all'}
ARMS = {'raw': 'A|full|text', 'correct_only': 'B|full|text', 'summary': 'C|sum_all|text'}
METHODS = ('twin_averaged_logit', 'per_copy_hard_choice')
CONTRASTS = {'correct_only_minus_raw': ('raw', 'correct_only'),
             'summary_minus_raw': ('raw', 'summary'),
             'summary_minus_correct_only': ('correct_only', 'summary')}


def attach_per_copy_accuracy(rows, paired):
    """Attach mean of the two individually scored copies to validated paired trials."""
    copies = defaultdict(dict)
    for r in rows:
        base, twin = am.split_twin(r['episode_id'])
        key = (r['_cell'], base, r['t'])
        if twin in copies[key]:
            raise am.AnalysisError(f'duplicate copy in readout sensitivity: {key}, twin={twin}')
        choose_a = .5 if r['_l'] == 0 else float(r['_l'] > 0)
        copies[key][twin] = choose_a * r['a_correct'] + (1 - choose_a) * (1 - r['a_correct'])
    for cell, episodes in paired.items():
        for base, trials in episodes.items():
            for tr in trials:
                scores = copies[(cell, base, tr['t'])]
                if set(scores) != {False, True}:
                    raise am.AnalysisError(f'missing counterbalanced copy: {cell}, {base}, {tr["t"]}')
                tr['per_copy_acc'] = float(np.mean(list(scores.values())))


def per_copy_metrics(trials):
    tr = [x for x in trials if x['t'] >= 2]
    if not tr:
        return None
    return dict(acc=float(np.mean([x['per_copy_acc'] for x in tr])), n_trials=len(tr))


def condition_summary(per_episode, n_boot, seed):
    values = [per_episode[b]['acc'] for b in sorted(per_episode)]
    return dict(mean=float(np.mean(values)), ci95=am.boot_mean_ci(values, n_boot, seed),
                n_pairs=len(values), n_paired_trials=sum(x['n_trials'] for x in per_episode.values()),
                per_pair={b: per_episode[b]['acc'] for b in sorted(per_episode)})


def analyze_runs(run_dirs, n_boot=10000, seed=0, expected_pairs=None):
    """Reuse main loading, completeness, twin alignment, pooling and bootstrap logic."""
    rows, info = am.load_rows(run_dirs, strict=True)
    rows = [r for r in rows if r['readout'] == 'choice' and r['_split'] == 'held_out'
            and r['_arm'] in ARMS.values()]
    if not rows or any(not r['_exact'] for r in rows):
        raise am.AnalysisError('readout sensitivity requires nonempty exact held-out choice logits')
    report = {}
    paired = am.pair_trials(rows, strict=True, report=report)
    groups = {cell[:4] for cell in paired}
    if len(groups) != 1:
        raise am.AnalysisError(f'exactly one model/dataset group required, found {groups}')
    group = next(iter(groups))
    for arm in ARMS.values():
        slots = {cell[5] for cell in paired if cell[4] == arm}
        if slots != {'pet', 'scenario'}:
            raise am.AnalysisError(f'{arm}: expected pet and scenario slots, found {slots}')
    attach_per_copy_accuracy(rows, paired)
    per = {
        METHODS[0]: am.arm_metrics(paired, group, list(ARMS.values()), 'all', strict=True),
        METHODS[1]: am.arm_metrics(paired, group, list(ARMS.values()), 'all', strict=True,
                                  metric_fn=per_copy_metrics),
    }
    if expected_pairs is not None and any(len(v) != expected_pairs for arms in per.values() for v in arms.values()):
        raise am.AnalysisError(f'expected {expected_pairs} complete episode pairs in every condition')
    results = {}
    for method, arms in per.items():
        conditions = {cond: condition_summary(arms[arm], n_boot, seed) for cond, arm in ARMS.items()}
        gaps = {name: am.paired_test(conditions[a]['per_pair'], conditions[b]['per_pair'], n_boot, seed)
                for name, (a, b) in CONTRASTS.items()}
        results[method] = dict(conditions=conditions, gaps=gaps)
    shift = {}
    for contrast in CONTRASTS:
        old, new = (results[method]['gaps'][contrast] for method in METHODS)
        shift[contrast] = am.paired_test(dict(zip(old['episodes'], old['diffs'])),
                                       dict(zip(new['episodes'], new['diffs'])), n_boot, seed)
    scored_rows = [r for r in rows if r['t'] >= 2]
    quality_by_slot = am.readout_quality(scored_rows)
    quality = {}
    for cond, arm in ARMS.items():
        entries = [r for r in quality_by_slot if r['arm'] == arm]
        mass = [float(r['candidate_mass']) for r in scored_rows if r['_arm'] == arm
                and r.get('candidate_mass') is not None]
        n_greedy = sum(r['n_greedy'] for r in entries)
        quality[cond] = dict(n_rows=sum(r['_arm'] == arm for r in scored_rows), n_mass=len(mass),
                             mass_mean=float(np.mean(mass)), mass_min=float(np.min(mass)),
                             frac_mass_lt_05=float(np.mean(np.array(mass) < .5)),
                             n_greedy=n_greedy,
                             greedy_follow_rate=(sum(r['greedy_follow_rate'] * r['n_greedy'] for r in entries) / n_greedy
                                                 if n_greedy else None), by_slot=entries)
    return dict(group=list(group), input_info=info, pairing=report,
                n_scored_copy_rows=len(scored_rows), methods=results,
                gap_shift_per_copy_minus_primary=shift, readout_quality=quality)


def render_summary(result):
    lines = ['All-trial held-out readout sensitivity; no candidate-mass selection.',
             't = 2..80; both counterbalanced copies; equal pet/scenario weights; 48 episode pairs.',
             'Accuracy %, paired gaps in percentage points; 95% percentile bootstrap CIs over pairs.',
             'Exact-zero logits receive half credit in both scoring methods.', '']
    for model, entry in result['models'].items():
        lines.append(model)
        for method, values in entry['methods'].items():
            lines.append(f'  {method}')
            for cond, x in values['conditions'].items():
                lines.append(f"    {cond:12s}: {100*x['mean']:.2f} [{100*x['ci95'][0]:.2f}, {100*x['ci95'][1]:.2f}]")
            for contrast, x in values['gaps'].items():
                lines.append(f"    {contrast}: {100*x['mean']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]")
        lines.append('  candidate mass (same all-trial mask, individual copies)')
        for cond, x in entry['readout_quality'].items():
            lines.append(f"    {cond:12s}: mean {x['mass_mean']:.3f}; mass < .5 {100*x['frac_mass_lt_05']:.2f}%; "
                         f"greedy valid candidate {100*x['greedy_follow_rate']:.2f}%; n={x['n_mass']}")
        lines.append('')
    return '\n'.join(lines).rstrip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'analysis/h200_ho/readout_sensitivity')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    result = dict(settings=dict(role='exploratory readout sensitivity', n_boot=args.n_boot, seed=args.seed,
                               mask='all complete held-out choice trials t >= 2; no mass-based selection',
                               format='text', slots=['pet', 'scenario'], expected_pairs=48,
                               aggregation='equal copies within pair, equal slots, equal episode pairs',
                               methods=dict(twin_averaged_logit='hard choice after averaging aligned twin logits (primary)',
                                            per_copy_hard_choice='hard choice per copy before averaging accuracy (sensitivity)'),
                               exact_logit_ties='half credit',
                               contrast_ci='paired percentile bootstrap over episode pairs'), models={})
    for model, key in MODELS.items():
        print(f'Analysing {model}', flush=True)
        result['models'][model] = analyze_runs([HERE / 'runs_h200' / key / run for run in RUNS.values()],
                                              args.n_boot, args.seed, expected_pairs=48)
    # Exploratory p-values: one clearly delimited five-model family per method/contrast.
    for method in METHODS:
        for contrast in CONTRASTS:
            gaps = [entry['methods'][method]['gaps'][contrast] for entry in result['models'].values()]
            for gap, p in zip(gaps, am.holm([x['p_wilcoxon'] for x in gaps])):
                gap.update(p_holm=p, holm_family=f'five models: {method}, {contrast}; exploratory')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'readout_sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = render_summary(result)
    (args.out / 'summary.txt').write_text(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main()
