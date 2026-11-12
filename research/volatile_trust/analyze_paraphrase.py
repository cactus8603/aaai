"""Exploratory prompt-wording robustness check (development episodes, text reports, choice readout).

Wordings: amd = MI300X dev_* runs with the original prompts; orig = original prompts re-run via paraphrase_probe.py;
p1 = reworded sentences; p2 = terse record; H<x>-L<x>-Q<x> = component mixes (x = o original, 2 terse), where
H = task description, L = history lines and count summary, Q = current-trial lines and question.

Run locations: 3090 runs in runs/paraphrase/<wording>_<fmt>_<q|i>; MI300X runs in runs_mi300x/<model>/para_<wording>_<fmt>.
For each model x wording with all formats present: analyze_main.py and analyze_mechanism.py (exploratory --partial
is only used with --partial), then a table of accuracy per format, summary - raw gap, positions, the
copy-previous-outcome weight and kappa under the raw history, summary-rule agreement, and, where the 2x2x2 design
is complete, the main effect of switching each component to the terse version (mean over the four cells with the
component terse minus the four with it original).

Usage: python analyze_paraphrase.py [--out-dir analysis/paraphrase] [--partial] [--models qwen3vl8b ...]
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = {'qwen3vl8b': 'q', 'internvl8b': 'i'}
NAMES = {'qwen3vl8b': 'Qwen3-VL-8B', 'internvl8b': 'InternVL3.5-8B', 'gemma3_12b': 'Gemma-3-12B',
         'qwen3vl32b': 'Qwen3-VL-32B', 'internvl38b': 'InternVL3.5-38B', 'gemma3_27b': 'Gemma-3-27B',
         'qwen25vl72b': 'Qwen2.5-VL-72B'}
FORMATS = ('raw', 'raw_coded', 'coded', 'sum_all')
AMD = {'raw': 'dev_full_text', 'raw_coded': 'dev_full_rawcoded', 'coded': 'dev_full_coded', 'sum_all': 'dev_sum_all'}
ARMS = {'raw': 'A|full|text', 'raw_coded': 'A+|full|text', 'coded': 'B|full|text', 'sum_all': 'C|sum_all|text'}
MIXES = [f'H{h}-L{l}-Q{q}' for h, l, q in itertools.product('o2', repeat=3)]
CANON = {'Ho-Lo-Qo': 'orig', 'H2-L2-Q2': 'p2'}       # corners of the cube already run under these names
WORDINGS = ['amd', 'orig', 'p1', 'p2'] + [m for m in MIXES if m not in CANON]


def run_dir(model, w, f):
    if w == 'amd':
        return HERE / 'runs_mi300x' / model / AMD[f]
    if model in LOCAL and (HERE / 'runs/paraphrase' / f'{w}_{f}_{LOCAL[model]}').exists():
        return HERE / 'runs/paraphrase' / f'{w}_{f}_{LOCAL[model]}'
    return HERE / 'runs_mi300x' / model / f'para_{w}_{f}'


def by_arm(d, suffix):
    return next((v for k, v in d.items() if k.endswith('/' + suffix)), None)


def analyze(model, w, out_dir, partial):
    dirs = [run_dir(model, w, f) for f in FORMATS]
    dirs = [d for d in dirs if d.exists()]
    done = [d for d in dirs if any(d.glob('_COMPLETE*'))]
    if not dirs or (len(done) < len(dirs) and not partial) or not any('sum_all' in d.name for d in done):
        return None
    tag = f'{w}_{model}'
    extra = ['--partial'] if partial else []
    for script, t in (('analyze_main.py', 'main'), ('analyze_mechanism.py', 'mech')):
        with open(out_dir / f'log_{t}_{tag}.txt', 'w') as log:
            r = subprocess.run([sys.executable, str(HERE / script), *map(str, done if not partial else dirs),
                                '--out-dir', str(out_dir), '--tag', f'{t}_{tag}', *extra], stdout=log, stderr=log)
        if r.returncode:
            print(f'{script} failed for {tag}; see {out_dir}/log_{t}_{tag}.txt')
    mj, kj = out_dir / f'main_{tag}.json', out_dir / f'mech_{tag}.json'
    if not mj.exists():
        return None
    M = json.loads(mj.read_text())
    K = json.loads(kj.read_text()) if kj.exists() else {}
    row = dict(model=model, wording=w)
    for f, arm in ARMS.items():
        c = by_arm(M['cells'], arm + '/all')
        row[f] = c['acc'] if c else None
    g = next((x for x in M['gaps'] if x['domain'] == 'all'), None)
    if g:
        row.update(gap=g['mean'], gap_ci=g['ci95'], gap_p=g['p_wilcoxon'])
    for f in ('raw_coded', 'coded'):
        p = next((x for x in M['positions'] if x['domain'] == 'all' and x['B'] == ARMS[f]), None)
        row[f'pos_{f}'] = p['position'] if p else None
    ag = next((x for x in M.get('agreement', []) if x['domain'] == 'all'), None)
    row['rule_agree'] = ag['agree_trial_pooled'] if ag else None
    kern = by_arm(K.get('kernels', {}), ARMS['raw'])
    row['copy_w'] = kern['coef']['copy_prev_truth']['w'] if kern else None
    heur = by_arm(K.get('heuristics', {}), ARMS['raw'])
    row['copy_kappa'] = heur['all']['strategies']['copy_prev_truth']['kappa'] if heur else None
    return row


def effects(rows):
    """Main effects of H, L, Q (terse minus original) per model, when all 8 cube corners are present."""
    out = []
    for model in dict.fromkeys(r['model'] for r in rows):
        cell = {}
        for mix in MIXES:
            r = next((x for x in rows if x['model'] == model and x['wording'] == CANON.get(mix, mix)), None)
            if r:
                cell[mix] = r
        if len(cell) < 8:
            continue
        for metric in ('raw', 'coded', 'sum_all', 'gap', 'rule_agree', 'copy_w'):
            if any(cell[m].get(metric) is None for m in MIXES):
                continue
            e = dict(model=model, metric=metric)
            for i, comp in enumerate('HLQ'):
                hi = [cell[m][metric] for m in MIXES if m.split('-')[i][1] == '2']
                lo = [cell[m][metric] for m in MIXES if m.split('-')[i][1] == 'o']
                e[comp] = sum(hi) / 4 - sum(lo) / 4
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=HERE / 'analysis/paraphrase')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--models', nargs='+', default=list(NAMES))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for m in a.models for w in WORDINGS if (r := analyze(m, w, a.out_dir, a.partial))]
    eff = effects(rows)
    (a.out_dir / 'paraphrase.json').write_text(json.dumps(dict(rows=rows, effects=eff), indent=1))
    f3 = lambda x: f'{x:7.3f}' if isinstance(x, (int, float)) else f"{'-':>7}"   # noqa: E731
    lines = [f"{'model':16}{'wording':10}{'raw':>7}{'raw+c':>7}{'coded':>7}{'summ':>7}  {'summ-raw [95% CI]':24}"
             f"{'p':>9}{'pos+c':>7}{'posC':>7}{'agree':>7}{'copyW':>7}{'copyK':>7}"]
    for r in rows:
        ci = r.get('gap_ci')
        gap = f"{r['gap']:+.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if 'gap' in r else '-'
        lines.append(f"{NAMES[r['model']]:16}{r['wording']:10}" + ''.join(f3(r[f]) for f in FORMATS) + f'  {gap:24}'
                     + (f"{r['gap_p']:9.1e}" if 'gap_p' in r else f"{'-':>9}")
                     + ''.join(f3(r.get(k)) for k in ('pos_raw_coded', 'pos_coded', 'rule_agree', 'copy_w',
                                                      'copy_kappa')))
    if eff:
        lines += ['', 'Main effects of switching a component to the terse version (2x2x2; terse minus original):',
                  f"{'model':16}{'metric':12}{'H':>8}{'L':>8}{'Q':>8}"]
        lines += [f"{NAMES[e['model']]:16}{e['metric']:12}" + ''.join(f"{e[c]:+8.3f}" for c in 'HLQ') for e in eff]
    (a.out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
