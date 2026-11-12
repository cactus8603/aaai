"""Exploratory prompt-wording robustness check (development episodes, text reports, choice readout).

For each model and wording (orig = runner prompts on this machine; p1, p2 = paraphrase_probe.py wordings; amd = the
MI300X development runs with the original wording, if present), runs analyze_main.py and analyze_mechanism.py on
the four history formats and tabulates: accuracy per format, summary - raw gap (Wilcoxon), position of the
raw+correctness and correctness-coded formats within that gap, the copy-previous-outcome weight under the raw
history, and agreement (kappa) of raw-history choices with copying the previous outcome.

Usage: python analyze_paraphrase.py [--out-dir analysis/paraphrase] [--partial]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs/paraphrase'
MODELS = {'q': ('qwen3vl8b', 'Qwen3-VL-8B'), 'i': ('internvl8b', 'InternVL3.5-8B')}
FORMATS = ('raw', 'raw_coded', 'coded', 'sum_all')
AMD = {'raw': 'dev_full_text', 'raw_coded': 'dev_full_rawcoded', 'coded': 'dev_full_coded', 'sum_all': 'dev_sum_all'}
ARMS = {'raw': 'A|full|text', 'raw_coded': 'A+|full|text', 'coded': 'B|full|text', 'sum_all': 'C|sum_all|text'}


def run_dirs(m, w):
    if w == 'amd':
        return [HERE / 'runs_mi300x' / MODELS[m][0] / AMD[f] for f in FORMATS]
    return [RUNS / f'{w}_{f}_{m}' for f in FORMATS]


def by_arm(d, arm):
    return next((v for k, v in d.items() if k.endswith('/' + arm)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=HERE / 'analysis/paraphrase')
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in MODELS:
        for w in ('amd', 'orig', 'p1', 'p2'):
            dirs = run_dirs(m, w)
            if not all((d / '_COMPLETE_shard0of1').exists() for d in dirs) and not a.partial:
                print(f'skip {m} {w}: incomplete')
                continue
            dirs = [d for d in dirs if d.exists()]
            if not dirs:
                continue
            tag = f'{w}_{MODELS[m][0]}'
            extra = ['--partial'] if a.partial else []
            for script, t in (('analyze_main.py', 'main'), ('analyze_mechanism.py', 'mech')):
                with open(a.out_dir / f'log_{t}_{tag}.txt', 'w') as log:
                    r = subprocess.run([sys.executable, str(HERE / script), *map(str, dirs), '--out-dir',
                                        str(a.out_dir), '--tag', f'{t}_{tag}', *extra], stdout=log, stderr=log)
                if r.returncode:
                    print(f'{script} failed for {tag}; see log_{t}_{tag}.txt')
            mj, kj = a.out_dir / f'main_{tag}.json', a.out_dir / f'mech_{tag}.json'
            if not mj.exists():
                continue
            M = json.loads(mj.read_text())
            K = json.loads(kj.read_text()) if kj.exists() else {}
            cells = M['cells']
            row = dict(model=MODELS[m][1], wording=w)
            for f, arm in ARMS.items():
                c = by_arm(cells, arm + '/all')
                row[f] = c['acc'] if c else None
            g = next((x for x in M['gaps'] if x['domain'] == 'all'), None)
            if g:
                row.update(gap=g['mean'], gap_ci=g['ci95'], gap_p=g['p_wilcoxon'])
            for f in ('raw_coded', 'coded'):
                p = next((x for x in M['positions'] if x['domain'] == 'all' and x['B'] == ARMS[f]), None)
                row[f'pos_{f}'] = p['position'] if p else None
            kern = by_arm(K.get('kernels', {}), ARMS['raw'])
            row['copy_w'] = kern['coef']['copy_prev_truth']['w'] if kern else None
            heur = by_arm(K.get('heuristics', {}), ARMS['raw'])
            if heur:
                row['copy_kappa'] = heur['all']['strategies']['copy_prev_truth']['kappa']
            rows.append(row)
    (a.out_dir / 'paraphrase.json').write_text(json.dumps(rows, indent=1))
    hdr = f"{'model':16}{'wording':8}{'raw':>7}{'raw+c':>7}{'coded':>7}{'summ':>7}  {'summ-raw [95% CI]':24}{'p':>9}" \
          f"{'pos+c':>7}{'posC':>7}{'copyW':>7}{'copyK':>7}"
    lines = [hdr]
    f3 = lambda x: f'{x:7.3f}' if isinstance(x, (int, float)) else f"{'-':>7}"   # noqa: E731
    for r in rows:
        ci = r.get('gap_ci')
        gap = f"{r['gap']:+.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if 'gap' in r else '-'
        lines.append(f"{r['model']:16}{r['wording']:8}" + ''.join(f3(r[f]) for f in FORMATS) + f'  {gap:24}'
                     + (f"{r['gap_p']:9.1e}" if 'gap_p' in r else f"{'-':>9}")
                     + ''.join(f3(r.get(k)) for k in ('pos_raw_coded', 'pos_coded', 'copy_w', 'copy_kappa')))
    (a.out_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
