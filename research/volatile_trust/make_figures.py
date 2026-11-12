"""Paper figures from analysis JSON (analyze_main / analyze_mechanism / analyze_controls).

Usage:
  python make_figures.py --analysis-dir analysis/dev3 --out ../../paper/figures [--split-label development]
Inputs: main_<model>.json, mech_<model>.json, controls.json in --analysis-dir.
Re-run unchanged on held-out analysis outputs to regenerate the final figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [('qwen3vl8b', 'Qwen3-VL-8B', 'o'), ('internvl8b', 'InternVL3.5-8B', 's'),
          ('gemma3_27b', 'Gemma-3-27B', 'D'), ('qwen3vl32b', 'Qwen3-VL-32B', '^'),
          ('internvl38b', 'InternVL3.5-38B', 'v'), ('qwen25vl72b', 'Qwen2.5-VL-72B', 'P')]
# history formats: (arm key, label, colour) -- colour follows the format in every figure
FORMATS = [('A|full|text', 'Raw', '#eb6834'), ('A+|full|text', 'Raw + correctness', '#eda100'),
           ('B|full|text', 'Correctness-coded', '#1baf7a'), ('C|sum_all|text', 'Summarized', '#2a78d6')]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'lines.linewidth': 1.4,
    'legend.frameon': False})


def cell(main, arm, slot='all'):
    for k, v in main['cells'].items():
        p = k.split('/')
        if p[-2] == arm and p[-1] == slot:
            return v
    return None


def load(ad):
    out = {}
    for key, name, mk in MODELS:
        p = ad / f'main_{key}.json'
        if p.exists():
            out[key] = dict(name=name, marker=mk, main=json.loads(p.read_text()),
                            mech=json.loads((ad / f'mech_{key}.json').read_text())
                            if (ad / f'mech_{key}.json').exists() else None)
    return out


def rule_alone(main):
    for b in main.get('baselines', []):
        t = b['table']
        return {k: t[k]['acc'] for k in ('count_all',) if k in t}
    return {}


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_gap(D, controls, out, split_label):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.28))
    n = len(D)
    offs = np.linspace(-0.18, 0.18, n) if n > 1 else [0]
    for i, (key, d) in enumerate(D.items()):
        for j, (arm, lab, col) in enumerate(FORMATS):
            c = cell(d['main'], arm)
            if c is None:
                continue
            lo, hi = c['acc_ci95']
            a.plot([j + offs[i]] * 2, [lo, hi], color=col, lw=1.0)
            a.plot(j + offs[i], c['acc'], marker=d['marker'], color=col, ms=4.5, mec='white', mew=0.6, ls='none')
    ra = rule_alone(next(iter(D.values()))['main']).get('count_all')
    a.axhline(0.5, color=INK2, lw=0.7, ls=':')
    a.text(3.45, 0.503, 'chance', va='bottom', ha='right', color=INK2, fontsize=6.3)
    if ra is not None:
        a.axhline(ra, color=INK2, lw=0.7, ls='--')
        a.text(1.0, ra + 0.004, 'summary rule alone', va='bottom', ha='center', color=INK2, fontsize=6.3)
    a.set_xticks(range(len(FORMATS)))
    a.set_xticklabels(['Raw', 'Raw +\ncorrectness', 'Correctness-\ncoded', 'Summarized'])
    a.set_xlim(-0.5, 3.5)
    a.set_ylim(0.40, 0.72)
    a.set_ylabel('Accuracy (main task)')
    a.set_title(f'(a) Main task by history format ({split_label})', loc='left')
    grid(a)
    for key, d in D.items():
        a.plot([], [], marker=d['marker'], color=INK2, ls='none', ms=4, label=d['name'])
    a.legend(loc='upper left', handletextpad=0.2, borderaxespad=0.1)

    # (b) control conditions, text reports
    conds = [('explicit_rel', 'Stated\n%'), ('explicit_counts', 'Stated\ncounts'), ('single_trial_who', 'Single\ntrial'),
             ('easy_short_coded', 'Short\ncoded'), ('easy_short', 'Short\nraw'), ('easy_long_coded', 'Long\ncoded'),
             ('easy_long', 'Long\nraw')]
    for i, (key, d) in enumerate(D.items()):
        vals = [controls.get(key, {}).get((c, 'text')) for c, _ in conds]
        xs = [j + offs[i] for j in range(len(conds))]
        b.plot(xs, [np.nan if v is None else v for v in vals], marker=d['marker'], color=INK, ms=4,
               mec='white', mew=0.6, ls='none')
    b.axhline(0.5, color=INK2, lw=0.7, ls=':')
    b.axhline(0.8, color=INK2, lw=0.7, ls='--')
    b.text(5.0, 0.805, 'prespecified gate', va='bottom', ha='center', color=INK2, fontsize=6.3)
    b.set_xticks(range(len(conds)))
    b.set_xticklabels([l for _, l in conds])
    b.set_xlim(-0.5, len(conds) - 0.5)
    b.set_ylim(0.4, 1.03)
    b.set_ylabel('Accuracy (control items)')
    b.set_title('(b) Control conditions, text reports (development)', loc='left')
    grid(b)
    fig.savefig(out / 'fig_gap.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_curves(D, out, split_label):
    keys = list(D)
    fig, axes = plt.subplots(1, len(keys), figsize=(7.0, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        d = D[key]
        for arm, lab, col in FORMATS:
            c = cell(d['main'], arm)
            if c is None:
                continue
            y = np.array(c['curve'], float)
            ax.plot(np.arange(len(y)), y, color=col, label=lab, marker='o', ms=2.5)
        ax.axhline(0.5, color=INK2, lw=0.7, ls=':')
        ax.set_title(d['name'], loc='left')
        ax.set_xlabel('Trials since reversal')
        ax.set_ylim(0.3, 0.75)
        grid(ax)
    axes[0].set_ylabel('P(follow better advisor)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.18), handlelength=1.6)
    fig.savefig(out / 'fig_curves.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_kernel(D, out, split_label):
    keys = [k for k in D if D[k]['mech']]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(7.0, 2.0),
                             gridspec_kw=dict(width_ratios=[1] * len(keys) + [1.15], wspace=0.3))
    for ax, key in zip(axes, keys):
        kern = D[key]['mech']['kernels']
        for arm, lab, col in FORMATS:
            kv = next((v for k, v in kern.items() if k.split('/')[-1] == arm), None)
            if kv is None:
                continue
            w = np.array(kv['kernel'][:10])
            lo, hi = np.array(kv['kernel_lo'][:10]), np.array(kv['kernel_hi'][:10])
            x = np.arange(1, len(w) + 1)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(x, w, color=col, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(D[key]['name'], loc='left')
        ax.set_xlabel('Trials in the past')
        ax.set_ylim(-0.25, 1.15)
        grid(ax)
    axes[0].set_ylabel('Regression weight (log-odds)')
    # last panel: weight on copying the previous outcome
    ax = axes[-1]
    for i, key in enumerate(keys):
        kern = D[key]['mech']['kernels']
        for j, (arm, lab, col) in enumerate(FORMATS):
            kv = next((v for k, v in kern.items() if k.split('/')[-1] == arm), None)
            if kv is None:
                continue
            c = kv['coef']['copy_prev_truth']
            x = j + (i - (len(keys) - 1) / 2) * 0.18
            ax.plot([x, x], c['ci95'], color=col, lw=1.0)
            ax.plot(x, c['w'], marker=D[key]['marker'], color=col, ms=4, mec='white', mew=0.6, ls='none')
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(FORMATS)))
    ax.set_xticklabels(['Raw', 'Raw +\ncorr.', 'Coded', 'Summ.'])
    ax.set_title('Weight on previous outcome', loc='left')
    grid(ax)
    ax.set_ylim(-0.25, 1.15)
    h, l = axes[0].get_legend_handles_labels()
    for key in keys:
        h.append(plt.Line2D([], [], marker=D[key]['marker'], color=INK2, ls='none', ms=4))
        l.append(D[key]['name'])
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.2), handlelength=1.5,
               columnspacing=1.2)
    fig.savefig(out / 'fig_kernel.pdf', bbox_inches='tight')
    plt.close(fig)


def load_controls(ad):
    p = ad / 'controls.json'
    if not p.exists():
        return {}
    res = {}
    for r in json.loads(p.read_text()):
        key = Path(r['run']).parent.name
        acc = res.setdefault(key, {})
        tmp = {}
        for t in r['table']:
            fmt = t['stratum'].split('/')[-1]
            if not t['n_items']:
                continue
            tmp.setdefault((t['control'], fmt), []).append((t['hard_acc'], t['n_items']))
        for k, v in tmp.items():
            acc[k] = sum(a * n for a, n in v) / sum(n for _, n in v)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis-dir', default='analysis/dev3')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'paper' / 'figures'))
    ap.add_argument('--split-label', default='development')
    a = ap.parse_args()
    ad, out = Path(a.analysis_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(ad)
    C = load_controls(ad)
    fig_gap(D, C, out, a.split_label)
    fig_curves(D, out, a.split_label)
    fig_kernel(D, out, a.split_label)
    print('wrote', sorted(p.name for p in out.glob('fig_*.pdf')))


if __name__ == '__main__':
    main()
