"""Paper figures from analysis JSON (analyze_main / analyze_mechanism / analyze_controls).

Usage:
  python make_figures.py --analysis-dir analysis/dev_v31 --out ../../paper/figures [--split-label development]
Inputs: main_<model>.json, mech_<model>.json, controls.json in --analysis-dir.
Re-run unchanged on held-out analysis outputs to regenerate the final figures.
"""
import argparse
import json
from pathlib import Path
from figure_pdf import save_outlined_pdf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# models: (key, name, marker, large). Shape = VLM series; filled = 27B and above.
MODELS = [('qwen3vl8b', 'Qwen3-VL-8B', 'o', False), ('internvl8b', 'InternVL3.5-8B', 's', False),
          ('gemma3_27b', 'Gemma-3-27B', 'D', True), ('qwen3vl32b', 'Qwen3-VL-32B', 'o', True),
          ('internvl38b', 'InternVL3.5-38B', 's', True)]
# family colours (Fig. 2 only; validated with the dataviz checks, all pairs, light surface)
FAMILY = {'qwen': '#2f6fd1', 'internvl': '#943f8e', 'gemma': '#a56710'}
# history formats: (arm key, label, colour) -- colour follows the format wherever formats are coloured
FORMATS = [('A|full|text', 'Raw', '#b44435'), ('A+|full|text', 'Raw+Correctness', '#95660e'),
           ('B|full|text', 'Correctness-only', '#078567'), ('C|sum_all|text', 'Summary', '#2366ab')]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
FULL_WIDTH, COLUMN_WIDTH = 7.0, 3.3125  # AAAI-27: 7in text, .375in gutter.

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'lines.linewidth': 1.4,
    'svg.fonttype': 'path', 'legend.frameon': False})


def save_figure(fig, out, name):
    """Export at the final included size; PDF text is vector outlines for AAAI.

    The fixed canvas must not be tightly cropped: that would change the font
    size when LaTeX rescales the PDF to columnwidth or textwidth.
    """
    from matplotlib.text import Text
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    if min(sizes, default=9) < 9:
        raise ValueError(f'{name}: text smaller than 9pt: {min(sizes)}')
    save_outlined_pdf(fig, out / f'{name}.pdf', facecolor='white')
    fig.savefig(out / f'{name}.svg', facecolor='white')
    svg = out / f'{name}.svg'
    svg.write_text('\n'.join(row.rstrip() for row in svg.read_text().splitlines()) + '\n')
    fig.savefig(out / f'{name}.png', dpi=300, facecolor='white')
    print(f'{name}: {fig.get_figwidth():.4g}in wide; minimum text {min(sizes):g}pt; vector outlines')
    plt.close(fig)


def cell(main, arm, slot='all'):
    for k, v in main['cells'].items():
        p = k.split('/')
        if p[-2] == arm and p[-1] == slot:
            return v
    return None


def load(ad):
    out = {}
    for key, name, mk, large in MODELS:
        p = ad / f'main_{key}.json'
        if p.exists():
            fam = next(f for f in FAMILY if key.startswith(f))
            out[key] = dict(name=name, marker=mk, large=large, color=FAMILY[fam], main=json.loads(p.read_text()),
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


def mstyle(d, ms=4.5):
    """Shape/colour identify the VLM series; filled = 27B and above, hollow = 8B."""
    return dict(marker=d['marker'], ms=ms, mec=d['color'], mew=0.9,
                mfc=d['color'] if d['large'] else 'white', ls='none')


def model_legend(ax, D, **kw):
    for d in D.values():
        ax.plot([], [], color=d['color'], lw=0.8, label=d['name'], **{k: v for k, v in mstyle(d, 4).items()
                                                                      if k != 'ls'})
    ax.legend(handletextpad=0.3, borderaxespad=0.1, **kw)


def fig_kernel_models(D, out, split_label):
    """Appendix: integration kernels per model and the weight on the previous outcome."""
    keys = [k for k in D if D[k]['mech']]
    fig, axs = plt.subplots(2, 3, figsize=(FULL_WIDTH, 5.0))
    axes = axs.ravel()
    fig.subplots_adjust(left=.09, right=.985, bottom=.34, top=.93, hspace=.65, wspace=.38)
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
        ax.set_yticks([0, .5, 1.0])
        ax.set_xticks([1, 5, 10])
        grid(ax)
    axes[0].set_ylabel('Weight (log-odds)')
    axes[3].set_ylabel('Weight (log-odds)')
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
            ax.plot(x, c['w'], **{**mstyle(D[key], 4), 'mec': col, 'mfc': col if D[key]['large'] else 'white'})
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(FORMATS)))
    ax.set_xticklabels(['Raw', 'Raw+\nCorrectness', 'Correctness-\nonly', 'Summary'], rotation=55, ha='right')
    ax.set_title('Previous-outcome weight', loc='left')
    grid(ax)
    ax.set_ylim(-0.25, 1.15)
    ax.set_yticks([0, .5, 1.0])
    ax.set_xlim(-.65, 3.65)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(.5, .10),
               handlelength=1.5, columnspacing=1.8)
    h, l = [], []
    for key in keys:
        h.append(plt.Line2D([], [], marker=D[key]['marker'], color=INK2, mfc=INK2 if D[key]['large'] else 'white', ls='none', ms=4))
        l.append(D[key]['name'])
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(.5, .005), handlelength=1.1,
               columnspacing=1.8)
    save_figure(fig, out, 'fig_kernel_models')


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
    ap.add_argument('--analysis-dir', default=str(Path(__file__).resolve().parent / 'analysis/dev_v31'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'paper' / 'figures'))
    ap.add_argument('--split-label', default='development')
    ap.add_argument('--masking', default=str(Path(__file__).resolve().parent / 'analysis/masking/masking.json'),
                    help='analyze_masking.py output')
    ap.add_argument('--masking-label', default='development, 8B')
    ap.add_argument('--skip-teaser', action='store_true', help='Leave the separately maintained teaser unchanged.')
    ap.add_argument('--api-analysis', default=None,
                    help='Matched per-copy analyze_api_behavior.py JSON for the redesigned results figure.')
    a = ap.parse_args()
    ad, out = Path(a.analysis_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(ad)
    C = load_controls(ad)
    if not a.skip_teaser:
        fig_teaser(D, out, a.split_label)
    from figure_results import fig_gap as results_figure, fig_controls
    from figure_mechanism import fig_mech as mechanism_figure
    from figure_dynamics import fig_curves as reversal_figure, fig_human as human_figure
    api_path = Path(a.api_analysis) if a.api_analysis else None
    if api_path is None and a.split_label == 'held-out':
        api_path = Path(__file__).resolve().parent / 'analysis/api_behavior/api_behavior.json'
    if api_path is not None:
        results_figure(D, C, out, a.split_label, json.loads(api_path.read_text()))
    else:
        fig_gap(D, C, out, a.split_label)
    fig_controls(D, C, out)
    reversal_figure(D, out, a.split_label)
    fig_kernel_models(D, out, a.split_label)
    mk = Path(a.masking)
    mechanism_figure(D, json.loads(mk.read_text()) if mk.exists() else {}, out, a.split_label, a.masking_label)
    fig_task(out)
    human_figure(ad, out)
    print('wrote', sorted(p.name for p in out.glob('fig_*.pdf')))



def fig_task(out):
    """Appendix: retain the original schedule and pre-trial cumulative accuracy."""
    from matplotlib.text import Text

    fig, b = plt.subplots(figsize=(COLUMN_WIDTH, 2.40))
    fig.subplots_adjust(left=.18, right=.98, top=.84, bottom=.36)
    t = np.arange(1, 81)
    scheduled = np.where(t <= 40, .8, np.where(((t - 41) // 10) % 2 == 0, .2, .8))
    b.axvspan(40.5, 80.5, color=GRID, alpha=.6, lw=0)
    b.step(t, scheduled, where='mid', color=INK2, ls='--', lw=.9, label='Scheduled')
    eps = [json.loads(x) for x in open(Path(__file__).resolve().parent / 'episodes_v3/episodes.jsonl')]
    ep = next(e for e in eps if e['split'] == 'development' and e['trials'][0]['phase'] == 'stable')
    observations = np.array([tr['a_correct'] for tr in ep['trials']], float)
    cumulative = [observations[:k].mean() if k else np.nan for k in range(80)]
    b.plot(t, cumulative, color=FORMATS[3][2], lw=1.2, label='Share correct so far')
    fig.legend(*b.get_legend_handles_labels(), loc='lower center', ncol=1,
               bbox_to_anchor=(.57, .025), handlelength=2,
               handletextpad=.6, labelspacing=.35, borderaxespad=0)
    b.text(20, 1.05, 'Stable', ha='center', color=INK2)
    b.text(60, 1.05, 'Volatile', ha='center', color=INK2)
    b.set_ylim(0, 1.16)
    b.set_xlim(.5, 80.5)
    b.set_yticks([0, .5, 1])
    b.set_yticklabels(['0', '50', '100'])
    b.set_xticks([1, 20, 40, 60, 80])
    b.set_xlabel('Trial', labelpad=3)
    b.set_ylabel('Advisor A accuracy (%)', labelpad=4)
    b.set_title('Example schedule (stable first)', loc='left', fontweight='bold', pad=9)
    grid(b)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < -.25 or bounds.y0 < -.25 or
                bounds.x1 > fig.bbox.width + .25 or bounds.y1 > fig.bbox.height + .25):
            raise ValueError(f'fig_task: text outside canvas: {artist.get_text()}')
    save_figure(fig, out, 'fig_task')


def fig_teaser(D, out, split_label):
    """Fig. 1: concrete feedback, aligned history excerpts, and model accuracy."""
    from make_teaser import draw_teaser, snapshot_from_models
    return draw_teaser(snapshot_from_models(D, split_label), out)


def fig_gap(D, controls, out, split_label):
    """Fig. 2: (a) main-task accuracy by history format, one line per model; (b) control items."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8),
                              gridspec_kw=dict(width_ratios=[1, 1.25]))
    fig.subplots_adjust(left=.08, right=.99, top=.89, bottom=.31, wspace=.28)
    n = len(D)
    offs = np.linspace(-0.1, 0.1, n) if n > 1 else [0]
    for i, (key, d) in enumerate(D.items()):
        xs, ys = [], []
        for j, (arm, lab, col) in enumerate(FORMATS):
            c = cell(d['main'], arm)
            if c is None:
                continue
            x = j + offs[i]
            a.plot([x, x], c['acc_ci95'], color=d['color'], lw=0.8, alpha=0.8)
            xs.append(x)
            ys.append(c['acc'])
        a.plot(xs, ys, color=d['color'], lw=0.7, alpha=0.6)
        a.plot(xs, ys, **mstyle(d, 4.2))
    ra = rule_alone(next(iter(D.values()))['main']).get('count_all')
    a.axhline(0.5, color=INK2, lw=0.6, ls=':')
    a.text(1.55, 0.503, 'chance', va='bottom', ha='center', color=INK2)
    if ra is not None:
        a.axhline(ra, color=INK2, lw=0.6, ls='--')
        a.text(-0.4, ra + 0.004, 'count rule', va='bottom', ha='left', color=INK2)
    a.set_xticks(range(len(FORMATS)))
    a.set_xticklabels(['Raw', 'Raw+\nCorrectness', 'Correctness-\nonly', 'Summary'])
    a.set_xlim(-0.45, 3.45)
    a.set_ylim(0.41, 0.69)
    a.set_ylabel('Accuracy')
    a.set_title(f'(a) Main task ({split_label})', loc='left', fontweight='bold')
    grid(a)

    conds = [('explicit_rel', 'Percent'), ('explicit_counts', 'Counts'),
             ('single_trial_who', 'Single\ntrial'), ('easy_short_coded', 'Short\nCorrectness-only'),
             ('easy_short', 'Short\nRaw'), ('easy_long_coded', 'Long\nCorrectness-only'), ('easy_long', 'Long\nRaw')]
    for i, (key, d) in enumerate(D.items()):
        vals = [controls.get(key, {}).get((c, 'text')) for c, _ in conds]
        xs = [j + offs[i] for j in range(len(conds))]
        b.plot(xs, [np.nan if v is None else v for v in vals], **mstyle(d, 4.2))
    b.axvspan(2.5, 6.5, color=GRID, alpha=0.35, lw=0)
    b.text(1.0, 0.43, 'Described / direct', ha='center', va='bottom', color=INK2)
    b.text(4.5, 0.43, 'Learned from history', ha='center', va='bottom', color=INK2)
    b.axhline(0.5, color=INK2, lw=0.6, ls=':')
    b.set_xticks(range(len(conds)))
    b.set_xticklabels([l for _, l in conds])
    b.set_xlim(-0.5, len(conds) - 0.5)
    b.set_ylim(0.41, 1.03)
    b.set_ylabel('Accuracy (control items)')
    b.set_title('(b) Control items (development)', loc='left', fontweight='bold')
    grid(b)
    handles = [plt.Line2D([], [], color=d['color'], **{k: v for k, v in mstyle(d, 4).items() if k != 'ls'})
               for d in D.values()]
    fig.legend(handles, [d['name'] for d in D.values()], loc='lower center', ncol=3,
               bbox_to_anchor=(.53, .005), handlelength=1.4, columnspacing=1.6)
    save_figure(fig, out, 'fig_gap')


STRATS = [('copy_prev_truth', 'Repeat'), ('follow_prev_correct', 'WSLS'), ('last5', 'Last-5\nrule'),
          ('count_all', 'Count\nrule'), ('bocpd', 'BOCPD'), ('label_prior', 'Label')]
MASK = [('A+', 'Raw+\nCorrectness'), ('out_ok', 'Reports\nmasked'), ('rep_ok', 'Outcome\nmasked'),
        ('mask_ok', 'Both\nmasked'), ('B', 'Correctness-\nonly')]


def arm_entry(block, arm):
    return next((v for k, v in block.items() if k.split('/')[-1] == arm), None)


def fig_mech(D, masking, out, split_label, mask_label):
    from figure_mechanism import fig_mech as draw
    return draw(D, masking, out, split_label, mask_label)


def fig_curves(D, out, split_label):
    from figure_dynamics import fig_curves as draw
    return draw(D, out, split_label)


def fig_human(ad, out):
    from figure_dynamics import fig_human as draw
    return draw(ad, out)


if __name__ == '__main__':
    main()
