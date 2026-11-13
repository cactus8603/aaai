"""Readable reversal trajectories and the descriptive Psych-101 comparison.

These panels use the stored analysis estimates only.  The reversal curves are
means of the five model-level curves; they do not estimate sampling uncertainty.
The Psych-101 curves and paired comparisons both use the sampled-choice fits
specified by the paper's protocol, rather than mixing soft and sampled fits.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from figure_pdf import save_outlined_pdf


FULL_WIDTH, COLUMN_WIDTH = 7.0, 3.3125
INK, MUTED, GRID = '#171717', '#777777', '#e4e3df'
MODELS = [
    ('qwen3vl8b', 'Qwen3-VL-8B', 'o', False, 'Qwen/Qwen3-VL-8B-Instruct'),
    ('internvl8b', 'InternVL3.5-8B', 's', False, 'OpenGVLab/InternVL3_5-8B-HF'),
    ('gemma3_27b', 'Gemma-3-27B', 'D', True, 'google/gemma-3-27b-it'),
    ('qwen3vl32b', 'Qwen3-VL-32B', 'o', True, 'Qwen/Qwen3-VL-32B-Instruct'),
    ('internvl38b', 'InternVL3.5-38B', 's', True, 'OpenGVLab/InternVL3_5-38B-HF'),
]
# Format colours match the other paper figures.  Marker and line styles also
# distinguish formats when printed without colour.
FORMATS = [
    ('A|full|text', 'Raw', '#b44435', '-', 'o'),
    ('A+|full|text', 'Raw + correctness', '#95660e', (0, (5, 2)), 's'),
    ('B|full|text', 'Correctness only', '#078567', '-', '^'),
    ('C|sum_all|text', 'All-trial summary', '#2366ab', '-', 'D'),
    ('C|sum_cp|text', 'Change-point', '#6a4fc4', (0, (4, 1.5)), 'v'),
    ('C|sum_window:5|text', 'Last 5 trials', '#4b4a47', (0, (1, 1)), 'P'),
]


def _style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'axes.edgecolor': '#52514e', 'axes.labelcolor': INK,
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'pdf.fonttype': 42, 'svg.fonttype': 'path', 'legend.frameon': False,
    })


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    if min(sizes, default=9) < 9:
        raise ValueError(f'{name}: text smaller than 9pt')
    save_outlined_pdf(fig, out / f'{name}.pdf', facecolor='white')
    fig.savefig(out / f'{name}.svg', facecolor='white')
    fig.savefig(out / f'{name}.png', dpi=300, facecolor='white')
    print(f'{name}: {fig.get_figwidth():g} x {fig.get_figheight():g} in; '
          f'minimum text {min(sizes):g}pt; vector outlines')
    plt.close(fig)


def _cell(main, arm):
    return next((v for k, v in main['cells'].items()
                 if k.split('/')[-2:] == [arm, 'all']), None)


def _curve(d, arm):
    values = (_cell(d['main'], arm) or {}).get('curve')
    return np.asarray(values, dtype=float) if values is not None else None


def _plot_formats(ax, series, *, markers=True):
    for (arm, label, color, ls, marker), values in zip(FORMATS, series):
        if values is None:
            continue
        ax.plot(np.arange(len(values)), values, color=color, lw=1.55, ls=ls,
                marker=marker if markers else None, ms=3.1,
                markeredgewidth=.6, mfc='white', mec=color, label=label)


def _trajectory_axes(ax):
    ax.axhline(.5, color=MUTED, lw=.75, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(-.15, 9.15)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.grid(axis='y', color=GRID, lw=.5)
    ax.set_axisbelow(True)


def fig_curves(D, out, split_label):
    """Fig. 4: six descriptive model means; per-model detail exported separately."""
    _style()
    series = []
    for arm, *_ in FORMATS:
        curves = [_curve(d, arm) for d in D.values()]
        curves = [c for c in curves if c is not None and np.isfinite(c).any()]
        series.append(np.nanmean(np.vstack(curves), axis=0) if curves else None)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    fig.subplots_adjust(left=.16, right=.985, bottom=.365, top=.95)
    _plot_formats(ax, series)
    _trajectory_axes(ax)
    ax.set_xlabel('Trials since reversal', labelpad=2)
    ax.set_ylabel('P(follow better advisor)', labelpad=2)
    ax.text(.02, .97, 'Mean across five models', transform=ax.transAxes,
            va='top', color=INK)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center',
               bbox_to_anchor=(.52, .015), handlelength=1.55,
               handletextpad=.45, columnspacing=1.0, labelspacing=.35)
    _save(fig, out, 'fig_curves')

    # Retain the original model-level information without 20 overlapping thin
    # lines in the main-text figure.  No unprovided confidence bands are added.
    fig, axs = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.985, bottom=.12, top=.94,
                        hspace=.40, wspace=.24)
    for ax, (key, label, *_rest) in zip(axs.flat, MODELS):
        if key not in D:
            ax.set_visible(False)
            continue
        _plot_formats(ax, [_curve(D[key], arm) for arm, *_ in FORMATS])
        _trajectory_axes(ax)
        ax.set_title(label, loc='left', fontweight='bold', pad=4)
    for ax in axs[:, 0]:
        ax.set_ylabel('P(follow better advisor)', labelpad=2)
    for ax in axs[-1, :2]:
        ax.set_xlabel('Trials since reversal', labelpad=3)
    axs[-1, -1].axis('off')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[-1, -1].legend(handles, labels, loc='center left', handlelength=2,
                      labelspacing=.65, handletextpad=.6, borderaxespad=.2)
    _save(fig, out, 'fig_curves_models')


def _source(sources, model_id, condition):
    prefix = f'{model_id}/text/{condition}/'
    values = [v for k, v in sources.items() if k.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f'Expected one Psych-101 source for {prefix}; got {len(values)}')
    return values[0]


def _holm(pvalues):
    """Holm adjusted p-values over the supplied family, retaining input order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ordered = np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1))
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, ordered)
    return adjusted


def fig_human(ad, out):
    """Fig. 5: sampled-choice hazard curves and paired slope CIs, descriptive."""
    _style()
    source_path = Path(ad) / 'psych101.json'
    if not source_path.exists():
        return
    sources = json.loads(source_path.read_text())['sources']
    hazards = ['0.1', '0.2', '0.4']
    hx = np.asarray([float(h) for h in hazards])
    human = sources['human/short100']['per_participant']['xiong_mf']
    model_sources = {(key, condition): _source(sources, model_id, condition)
                     for key, _, _, _, model_id in MODELS
                     for condition in ('hidden', 'told')}
    adjusted = _holm([model_sources[key, 'hidden']['model_vs_human_phi_slope']['wilcoxon_p']
                      for key, *_ in MODELS])

    fig = plt.figure(figsize=(FULL_WIDTH, 2.65))
    left = fig.add_axes([.065, .26, .19, .60])
    mid = fig.add_axes([.305, .26, .19, .60], sharey=left)
    forest = fig.add_axes([.715, .26, .27, .60])
    for ax, condition, panel in [(left, 'hidden', '(a) $h$ hidden'),
                                  (mid, 'told', '(b) $h$ stated')]:
        ax.plot(hx, hx, color='#b5b5b5', ls=(0, (4, 2)), lw=1.0)
        # Plot the models first so the human curve is unambiguous.
        for key, _, marker, filled, _ in MODELS:
            sampled = model_sources[key, condition]['sampled_fit']['median']
            ax.plot(hx, [sampled[h]['phi'] for h in hazards], color=MUTED,
                    marker=marker, ms=3.6, lw=.9, mec=MUTED, mew=.7,
                    mfc=MUTED if filled else 'white')
        ax.plot(hx, [human[h]['phi'] for h in hazards], color=INK,
                marker='o', ms=3.8, lw=1.7, zorder=8)
        ax.set_title(panel, loc='left', fontweight='bold', pad=7)
        ax.set_xlim(.075, .425)
        ax.set_xticks(hx)
        ax.set_xticklabels(['.1', '.2', '.4'])
        ax.set_xlabel('Hazard rate $h$', labelpad=3)
        ax.set_ylim(-.012, .425)
        ax.set_yticks([0, .1, .2, .3, .4])
        ax.set_yticklabels(['0', '.1', '.2', '.3', '.4'])
        ax.grid(axis='y', color=GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.text(.4, .27, 'Humans', ha='right', va='bottom', color=INK)
        ax.text(.4, .085, 'Models', ha='right', va='bottom', color=MUTED)
    left.set_ylabel('Median forgetting rate $\phi$', labelpad=2)
    mid.tick_params(axis='y', labelleft=False)

    forest.axvline(0, color=MUTED, lw=.75, ls=(0, (2, 2)), zorder=0)
    for i, (key, _, marker, filled, _) in enumerate(MODELS):
        for condition, offset, color, ls in [('hidden', .12, INK, '-'),
                                               ('told', -.12, MUTED, (0, (3, 1.5)))]:
            test = model_sources[key, condition]['model_vs_human_phi_slope']
            estimate = test['median_diff']
            lo, hi = test['median_diff_ci95']
            y = len(MODELS) - 1 - i + offset
            forest.plot([lo, hi], [y, y], color=color, lw=1.1, ls=ls)
            forest.plot([lo, lo], [y-.035, y+.035], color=color, lw=.8)
            forest.plot([hi, hi], [y-.035, y+.035], color=color, lw=.8)
            forest.plot(estimate, y, marker=marker, ms=4.4, mec=color, mew=.8,
                        mfc=color if filled else 'white', ls='none')
    forest.set_title('(c) Paired slope difference', loc='left',
                     fontweight='bold', pad=7)
    forest.set_yticks(np.arange(len(MODELS)))
    forest.set_yticklabels([label for _, label, *_ in MODELS][::-1])
    forest.tick_params(axis='y', length=0, pad=5)
    forest.set_ylim(-.48, len(MODELS)-.52)
    forest.set_xlim(-1.65, .25)
    forest.set_xticks([-1.5, -1, -.5, 0])
    forest.set_xticklabels(['-1.5', '-1.0', '-.5', '0'])
    forest.set_xlabel('Model $-$ human ($\phi$ per unit $h$)', labelpad=3)
    forest.spines['left'].set_visible(False)
    forest.grid(axis='x', color=GRID, lw=.5)
    forest.set_axisbelow(True)

    curve_handles = [Line2D([], [], color='#b5b5b5', lw=1, ls=(0, (4, 2)))]
    fig.legend(curve_handles, ['Reference: $\phi=h$'], loc='lower left',
               bbox_to_anchor=(.065, .025), handlelength=1.5, borderaxespad=0)
    contrast_handles = [Line2D([], [], color=INK, lw=1.2),
                        Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 1.5)))]
    fig.legend(contrast_handles, ['$h$ hidden', '$h$ stated'], loc='upper center',
               bbox_to_anchor=(.80, .995), ncol=2, handlelength=1.5,
               columnspacing=1, handletextpad=.45, borderaxespad=0)
    if np.all(adjusted > .05):
        inference_note = 'Unadjusted 95% CIs; all hidden $p_{\mathrm{Holm}}>.05$'
    else:
        inference_note = ('Unadjusted 95% CIs; min. hidden $p_{\mathrm{Holm}}=$'
                          + f'{adjusted.min():.3f}')
    fig.text(.535, .045, inference_note, ha='left', va='center', color=INK)
    _save(fig, out, 'fig_human')
