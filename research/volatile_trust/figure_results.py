"""Primary accuracy figures, separating full-set and matched-subset estimates."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_figures import FORMATS, FULL_WIDTH, GRID, INK, INK2, cell, grid, save_figure


def _marker(model, color, size=4):
    return dict(marker=model['marker'], ms=size, mec=color, mew=.85,
                mfc=color if model['large'] else 'white', ls='none')


def fig_gap(D, controls, out, split_label, api=None):
    """Full-set estimates above; matched per-copy subset in three format facets.

    The matched panel shares seven model rows across facets, so each row carries
    one estimate rather than three vertically offset intervals. The full-set
    panel gives each open model a separate four-row facet, with no overlaid
    model trajectories. Both panels use the same horizontal accuracy scale.
    """
    if not api:
        raise ValueError('The redesigned results figure requires the matched API analysis; '
                         'pass --api-analysis with analyze_api_behavior.py output.')
    fig = plt.figure(figsize=(FULL_WIDTH, 4.3))
    models = list(D.values())
    if len(models) != 5:
        raise ValueError('The primary accuracy figure requires all five open VLMs')
    full_count = 0
    labels = ['Raw', 'Raw + correctness', 'Correctness only', 'Summary']
    fig.text(.095, .96, '(a) Full test set: 48 episode pairs', fontsize=9.5, fontweight='bold')
    fig.text(.095, .918, 'Copies combined before scoring', fontsize=9, color=INK2)
    fig.text(.985, .918, 'Dashed lines: 50% chance', fontsize=9, color=INK2, ha='right')
    for i, d in enumerate(models):
        a = fig.add_axes([.225 + i * .154, .595, .132, .215])
        for j, (arm, label, color) in enumerate(FORMATS):
            result = cell(d['main'], arm)
            if result is None:
                raise ValueError(f'{d["name"]}/{arm}: missing full-set result')
            if result['n_episodes'] != 48:
                raise ValueError('The full-set panel requires all 48 held-out episode pairs')
            ci = np.asarray(result['acc_ci95']) * 100
            a.plot(ci, [j, j], color=color, lw=.9, zorder=2)
            # Short caps keep genuinely small CIs visible beside each mean;
            # they preserve the exact endpoints without inflating uncertainty.
            for bound in ci:
                a.plot([bound, bound], [j - .15, j + .15], color=color, lw=.7, zorder=2)
            a.plot(result['acc'] * 100, j, **_marker(d, color, 3.7), zorder=4)
            full_count += 1
        a.axvline(50, color=INK2, lw=.7, ls='--')
        a.set(yticks=range(4), yticklabels=labels if i == 0 else [],
              xlim=(40, 72), xticks=[40, 50, 60, 70], ylim=(3.5, -.5))
        a.tick_params(axis='y', length=0, pad=5)
        a.tick_params(axis='x', pad=3)
        a.set_title(d['name'], fontsize=9, fontweight='bold', pad=7)
        a.spines['left'].set_visible(False)
        grid(a, 'x')
    fig.text(.615, .523, 'Accuracy (%)', fontsize=9, ha='center', color=INK2)

    entries = [(d['name'], d) for d in models]
    entries += [('GPT-6-Astra', dict(marker='^', large=True)),
                ('Claude Fable 5.2', dict(marker='v', large=True))]
    arms = [('raw', 'Raw', FORMATS[0][2]),
            ('correct_only', 'Correctness only', FORMATS[2][2]),
            ('summary', 'Summary', FORMATS[3][2])]
    fig.text(.095, .475, '(b) Same test trials across systems',
             fontsize=9.5, fontweight='bold')
    fig.text(.095, .434, '24 episode pairs; copies scored separately', fontsize=9, color=INK2)
    fig.text(.985, .434, 'Dashed lines: 50% chance', fontsize=9, color=INK2, ha='right')
    bs, matched_count = [], 0
    for j, (arm, label, color) in enumerate(arms):
        b = fig.add_axes([.225 + j * .26, .108, .24, .27])
        bs.append(b)
        for i, (name, d) in enumerate(entries):
            result = api['agents'][name][arm]
            mean, lo, hi = np.asarray(result['acc'], dtype=float) * 100
            if result['n_items'] != 960:
                raise ValueError(f'{name}/{arm}: expected the 960-item matched subset')
            b.plot([lo, hi], [i] * 2, color=color, lw=.9, zorder=2)
            b.plot(mean, i, **_marker(d, color, 4), zorder=3)
            matched_count += 1
        b.axhspan(4.5, 6.5, color='#f3f3f0', zorder=0)
        b.axhline(4.5, color='#b4b4b0', lw=.6)
        b.axvline(50, color=INK2, lw=.7, ls='--')
        b.set(yticks=range(len(entries)),
              yticklabels=[name for name, _ in entries] if j == 0 else [],
              ylim=(len(entries) - .45, -.55), xlim=(40, 72), xticks=[40, 50, 60, 70])
        b.tick_params(axis='y', length=0, pad=5)
        b.set_title(label, color=color, fontweight='bold', pad=6)
        grid(b, 'x')
        b.spines['left'].set_visible(False)
        if j == 0:
            for tick_label in b.get_yticklabels()[-2:]:
                tick_label.set_fontweight('bold')

    # Model names are printed once for all three facets. A separate group label
    # and the divider identify API rows even when colour is unavailable.
    for label, row in [('Open', 2), ('API', 5.5)]:
        y = bs[0].transData.transform((40, row))[1] / fig.bbox.height
        fig.text(.025, y, label, fontsize=9, color=INK2, ha='left', va='center')
    fig.text(.615, .018, 'Accuracy (%)', fontsize=9, ha='center', color=INK2)

    # Do not silently omit an estimate or allow text to leave the fixed canvas.
    if (full_count, matched_count) != (20, 21):
        raise ValueError('Expected 20 full-set and 21 matched-subset estimates')
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < -.5 or bounds.y0 < -.5 or bounds.x1 > fig.bbox.width + .5
                or bounds.y1 > fig.bbox.height + .5):
            raise ValueError(f'fig_gap: text outside canvas: {artist.get_text()}')
    save_figure(fig, Path(out), 'fig_gap')


def fig_controls(D, controls, out):
    """Retain the development controls as a self-contained appendix figure."""
    conds = [('explicit_rel', 'Described\naccuracy'), ('explicit_counts', 'Described\ncounts'),
             ('single_trial_who', 'Single-trial\njudgment'),
             ('easy_short_coded', 'Short\ncorrect only'), ('easy_short', 'Short\nraw'),
             ('easy_long_coded', 'Long\ncorrect only'), ('easy_long', 'Long\nraw')]
    colors = [INK2, INK2, INK2, FORMATS[2][2], FORMATS[0][2], FORMATS[2][2], FORMATS[0][2]]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.55))
    fig.subplots_adjust(left=.09, right=.99, top=.88, bottom=.37)
    ax.axvspan(2.5, 6.5, color=GRID, alpha=.35, lw=0)
    offsets = np.linspace(-.14, .14, len(D))
    for i, (key, d) in enumerate(D.items()):
        for j, ((condition, _), color) in enumerate(zip(conds, colors)):
            value = controls.get(key, {}).get((condition, 'text'))
            if value is not None:
                ax.plot(j + offsets[i], value * 100, **_marker(d, color, 4.2))
    ax.axhline(50, color=INK2, lw=.65, ls=':')
    ax.set(xticks=range(len(conds)), xticklabels=[label for _, label in conds],
           ylim=(40, 104), yticks=[50, 75, 100], xlim=(-.5, 6.5),
           ylabel='Control-item accuracy (%)')
    ax.set_title('Development controls: described evidence and easy histories',
                 loc='left', fontweight='bold', pad=10)
    grid(ax)
    handles = [plt.Line2D([], [], color=INK2, **_marker(d, INK2)) for d in D.values()]
    fig.legend(handles, [d['name'] for d in D.values()], loc='lower center', ncol=3,
               bbox_to_anchor=(.53, .005), handlelength=.9, columnspacing=1.2)
    save_figure(fig, Path(out), 'fig_controls')
