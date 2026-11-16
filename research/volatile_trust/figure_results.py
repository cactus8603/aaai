"""Matched-subset accuracy and development-control figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_figures import FORMATS, FULL_WIDTH, GRID, INK, INK2, grid, save_figure


def _marker(model, color, size=4):
    return dict(marker=model['marker'], ms=size, mec=color, mew=.85,
                mfc=color if model['large'] else 'white', ls='none')


def fig_gap(D, controls, out, split_label, api=None):
    """Compare all seven systems on the same per-copy-scored subset.

    Three history-format panels share model rows and an accuracy scale. Full-set
    estimates are reported separately in the manuscript's main table.
    """
    if not api:
        raise ValueError('The redesigned results figure requires the matched API analysis; '
                         'pass --api-analysis with analyze_api_behavior.py output.')
    fig = plt.figure(figsize=(FULL_WIDTH, 2.80))
    models = list(D.values())
    if len(models) != 5:
        raise ValueError('The primary accuracy figure requires all five open VLMs')
    entries = [(d['name'], d) for d in models]
    entries += [('GPT-6-Astra', dict(marker='^', large=True)),
                ('Claude Fable 5.2', dict(marker='v', large=True))]
    arms = [('raw', 'Raw', FORMATS[0][2]),
            ('correct_only', 'Correctness-only', FORMATS[2][2]),
            ('summary', 'Summary', FORMATS[3][2])]
    fig.text(.095, .96, 'Same test trials across systems',
             fontsize=9.5, fontweight='bold')
    fig.text(.095, .895, '24 episode pairs; copies scored separately', fontsize=9, color=INK2)
    fig.text(.985, .895, 'Dashed lines: 50% chance', fontsize=9, color=INK2, ha='right')
    bs, matched_count = [], 0
    for j, (arm, label, color) in enumerate(arms):
        b = fig.add_axes([.225 + j * .26, .22, .24, .55])
        bs.append(b)
        for i, (name, d) in enumerate(entries):
            result = api['agents'][name][arm]
            mean, lo, hi = np.asarray(result['acc'], dtype=float) * 100
            if result['n_items'] != 960:
                raise ValueError(f'{name}/{arm}: expected the 960-item matched subset')
            if not np.isfinite([mean, lo, hi]).all() or not 40 <= lo <= mean <= hi <= 72:
                raise ValueError(f'{name}/{arm}: invalid estimate or CI outside the accuracy scale')
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
    fig.text(.615, .035, 'Accuracy (%)', fontsize=9, ha='center', color=INK2)

    # Do not silently omit an estimate or allow text to leave the fixed canvas.
    if matched_count != 21:
        raise ValueError('Expected 21 matched-subset estimates')
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
             ('easy_short_coded', 'Short\nCorrectness-only'), ('easy_short', 'Short\nRaw'),
             ('easy_long_coded', 'Long\nCorrectness-only'), ('easy_long', 'Long\nRaw')]
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
