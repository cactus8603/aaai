"""Draw Figure 1 from a compact, versioned snapshot of the analysis results.

From the repository root:
  .venv/bin/python research/volatile_trust/make_teaser.py --verify-sources

Omit --verify-sources when the original analysis JSON files are unavailable.
The committed snapshot is sufficient to regenerate PDF, SVG and PNG outputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from figure_pdf import save_outlined_pdf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('A|full|text', 'A+|full|text', 'B|full|text', 'C|sum_all|text')
LABELS = ('Raw', 'Raw + correctness', 'Correctness only')
COLORS = ('#9F4232', '#865B12', '#006B54')
INK, MUTED, RULE, BLUE = '#202A36', '#56616E', '#D8DDE2', '#17619C'
WIDTH, HEIGHT = 504, 302.4  # points; exactly 7 × 4.2 inches at the final paper width


def validate(data):
    """Do not silently average different sets of models, splits or episode counts."""
    if data['schema_version'] != 1 or not data['models']:
        raise ValueError('Expected a nonempty version-1 figure snapshot')
    if data['slot'] != 'all' or data['report_modality'] != 'text':
        raise ValueError('Figure 1 requires all cover-story slots with text reports')
    if data['readout'] != 'choice':
        raise ValueError('Figure 1 requires the choice readout')
    if data['aggregation'] != 'equal-weight mean over models':
        raise ValueError('Unsupported aggregation')
    keys = [m['key'] for m in data['models']]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate model in figure snapshot')
    for model in data['models']:
        for arm in ARMS:
            cell = model['cells'][arm]
            if not math.isfinite(cell['acc']) or not 0 <= cell['acc'] <= 1:
                raise ValueError(f'Invalid accuracy: {model["key"]} / {arm}')
            if cell['n_episodes'] != data['n_episode_pairs']:
                raise ValueError('All cells must use the same number of episode pairs')


def verify_sources(data):
    """Check the snapshot against the exact analysis files from which it was made."""
    for model in data['models']:
        source = ROOT / model['source']
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != model['source_sha256']:
            raise ValueError(f'Analysis source hash changed: {source}')
        prefix = model['source_cell_prefix'].split('/')
        if prefix[-3:] != [data['dataset'], data['split'], data['readout']]:
            raise ValueError(f'Dataset, split or readout mismatch: {source}')
        cells = json.loads(raw)['cells']
        for arm in ARMS:
            original = cells[f'{model["source_cell_prefix"]}/{arm}/{data["slot"]}']
            for field, value in model['cells'][arm].items():
                if original[field] != value:
                    raise ValueError(f'Snapshot differs from analysis: {source} / {arm} / {field}')


def snapshot_from_models(models, split):
    """Adapter for make_figures.py; require one matching cell per model and arm."""
    split = split.replace('-', '_')  # display label held-out corresponds to stored held_out
    rows, counts, datasets = [], set(), set()
    for key, model in models.items():
        cells = {}
        for arm in ARMS:
            matches = [(k, v) for k, v in model['main']['cells'].items()
                       if k.split('/')[-4:] == [split, 'choice', arm, 'all']]
            if len(matches) != 1:
                raise ValueError(f'Expected one {split} cell for {key} / {arm}')
            cell_key, cell = matches[0]
            datasets.add(cell_key.split('/')[-5])
            cells[arm] = {field: cell[field] for field in ('acc', 'n_episodes')}
            counts.add(cell['n_episodes'])
        rows.append(dict(key=key, label=model['name'], cells=cells))
    if len(counts) != 1 or len(datasets) != 1:
        raise ValueError('Figure 1 requires a shared dataset and episode-pair count')
    return dict(schema_version=1, split=split, slot='all', report_modality='text',
                readout='choice', dataset=datasets.pop(), n_episode_pairs=counts.pop(),
                aggregation='equal-weight mean over models', models=rows)


def draw_teaser(data, out):
    validate(data)
    values = [[model['cells'][arm]['acc'] for model in data['models']] for arm in ARMS]
    means = [sum(row) / len(row) for row in values]
    # A fixed, shared accuracy axis, with an explicit failure if future data exceed it.
    if any(not .4 <= v <= .7 for row in values for v in row):
        raise ValueError('Accuracy lies outside the .40–.70 axis; revise the axis before plotting')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 10, 'text.color': INK, 'pdf.fonttype': 42,
                'svg.fonttype': 'none', 'svg.hashsalt': 'volatile-trust-teaser',
                'savefig.bbox': None}
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        ax.axis('off')
        texts = []

        def text(x, y, label, size=10, color=INK, bold=False, ha='left', **kw):
            artist = ax.text(x, y, label, fontsize=size, color=color, ha=ha, va='center',
                             weight='bold' if bold else 'normal', **kw)
            texts.append(artist)
            return artist

        def line(x1, y1, x2, y2, color=RULE, lw=.7, **kw):
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle='butt', **kw)

        def box(x, y, w, h, fill='white', edge=RULE, lw=.7, radius=4):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle=f'round,pad=0,rounding_size={radius}',
                         fc=fill, ec=edge, lw=lw))

        def arrow(x1, y1, x2, y2):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                         mutation_scale=7, color=MUTED, lw=.8, shrinkA=0, shrinkB=0))

        def bubble(y, h, col, fill, title, excerpts):
            # Separate input messages, each with the actual history wording.
            # No answer bubbles or invented multi-turn dialogue are shown.
            x, width = 12, 300
            box(x, y, width, h, fill=fill, edge=col, lw=.75, radius=7)
            ax.add_patch(Polygon([(x + .5, y + 14), (x - 7, y + 21),
                                  (x + .5, y + 23)], closed=True,
                                 fc=fill, ec=col, lw=.75))
            line(x + .55, y + 14.3, x + .55, y + 22.6, fill, 1.7)
            text(x + 10, y + 10.5, title, 10.2, col, bold=True)
            for i, excerpt in enumerate(excerpts):
                text(x + 10, y + 25 + i * 11.8, excerpt, 9.5,
                     COLORS[2] if 'was correct' in excerpt else INK)

        # The original boxed trial sequence keeps the temporal boundary visible:
        # feedback arrives after prediction and is reused only on the next trial.
        text(2, 8, '(a) Predict first, observe feedback afterwards', 10.5, bold=True)
        box(2, 25, 128, 44, fill='#F7F8FA')
        text(12, 38, 'Ken reports: rain', 10)
        text(12, 55, 'Sam reports: dry', 10)
        arrow(134, 47, 149, 47)
        box(153, 25, 138, 44, fill='#F0F3F7', edge='#AAB5C2')
        text(222, 38, 'Model predicts', 10, bold=True, ha='center')
        text(222, 55, 'rain or dry?', 11, ha='center')
        arrow(295, 47, 310, 47)
        box(314, 25, 96, 44)
        text(362, 38, 'Then reveal', 10, bold=True, ha='center')
        text(362, 55, 'Actual: rain', 10, ha='center')
        arrow(414, 47, 429, 47)
        box(433, 25, 69, 44, fill='#F7F8FA')
        text(467.5, 38, 'Updated', 10, ha='center')
        text(467.5, 55, 'history', 10, ha='center')
        line(467.5, 70, 467.5, 82, MUTED, .7)
        line(467.5, 82, 222, 82, MUTED, .7)
        arrow(222, 82, 222, 71)
        text(345, 82, 'Reformat history for the next trial', 9, MUTED, ha='center',
             bbox=dict(fc='white', ec='none', pad=2))

        text(2, 105, '(b) Three history formats', 10.5, bold=True)
        text(2, 119, 'One past trial shown; each format is queried separately', 9, MUTED)
        bubble(131, 47, COLORS[0], '#FFF7F4', LABELS[0], (
            'Day 3: Ken forecasts rain; Sam forecasts dry.',
            'Actual weather: rain.',
        ))
        bubble(188, 59, COLORS[1], '#FFFBF1', LABELS[1], (
            'Day 3: Ken forecasts rain; Sam forecasts dry.',
            'Actual weather: rain.',
            '(Ken was correct; Sam was wrong.)',
        ))
        bubble(257, 35, COLORS[2], '#F1FAF6', LABELS[2], (
            'Day 3: Ken was correct; Sam was wrong.',
        ))

        # The teaser shows one directly labelled metric. The summary is a
        # separate reference row, not a fourth trial-level message. Model-level
        # variation and paired episode CIs are reported in Figure 2; these mean
        # points have no independently computed uncertainty intervals.
        text(328, 105, '(c) Prediction accuracy', 10.5, bold=True)
        text(328, 119, f'Mean across {len(data["models"])} open models', 9, MUTED)
        left, right, top, bottom = 334, 494, 137, 279
        x = lambda value: left + (value - .4) / .3 * (right - left)
        box(326, 244, 177, 34, fill='#F3F7FC', edge='none', radius=3)
        for tick in (.4, .5, .6, .7):
            line(x(tick), top, x(tick), bottom, RULE, .5)
            text(x(tick), 291, f'{tick:.0%}', 9, MUTED, ha='center')
        line(x(.5), top, x(.5), bottom, MUTED, .8, ls=(0, (2.5, 2.5)))
        text(x(.5), 131, 'Chance', 9, MUTED, ha='center')
        labels = (*LABELS, 'Summary (reference)')
        for j, (label, row, col, mean) in enumerate(zip(
                labels, (156, 194, 232, 270), (*COLORS, BLUE), means)):
            fill = '#F3F7FC' if j == 3 else 'white'
            text(334, row - 15, label, 9.2, col, bold=j < 3,
                 bbox=dict(fc=fill, ec='none', pad=.3))
            ax.plot(x(mean), row, 'D' if j == 3 else 'o', ms=6.8,
                    color=col, mec=col if j == 3 else 'white',
                    mfc='white' if j == 3 else col, mew=.9, zorder=4)
            text(x(mean) + 8, row, f'{mean:.1%}', 9.5, col, bold=True, zorder=5,
                 bbox=dict(fc=fill, ec='none', pad=.3))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in texts:
            if artist.get_fontsize() < 9:
                raise ValueError(f'Figure text smaller than 9 pt: {artist.get_text()}')
            bounds = artist.get_window_extent(renderer)
            if (bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width
                    or bounds.y1 > fig.bbox.height):
                raise ValueError(f'Text outside canvas: {artist.get_text()}')
        for extension in ('pdf', 'svg', 'png'):
            metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
            if extension == 'svg':
                metadata = {'Date': None}
            path = out / f'fig_teaser.{extension}'
            if extension == 'pdf':
                save_outlined_pdf(fig, path, dpi=300, metadata=metadata)
            else:
                fig.savefig(path, dpi=300, metadata=metadata)
            if extension == 'svg':
                # Matplotlib leaves spaces at the ends of multiline path commands.
                path.write_text('\n'.join(row.rstrip() for row in path.read_text().splitlines()) + '\n')
        plt.close(fig)
    return dict(zip(ARMS, means))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('teaser_data.json'))
    parser.add_argument('--out', type=Path, default=ROOT / 'paper/figures')
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    validate(data)
    if args.verify_sources:
        verify_sources(data)
    print(json.dumps(draw_teaser(data, args.out), indent=2))


if __name__ == '__main__':
    main()
