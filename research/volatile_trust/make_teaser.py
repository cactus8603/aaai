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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('A|full|text', 'A+|full|text', 'B|full|text', 'C|sum_all|text')
LABELS = ('Raw', 'Raw + correctness', 'Correctness only')
COLORS = ('#9F4232', '#865B12', '#006B54')
INK, MUTED, RULE, BLUE = '#202A36', '#56616E', '#D8DDE2', '#17619C'
WIDTH, HEIGHT = 504, 344  # points; exactly the paper's seven-inch text width


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
        raise ValueError('Accuracy lies outside the 40–70% axis; revise the axis before plotting')
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

        # (a) A concrete trial, with the prediction before any outcome feedback.
        text(2, 8, '(a) Predict first, observe feedback afterwards', 10.5, bold=True)
        box(2, 25, 128, 44, fill='#F7F8FA')
        text(12, 38, 'Ken reports: rain', 10)
        text(12, 55, 'Sam reports: dry', 10)
        arrow(134, 47, 149, 47)
        box(153, 25, 138, 44, fill='#F0F3F7', edge='#AAB5C2')
        text(222, 38, 'Frozen VLM predicts', 10, bold=True, ha='center')
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

        # (b) Aligned schematic excerpts. Struck text is an editorial omission cue,
        # never a claim about tokens included in the correctness-only prompt.
        text(2, 103, '(b) Reformat the same past feedback', 10.5, bold=True)
        text(502, 103, 'One past trial shown', 9, MUTED, ha='right')
        lefts, centers = (2, 173, 344), (81, 252, 423)
        removed = []
        for j, (x, col, label) in enumerate(zip(lefts, COLORS, LABELS)):
            box(x, 117, 158, 91, fill='white', edge=col, lw=.8)
            text(x + 10, 131, label, 11, col, bold=True)
            line(x + 10, 141, x + 148, 141, RULE)
            report = text(x + 10, 154, 'Ken: rain; Sam: dry.', 10, MUTED if j == 2 else INK)
            outcome = text(x + 10, 170, 'Outcome: rain.', 10, MUTED if j == 2 else INK)
            if j > 0:
                box(x + 6, 181, 146, 21, fill='#EAF4EF', edge='none', radius=2)
                text(x + 10, 191.5, 'Ken correct; Sam wrong.', 10, COLORS[2])
            if j == 2:
                removed += [report, outcome]

        # (c) Equal-weight model means and the underlying model values. The blue
        # reference uses exactly those models in C|sum_all|text, not a count agent.
        text(2, 227, '(c) Accuracy across four VLMs' if len(data['models']) == 4
             else f'(c) Accuracy across {len(data["models"])} VLMs', 10.5, bold=True)
        ax.plot(349, 227, 'o', ms=6, color=INK)
        text(358, 227, 'Mean', 9, MUTED)
        ax.plot(400, 227, 'o', ms=2.8, color=INK)
        text(407, 227, 'Individual model', 9, MUTED)
        y = lambda v: 314 - (v - .4) * 210
        for tick in (.4, .5, .6, .7):
            text(27, y(tick), f'{tick:.0%}', 9, MUTED, ha='right')
            if tick in (.4, .6):
                line(36, y(tick), 502, y(tick), '#E5E8EB', .55)
        line(36, y(.5), 502, y(.5), MUTED, .65, ls=(0, (1, 2.5)))
        text(498, y(.5) - 6, 'Chance', 9, MUTED, ha='right',
             bbox=dict(fc='white', ec='none', pad=1))
        line(36, y(means[3]), 502, y(means[3]), BLUE, .9, ls=(0, (4, 3)))
        text(42, y(means[3]) - 9, f'Count-summary mean: {means[3]:.1%}', 9, BLUE,
             bbox=dict(fc='white', ec='none', pad=1))
        for x, col, vals, mean in zip(centers, COLORS, values, means):
            offsets = [0] if len(vals) == 1 else [-18 + 36 * i / (len(vals) - 1) for i in range(len(vals))]
            ax.plot([x + off for off in offsets], [y(v) for v in vals], 'o', ms=2.8,
                    color=col, ls='none', zorder=3)
            ax.plot(x, y(mean), 'o', ms=7, color=col, mec='white', mew=.9, zorder=4)
            text(x, y(mean) - 13, f'{mean:.1%}', 12, col, bold=True, ha='center',
                 bbox=dict(fc='white', ec='none', pad=.6), zorder=5)
        for x, col, label in zip(centers, COLORS, LABELS):
            text(x, 326, label, 10, col, ha='center')
        split_label = data['split'].replace('_', '-')
        text(502, 338, f'Text reports · {data["n_episode_pairs"]} episode pairs · {split_label}',
             9, MUTED, ha='right')

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in removed:
            bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
            line(bounds.x0, artist.get_position()[1], bounds.x1, artist.get_position()[1], MUTED, .7)
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
