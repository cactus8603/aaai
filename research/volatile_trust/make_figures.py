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
FORMATS = [('A|full|text', 'Raw', '#b44435'), ('A+|full|text', 'Raw + correct', '#95660e'),
           ('B|full|text', 'Correct only', '#078567'), ('C|sum_all|text', 'Summary', '#2366ab')]
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
    fig, axs = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.4))
    axes = axs.ravel()
    fig.subplots_adjust(left=.09, right=.985, bottom=.27, top=.93, hspace=.65, wspace=.38)
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
    ax.set_xticklabels(['Raw', 'Raw +\ncorr.', 'Corr.\nonly', 'Summ.'])
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
    a = ap.parse_args()
    ad, out = Path(a.analysis_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(ad)
    C = load_controls(ad)
    if not a.skip_teaser:
        fig_teaser(D, out, a.split_label)
    fig_gap(D, C, out, a.split_label)
    fig_curves(D, out, a.split_label)
    fig_kernel_models(D, out, a.split_label)
    mk = Path(a.masking)
    fig_mech(D, json.loads(mk.read_text()) if mk.exists() else {}, out, a.split_label, a.masking_label)
    fig_task(out)
    fig_human(ad, out)
    print('wrote', sorted(p.name for p in out.glob('fig_*.pdf')))



def fig_task(out):
    """Appendix: trial, schedule, and format-specific computation requirements."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scenarios import icon
    fig = plt.figure(figsize=(FULL_WIDTH, 4.05))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.13], width_ratios=[1, 1.3],
                          left=.075, right=.985, top=.94, bottom=.08, hspace=.42, wspace=.3)
    a = fig.add_subplot(gs[0, 0])
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis('off')
    a.set_title('(a) One trial (weather)', loc='left', fontweight='bold')
    a.text(0, .88, 'Day 6 of 80', color=INK2)
    for y, name, lab in ((.67, 'Joe', 'dry'), (.43, 'Tim', 'rain')):
        a.text(0, y, f'{name} forecasts:', va='center')
        ia = a.inset_axes([.45, y - .1, .13, .2])
        ia.imshow(np.asarray(icon('weather', lab, 112)))
        ia.axis('off')
    a.text(0, .20, 'Predict rain or dry, then observe feedback.', va='center')
    a.text(0, .02, 'Record feedback for subsequent trials.', color=INK2)
    b = fig.add_subplot(gs[0, 1])
    t = np.arange(1, 81)
    scheduled = np.where(t <= 40, .8, np.where(((t - 41) // 10) % 2 == 0, .2, .8))
    b.axvspan(40.5, 80.5, color=GRID, alpha=.6, lw=0)
    b.step(t, scheduled, where='mid', color=INK2, ls='--', lw=.9, label='Scheduled')
    eps = [json.loads(x) for x in open(Path(__file__).resolve().parent / 'episodes_v3/episodes.jsonl')]
    ep = next(e for e in eps if e['split'] == 'development' and e['trials'][0]['phase'] == 'stable')
    observations = np.array([tr['a_correct'] for tr in ep['trials']], float)
    cumulative = [observations[:k].mean() if k else np.nan for k in range(80)]
    b.plot(t, cumulative, color=FORMATS[3][2], lw=1.2, label='Share correct so far')
    b.legend(loc='lower center', handlelength=1.5, ncol=2, columnspacing=.9)
    b.text(20, 1.02, 'Stable', ha='center', color=INK2)
    b.text(60, 1.02, 'Volatile', ha='center', color=INK2)
    b.set_ylim(0, 1.16)
    b.set_xlim(.5, 80.5)
    b.set_yticks([0, .5, 1])
    b.set_xticks([1, 20, 40, 60, 80])
    b.set_xlabel('Trial')
    b.set_ylabel('Advisor A accuracy')
    b.set_title('(b) Example schedule (stable first)', loc='left', fontweight='bold')
    grid(b)
    c = fig.add_subplot(gs[1, :])
    c.axis('off')
    c.set_xlim(0, 1)
    c.set_ylim(0, 1)
    c.set_title('(c) History formats and steps left to the model', loc='left', fontweight='bold')
    rows = [('Raw', 'Joe: rain; Tim: dry; actual: rain', (1, 1, 1), FORMATS[0][2]),
            ('Raw + correctness', '...rain (Joe correct; Tim wrong)', (0, 1, 1), FORMATS[1][2]),
            ('Correctness only', 'Joe correct; Tim wrong', (0, 1, 1), FORMATS[2][2]),
            ('Summarized', '5 days: Joe correct 4, Tim 1', (0, 0, 1), FORMATS[3][2]),
            ('Described accuracy', 'Joe correct on 80% of days', (0, 0, 1), INK2)]
    for x, h in ((.02, 'Format'), (.285, 'Example (abbreviated)'), (.82, '(i)'), (.9, '(ii)'), (.98, '(iii)')):
        c.text(x, .98, h, color=INK2, va='top', ha='left' if x < .8 else 'center')
    for i, (name, example, steps, col) in enumerate(rows):
        y = .78 - i * .15
        c.plot(0, y, marker='s', color=col, ms=5, clip_on=False)
        c.text(.02, y, name, va='center')
        c.text(.285, y, example, va='center')
        for j, on in enumerate(steps):
            c.plot(.82 + j * .08, y, marker='o', ms=5, mfc=INK if on else 'white',
                   mec=INK2, mew=.7, clip_on=False)
    c.text(0, -.06, 'Filled: step left to the model. (i) Judge correctness; (ii) integrate history; (iii) use estimate.',
           color=INK2, va='top')
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
    a.set_xticklabels(['Raw', 'Raw +\ncorrect', 'Correct\nonly', 'Summary'])
    a.set_xlim(-0.45, 3.45)
    a.set_ylim(0.41, 0.69)
    a.set_ylabel('Accuracy')
    a.set_title(f'(a) Main task ({split_label})', loc='left', fontweight='bold')
    grid(a)

    conds = [('explicit_rel', 'Percent'), ('explicit_counts', 'Counts'),
             ('single_trial_who', 'Single\ntrial'), ('easy_short_coded', 'Short\ncorrect'),
             ('easy_short', 'Short\nraw'), ('easy_long_coded', 'Long\ncorrect'), ('easy_long', 'Long\nraw')]
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


STRATS = [('copy_prev_truth', 'Repeat'), ('follow_prev_correct', 'WSLS'), ('last5', 'Last 5'),
          ('count_all', 'Count'), ('bocpd', 'BOCPD'), ('label_prior', 'Label')]
MASK = [('A+', 'Raw +\ncorr.'), ('out_ok', 'Reports\nmasked'), ('rep_ok', 'Outcome\nmasked'),
        ('mask_ok', 'Both\nmasked'), ('B', 'Corr.\nonly')]


def arm_entry(block, arm):
    return next((v for k, v in block.items() if k.split('/')[-1] == arm), None)


def fig_mech(D, masking, out, split_label, mask_label):
    """Fig. 3: (a) weight on past correctness by format (mean over models, thin lines per model, direct labels);
    (b) agreement (kappa) with simple strategies under raw histories; (c) masking the raw history with correctness."""
    keys = [k for k in D if D[k]['mech']]
    fig = plt.figure(figsize=(FULL_WIDTH, 3.65))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1.0, .85],
                          left=.15, right=.985, bottom=.23, top=.93, wspace=.3, hspace=.75)
    a = fig.add_subplot(gs[0, 0])
    for arm, lab, col in FORMATS:
        ks = [arm_entry(D[k]['mech']['kernels'], arm) for k in keys]
        ks = [np.array(k['kernel'][:10]) for k in ks if k]
        if not ks:
            continue
        K = np.vstack(ks)
        x = np.arange(1, K.shape[1] + 1)
        for row in K:
            a.plot(x, row, color=col, lw=0.4, alpha=0.3)
        a.plot(x, K.mean(0), color=col, lw=1.6, label=lab)
    a.legend(loc='upper right', ncol=2, handlelength=1.1, labelspacing=.25, columnspacing=.7)
    a.axhline(0, color=INK2, lw=0.5)
    a.set_xlabel('Trials back')
    a.set_ylabel('Correctness weight')
    a.set_title('(a) Integration kernel', loc='left', fontweight='bold')
    a.set_xticks([1, 5, 10])
    a.set_xlim(0.7, 10.3)
    a.set_yticks([0, .4, .8])
    grid(a)

    b = fig.add_subplot(gs[1, 0])
    M = np.full((len(keys), len(STRATS)), np.nan)
    for i, k in enumerate(keys):
        h = arm_entry(D[k]['mech']['heuristics'], FORMATS[0][0])
        if h:
            for j, (st, _) in enumerate(STRATS):
                M[i, j] = h['all']['strategies'][st]['kappa']
    lim = max(0.4, np.nanmax(np.abs(M)))
    b.imshow(M, cmap='PuOr_r', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                b.text(j, i, f'{M[i, j]:.2f}'.replace('0.', '.').replace('-.', '−.'), ha='center',
                       va='center', color='white' if abs(M[i, j]) > 0.6 * lim else INK)
    b.set_xticks(range(len(STRATS)))
    b.set_xticklabels([l for _, l in STRATS])
    b.set_yticks(range(len(keys)))
    b.set_yticklabels([D[k]['name'] for k in keys])
    b.tick_params(length=0)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_title("(b) Strategy agreement ($\\kappa$)", loc='left', fontweight='bold')

    c = fig.add_subplot(gs[:, 1])
    for i, d in enumerate(D.values()):
        r = masking.get(d['name'])
        if r is None:
            continue
        ys = [r['acc'].get(k, np.nan) for k, _ in MASK]
        xs = np.arange(len(MASK)) + (i - (len(masking) - 1) / 2) * 0.1
        c.plot(xs, ys, color=d['color'], lw=0.7, alpha=0.7)
        c.plot(xs, ys, **mstyle(d, 4.2))
    c.axhline(0.5, color=INK2, lw=0.6, ls=':')
    c.set_xticks(range(len(MASK)))
    c.set_xticklabels([l for _, l in MASK])
    c.set_xlim(-0.5, len(MASK) - 0.5)
    c.set_ylim(0.47, 0.67)
    c.set_ylabel('Accuracy')
    c.set_title(f'(c) Masking ({mask_label})', loc='left', fontweight='bold')
    grid(c)
    handles = [plt.Line2D([], [], color=d['color'], **{k: v for k, v in mstyle(d, 4).items() if k != 'ls'})
               for d in D.values() if d['name'] in masking]
    labels = [d['name'] for d in D.values() if d['name'] in masking]
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(.53, .005), handlelength=1.4, columnspacing=1.6)
    save_figure(fig, out, 'fig_kernel')


def fig_curves(D, out, split_label):
    """Fig. 4 (single column): following the better advisor after a reversal; mean over models (thick) and
    models (thin), by history format, with direct labels."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    fig.subplots_adjust(left=.17, right=.985, top=.96, bottom=.36)
    for arm, lab, col in FORMATS:
        cs = [np.array((cell(d['main'], arm) or {}).get('curve') or [np.nan] * 10, float) for d in D.values()]
        cs = [c for c in cs if np.isfinite(c).any()]
        if not cs:
            continue
        C = np.vstack(cs)
        x = np.arange(C.shape[1])
        for row in C:
            ax.plot(x, row, color=col, lw=0.4, alpha=0.3)
        m = np.nanmean(C, 0)
        ax.plot(x, m, color=col, lw=1.6, label=lab)
    # summaries over recent trials: model means only (dashed), to keep the panel readable
    for arm, lab, col, ls in [('C|sum_cp|text', 'Change-point summ.', '#6a4fc4', (0, (3, 1.5))),
                              ('C|sum_window:5|text', 'Last-5 summ.', '#4b4a47', (0, (1, 1)))]:
        cs = [np.array((cell(d['main'], arm) or {}).get('curve') or [np.nan] * 10, float) for d in D.values()]
        cs = [c for c in cs if np.isfinite(c).any()]
        if cs:
            ax.plot(np.arange(len(cs[0])), np.nanmean(np.vstack(cs), 0), color=col, lw=1.4, ls=ls, label=lab)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .005),
               handlelength=1.5, labelspacing=.25, ncol=2, columnspacing=.9)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=':')
    ax.set_xlabel('Trials since reversal')
    ax.set_ylabel('P(follow better advisor)')
    ax.set_xlim(-0.3, 9.3)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(0, 1.0)
    grid(ax)
    save_figure(fig, out, 'fig_curves')


def fig_human(ad, out):
    """Fig. 5 (single column): median forgetting rate phi by hazard rate, humans vs models (Psych-101)."""
    pj = ad / 'psych101.json'
    if not pj.exists():
        return
    S = json.loads(pj.read_text())['sources']
    H = ['0.1', '0.2', '0.4']
    hx = [float(h) for h in H]
    fig, axs = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.05), sharey=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.25, top=.83, wspace=.18)
    hum = S['human/short100']['per_participant']['xiong_mf']
    names = {'Qwen/Qwen3-VL-8B-Instruct': 'qwen3vl8b', 'OpenGVLab/InternVL3_5-8B-HF': 'internvl8b',
             'google/gemma-3-27b-it': 'gemma3_27b', 'Qwen/Qwen3-VL-32B-Instruct': 'qwen3vl32b',
             'OpenGVLab/InternVL3_5-38B-HF': 'internvl38b'}
    style = {k: (mk, large) for k, _, mk, large in MODELS}
    for ax, cond, title in zip(axs, ['text/hidden', 'text/told'], ['(a) $h$ hidden', '(b) $h$ stated']):
        ax.plot(hx, hx, color=GRID, lw=1.0, ls='--', zorder=0)
        ax.plot(hx, [hum[h]['phi'] for h in H], color=INK, lw=1.8, marker='o', ms=3.5, label='Humans')
        for m, key in names.items():
            k = [k for k in S if k.startswith(m + '/' + cond)]
            if not k:
                continue
            pp = S[k[0]]['per_participant']['xiong_mf']
            fam = next(f for f in FAMILY if key.startswith(f))
            mk, large = style[key]
            ax.plot(hx, [pp[h]['phi'] for h in H], color=FAMILY[fam], lw=0.9, marker=mk, ms=3,
                    mfc=FAMILY[fam] if large else 'white', mec=FAMILY[fam], mew=0.8)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xticks(hx)
        ax.set_xticklabels(['.1', '.2', '.4'])
        ax.set_xlabel('Hazard rate $h$')
        grid(ax)
    axs[0].set_ylabel('Forgetting rate $\\phi$')
    axs[0].set_ylim(-0.02, 0.42)
    axs[0].text(0.4, 0.27, 'Humans', ha='right', va='bottom')
    axs[0].text(0.4, 0.06, 'VLMs', ha='right', va='bottom', color=INK2)
    save_figure(fig, out, 'fig_human')


if __name__ == '__main__':
    main()
