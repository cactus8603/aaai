"""Paper figures from analysis JSON (analyze_main / analyze_mechanism / analyze_controls).

Usage:
  python make_figures.py --analysis-dir analysis/dev_v31 --out ../../paper/figures [--split-label development]
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

# models: (key, name, marker, large). Marker shape = family; filled = 27B and above, hollow = 8B.
MODELS = [('qwen3vl8b', 'Qwen3-VL-8B', 'o', False), ('internvl8b', 'InternVL3.5-8B', 's', False),
          ('gemma3_27b', 'Gemma-3-27B', 'D', True), ('qwen3vl32b', 'Qwen3-VL-32B', 'o', True),
          ('internvl38b', 'InternVL3.5-38B', 's', True)]
# family colours (Fig. 2 only; validated with the dataviz checks, all pairs, light surface)
FAMILY = {'qwen': '#2f6fd1', 'internvl': '#a3449b', 'gemma': '#e0891f'}
# history formats: (arm key, label, colour) -- colour follows the format wherever formats are coloured
FORMATS = [('A|full|text', 'Raw', '#d0443a'), ('A+|full|text', 'Raw + correct', '#e3a008'),
           ('B|full|text', 'Correct only', '#1baf7a'), ('C|sum_all|text', 'Summary', '#2a78d6')]
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
    """Marker style: shape = family, colour = family, filled = 27B and above, hollow = 8B."""
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
            ax.plot(x, c['w'], **{**mstyle(D[key], 4), 'mec': col, 'mfc': col if D[key]['large'] else 'white'})
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(FORMATS)))
    ax.set_xticklabels(['Raw', 'Raw +\ncorr.', 'Corr.\nonly', 'Summ.'])
    ax.set_title('Weight on previous outcome', loc='left')
    grid(ax)
    ax.set_ylim(-0.25, 1.15)
    h, l = axes[0].get_legend_handles_labels()
    for key in keys:
        h.append(plt.Line2D([], [], marker=D[key]['marker'], color=INK2, mfc=INK2 if D[key]['large'] else 'white', ls='none', ms=4))
        l.append(D[key]['name'])
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.2), handlelength=1.5,
               columnspacing=1.2)
    fig.savefig(out / 'fig_kernel_models.pdf', bbox_inches='tight')
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
    ap.add_argument('--analysis-dir', default=str(Path(__file__).resolve().parent / 'analysis/dev_v31'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'paper' / 'figures'))
    ap.add_argument('--split-label', default='development')
    ap.add_argument('--masking', default=str(Path(__file__).resolve().parent / 'analysis/masking/masking.json'),
                    help='analyze_masking.py output')
    ap.add_argument('--masking-label', default='development, 8B')
    a = ap.parse_args()
    ad, out = Path(a.analysis_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(ad)
    C = load_controls(ad)
    fig_teaser(D, out, a.split_label)
    fig_gap(D, C, out, a.split_label)
    fig_curves(D, out, a.split_label)
    fig_kernel_models(D, out, a.split_label)
    mk = Path(a.masking)
    fig_mech(D, json.loads(mk.read_text()) if mk.exists() else {}, out, a.split_label, a.masking_label)
    fig_task(out)
    print('wrote', sorted(p.name for p in out.glob('fig_*.pdf')))



def fig_task(out):
    """Fig. 1: (a) one trial, (b) accuracy schedule, (c) history formats and the steps left to the model."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scenarios import icon
    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 0.9, 1.85], wspace=0.25)
    # (a) one trial
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis('off')
    a.set_title('(a) One trial (weather)', loc='left')
    a.text(0.0, 0.93, 'Day 6 of 80', fontsize=7, color=INK2, va='top')
    for y, name, lab in ((0.70, 'Joe', 'dry'), (0.44, 'Tim', 'rain')):
        a.text(0.0, y, f'{name} forecasts:', fontsize=7.5, va='center')
        ia = a.inset_axes([0.68, y - 0.11, 0.22, 0.22])
        ia.imshow(np.asarray(icon('weather', lab, 112)))
        ia.axis('off')
    a.text(0.0, 0.2, 'Model predicts the\nweather = chooses\nwhom to follow', fontsize=6.8, va='center', color=INK)
    a.text(0.0, -0.02, 'Outcome revealed,\nadded to history', fontsize=6.8,
           va='bottom', color=INK2)
    # (b) schedule
    b = fig.add_subplot(gs[1])
    t = np.arange(1, 81)
    p = np.where(t <= 40, 0.8, np.where(((t - 41) // 10) % 2 == 0, 0.2, 0.8))
    b.axvspan(40.5, 80.5, color=GRID, alpha=0.6, lw=0)
    b.step(t, p, where='mid', color=INK2, ls='--', lw=0.9, label='scheduled')
    # what the summary rules see on one stable-first development episode: share of past trials A was correct
    eps = [json.loads(x) for x in open(Path(__file__).resolve().parent / 'episodes_v3/episodes.jsonl')]
    ep = next(e for e in eps if e['split'] == 'development' and e['trials'][0]['phase'] == 'stable')
    o = np.array([tr['a_correct'] for tr in ep['trials']], float)
    cum = np.array([o[:k].mean() if k else np.nan for k in range(80)])
    b.plot(t, cum, color='#2a78d6', lw=1.2, label='share correct so far\n(what the summary shows)')
    b.legend(loc='lower left', fontsize=5.6, handlelength=1.4, borderaxespad=0.2)
    b.text(20, 1.02, 'stable', ha='center', fontsize=6.8, color=INK2)
    b.text(60, 1.02, 'volatile', ha='center', fontsize=6.8, color=INK2)
    b.set_ylim(0, 1.1)
    b.set_xlim(0.5, 80.5)
    b.set_xlabel('Trial')
    b.set_ylabel('Advisor A accuracy')
    b.set_title('(b) Schedule (stable-first)', loc='left')
    grid(b)
    # (c) formats
    c = fig.add_subplot(gs[2])
    c.axis('off')
    c.set_xlim(0, 1)
    c.set_ylim(0, 1)
    c.set_title('(c) History formats', loc='left')
    rows = [('Raw', 'Joe: rain; Tim: dry; actual: rain', 'iii', '#d0443a'),
            ('Raw+correctness', '...rain (Joe correct; Tim wrong)', 'iii', '#e3a008'),
            ('Correctness only', 'Joe correct; Tim wrong', ' ii', '#1baf7a'),
            ('Summarized', '5 days: Joe correct 4, Tim 1', '  i', '#2a78d6'),
            ('Described accuracy', 'Joe correct on 80% of days', '  i', INK2)]
    hdr_y = 0.95
    for x, h in ((0.0, 'Format'), (0.355, 'Example (abbreviated)'), (0.865, '(i)'), (0.93, '(ii)'), (0.995, '(iii)')):
        c.text(x, hdr_y, h, fontsize=6.5, color=INK2, va='top', ha='left' if x < 0.8 else 'center')
    steps = {'iii': (1, 1, 1), ' ii': (0, 1, 1), '  i': (0, 0, 1)}
    for k, (name, ex, st, col) in enumerate(rows):
        y = 0.78 - k * 0.17
        c.plot([0.0], [y], marker='s', color=col, ms=5)
        c.text(0.03, y, name, fontsize=6.6, va='center')
        c.text(0.355, y, ex, fontsize=6.2, va='center', color=INK, linespacing=1.15)
        for j, on in enumerate(steps[st]):
            c.plot(0.865 + j * 0.065, y, marker='o', ms=5, color=INK if on else 'white', mec=INK2, mew=0.6, clip_on=False)
    c.text(1.0, 0.0, 'Filled: step left to the model. (i) outcome evaluation, (ii) integration, (iii) use',
           fontsize=6, color=INK2, ha='right')
    fig.savefig(out / 'fig_task.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_teaser(D, out, split_label):
    """Fig. 1: concrete feedback, aligned history excerpts, and model accuracy."""
    from make_teaser import draw_teaser, snapshot_from_models
    return draw_teaser(snapshot_from_models(D, split_label), out)


def fig_gap(D, controls, out, split_label):
    """Fig. 2: (a) main-task accuracy by history format, one line per model; (b) control items."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.22))
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
    a.text(1.5, 0.503, 'chance', va='bottom', ha='center', color=INK2, fontsize=6.0)
    if ra is not None:
        a.axhline(ra, color=INK2, lw=0.6, ls='--')
        a.text(-0.4, ra + 0.004, 'count rule', va='bottom', ha='left', color=INK2, fontsize=6.0)
    a.set_xticks(range(len(FORMATS)))
    a.set_xticklabels(['Raw', 'Raw +\ncorrect', 'Correct\nonly', 'Summary'])
    a.set_xlim(-0.45, 3.45)
    a.set_ylim(0.41, 0.69)
    a.set_ylabel('Accuracy')
    a.set_title(f'(a) Main task ({split_label})', loc='left', fontweight='bold')
    grid(a)
    model_legend(a, D, loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=5.8, handlelength=1.6,
                 labelspacing=0.25)

    conds = [('explicit_rel', 'Percent'), ('explicit_counts', 'Counts'),
             ('single_trial_who', 'Single\ntrial'), ('easy_short_coded', 'Short\ncorrect'),
             ('easy_short', 'Short\nraw'), ('easy_long_coded', 'Long\ncorrect'), ('easy_long', 'Long\nraw')]
    for i, (key, d) in enumerate(D.items()):
        vals = [controls.get(key, {}).get((c, 'text')) for c, _ in conds]
        xs = [j + offs[i] for j in range(len(conds))]
        b.plot(xs, [np.nan if v is None else v for v in vals], **mstyle(d, 4.2))
    b.axvspan(2.5, 6.5, color=GRID, alpha=0.35, lw=0)
    b.text(1.0, 0.43, 'accuracy described', ha='center', va='bottom', fontsize=6.0, color=INK2)
    b.text(4.5, 0.43, 'learned from experience', ha='center', va='bottom', fontsize=6.0, color=INK2)
    b.axhline(0.5, color=INK2, lw=0.6, ls=':')
    b.set_xticks(range(len(conds)))
    b.set_xticklabels([l for _, l in conds])
    b.set_xlim(-0.5, len(conds) - 0.5)
    b.set_ylim(0.41, 1.03)
    b.set_ylabel('Accuracy (control items)')
    b.set_title('(b) Control items (development)', loc='left', fontweight='bold')
    grid(b)
    fig.savefig(out / 'fig_gap.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


STRATS = [('copy_prev_truth', 'Repeat'), ('follow_prev_correct', 'WSLS'), ('last5', 'Last 5'),
          ('count_all', 'Count'), ('bocpd', 'BOCPD'), ('label_prior', 'Label')]
MASK = [('A+', 'Raw +\ncorrect'), ('out_ok', 'No\nreports'), ('rep_ok', 'No\noutcome'),
        ('mask_ok', 'No\nlabels'), ('B', 'Correct\nonly')]


def arm_entry(block, arm):
    return next((v for k, v in block.items() if k.split('/')[-1] == arm), None)


def fig_mech(D, masking, out, split_label, mask_label):
    """Fig. 3: (a) weight on past correctness by format (mean over models, thin lines per model, direct labels);
    (b) agreement (kappa) with simple strategies under raw histories; (c) masking the raw history with correctness."""
    keys = [k for k in D if D[k]['mech']]
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.2, 0.95], wspace=0.42)
    a = fig.add_subplot(gs[0])
    ends = []
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
    a.legend(loc='upper right', fontsize=5.6, handlelength=1.3, labelspacing=0.25)
    a.axhline(0, color=INK2, lw=0.5)
    a.set_xlabel('Trials back')
    a.set_ylabel('Weight on who was correct')
    a.set_title('(a) Integration kernel', loc='left', fontweight='bold')
    a.set_xticks([1, 5, 10])
    a.set_xlim(0.7, 10.3)
    grid(a)

    b = fig.add_subplot(gs[1])
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
                       va='center', fontsize=5.8, color='white' if abs(M[i, j]) > 0.6 * lim else INK)
    b.set_xticks(range(len(STRATS)))
    b.set_xticklabels([l for _, l in STRATS], fontsize=5.8)
    b.set_yticks(range(len(keys)))
    b.set_yticklabels([D[k]['name'] for k in keys], fontsize=5.8)
    b.tick_params(length=0)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_title("(b) Agreement with strategies ($\\kappa$)", loc='left', fontweight='bold')

    c = fig.add_subplot(gs[2])
    name2d = {d['name']: d for d in D.values()}
    for i, (name, r) in enumerate(masking.items()):
        d = name2d.get(name)
        if d is None:
            continue
        ys = [r['acc'].get(k, np.nan) for k, _ in MASK]
        xs = np.arange(len(MASK)) + (i - (len(masking) - 1) / 2) * 0.1
        c.plot(xs, ys, color=d['color'], lw=0.7, alpha=0.7)
        c.plot(xs, ys, **mstyle(d, 4.2))
    c.axhline(0.5, color=INK2, lw=0.6, ls=':')
    c.set_xticks(range(len(MASK)))
    c.set_xticklabels([l for _, l in MASK], fontsize=5.6)
    c.set_xlim(-0.5, len(MASK) - 0.5)
    c.set_ylim(0.47, 0.67)
    c.set_ylabel('Accuracy')
    c.set_title(f'(c) Masking ({mask_label})', loc='left', fontweight='bold')
    grid(c)
    model_legend(c, {k: d for k, d in D.items() if d['name'] in masking}, loc='upper left', fontsize=5.6,
                 handlelength=1.4, labelspacing=0.25)
    fig.savefig(out / 'fig_kernel.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_curves(D, out, split_label):
    """Fig. 4 (single column): following the better advisor after a reversal; mean over models (thick) and
    models (thin), by history format, with direct labels."""
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
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
    ax.legend(loc='lower right', fontsize=5.6, handlelength=1.3, labelspacing=0.25, ncol=2, columnspacing=0.8)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=':')
    ax.set_xlabel('Trials since reversal')
    ax.set_ylabel('P(follow better advisor)')
    ax.set_xlim(-0.3, 9.3)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(0.27, 0.66)
    grid(ax)
    fig.savefig(out / 'fig_curves.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == '__main__':
    main()
