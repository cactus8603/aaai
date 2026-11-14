"""Mechanism figure, paired masking effects and strategy-agreement appendices.

This module changes only the visual presentation of the stored analyses.  In
particular, masking intervals come from the episode-paired contrasts in
``analyze_masking.py``; intervals are never inferred from marginal accuracies.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Arm names describe the information *shown*.  The plot labels instead name
# the intervention, so out_ok masks reports and rep_ok masks the outcome.
INTERVENTIONS = (
    ("out_ok", "Reports masked", "#77736b", ":"),
    ("rep_ok", "Outcome masked", "#6a4fc4", "--"),
    ("mask_ok", "Both masked", "#303f49", "-"),
)


def _check_text_bounds(fig):
    """Reject clipped text on the fixed publication canvas before export."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < canvas.x0 - .25 or bounds.y0 < canvas.y0 - .25 or
                bounds.x1 > canvas.x1 + .25 or bounds.y1 > canvas.y1 + .25):
            raise ValueError(f"Text exceeds publication canvas: {artist.get_text()!r}, {bounds}")


def _masking_contrast(result, arm):
    """Return the stored intervention-minus-Raw+correctness contrast."""
    matches = [v for k, v in result["contrasts"].items()
               if k.split("(")[0].strip() == f"{arm} - A+"]
    if len(matches) != 1:
        raise ValueError(f"Expected one paired {arm} - A+ contrast, got {len(matches)}")
    contrast = matches[0]
    mean = float(contrast["mean"])
    lo, hi = map(float, contrast["ci95"])
    if not lo <= mean <= hi:
        raise ValueError(f"Invalid stored interval for {arm} - A+: {(lo, mean, hi)}")
    return 100 * mean, 100 * lo, 100 * hi


def _kappa_label(value):
    """Keep the original two-decimal precision without displaying minus zero."""
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}".replace("0.", ".").replace("-", "−")


def _strategy_appendix(D, keys, out, split_label, common):
    """Preserve every cell of the former main-figure Raw strategy heatmap."""
    matrix = np.full((len(keys), len(common.STRATS)), np.nan)
    for i, key in enumerate(keys):
        entry = common.arm_entry(D[key]["mech"]["heuristics"], common.FORMATS[0][0])
        if entry:
            for j, (strategy, _) in enumerate(common.STRATS):
                matrix[i, j] = entry["all"]["strategies"][strategy]["kappa"]

    fig = plt.figure(figsize=(common.FULL_WIDTH, 2.55))
    ax = fig.add_axes([.215, .22, .62, .65])
    lim = max(.4, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, _kappa_label(matrix[i, j]), ha="center", va="center",
                        fontsize=9.5, color="white" if abs(matrix[i, j]) > .6 * lim else common.INK)

    ax.set_xticks(range(len(common.STRATS)))
    ax.set_xticklabels([label for _, label in common.STRATS])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([D[key]["name"] for key in keys])
    ax.tick_params(length=0, pad=7)
    ax.set_xticks(np.arange(-.5, len(common.STRATS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Strategy agreement under Raw ({split_label})",
                 loc="left", fontweight="bold", pad=12)
    cax = fig.add_axes([.86, .22, .018, .65])
    cb = fig.colorbar(im, cax=cax, ticks=[-.4, -.2, 0, .2, .4])
    cb.set_label("Cohen's $\\kappa$", labelpad=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=.6)
    common.save_figure(fig, out, "fig_strategies")


def _correctness_kernel(ax, D, keys, common):
    """Retain the original fixed-model mean kernel and format encodings."""
    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "s", "D", "^"]
    format_labels = [label for _, label, _ in common.FORMATS]
    for (arm, _, color), style, marker, label in zip(
            common.FORMATS, line_styles, marker_styles, format_labels):
        kernels = [common.arm_entry(D[key]["mech"]["kernels"], arm) for key in keys]
        if any(entry is None for entry in kernels):
            raise ValueError(f"Missing {arm} kernel in the fixed model set")
        values = np.vstack([np.array(entry["kernel"][:10]) for entry in kernels])
        xs = np.arange(1, values.shape[1] + 1)
        ax.plot(xs, values.mean(axis=0), color=color, ls=style, marker=marker,
                ms=3.3, markevery=[0, 4, 9], lw=1.5, label=label)
    ax.axhline(0, color=common.INK2, lw=.65)
    ax.set_xlim(.65, 10.35)
    # The legend sits outside the data area; retain zero and every observed
    # weight without reserving empty plotting space for legend text.
    ax.set_ylim(-.03, .50)
    ax.set_xticks([1, 5, 10])
    ax.set_yticks([0, .2, .4])
    ax.set_xlabel("Trials in the past", labelpad=3)
    ax.set_ylabel("Correctness weight\n(log-odds)", labelpad=4)
    common.grid(ax)
    ax.legend(loc="center", bbox_to_anchor=(.5, 1.11), ncol=2,
              handlelength=1.5, handletextpad=.45, columnspacing=1.1,
              borderaxespad=0, labelspacing=.25)


def _masking_effects_appendix(D, masking, mask_keys, out, mask_label, common):
    """Preserve all 15 stored paired contrasts and confidence intervals."""
    fig = plt.figure(figsize=(common.FULL_WIDTH, 3.10))
    right = fig.add_axes([.245, .22, .72, .565])
    fig.text(.065, .965, "Masking effects relative to Raw+Correctness",
             fontweight="bold", fontsize=9.5, va="top")
    fig.text(.065, .902, f"Paired 95% bootstrap confidence intervals · {'test' if mask_label == 'held-out' else mask_label} episodes",
             color=common.INK2, va="top", fontsize=9)

    # Three vertically ordered contrasts per named model: reports, outcome,
    # then both.  Colours and CI line styles identify interventions; marker
    # shape/fill retains the established model encoding.
    offsets = [.23, 0, -.23]
    for i, key in enumerate(mask_keys):
        model = D[key]
        y = len(mask_keys) - 1 - i
        if i % 2 == 0:
            right.axhspan(y - .46, y + .46, color=common.GRID, alpha=.22, lw=0, zorder=0)
        for (arm, _, color, style), offset in zip(INTERVENTIONS, offsets):
            mean, lo, hi = _masking_contrast(masking[model["name"]], arm)
            right.plot([lo, hi], [y + offset, y + offset], color=color, ls=style, lw=1.15, zorder=2)
            right.plot([lo, lo], [y + offset - .055, y + offset + .055], color=color, lw=.7, zorder=2)
            right.plot([hi, hi], [y + offset - .055, y + offset + .055], color=color, lw=.7, zorder=2)
            right.plot(mean, y + offset, marker=model["marker"], ms=4.2,
                       mfc=color if model["large"] else "white", mec=color, mew=.95, ls="none", zorder=3)
    right.axvline(0, color=common.INK2, lw=.8, zorder=1)
    right.set_ylim(-.52, len(mask_keys) - .48)
    right.set_xlim(-2.6, 13.5)
    right.set_xticks([0, 5, 10])
    right.set_yticks(range(len(mask_keys)))
    right.set_yticklabels([D[key]["name"] for key in reversed(mask_keys)])
    right.tick_params(axis="y", length=0, pad=6)
    right.spines["left"].set_visible(False)
    right.set_xlabel("Accuracy change (percentage points)\nvs Raw+Correctness", labelpad=5)
    common.grid(right, axis="x")

    handles = [Line2D([], [], color=color, ls=style, lw=1.4)
               for _, _, color, style in INTERVENTIONS]
    fig.legend(handles, [label for _, label, _, _ in INTERVENTIONS],
               loc="center", bbox_to_anchor=(.59, .835), ncol=3,
               handlelength=1.4, handletextpad=.4, columnspacing=1.8, borderaxespad=0)
    _check_text_bounds(fig)
    common.save_figure(fig, out, "fig_masking_effects")


def fig_mech(D, masking, out, split_label, mask_label, *, include_strategy=True, include_effects=True):
    """Export main Figure 3 and its paired-effects and strategy appendices.

    ``D`` and ``masking`` are the existing, unchanged analysis objects. Set
    ``include_strategy=False, include_effects=False`` to regenerate only the
    main mechanism figure without modifying either appendix.
    """
    # Lazy import avoids an import-time circular dependency with make_figures.
    import make_figures as common

    keys = [key for key in D if D[key]["mech"]]
    if not keys:
        raise ValueError("The mechanism figure requires stored kernel analyses")
    mask_keys = [key for key in keys if D[key]["name"] in masking]
    if not mask_keys:
        raise ValueError("The masking figure requires stored accuracy means and paired contrasts")

    fig = plt.figure(figsize=(common.COLUMN_WIDTH, 2.80))
    left = fig.add_axes([.190, .170, .770, .600])
    fig.text(.190, .965, "Correctness integration", fontweight="bold",
             va="top", fontsize=9.5)
    _correctness_kernel(left, D, keys, common)
    _check_text_bounds(fig)
    common.save_figure(fig, out, "fig_kernel")
    if include_effects:
        _masking_effects_appendix(D, masking, mask_keys, out, mask_label, common)
    if include_strategy:
        _strategy_appendix(D, keys, out, split_label, common)
