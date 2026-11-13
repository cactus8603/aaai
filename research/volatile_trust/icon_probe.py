"""Exploratory (not part of the frozen plan): icon legibility probe for the medical and sports cover stories.

In the prereg-v3 runs, every model named the medical icon at chance (.50) and InternVL3.5-8B also the sports icon.
This script draws alternative text-free icons and asks the single-image perception question used by
run_controls.py ("Which <noun> does this image show? Answer with exactly one lowercase word: a or b."), with both
answer orders, so a variant can be chosen for future runs. Main-study data and results are unchanged.

Usage: python icon_probe.py --model qwen3vl8b --out runs/iconprobe [--save-png]
       python icon_probe.py --model internvl8b --out runs/iconprobe
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'source_trust'))
from scenarios import SCENARIOS, icon as icon_v3  # noqa: E402

MODELS = {
    'qwen3vl8b': ('Qwen/Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 224),
    'internvl8b': ('OpenGVLab/InternVL3_5-8B-HF', 'InternVL3_5-8B-HF/741a7d03020411e666c6109218ab71e08151ef86', 448),
}
RED, GREEN, BLUE, GOLD, GRAY = (210, 40, 40), (30, 160, 60), (40, 90, 220), (220, 175, 30), (150, 150, 150)


def canvas(P):
    img = Image.new('RGB', (P, P), 'white')
    return img, ImageDraw.Draw(img), P / 224


def m_strip_big(lab, P):
    """Enlarged lateral-flow strip: wide window, thick lines (2 lines = positive)."""
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    d.rounded_rectangle(S(40, 12, 184, 212), radius=int(16 * s), fill=(235, 235, 235), outline=(80, 80, 80),
                        width=int(4 * s))
    d.rectangle(S(62, 50, 162, 174), fill='white', outline=(120, 120, 120), width=int(3 * s))
    d.rectangle(S(66, 78, 158, 92), fill=(200, 30, 60))
    if lab == 'positive':
        d.rectangle(S(66, 132, 158, 146), fill=(200, 30, 60))
    return img


def m_plus_minus(lab, P):
    """Thick plus (positive) or minus (negative) sign, same colour."""
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    d.ellipse(S(22, 22, 202, 202), outline=(80, 80, 80), width=int(6 * s))
    d.rectangle(S(52, 98, 172, 126), fill=(60, 60, 60))
    if lab == 'positive':
        d.rectangle(S(98, 52, 126, 172), fill=(60, 60, 60))
    return img


def m_plus_minus_color(lab, P):
    """Red plus on pink (positive) vs green minus on pale green (negative)."""
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    if lab == 'positive':
        d.ellipse(S(22, 22, 202, 202), fill=(250, 215, 215))
        d.rectangle(S(52, 98, 172, 126), fill=RED)
        d.rectangle(S(98, 52, 126, 172), fill=RED)
    else:
        d.ellipse(S(22, 22, 202, 202), fill=(215, 240, 215))
        d.rectangle(S(52, 98, 172, 126), fill=GREEN)
    return img


def m_virus(lab, P):
    """Red virus particle (positive) vs empty dashed circle (negative)."""
    import math
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    if lab == 'positive':
        for k in range(12):
            a = 2 * math.pi * k / 12
            x0, y0 = 112 + 55 * math.cos(a), 112 + 55 * math.sin(a)
            x1, y1 = 112 + 88 * math.cos(a), 112 + 88 * math.sin(a)
            d.line(S(x0, y0, x1, y1), fill=RED, width=int(7 * s))
            d.ellipse(S(x1 - 9, y1 - 9, x1 + 9, y1 + 9), fill=RED)
        d.ellipse(S(57, 57, 167, 167), fill=RED)
    else:
        d.ellipse(S(57, 57, 167, 167), outline=GRAY, width=int(5 * s))
    return img


def s_trophy_sad(lab, P):
    """Gold trophy (win) vs sad face (lose)."""
    if lab == 'win':
        return icon_v3('sports', 'win', P, version='v3')
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    d.ellipse(S(32, 32, 192, 192), fill=(160, 170, 190))
    d.ellipse(S(75, 80, 95, 100), fill=(40, 40, 40))
    d.ellipse(S(129, 80, 149, 100), fill=(40, 40, 40))
    d.arc(S(70, 125, 154, 185), start=200, end=340, fill=(40, 40, 40), width=int(8 * s))
    return img


def s_faces(lab, P):
    """Happy face (win) vs sad face (lose)."""
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    d.ellipse(S(32, 32, 192, 192), fill=(250, 210, 60) if lab == 'win' else (160, 170, 190))
    d.ellipse(S(75, 80, 95, 100), fill=(40, 40, 40))
    d.ellipse(S(129, 80, 149, 100), fill=(40, 40, 40))
    if lab == 'win':
        d.arc(S(70, 95, 154, 160), start=20, end=160, fill=(40, 40, 40), width=int(8 * s))
    else:
        d.arc(S(70, 125, 154, 185), start=200, end=340, fill=(40, 40, 40), width=int(8 * s))
    return img


def s_medal(lab, P):
    """Gold medal with ribbon (win) vs grey medal with a crack (lose)."""
    img, d, s = canvas(P)
    S = lambda *xy: [v * s for v in xy]  # noqa: E731
    col = GOLD if lab == 'win' else GRAY
    d.polygon(S(80, 10, 112, 10, 125, 95, 100, 95), fill=BLUE if lab == 'win' else (110, 110, 110))
    d.polygon(S(112, 10, 144, 10, 124, 95, 99, 95), fill=RED if lab == 'win' else (130, 130, 130))
    d.ellipse(S(57, 85, 167, 195), fill=col)
    if lab == 'lose':
        d.line(S(90, 90, 118, 135, 100, 150, 130, 192), fill=(60, 60, 60), width=int(6 * s))
    else:
        d.ellipse(S(82, 110, 142, 170), outline=(180, 135, 15), width=int(5 * s))
    return img


VARIANTS = {
    'medical': {'v3_strip': lambda lab, P: icon_v3('medical', lab, P, version='v3'), 'strip_big': m_strip_big,
                'plus_minus': m_plus_minus, 'plus_minus_color': m_plus_minus_color, 'virus': m_virus},
    'sports': {'v3_trophy_x': lambda lab, P: icon_v3('sports', lab, P, version='v3'), 'trophy_sad': s_trophy_sad,
               'faces': s_faces, 'medal': s_medal},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=list(MODELS), required=True)
    ap.add_argument('--models-root', default=str(HERE.parents[1] / '.cache' / 'models'))
    ap.add_argument('--out', default=str(HERE / 'runs' / 'iconprobe'))
    ap.add_argument('--save-png', action='store_true')
    a = ap.parse_args()
    mid, sub, P = MODELS[a.model]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.save_png:
        for dom, vs in VARIANTS.items():
            for v, fn in vs.items():
                for lab in SCENARIOS[dom]['labels']:
                    fn(lab, 224).save(out / f'{dom}_{v}_{lab}.png')
    from model_backends import FrozenVLM
    vlm = FrozenVLM(str(Path(a.models_root) / sub), mid, pixels=P)
    rows = []
    for dom, vs in VARIANTS.items():
        labs = list(SCENARIOS[dom]['labels'])
        noun = SCENARIOS[dom]['noun']
        for v, fn in vs.items():
            for lab in labs:
                img = fn(lab, P)
                for order in (labs, labs[::-1]):
                    q = (f"\nWhich {noun} does this image show? Answer with exactly one lowercase word: "
                         f"{order[0]} or {order[1]}.")
                    r = vlm.score(dict(content=[{'type': 'image'}, {'type': 'text', 'text': q}], images=[img]),
                                  labs, max_new_tokens=1)
                    other = [x for x in labs if x != lab][0]
                    diff = r['logits'][lab] - r['logits'][other]
                    rows.append(dict(model=a.model, domain=dom, variant=v, label=lab, order='/'.join(order),
                                     logit_diff=diff, correct=bool(diff > 0), p_target=r['probabilities'][lab]))
    (out / f'{a.model}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    print(f'== {a.model}')
    for dom, vs in VARIANTS.items():
        for v in vs:
            rr = [x for x in rows if x['domain'] == dom and x['variant'] == v]
            print(f"{dom:8s} {v:18s} acc={sum(x['correct'] for x in rr)}/{len(rr)} "
                  f"mean p(target)={sum(x['p_target'] for x in rr) / len(rr):.2f} "
                  f"min margin={min(x['logit_diff'] for x in rr):+.2f}")


if __name__ == '__main__' and '--figure' not in sys.argv:
    main()


def make_figure(out_pdf):
    """All probed icon pairs, with readable labels at a 7-inch print width."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'v3_strip': 'test strip\n(original)', 'strip_big': 'large strip',
             'plus_minus': 'plus / minus\n(selected)',
             'plus_minus_color': 'plus / minus,\ncoloured', 'virus': 'virus / empty',
             'v3_trophy_x': 'trophy / cross\n(original)',
             'trophy_sad': 'trophy / sad face',
             'faces': 'happy / sad face\n(selected)', 'medal': 'medal /\ncracked medal'}
    style = {'font.family': 'serif',
             'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
             'font.size': 9, 'pdf.fonttype': 42}
    # Positions are in inches so the PDF retains its intended label sizes when
    # included at full text width. Native 224-pixel icons print above 380 dpi.
    width, height = 7.0, 3.15
    pair_width, icon_size, icon_gap = 1.34, 0.58, 0.04
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width, height))
        for dom, heading_y, title_y, image_y in (
                ('medical', 3.04, 2.78, 1.91),
                ('sports', 1.52, 1.27, 0.40)):
            variants = VARIANTS[dom]
            left = (width - len(variants) * pair_width) / 2
            fig.text(0.15 / width, heading_y / height, dom.capitalize(),
                     fontsize=10, fontweight='bold', va='top')
            for j, (variant, draw_icon) in enumerate(variants.items()):
                center = left + (j + 0.5) * pair_width
                fig.text(center / width, title_y / height, names[variant],
                         ha='center', va='top', fontsize=9,
                         fontweight='bold' if variant in ('plus_minus', 'faces') else 'normal')
                pair_left = center - (2 * icon_size + icon_gap) / 2
                for i, label in enumerate(SCENARIOS[dom]['labels']):
                    icon_left = pair_left + i * (icon_size + icon_gap)
                    ax = fig.add_axes([icon_left / width, image_y / height,
                                       icon_size / width, icon_size / height])
                    # Match the saved probe previews exactly; do not redraw at
                    # a smaller resolution or interpolate their source pixels.
                    ax.imshow(draw_icon(label, 224), interpolation='none')
                    ax.set_axis_off()
                    fig.text((icon_left + icon_size / 2) / width,
                             (image_y - 0.07) / height, label,
                             ha='center', va='top', fontsize=9)
        from figure_pdf import save_outlined_pdf
        save_outlined_pdf(fig, out_pdf)
        plt.close(fig)


if __name__ == '__main__' and '--figure' in sys.argv:
    make_figure(sys.argv[sys.argv.index('--figure') + 1])
