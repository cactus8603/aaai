"""Deterministic prompt + PIL renderers for the xiong2023 changing-bandit replay (no generative models).

Formats (history = the human's trials 1..t-1; query = which machine on trial t):
  text          : Psych-101 text history, verbatim lines "You press <<M>> and get 65 points."
  image_outcome : same text history, but the outcomes of the LAST N_IMG = 5 trials are shown as a rendered
                  slot-machine picture whose digital display shows the points (earlier trials stay text).
                  Why last-5 rather than only the previous outcome: under h = 0.4 most of the decision-relevant
                  evidence (incl. the current value of the other machine) lies in the last few trials, so a
                  single picture would let the model ignore the image channel whenever the decisive outcome is
                  2+ trials back; 5 pictures cost ~5 x 49-64 = 245-320 image tokens at 224 px.
  chart         : the whole history replaced by ONE scatter chart (x = trial 1..100, y = points 0..100,
                  machine 1 = blue filled circles, machine 2 = orange filled triangles, legend with the machine
                  letters), plus the instruction text and the trial count.
Perception controls:
  perc_outcome  : one outcome picture -> which of 4 numbers is on the display (answer A/B/C/D).
  perc_chart    : chart of trials 1..K -> (compare) at trials k-1 and k different machines were pressed; which
                  machine gave more points (|diff| >= 10)?  (identify) which machine was pressed at trial k?
Hazard variants: told = original Psych-101 wording; hidden = the hazard-rate clauses removed.

CLI (CPU only; writes examples/ and token counts):
    .venv/bin/python research/volatile_trust/psych101/render_xiong.py --examples
"""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N_IMG = 5
GAME_LEN = 100
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
COLORS = [(31, 102, 189), (230, 120, 20)]          # machine 1 blue, machine 2 orange (color-blind safe pair)

# ---------------------------------------------------------------- text
LINES_TOLD = [
    "You are participating in multiple games involving two slot machines, labeled {A} and {B}.",
    "The two slot machines are different in different games.",
    "Each time you choose a slot machine, you get points (choosing the same slot machine will not always give you the same points).",
    "You select a slot machine by pressing the corresponding key.",
    "The expected points change randomly, abruptly, and independently with a hazard rate (which you will be told).",
    "When the points change, the new expected point value assigned to that slot machine is sampled from a uniform distribution (from 1 to 99 points).",
    "For example, if the hazard rate is 0.1, the expected points of the machines change with 10%.",
    "Your goal is to choose the slot machine that will give you the most points."]
HIDDEN_REPLACE = {4: "The expected points change randomly, abruptly, and independently.", 6: None}


def letters(ep):
    """Alphabetical order of the two letters for instructions/questions (independent of slot index)."""
    return sorted(ep['labels'])


def instructions(ep, hazard_info='told'):
    a, b = letters(ep)
    lines = []
    for i, line in enumerate(LINES_TOLD):
        if hazard_info == 'hidden' and i in HIDDEN_REPLACE:
            line = HIDDEN_REPLACE[i]
            if line is None:
                continue
        lines.append(line.format(A=a, B=b))
    return '\n'.join(lines)


def game_header(ep, hazard_info='told'):
    if hazard_info == 'told':
        return f"Game 1. The hazard rate is {ep['hazard']}. There are {ep['n_trials']} trials in this game."
    return f"Game 1. There are {ep['n_trials']} trials in this game."


def trial_line(ep, tr):
    return f"You press <<{ep['labels'][tr['choice']]}>> and get {tr['reward']} points."


def question(ep, t):
    a, b = letters(ep)
    return (f"\nTrial {t} of this game: which slot machine do you press? "
            f"Answer with exactly one letter: {a} or {b}.")


# ---------------------------------------------------------------- images
def _font(name, size):
    p = FONT_DIR / name
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default(size=size)


def font_provenance():
    out = {}
    for n in ('DejaVuSans-Bold.ttf', 'DejaVuSansMono-Bold.ttf', 'DejaVuSans.ttf'):
        p = FONT_DIR / n
        out[n] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'PIL-default'
    return out


def machine_image(letter, points, size=224):
    """A slot machine cabinet with its letter and a digital display showing the points."""
    s = size / 224
    im = Image.new('RGB', (size, size), (245, 245, 240))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([28 * s, 14 * s, 186 * s, 212 * s], radius=int(16 * s), fill=(150, 30, 40), outline=(60, 10, 15), width=max(1, int(3 * s)))
    d.rounded_rectangle([38 * s, 22 * s, 176 * s, 62 * s], radius=int(8 * s), fill=(250, 210, 60))
    d.text((107 * s, 42 * s), f'Machine {letter}', fill=(40, 20, 10), font=_font('DejaVuSans-Bold.ttf', int(19 * s)), anchor='mm')
    d.rectangle([46 * s, 78 * s, 168 * s, 150 * s], fill=(10, 10, 10), outline=(200, 200, 200), width=max(1, int(3 * s)))
    d.text((107 * s, 114 * s), f'{points:>2d}', fill=(60, 255, 90), font=_font('DejaVuSansMono-Bold.ttf', int(58 * s)), anchor='mm')
    d.text((107 * s, 168 * s), 'POINTS', fill=(255, 235, 200), font=_font('DejaVuSans-Bold.ttf', int(14 * s)), anchor='mm')
    # lever
    d.line([194 * s, 70 * s, 194 * s, 130 * s], fill=(90, 90, 90), width=max(1, int(6 * s)))
    d.ellipse([186 * s, 58 * s, 202 * s, 74 * s], fill=(220, 30, 30))
    return im


def _marker(d, x, y, k, r):
    if k == 0:
        d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[0], outline=(0, 0, 0))
    else:
        d.polygon([(x, y - r * 1.2), (x - r * 1.1, y + r * 0.8), (x + r * 1.1, y + r * 0.8)], fill=COLORS[1], outline=(0, 0, 0))


def chart_image(ep, upto, size=448, highlight=None):
    """Scatter chart of trials 1..upto (chosen machine's points). x axis fixed to 1..GAME_LEN."""
    im = Image.new('RGB', (size, size), (255, 255, 255))
    d = ImageDraw.Draw(im)
    s = size / 448
    L, R, T, B = 58 * s, 436 * s, 64 * s, 400 * s
    f = _font('DejaVuSans.ttf', int(13 * s))
    fb = _font('DejaVuSans-Bold.ttf', int(15 * s))
    X = lambda t: L + (t - 0.5) / GAME_LEN * (R - L)
    Y = lambda v: B - v / 100 * (B - T)
    for v in (0, 25, 50, 75, 100):
        d.line([L, Y(v), R, Y(v)], fill=(225, 225, 225), width=1)
        d.text((L - 6 * s, Y(v)), str(v), fill=(0, 0, 0), font=f, anchor='rm')
    for t in (1, 20, 40, 60, 80, 100):
        d.line([X(t), B, X(t), B + 5 * s], fill=(0, 0, 0))
        d.text((X(t), B + 8 * s), str(t), fill=(0, 0, 0), font=f, anchor='mt')
    d.rectangle([L, T, R, B], outline=(0, 0, 0), width=max(1, int(2 * s)))
    d.text(((L + R) / 2, size - 12 * s), 'Trial', fill=(0, 0, 0), font=fb, anchor='mb')
    lab = Image.new('RGB', (int(80 * s), int(20 * s)), (255, 255, 255))
    ImageDraw.Draw(lab).text((40 * s, 10 * s), 'Points', fill=(0, 0, 0), font=fb, anchor='mm')
    lab = lab.rotate(90, expand=True)
    im.paste(lab, (int(2 * s), int((T + B) / 2 - lab.size[1] / 2)))
    # legend (above the plot, never overlaps data)
    for k, lab in enumerate(ep['labels']):
        x0 = L + 20 * s + k * 170 * s
        _marker(d, x0, 32 * s, k, 7 * s)
        d.text((x0 + 14 * s, 32 * s), f'Machine {lab}', fill=(0, 0, 0), font=fb, anchor='lm')
    for tr in ep['trials'][:upto]:
        _marker(d, X(tr['t']), Y(tr['reward']), tr['choice'], 4.2 * s)
    if highlight is not None:
        for t in highlight:
            d.line([X(t), T, X(t), B], fill=(150, 150, 150), width=1)
    return im


# ---------------------------------------------------------------- rendered queries
def choice_query(ep, t, fmt, hazard_info='told', n_img=N_IMG, chart_px=448, img_px=224):
    """Returns rendered={content, images}, labels (alphabetical letters)."""
    hist = ep['trials'][:t - 1]
    head = instructions(ep, hazard_info) + '\n\n' + game_header(ep, hazard_info) + '\n'
    content, images = [], []
    if fmt == 'text':
        body = ''.join(trial_line(ep, tr) + '\n' for tr in hist)
        content.append({'type': 'text', 'text': head + body + question(ep, t)})
    elif fmt == 'image_outcome':
        note = (f"(For the most recent {n_img} trials, the points you got are shown as a picture of the "
                f"machine's display.)\n")
        k0 = max(0, len(hist) - n_img)
        txt = head + note + ''.join(trial_line(ep, tr) + '\n' for tr in hist[:k0])
        for tr in hist[k0:]:
            txt += f"You press <<{ep['labels'][tr['choice']]}>> and get the points shown here: "
            content.append({'type': 'text', 'text': txt})
            content.append({'type': 'image'})
            images.append(machine_image(ep['labels'][tr['choice']], tr['reward'], img_px))
            txt = '\n'
        content.append({'type': 'text', 'text': txt + question(ep, t)})
    elif fmt == 'chart':
        txt = (head + f"You have played {t - 1} trials of this game so far. The chart shows the points you got on "
               "every past trial of this game (x axis: trial number; y axis: points; the marker shape and color "
               "show which slot machine you pressed, see the legend).\n")
        content += [{'type': 'text', 'text': txt}, {'type': 'image'},
                    {'type': 'text', 'text': question(ep, t)}]
        images.append(chart_image(ep, t - 1, chart_px))
    else:
        raise ValueError(fmt)
    return dict(content=content, images=images), letters(ep)


def perception_items(ep, kind, n_items=3, seed=20261111):
    """Deterministic perception-control items for one episode."""
    rng = random.Random(f'{seed}-{ep["episode_id"]}-{kind}')
    items = []
    if kind == 'perc_outcome':
        for t in rng.sample(range(1, ep['n_trials'] + 1), 5):
            tr = ep['trials'][t - 1]
            v = tr['reward']
            near = [x for x in range(max(1, v - 10), min(99, v + 10) + 1) if x != v and str(x) != str(v)[::-1]]
            pool = [x for x in range(1, 100) if abs(x - v) > 10]
            opts = [v, rng.choice(near)] + rng.sample(pool, 2)
            rng.shuffle(opts)
            letter = 'ABCD'[opts.index(v)]
            items.append(dict(item_id=f'{ep["episode_id"]}_po{t}', t=t, options=opts, answer=letter,
                              q=('The picture shows a slot machine. What number of points is shown on its display? '
                                 'Answer with exactly one letter: ' +
                                 ' '.join(f'{L}) {o}' for L, o in zip('ABCD', opts)))))
    elif kind == 'perc_chart':
        tr = ep['trials']
        sw = [k for k in range(2, ep['n_trials'] + 1)
              if tr[k - 1]['choice'] != tr[k - 2]['choice'] and abs(tr[k - 1]['reward'] - tr[k - 2]['reward']) >= 10]
        a, b = letters(ep)
        for k in rng.sample(sw, min(n_items, len(sw))):
            upto = rng.randint(k, ep['n_trials'])
            hi = tr[k - 1] if tr[k - 1]['reward'] > tr[k - 2]['reward'] else tr[k - 2]
            items.append(dict(item_id=f'{ep["episode_id"]}_pc{k}', subtask='compare', t=k, upto=upto,
                              answer=ep['labels'][hi['choice']],
                              q=(f'The chart shows points (y axis) over trials (x axis); marker shape and color show which '
                                 f'slot machine was pressed (see legend). At trials {k - 1} and {k} different machines '
                                 f'were pressed. Which machine gave more points on those two trials? '
                                 f'Answer with exactly one letter: {a} or {b}.')))
        for k in rng.sample(range(1, ep['n_trials'] + 1), n_items):
            upto = rng.randint(k, ep['n_trials'])
            items.append(dict(item_id=f'{ep["episode_id"]}_pi{k}', subtask='identify', t=k, upto=upto,
                              answer=ep['labels'][tr[k - 1]['choice']],
                              q=(f'The chart shows points (y axis) over trials (x axis); marker shape and color show which '
                                 f'slot machine was pressed (see legend). Which machine was pressed at trial {k}? '
                                 f'Answer with exactly one letter: {a} or {b}.')))
    return items


def perception_query(ep, item, kind, chart_px=448, img_px=224):
    if kind == 'perc_outcome':
        tr = ep['trials'][item['t'] - 1]
        img = machine_image(ep['labels'][tr['choice']], tr['reward'], img_px)
        return dict(content=[{'type': 'image'}, {'type': 'text', 'text': item['q']}], images=[img]), list('ABCD')
    img = chart_image(ep, item['upto'], chart_px)
    return dict(content=[{'type': 'image'}, {'type': 'text', 'text': item['q']}], images=[img]), letters(ep)


def image_digest(images):
    h = hashlib.sha256()
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        h.update(buf.getvalue())
    return h.hexdigest()


# ---------------------------------------------------------------- examples + token counts (CPU)
def flat_text(rendered):
    return ''.join(p['text'] if p['type'] == 'text' else '<IMAGE>' for p in rendered['content'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--examples', action='store_true')
    ap.add_argument('--model-dir', type=Path,
                    default=ROOT / '.cache/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
    ap.add_argument('--episode', default='p00_g5')
    args = ap.parse_args()
    eps = {json.loads(l)['episode_id']: json.loads(l) for l in open(HERE / 'data/episodes.jsonl')}
    ep = eps[args.episode]
    ex = HERE / 'examples'
    ex.mkdir(exist_ok=True)
    # deterministic re-render check
    assert image_digest([chart_image(ep, 60)]) == image_digest([chart_image(ep, 60)])
    machine_image(ep['labels'][0], 65).save(ex / 'outcome_machine_224.png')
    machine_image(ep['labels'][1], 7, 448).save(ex / 'outcome_machine_448.png')
    chart_image(ep, 99).save(ex / 'chart_t100_448.png')
    chart_image(ep, 30).save(ex / 'chart_t31_448.png')
    chart_image(ep, 99, 224).save(ex / 'chart_t100_224.png')
    for fmt in ('text', 'image_outcome', 'chart'):
        for hz in ('told', 'hidden'):
            r, labs = choice_query(ep, 100, fmt, hz)
            (ex / f'prompt_{fmt}_{hz}_t100.txt').write_text(flat_text(r) + f'\n\n[candidates: {labs}; images: {len(r["images"])}]\n')
    it = perception_items(ep, 'perc_chart')[0]
    r, _ = perception_query(ep, it, 'perc_chart')
    r['images'][0].save(ex / 'perc_chart_example.png')
    (ex / 'perc_chart_example.txt').write_text(flat_text(r) + f"\n[answer: {it['answer']}]\n")
    it = perception_items(ep, 'perc_outcome')[0]
    r, _ = perception_query(ep, it, 'perc_outcome')
    (ex / 'perc_outcome_example.txt').write_text(flat_text(r) + f"\n[answer: {it['answer']}]\n")

    # token counts with the Qwen3-VL processor on CPU, using the backend's preprocessing settings
    import sys
    sys.path.insert(0, str(ROOT / 'research/source_trust'))
    from model_backends import local_preprocessing
    from transformers import AutoProcessor
    report = dict(model_dir=str(args.model_dir), episode=args.episode, fonts=font_provenance(), formats={})
    procs = {}
    for px in (224, 448):
        pre = local_preprocessing(args.model_dir, px)
        procs[px] = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False,
                                                  use_fast=False, **pre['image_processor_settings'])

    def count(rendered, px):
        proc = procs[px]
        prompt = proc.apply_chat_template([{'role': 'user', 'content': rendered['content']}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        kw = dict(text=[prompt], return_tensors='pt')
        if rendered['images']:
            kw['images'] = rendered['images']
        return int(proc(**kw)['input_ids'].shape[1])

    for fmt, px in (('text', 224), ('image_outcome', 224), ('image_outcome', 448), ('chart', 448), ('chart', 224)):
        for hz in ('told', 'hidden'):
            if hz == 'hidden' and fmt != 'text':
                continue
            per_t = {t: count(choice_query(ep, t, fmt, hz, chart_px=px, img_px=px)[0], px) for t in range(1, 101)}
            report['formats'][f'{fmt}@{px}px/{hz}'] = dict(t1=per_t[1], t50=per_t[50], t100=per_t[100],
                                                          mean=sum(per_t.values()) / 100,
                                                          total_100_queries=sum(per_t.values()))
    report['perception'] = {
        'perc_outcome@224': count(perception_query(ep, perception_items(ep, 'perc_outcome')[0], 'perc_outcome')[0], 224),
        'perc_chart@448': count(perception_query(ep, perception_items(ep, 'perc_chart')[0], 'perc_chart')[0], 448)}
    (ex / 'token_counts.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
