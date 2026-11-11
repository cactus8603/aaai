"""Trial-by-trial frozen-VLM readout on volatile-trust episodes.

Readouts per episode and trial t (history = trials 1..t-1, always text):
  choice   : domain x format(text|image)  -> P(predict A's current report)
  judgment : domain, text only            -> P(name A as currently more reliable)
Resumable; shard episodes across GPUs with --shard i/n.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/source_trust'))
from model_backends import FrozenVLM  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import history_window  # noqa: E402

DOMAINS = {'color': ('red', 'blue'), 'pet': ('cat', 'dog')}
NOUN = {'color': 'color', 'pet': 'animal'}
COLOR_RGB = {'red': (220, 30, 30), 'blue': (30, 60, 220)}


def header(ep, domain, hint=True):
    n = NOUN[domain]
    first, second = ((ep['a_name'], ep['b_name']) if ep.get('header_a_first', True)
                     else (ep['b_name'], ep['a_name']))
    s = (f"You are observing a sequence of trials. On every trial, two reporters, {first} and "
         f"{second}, give conflicting reports about a hidden {n}. After each trial, the true {n} "
         f"is revealed.")
    if hint:
        s += " A reporter's accuracy may change over time."
    return s + '\n'


def label(domain, bit, polarity):
    return DOMAINS[domain][bit ^ polarity]


def history(ep, domain, t, start=1, marker=None, as_summary=False):
    if as_summary:
        return summary(ep, t, start)
    if t == 1 or start >= t:
        return 'Past trials: none yet.\n'
    lines = ['Past trials:' if start == 1 else f'Recent past trials (trials {start}-{t - 1}):']
    pol = ep['label_polarity']
    for tr in ep['trials'][start - 1:t - 1]:
        if marker is not None and tr['t'] == marker:
            lines.append('--- Note: reporter accuracy probably changed around here ---')
        a = f"{ep['a_name']} reports {label(domain, tr['a_report_bit'], pol)}"
        b = f"{ep['b_name']} reports {label(domain, tr['b_report_bit'], pol)}"
        first, second = (a, b) if tr['a_listed_first'] else (b, a)
        lines.append(f"Trial {tr['t']}: {first}; {second}. True {NOUN[domain]}: "
                     f"{label(domain, tr['truth_bit'], pol)}.")
    return '\n'.join(lines) + '\n'


def summary(ep, t, start):
    """Explicit counts over trials start..t-1 (controller-computed accumulator)."""
    if t == 1 or start >= t:
        return 'Past trials: none yet.\n'
    seg = ep['trials'][start - 1:t - 1]
    a = sum(x['a_correct'] for x in seg)
    n = len(seg)
    names = [(ep['a_name'], a), (ep['b_name'], n - a)]
    if not ep['trials'][t - 1]['a_listed_first']:
        names.reverse()
    rng = f'all {n} past trials' if start == 1 else f'the last {n} trials (trials {start}-{t - 1})'
    return (f'Summary of {rng}: {names[0][0]} was correct {names[0][1]} times; '
            f'{names[1][0]} was correct {names[1][1]} times.\n')


class Renderer:
    def __init__(self, image_index, pixels):
        self.index, self.pixels, self.cache = image_index, pixels, {}

    def image(self, domain, lab, ep, t):
        if domain == 'color':
            key = ('color', lab)
            if key not in self.cache:
                self.cache[key] = Image.new('RGB', (self.pixels, self.pixels), COLOR_RGB[lab])
            return self.cache[key]
        iid = ep['pet_image_ids'][str(t)][lab]
        rec = self.index[iid]
        assert rec['label'] == lab
        path = ROOT / rec['path224' if self.pixels == 224 else 'path448']
        with Image.open(path) as im:
            return im.convert('RGB')

    def _font(self, size):
        from PIL import ImageFont
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def banner(self, img, name):
        """Print the reporter name in a white banner above the image (same final size)."""
        from PIL import ImageDraw
        P = self.pixels
        h = P // 6
        out = Image.new('RGB', (P, P), 'white')
        out.paste(img.resize((P, P - h)), (0, h))
        ImageDraw.Draw(out).text((P // 2, h // 2), name, fill='black', anchor='mm', font=self._font(h * 3 // 4))
        return out

    def composite(self, domain, reps, ep, t):
        """Two labelled panels side by side, in listing order."""
        from PIL import ImageDraw
        P = self.pixels
        out = Image.new('RGB', (2 * P, P), 'white')
        for i, (name, lab) in enumerate(reps):
            out.paste(self.banner(self.image(domain, lab, ep, t), name), (i * P, 0))
        ImageDraw.Draw(out).line([(P, 0), (P, P)], fill='black', width=3)
        return out

    def chart(self, ep, t):
        """Chart of trials 1..t-1, wrapped as 4 lines x 20 trials. Each line has one row per reporter
        (listing order of trial t); green = that reporter was correct, gray = wrong, blank = not yet."""
        from PIL import ImageDraw
        n = t - 1
        cw, ch, left, line_gap, top = 22, 26, 64, 22, 8
        per_line, lines = 20, 4
        W = left + per_line * cw + 12
        H = top + lines * (2 * ch + line_gap)
        img = Image.new('RGB', (W, H), 'white')
        d = ImageDraw.Draw(img)
        names = [ep['a_name'], ep['b_name']]
        if not ep['trials'][t - 1]['a_listed_first']:
            names.reverse()
        fn, fs = self._font(16), self._font(11)
        for L in range(lines):
            y0 = top + L * (2 * ch + line_gap)
            for row, name in enumerate(names):
                y = y0 + row * ch
                d.text((4, y + 5), name, fill='black', font=fn)
                for c in range(per_line):
                    k = L * per_line + c          # 0-based trial index
                    x = left + c * cw
                    if k < n:
                        tr = ep['trials'][k]
                        correct = tr['a_correct'] if name == ep['a_name'] else 1 - tr['a_correct']
                        d.rectangle([x, y + 2, x + cw - 3, y + ch - 3],
                                    fill=(40, 170, 60) if correct else (190, 190, 190))
                    else:
                        d.rectangle([x, y + 2, x + cw - 3, y + ch - 3], outline=(230, 230, 230))
            d.text((left, y0 + 2 * ch + 2), f'trials {L * per_line + 1}-{(L + 1) * per_line}',
                   fill='black', font=fs)
        return img

    def choice(self, ep, domain, fmt, t, hint=True, window=(1, None, False)):
        tr = ep['trials'][t - 1]
        pol = ep['label_polarity']
        labels = DOMAINS[domain]
        reps = [(ep['a_name'], label(domain, tr['a_report_bit'], pol)),
                (ep['b_name'], label(domain, tr['b_report_bit'], pol))]
        if not tr['a_listed_first']:
            reps.reverse()
        content = [{'type': 'text', 'text': header(ep, domain, hint) + history(ep, domain, t, *window) +
                    f"Current trial {t} (true {NOUN[domain]} not yet revealed):\n"}]
        images = []
        if fmt == 'chart':   # method 2: history as one raster chart; current reports as text
            content = [{'type': 'text', 'text': header(ep, domain, hint) +
                        ('Past trials: none yet.\n' if t == 1 else
                         f'Past trials 1-{t - 1} are summarized in this chart (trials run left to right in '
                         'four lines of 20; for each trial, a green square means that reporter was correct, '
                         'a gray square means that reporter was wrong, an empty square means not yet played):\n')}]
            if t > 1:
                content.append({'type': 'image'})
                images.append(self.chart(ep, t))
            content.append({'type': 'text', 'text': f"\nCurrent trial {t} (true {NOUN[domain]} not yet revealed):\n"})
        if fmt == 'image_composite':   # binding control: one image, two labelled panels
            content.append({'type': 'text', 'text': 'Reports (one panel per reporter, name above each panel): '})
            content.append({'type': 'image'})
            content.append({'type': 'text', 'text': '\n'})
            images.append(self.composite(domain, reps, ep, t))
        for name, lab in ([] if fmt == 'image_composite' else reps):
            if fmt in ('text', 'chart'):
                content.append({'type': 'text', 'text': f'{name} reports: {lab}\n'})
            elif fmt == 'image_after':    # binding control: image first, then attribution
                content.append({'type': 'image'})
                content.append({'type': 'text', 'text': f' (the image above was reported by {name})\n'})
                images.append(self.image(domain, lab, ep, t))
            else:                          # 'image' or 'image_named' (name printed on the image)
                content.append({'type': 'text', 'text': f'{name} reports: '})
                content.append({'type': 'image'})
                content.append({'type': 'text', 'text': '\n'})
                img = self.image(domain, lab, ep, t)
                images.append(self.banner(img, name) if fmt == 'image_named' else img)
        content.append({'type': 'text', 'text':
                        f"Predict the true {NOUN[domain]} on the current trial. "
                        f"Answer with exactly one lowercase word: {labels[0]} or {labels[1]}."})
        a_label = label(domain, tr['a_report_bit'], pol)
        return dict(content=content, images=images), list(labels), a_label

    def judgment(self, ep, domain, t, hint=True, window=(1, None, False)):
        tr = ep['trials'][t - 1]
        names = [ep['a_name'], ep['b_name']]
        if not tr['a_listed_first']:
            names.reverse()
        text = (header(ep, domain, hint) + history(ep, domain, t, *window) +
                f"Before trial {t}: which reporter is currently more likely to be correct? "
                f"Answer with exactly one name: {names[0]} or {names[1]}.")
        return dict(content=[{'type': 'text', 'text': text}], images=[]), names, ep['a_name']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'episodes_v1')
    ap.add_argument('--split', choices=('development', 'held_out'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--pixels', type=int, choices=(224, 448), default=224)
    ap.add_argument('--readouts', nargs='+', default=['choice', 'judgment'])
    ap.add_argument('--domains', nargs='+', default=['color', 'pet'])
    ap.add_argument('--formats', nargs='+', default=['text', 'image'])
    ap.add_argument('--max-episodes', type=int)
    ap.add_argument('--trials', type=int, nargs='+', help='subset of trial indices (debug)')
    ap.add_argument('--sharded', action='store_true', help='shard weights over all visible GPUs')
    ap.add_argument('--history-mode', default='full', help='full | window:K | cp_hard | cp_soft')
    ap.add_argument('--hazard', type=float, default=1 / 15)
    ap.add_argument('--no-hint', action='store_true', help='omit the "accuracy may change" sentence')
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(args.data_dir / 'episodes.jsonl')]
    eps = [e for e in eps if e['split'] == args.split]
    i, n = map(int, args.shard.split('/'))
    eps = eps[i::n][:args.max_episodes]
    renderer = Renderer(json.loads((args.data_dir / 'image_index.json').read_text()), args.pixels)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f'results_shard{i}of{n}.jsonl'
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r['episode_id'], r['t'], r['readout'], r['domain'], r['format'],
                          r.get('history_mode', 'full'), r.get('hint', True), r.get('pixels', args.pixels)))
            except json.JSONDecodeError:
                pass
    if args.sharded:
        from backend_multi import ShardedVLM
        backend = ShardedVLM(args.model_dir, args.model_id, pixels=args.pixels)
    else:
        backend = FrozenVLM(args.model_dir, args.model_id, pixels=args.pixels)
    meta = dict(backend.metadata, args={k: str(v) for k, v in vars(args).items()},
                episodes=[e['episode_id'] for e in eps])
    meta['started'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(args.output / f'run_metadata_shard{i}of{n}.jsonl', 'a') as mf:   # one line per launch (resume-safe)
        mf.write(json.dumps(meta, default=str) + '\n')
    hint = not args.no_hint
    started, count = time.time(), 0
    with open(out_path, 'a') as out:
        for ep in eps:
            o_seq = [tr['a_correct'] for tr in ep['trials']]
            for t in (args.trials or range(1, 81)):
                mode = args.history_mode
                summ = mode.startswith('sum_')
                base = {'sum_all': 'full', 'sum_cp': 'cp_hard', 'sum_oracle': 'oracle'}.get(
                    mode, mode.replace('sum_window', 'window') if summ else mode)
                if base == 'oracle':   # diagnostic upper bound: true last reversal
                    window = (max([r for r in ep['reversals'] if r <= t - 1], default=1), None)
                else:
                    window = history_window(o_seq, t, base, h=args.hazard)
                window = (window[0], window[1], summ)
                jobs = []
                if 'choice' in args.readouts:
                    jobs += [('choice', d, f) for d in args.domains for f in args.formats]
                if 'judgment' in args.readouts:
                    jobs += [('judgment', d, 'text') for d in args.domains]
                for readout, domain, fmt in jobs:
                    key = (ep['episode_id'], t, readout, domain, fmt, args.history_mode, hint, args.pixels)
                    if key in done:
                        continue
                    if readout == 'choice':
                        rendered, labels, target = renderer.choice(ep, domain, fmt, t, hint, window)
                    else:
                        rendered, labels, target = renderer.judgment(ep, domain, t, hint, window)
                    row = backend.score(rendered, labels, max_new_tokens=1)
                    tr = ep['trials'][t - 1]
                    rec = dict(episode_id=ep['episode_id'], order=ep['order'], t=t, readout=readout,
                               domain=domain, format=fmt, p_a=tr['p_a'], phase=tr['phase'],
                               a_correct=tr['a_correct'], y=row['probabilities'][target],
                               candidate_mass=row['canonical_candidate_mass'],
                               greedy=row['greedy_answer'], target=target,
                               input_tokens=row['input_tokens'], prompt_sha256=row['prompt_sha256'],
                               model_id=args.model_id, hint=hint, history_mode=args.history_mode,
                               pixels=args.pixels, argmax=row.get('argmax_label'),
                               window_start=window[0], marker=window[1], summary=window[2])
                    out.write(json.dumps(rec) + '\n')
                    count += 1
                out.flush()
            print(json.dumps(dict(episode=ep['episode_id'], rows=count,
                                  sec_per_row=(time.time() - started) / max(count, 1))), flush=True)
    # Completion check: every expected key for this configuration must be present exactly once.
    n_jobs = (len(args.domains) * len(args.formats) if 'choice' in args.readouts else 0) + \
             (len(args.domains) if 'judgment' in args.readouts else 0)
    expected = len(eps) * len(args.trials or range(1, 81)) * n_jobs
    keys = []
    for line in open(out_path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get('history_mode', 'full'), r.get('hint', True), r.get('pixels', args.pixels)) == \
                (args.history_mode, hint, args.pixels):
            keys.append((r['episode_id'], r['t'], r['readout'], r['domain'], r['format']))
    if len(set(keys)) == expected and len(keys) == expected and expected > 0:
        (args.output / f'_COMPLETE_shard{i}of{n}').write_text(json.dumps(dict(rows=expected, at=time.strftime('%F %T'))))
        print(f'COMPLETE {expected} rows', flush=True)
    else:
        print(f'INCOMPLETE: {len(set(keys))} unique / {len(keys)} rows, expected {expected}', flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
