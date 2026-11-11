"""Frozen-VLM replay of Psych-101 xiong2023neural/exp1 (changing bandit), mirroring run_volatile.py.

Per episode (participant x 100-trial game) and trial t = 1..100: the model sees the human's trials 1..t-1
and is scored on the two machine letters (raw first-token logits, normalized over the 2 letters).
One --format per process (the image resolution is fixed per backend):
  text | image_outcome | chart                     (choice readout; --hazard-info told|hidden)
  perc_outcome | perc_chart                         (perception controls; hazard text irrelevant)
Resumable (skips rows already in the shard file); --shard i/n splits episodes across processes/GPUs;
--sharded spreads one large model over all visible GPUs (backend_multi.ShardedVLM).

Example:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/volatile_trust/psych101/run_xiong.py \
     --model-dir .cache/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
     --model-id Qwen/Qwen3-VL-8B-Instruct --format text --output research/volatile_trust/psych101/runs/qwen3vl8b_text
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'research/source_trust'))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import render_xiong as rx  # noqa: E402

FORMATS = ('text', 'image_outcome', 'chart', 'perc_outcome', 'perc_chart')
DEFAULT_PX = {'text': 224, 'image_outcome': 224, 'perc_outcome': 224, 'chart': 448, 'perc_chart': 448}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--format', choices=FORMATS, required=True)
    ap.add_argument('--hazard-info', choices=('told', 'hidden'), default='told')
    ap.add_argument('--data', type=Path, default=HERE / 'data/episodes.jsonl')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--pixels', type=int, choices=(224, 448),
                    help='backend image size (default 224 for text/image_outcome, 448 for chart; InternVL needs 448)')
    ap.add_argument('--n-img', type=int, default=rx.N_IMG, help='image_outcome: pictures for the last N trials')
    ap.add_argument('--max-participants', type=int)
    ap.add_argument('--hazards', type=float, nargs='+', default=[0.1, 0.2, 0.4])
    ap.add_argument('--trials', type=int, nargs='+', help='subset of trial indices (debug)')
    ap.add_argument('--sharded', action='store_true', help='shard weights over all visible GPUs')
    args = ap.parse_args()
    px = args.pixels or DEFAULT_PX[args.format]

    eps = [json.loads(l) for l in open(args.data)]
    pids = sorted({e['participant'] for e in eps})[:args.max_participants]
    eps = [e for e in eps if e['participant'] in pids and e['hazard'] in args.hazards]
    eps.sort(key=lambda e: (e['participant'], e['game']))
    i, n = map(int, args.shard.split('/'))
    eps = eps[i::n]
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.format}_{args.hazard_info}' if not args.format.startswith('perc') else args.format
    out_path = args.output / f'results_{tag}_shard{i}of{n}.jsonl'
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r['episode_id'], r['item']))
            except json.JSONDecodeError:
                pass
    if args.sharded:
        from backend_multi import ShardedVLM
        backend = ShardedVLM(args.model_dir, args.model_id, pixels=px)
    else:
        from model_backends import FrozenVLM
        backend = FrozenVLM(args.model_dir, args.model_id, pixels=px)
    meta = dict(backend.metadata, args={k: str(v) for k, v in vars(args).items()}, image_px=px,
                episodes=[e['episode_id'] for e in eps], fonts=rx.font_provenance(), n_img=args.n_img,
                protocol='teacher-forced human history; P(letter) from raw first-token logits over 2 letters')
    (args.output / f'run_metadata_{tag}_shard{i}of{n}.json').write_text(json.dumps(meta, indent=1, default=str))
    started, count = time.time(), 0
    with open(out_path, 'a') as out:
        for ep in eps:
            if args.format.startswith('perc'):
                jobs = [(it['item_id'], it) for it in rx.perception_items(ep, args.format)]
            else:
                jobs = [(t, None) for t in (args.trials or range(1, ep['n_trials'] + 1))]
            for item, it in jobs:
                if (ep['episode_id'], item) in done:
                    continue
                if it is None:
                    rendered, labels = rx.choice_query(ep, item, args.format, args.hazard_info,
                                                       n_img=args.n_img, chart_px=px, img_px=px)
                else:
                    rendered, labels = rx.perception_query(ep, it, args.format, chart_px=px, img_px=px)
                row = backend.score(rendered, labels, max_new_tokens=1)
                probs = row['probabilities']
                rec = dict(episode_id=ep['episode_id'], participant=ep['participant'], game=ep['game'],
                           hazard=ep['hazard'], item=item, format=args.format, hazard_info=args.hazard_info,
                           model_id=args.model_id, image_px=px, n_images=len(rendered['images']),
                           probabilities=probs, candidate_mass=row['canonical_candidate_mass'],
                           greedy=row['greedy_answer'], input_tokens=row['input_tokens'],
                           prompt_sha256=row['prompt_sha256'], images_sha256=rx.image_digest(rendered['images']))
                if it is None:
                    tr = ep['trials'][item - 1]
                    rec.update(t=item, p_slot1=probs[ep['labels'][0]], human_choice=tr['choice'],
                               labels=ep['labels'])
                else:
                    rec.update(t=it['t'], answer=it['answer'], subtask=it.get('subtask', 'read'),
                               correct_prob=probs[it['answer']],
                               correct_argmax=max(probs, key=probs.get) == it['answer'])
                out.write(json.dumps(rec) + '\n')
                count += 1
            out.flush()
            print(json.dumps(dict(episode=ep['episode_id'], rows=count,
                                  sec_per_row=(time.time() - started) / max(count, 1))), flush=True)


if __name__ == '__main__':
    main()
