"""Run the API advice-task subset for an Anthropic model through the Message Batches API (50% price).

Same prompts, conditions and parsing as api_runner.py; writes the same JSONL record format.
  submit:  python api_batch.py submit  --model claude-fable-5-2 --key-file KEY --episodes ... --trials ... --state S.json
  collect: python api_batch.py collect --key-file KEY --state S.json --out runs_api/X.jsonl   (no-op until ended)
Credential: read from --key-file (never printed or written). No refusal fallbacks (a fallback would change the
model); refusals and errors are recorded as invalid answers.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import api_runner as ar  # noqa: E402
from run_volatile import Renderer, resolve_domain  # noqa: E402


def items(data, episodes, trials, conditions):
    eps = {}
    for line in open(data):
        e = json.loads(line)
        eps[e['episode_id']] = e
    rend = Renderer(None, 224)
    for eid in [i for b in episodes for i in (b, b + 't')]:
        ep = eps[eid]
        dom = resolve_domain('scenario', ep)
        for t in trials:
            for cond in conditions:
                mode, hfmt = ar.CONDITIONS[cond]
                rendered, labels, a_label = rend.choice(ep, dom, 'text', t, True, ar.window_for(ep, t, mode), hfmt)
                tr = ep['trials'][t - 1]
                yield dict(custom_id=f'{eid}-{t}-{cond}', text=''.join(c['text'] for c in rendered['content']),
                           meta=dict(episode_id=eid, t=t, condition=cond, history_mode=mode, history_format=hfmt,
                                     domain=dom, domain_slot='scenario', format='text', readout='choice_generated',
                                     p_a=tr['p_a'], phase=tr['phase'], a_correct=tr['a_correct'], target=a_label,
                                     labels=labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['submit', 'collect'])
    ap.add_argument('--model', default='claude-fable-5-2')
    ap.add_argument('--key-file', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=HERE / 'episodes_v3/episodes.jsonl')
    ap.add_argument('--episodes', nargs='+')
    ap.add_argument('--trials', nargs='+', type=int)
    ap.add_argument('--conditions', nargs='+', default=list(ar.CONDITIONS))
    ap.add_argument('--effort', default='low')
    ap.add_argument('--max-output-tokens', type=int, default=4000)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic(api_key=a.key_file.read_text().strip().splitlines()[0].strip())

    if a.action == 'submit':
        if a.state.exists():
            raise SystemExit('State file exists (batch already submitted?); choose a new path.')
        its = list(items(a.data, a.episodes, a.trials, a.conditions))
        reqs = [Request(custom_id=i['custom_id'], params=MessageCreateParamsNonStreaming(
            model=a.model, max_tokens=a.max_output_tokens, output_config={'effort': a.effort},
            messages=[{'role': 'user', 'content': i['text']}])) for i in its]
        b = client.messages.batches.create(requests=reqs)
        a.state.parent.mkdir(parents=True, exist_ok=True)
        a.state.write_text(json.dumps(dict(batch_id=b.id, model=a.model, effort=a.effort, n=len(its),
                                           items={i['custom_id']: i['meta'] for i in its}), indent=0))
        print('submitted', b.id, len(its), b.processing_status)
        return

    st = json.loads(a.state.read_text())
    b = client.messages.batches.retrieve(st['batch_id'])
    print('status', b.processing_status, dict(b.request_counts))
    if b.processing_status != 'ended':
        return
    n = 0
    with open(a.out, 'w') as f:
        for r in client.messages.batches.results(st['batch_id']):
            meta = dict(st['items'][r.custom_id])
            ans, usage, status, served = '', {}, r.result.type, None
            if r.result.type == 'succeeded':
                m = r.result.message
                ans = ''.join(x.text for x in m.content if x.type == 'text')
                usage = dict(input_tokens=m.usage.input_tokens, output_tokens=m.usage.output_tokens)
                status, served = m.stop_reason, m.model
            choice, how = ar.parse(ans, meta['labels']) if ans else (None, 'invalid')
            y = None if choice is None else float(choice == meta['target'])
            meta.update(answer=ans, parsed=choice, parse=how, y=y,
                        correct=None if y is None else float(y == meta['a_correct']),
                        model=st['model'], served_model=served, effort=st['effort'], status=status, usage=usage,
                        batch=True)
            f.write(json.dumps(meta) + '\n')
            n += 1
    print('wrote', n, a.out)


if __name__ == '__main__':
    main()
