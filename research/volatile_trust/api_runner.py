"""Query a proprietary model through its API on the advice task (text reports; Appendix K).

Prompts are rendered by run_volatile.Renderer exactly as for the open models. The choice is the answer word in
the generated text (no logits): y = 1 if the answer is advisor A's report, 0 if B's, None if invalid.
Writes one JSON line per call (resume-safe) with token usage, so cost can be measured on a pilot first.

Credential: read from --key-file (never printed or written). Providers: openai (Responses API), anthropic (SDK).

Usage (pilot, 2 episode pairs x 3 formats x 4 trials = 48 calls):
  python api_runner.py --provider openai --model gpt-6-astra --key-file KEY \
      --episodes hel000 hel024 --trials 20 40 60 80 --out runs_api/pilot_gpt6astra.jsonl
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from controller import history_window  # noqa: E402
from run_volatile import Renderer, resolve_domain  # noqa: E402

# (condition name, history mode, history format) -- same arms as the open models
CONDITIONS = {'raw': ('full', 'raw'), 'correct_only': ('full', 'coded'), 'summary': ('sum_all', 'raw')}
EFFORTS = ['none', 'minimal', 'low']   # lowest reasoning effort the API accepts is used and recorded


def window_for(ep, t, mode):
    summ = mode.startswith('sum_')
    base = {'sum_all': 'full'}.get(mode, mode)
    w = history_window([tr['a_correct'] for tr in ep['trials']], t, base)
    return (w[0], w[1], summ)


def call_openai(model, text, key, effort, max_out):
    payload = dict(model=model, store=False, input=text, max_output_tokens=max_out, reasoning=dict(effort=effort))
    req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=600) as s:
        r = json.load(s)
    out = ''.join(c.get('text', '') for o in r.get('output', []) if o.get('type') == 'message'
                  for c in o.get('content', []))
    return out, r.get('usage', {}), r.get('status'), r.get('model')


def call_anthropic(client, model, text, effort, max_out):
    # No server-side refusal fallbacks on purpose: a fallback would silently answer with a different model.
    # Refusals are recorded as invalid answers instead.
    r = client.messages.create(model=model, max_tokens=max_out, output_config={'effort': effort},
                               messages=[{'role': 'user', 'content': text}])
    out = ''.join(b.text for b in r.content if b.type == 'text')
    u = r.usage
    usage = dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                 cache_read_input_tokens=getattr(u, 'cache_read_input_tokens', None))
    return out, usage, r.stop_reason, r.model


def parse(answer, labels):
    words = re.findall(r'[a-z]+', answer.lower())
    hits = [w for w in words if w in labels]
    if len(set(hits)) == 1 and len(words) <= 3:
        return hits[0], 'exact'
    last = re.findall(r'[a-z]+', answer.strip().splitlines()[-1].lower()) if answer.strip() else []
    if len(last) == 1 and last[0] in labels:
        return last[0], 'final_line'   # one-word answer on its own last line, after visible reasoning
    if hits:
        return hits[-1], 'loose'    # recorded separately; analysis treats as invalid
    return None, 'invalid'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--provider', choices=['openai', 'anthropic'], default='openai')
    ap.add_argument('--model', required=True)
    ap.add_argument('--key-file', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=HERE / 'episodes_v3/episodes.jsonl')
    ap.add_argument('--episodes', nargs='+', required=True, help='base ids; the counterbalanced copy (…t) is added')
    ap.add_argument('--trials', nargs='+', type=int, required=True)
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS))
    ap.add_argument('--max-output-tokens', type=int, default=4000)
    ap.add_argument('--max-calls', type=int, default=200, help='hard stop (cost guard)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    key = a.key_file.read_text().strip().splitlines()[0].strip()
    client = None
    if a.provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=key)
    eps = {}
    for line in open(a.data):
        e = json.loads(line)
        eps[e['episode_id']] = e
    ids = [i for b in a.episodes for i in (b, b + 't')]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in open(a.out):
            r = json.loads(line)
            done.add((r['episode_id'], r['t'], r['condition']))
    rend = Renderer(None, 224)
    effort = None
    n = 0
    with open(a.out, 'a') as f:
        for eid in ids:
            ep = eps[eid]
            dom = resolve_domain('scenario', ep)
            for t in a.trials:
                for cond in a.conditions:
                    if (eid, t, cond) in done:
                        continue
                    if n >= a.max_calls:
                        print('max calls reached'); return
                    mode, hfmt = CONDITIONS[cond]
                    rendered, labels, a_label = rend.choice(ep, dom, 'text', t, True, window_for(ep, t, mode), hfmt)
                    text = ''.join(c['text'] for c in rendered['content'])
                    t0 = time.time()
                    if client is not None:
                        effort = 'low'
                        ans, usage, status, served = call_anthropic(client, a.model, text, effort, a.max_output_tokens)
                    for eff in ([] if client is not None else [effort] if effort else EFFORTS):
                        try:
                            ans, usage, status, served = call_openai(a.model, text, key, eff, a.max_output_tokens)
                            effort = eff
                            break
                        except urllib.error.HTTPError as e:
                            body = e.read().decode(errors='replace')[:300]
                            if e.code == 400 and 'effort' in body and not effort:
                                continue
                            raise SystemExit(f'HTTP {e.code}: {body}')
                    choice, how = parse(ans, labels)
                    tr = ep['trials'][t - 1]
                    y = None if choice is None else float(choice == a_label)
                    rec = dict(episode_id=eid, t=t, condition=cond, history_mode=mode, history_format=hfmt,
                               domain=dom, domain_slot='scenario', format='text', readout='choice_generated',
                               p_a=tr['p_a'], phase=tr['phase'], a_correct=tr['a_correct'], target=a_label,
                               labels=labels, answer=ans, parsed=choice, parse=how, y=y,
                               correct=None if y is None else float(y == tr['a_correct']),
                               model=a.model, served_model=served, effort=effort, status=status, usage=usage,
                               seconds=round(time.time() - t0, 2))
                    f.write(json.dumps(rec) + '\n'); f.flush()
                    n += 1
                    print(eid, t, cond, repr(ans[:30]), how, usage.get('input_tokens'), usage.get('output_tokens'),
                          flush=True)
    key = None


if __name__ == '__main__':
    main()
