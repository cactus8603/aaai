import sys, anthropic
from pathlib import Path
key = Path(sys.argv[1]).read_text().strip().splitlines()[0].strip()
client = anthropic.Anthropic(api_key=key)
text = Path(sys.argv[2]).read_text()
with client.messages.stream(model='claude-fable-5-2', max_tokens=int(sys.argv[4]) if len(sys.argv) > 4 else 32000, output_config={'effort': 'high'},
                            messages=[{'role': 'user', 'content': text}]) as s:
    m = s.get_final_message()
out = ''.join(b.text for b in m.content if b.type == 'text')
Path(sys.argv[3]).write_text(out)
print(m.model, m.stop_reason, m.usage.input_tokens, m.usage.output_tokens)
