"""Cross-platform check H200 vs MI300X (PLAN_H200.md): for every job present in both runs_h200 and runs_mi300x with
_COMPLETE, match rows by (episode, order, t, domain, readout, format, history_format) and report prompt-hash equality,
median |dy|, argmax agreement and accuracy. Thresholds (EXPERIMENTS.md E5): median |dy| < 0.02, agreement > 98%.
Writes results/h200_xplat/README.md.  python research/volatile_trust/portable/h200/xplat_check.py"""
import glob
import json
import statistics
from pathlib import Path

VT = Path(__file__).resolve().parents[2]


def load(d):
    out = {}
    for f in glob.glob(str(d / 'results*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            out[(r['episode_id'], r.get('order'), r['t'], r['domain'], r['readout'], r['format'],
                 r.get('history_format'))] = r
    return out


rows = []
for h in sorted((VT / 'runs_h200').glob('*/*/')):
    m = VT / 'runs_mi300x' / h.parent.name / h.name
    if not (list(h.glob('_COMPLETE*')) and list(m.glob('_COMPLETE*'))):
        continue
    a, b = load(h), load(m)
    k = sorted(set(a) & set(b))
    if not k:
        continue
    dy = [abs(a[x]['y'] - b[x]['y']) for x in k]
    agree = sum(a[x]['argmax'] == b[x]['argmax'] for x in k) / len(k)
    same = sum(a[x]['prompt_sha256'] == b[x]['prompt_sha256'] for x in k)
    acc = lambda d: sum(d[x]['argmax'] == d[x]['target'] for x in k) / len(k)
    ok = statistics.median(dy) < 0.02 and agree > 0.98 and same == len(k)
    rows.append(f'| {h.parent.name} | {h.name} | {len(k)} | {same} | {statistics.median(dy):.4f} | {max(dy):.4f} | '
                f'{100 * agree:.2f}% | {100 * acc(a):.2f}% | {100 * acc(b):.2f}% | {"通過" if ok else "**未通過**"} |')
md = ['# H200 與 MI300X 的跨平台對照', '',
      '同一個工作在兩個平台的逐筆比對。門檻沿用 E5：|Δy| 中位數 < 0.02、選擇一致率 > 98%，且提示雜湊全部相同。'
      '由 `portable/h200/xplat_check.py` 產生。', '',
      '| 模型 | 工作 | 對齊筆數 | 提示相同 | \\|Δy\\| 中位數 | \\|Δy\\| 最大 | 選擇一致率 | 準確率 H200 | 準確率 MI300X | 判定 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---|', *rows]
out = VT / 'results/h200_xplat/README.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(md) + '\n')
print('\n'.join(md))
