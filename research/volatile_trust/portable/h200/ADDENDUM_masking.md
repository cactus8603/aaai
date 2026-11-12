# H200 附加工作與遮罩實驗的事先預測（2026-11-13，跑 held-out 之前寫定）

H200 不跑大於 38B 的模型（Qwen2.5-VL-72B 不送；`submit.sh` 的 `MODELS` 不要放 `qwen25vl72b`）。
`jobs_h200.jsonl` 前 343 行不變，後面附加 37 個 tier 4 工作（`make_jobs_h200.py`），依優先順序：

1. **held-out 遮罩確認**（15 個）：`<model>__ho_mask_{rep_ok,out_ok,mask_ok}`，模型為 Qwen3-VL-8B、InternVL3.5-8B、
   Qwen3-VL-32B、InternVL3.5-38B、Gemma-3-27B。對照組是同模型的 `ho_full_rawcoded`、`ho_full_coded`。
2. **8B 的 dev 遮罩**（6 個）：之前只在 3090 跑過，這裡重跑，讓同一模型的所有條件都在 H200 上。
3. **8B 的措辭關鍵對比**（16 個）：p1／p2 × 四種歷史，理由同上；2×2×2 拆解仍以 3090 結果為準（探索性）。

## 遮罩實驗設計（`masking_probe.py`）
沿用「原始歷史＋正確性註記」的每一行排版和長度，把標籤換成 `---`：
rep_ok 遮結果、out_ok 遮報告、mask_ok 兩者都遮（資訊等同逐回合標正確）。

## dev 的發現（3090，Qwen3-VL-8B／InternVL3.5-8B）
遮結果的主效果 +.074／+.096，遮報告約 0，mask_ok−raw+correct +.085／+.095，mask_ok−coded −.021／−.015；
copy 權重只要顯示結果就是 .27–.50，遮掉後 ≤ .12。

## held-out 的事先預測（每個模型分開，episode 層級 bootstrap 95% CI、twin 平均、t ≥ 2）
- P1：遮結果的主效果 [(rep_ok + mask_ok) − (raw+correct + out_ok)] / 2 > 0（CI 下界 > 0）。
- P2：mask_ok − raw+correct > 0。
- P3：遮報告的主效果小於遮結果的主效果（配對差的 CI 下界 > 0）。
- P4（描述性）：raw+correct 與 out_ok 的 copy 權重高於 rep_ok 與 mask_ok。
- 每個模型的結果都報告，包括不符合預測的。這組仍屬探索性附加，不是預先登記的主檢定。

分析：`python research/volatile_trust/analyze_masking.py --root runs_h200 --split ho`（dev 用 `--split dev`）。
