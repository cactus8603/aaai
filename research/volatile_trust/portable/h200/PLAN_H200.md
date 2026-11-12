# H200 執行計畫（2026-11-13 重新整理；在看到任何 held-out 模型結果之前決定）

原本的清單會跑不完（全部約 240 GPU 小時，其中 E9 推理約 80 小時、72B 約 54 小時）。
這裡依「論文主張需要什麼」重新分級，`jobs_h200.jsonl` 由 `make_jobs_h200.py` 產生，共 177 個工作、約 97 GPU 小時（同日修訂多次：第一版 161 個、約 74 小時，砍太多；折衷版 241 個、約 124 小時；再刪掉貢獻最小的約 27 小時）。

**2 張 H200 的時程**（工作可分給任一張空卡；H200 速度設為 MI300X 的 0.8／1.0／1.3 倍）：tier 1 完成約 11／13／18 小時，tier 2 約 28／35／46 小時，全部約 39／48／63 小時。不含排隊時間。
工作 id、參數、輸出目錄名稱都和原清單相同（只有輸出移到 `runs_h200/`、`tier` 改成優先序），所以續跑會自動跳過已完成的工作。

模型：Qwen3-VL-8B、InternVL3.5-8B、Qwen3-VL-32B、InternVL3.5-38B、Gemma-3-27B（≤ 38B）；`qwen3vl8b_thinking` 只在 tier 3。不跑 Qwen2.5-VL-72B。

GPU 小時是用 MI300X 實測的每筆秒數推估的（8B 約 .06–.07 秒、32/38B 約 .17 秒、Gemma-3-27B 約 .28 秒），H200 實際速度請以 stage 1 的 log 為準。

## 平台分工（論文寫法）
- **MI300X 暫停使用（2026-11-13）**：不再送任何工作；只沿用已完成的結果（dev 主任務，以及跨平台對照用的 `dev_full_text`，5 個模型都完整）。MI300X 清單上未完成的工作（InternVL3.5-38B 的部分 dev、tier 2–4）不補跑，需要的部分都已在本清單。
- **開發**在 MI300X 與 RTX 3090：提示與 icon 設計、dev 上的主任務與探索性檢查（措辭、遮罩）。
- **正式評估全部在 H200**：held-out、穩健性資料、Psych-101，以及論文會報告的 dev-only 對照組（明示可靠度、綁定、圖表可讀性）和選 K 用的 dev 視窗。
- **跨平台對照**：每個模型在 H200 重跑 `dev_full_text`（stage 1），和 MI300X 的同一工作逐題比對（y 的差、選擇一致率、準確率差）；`xcheck` 另外對照 3090。
  論文寫法：「Development runs were done on MI300X and RTX 3090 GPUs; all held-out, robustness and human-comparison evaluations were run on H200 GPUs. The development main task, rerun on H200 for every model, agreed with the MI300X run (...).」

## tier 1（必要，約 27 GPU 小時；Gemma-3-27B 約 10 小時最久）
| 工作 | 用途 |
|---|---|
| `dev_full_text`、`xcheck` | 跨平台對照（見上） |
| `ho_full_text`、`ho_sum_all` | **唯一的主要檢定**：held-out 摘要 − 原始（C1）、Table main、Fig. gap、積分核 |
| `ho_full_rawcoded`、`ho_full_coded` | 落差拆解（C1b）：原始＋標正確、逐回合標正確（含圖片版） |
| `dev_controls`、`dev_binding`、`dev_chartread` | 明示可靠度（D−A，E1）、綁定、圖表可讀性；依設計只用 dev |
| `rob_full`、`rob_sum_all` | 穩健性資料上的主要比較（Table robust） |

## tier 2（次要檢定、人機比較與表格，約 43 GPU 小時）
依序：`ho_sum_cp`（變化點摘要的 TOST）→ `dev_sum_window{5,10,20}`（在 dev 選 K）→ `ho_sum_window{5,10,20}`（選定視窗的 TOST、Table variants）→ `ho_full_chart`（Table main 的 Chart 欄）→ `ho_full_image`（多模態）→ **Psych-101 全部六種**（`text_hidden`、`text_told`、`image_outcome`、`chart`、`perc_outcome`、`perc_chart`；人與 LLM 的比較，約 15 小時）→ `ho_mask_*`（遮罩結果的 held-out 確認，預測見 `ADDENDUM_masking.md`）。

## tier 3（有時間才跑，約 27 GPU 小時，依序）
1. `ho_sum_oracle`、`ho_window10`、`ho_cp_hard`（Table variants 的「截斷原始歷史」欄；也回答「是不是歷史太長」）。
2. 三個大模型的精簡表格措辭（p2；Limitations 的排版依賴）。
3. 推理模式（約 16 GPU 小時，放最後）：`qwen3vl8b_thinking` 全部（4 個 shard × 原始／摘要）。

## 不在 H200 跑
- Qwen2.5-VL-72B；E9 的 Qwen3-VL-32B CoT 與 32B thinking（16 小時；一個 thinking 模型就足以回答「推理能不能補救」）。
- Gemma-3-12B 的措辭拆解與遮罩（約 5 小時；探索性，8B 的拆解已在 3090 做完）。
- 大模型的措辭 p1（約 3 小時；8B 上 p1 和原始措辭幾乎相同）。
- 穩健性資料的額外條件 `rob_full_coded`、`rob_sum_cp`、`rob_sum_window10`（約 3 小時；主要比較 `rob_full`／`rob_sum_all` 已在 tier 1）。
- dev 主任務的其他格式：屬於開發，沿用 MI300X。
- 8B 的 dev 遮罩和措辭（3090 已做，探索性）；held-out 遮罩確認在 tier 2。

## 送出
**2026-11-13 改用工作者（`worker.py`、`submit_workers.sh`）**：每個工作者 1 張卡，依本清單的順序跨模型挑下一個工作（重新讀清單、用 `_LOCK` 記 Slurm job id 認領、權重沒下載的先跳過），實驗室有幾張空卡就跑幾個。`NWORK=4 NCHAIN=3 bash research/volatile_trust/portable/h200/submit_workers.sh`。跨平台對照不再擋後面的工作：`dev_full_text` 一完成就和 MI300X 比，不一致就 `scancel` 所有工作者。以下是舊的每模型送法，保留備查。

`submit.sh` 不用改：stage 1 跑 tier 1 的 `dev_` 與 `xcheck`，stage 2 依 tier 1 → 2 → 3 →（4，現在沒有）。
`MODELS="qwen3vl8b internvl8b qwen3vl32b internvl38b gemma3_27b"`。`qwen3vl8b_thinking` 沒有 tier 1 工作（stage 1 會沒事可做），建議直接送 `--tier 3` 的工作，或等主要模型跑完再用空出的卡跑。
建議 Gemma-3-27B 用 `NGPU=2`（最慢）。截稿前若 tier 2 跑不完，論文以 tier 1 為主，tier 2 未完成的表格照實標示。
