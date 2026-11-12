# H200 執行計畫（2026-11-13 重新整理；在看到任何 held-out 模型結果之前決定）

原本的清單會跑不完（全部約 240 GPU 小時，其中 E9 推理約 80 小時、72B 約 54 小時）。
這裡依「論文主張需要什麼」重新分級，`jobs_h200.jsonl` 由 `make_jobs_h200.py` 產生，共 161 個工作。
工作 id、參數、輸出目錄名稱都和原清單相同（只有輸出移到 `runs_h200/`、`tier` 改成優先序），所以續跑會自動跳過已完成的工作。

模型：Qwen3-VL-8B、InternVL3.5-8B、Qwen3-VL-32B、InternVL3.5-38B、Gemma-3-27B（≤ 38B）。不跑 Qwen2.5-VL-72B、Gemma-3-12B。

GPU 小時是用 MI300X 實測的每筆秒數推估的（8B 約 .06–.07 秒、32/38B 約 .17 秒、Gemma-3-27B 約 .28 秒），H200 實際速度請以 stage 1 的 log 為準。

## tier 1（必要，約 25 GPU 小時；Gemma-3-27B 約 9 小時最久）
| 工作 | 用途 |
|---|---|
| `ho_full_text`、`ho_sum_all` | **唯一的主要檢定**：held-out 摘要 − 原始（C1）、Table main、Fig. gap、積分核 |
| `ho_full_rawcoded`、`ho_full_coded` | 落差拆解（C1b）：原始＋標正確、逐回合標正確（含圖片版） |
| `dev_controls`、`dev_binding`、`dev_chartread` | 明示可靠度（D−A，E1）、綁定、圖表可讀性；依設計只用 dev |
| `rob_full`、`rob_sum_all` | 穩健性資料上的主要比較（Table robust） |
| `xcheck` | 跨平台檢查（和 MI300X 對照） |

## tier 2（次要檢定與表格，約 31 GPU 小時）
依序：`ho_sum_cp`（變化點摘要的 TOST）→ `dev_sum_window{5,10,20}`（在 dev 選 K）→ `ho_sum_window{5,10,20}`（選定視窗的 TOST、Table variants）→ `ho_full_chart`（Table main 的 Chart 欄）→ `ho_full_image`（多模態）→ `psy_text_{hidden,told}`（Psych-101 主要擬合）→ `ho_mask_*`（遮罩結果的 held-out 確認，預測見 `ADDENDUM_masking.md`）。

## tier 3（有時間才跑，約 18 GPU 小時）
`ho_sum_oracle`、`rob_sum_cp`、`rob_sum_window10`、`ho_window10`、`ho_cp_hard`、`rob_full_coded`、三個大模型的精簡表格措辭（`para_p2_*`），
以及推理模式的縮小版：只跑 `qwen3vl8b_thinking` 的第 0 個 shard（四分之一題目，原始與摘要各一）。

## 不在 H200 跑
- **dev 主任務**（`dev_full_*`、`dev_sum_all` 等）：論文正文改用 held-out；dev 結果沿用 MI300X 的，只作開發用途。
- **E9 其餘部分**（CoT、32B thinking、其餘 shard）：每題自由生成上萬 token，成本最高；論文附錄 H 改為小規模子集。
- **Psych-101 的圖片／圖表／知覺變體**、dev 的圖片／圖表／cp_hard／window 變體。
- **措辭**：p1 和 2×2×2 拆解（8B 在 3090 已做完，屬探索性）；Gemma-3-12B 全部。
- **8B 的 dev 遮罩**：3090 已做；held-out 確認在 tier 2。

## 送出
`submit.sh` 不用改：stage 1 跑 tier 1 的 `dev_` 與 `xcheck`，stage 2 依 tier 1 → 2 → 3 →（4，現在沒有）。
`MODELS="qwen3vl8b internvl8b qwen3vl32b internvl38b gemma3_27b"`；要跑推理子集時另外加 `qwen3vl8b_thinking`。
建議 Gemma-3-27B 用 `NGPU=2`（最慢）。截稿前若 tier 2 跑不完，論文以 tier 1 為主，tier 2 未完成的表格照實標示。
