# AAAI v3 MI300X 開發集結果

快照時間為 2026-11-12 09:45:57（台灣時間），使用單張 AMD Instinct MI300X。
執行與分析規格固定在 `cce5624703fe60d7eb8411aa4b26f3d4381861fb`（`prereg-v3`）。
原始快照共 23 個完成工作、137,599 筆 JSONL 結果、115 個檔案（80.5 MB）。

## 進度

| 模型 | Tier 1 dev 完成 | 快照狀態 |
|---|---:|---|
| Qwen3-VL-8B | 7/7 | 已完成 |
| Qwen3-VL-32B | 7/7 | 已完成 |
| InternVL3.5-8B | 7/7 | 已完成 |
| InternVL 38B | 0/7 | `dev_controls` 執行中 |
| Gemma-3 27B IT | 0/7 | 權重已就緒，GPU 實驗待排 |

Tier 1 dev 已完成 21 組。AMD 端 xcheck 已完成 2/2 組，Qwen3-VL-8B 和 32B 各有 320 筆 `episodes_v1` 紀錄。E5 的跨平台差值與 argmax 一致率待配對 3090 輸出後計算。

## 主任務

下表是 `episodes_v3` development split 的 choice accuracy。每格有 16 對 episode，對齊孿生的候選 logits 後取平均，排除第 1 回合，pet 與文字情境各佔一半。95% CI、兩種領域各自的結果、log score、FB、AUC₁₀、反轉曲線與配對差值均記在指標檔。

| 模型 | A 原始紀錄 | A+ 原始紀錄加正確性 | B 逐回合正確性 | C 累積摘要 | B−A | C−A |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 48.66% | 53.60% | 64.16% | 61.55% | +15.51 pp | +12.90 pp |
| Qwen3-VL-32B | 46.32% | 52.85% | 62.74% | 63.21% | +16.42 pp | +16.89 pp |
| InternVL3.5-8B | 47.11% | 53.32% | 63.92% | 62.82% | +16.81 pp | +15.70 pp |

三個模型的 B 與 C 在開發集的 accuracy 均高於 A。C 的穩定期 FB 為 87.2–97.7%，波動期 FB 為 49.2–49.6%。Qwen3-VL-32B 的 C accuracy 為 63.21%，log score 為 −4.477，A 的 log score 為 −1.262。這批數字用於開發集診斷，held-out 的主要檢定仍待執行。

## 正向對照

| 模型 | E1 明示可靠度（896 題） | E4 圖表讀取（192 題） | E4 面板讀取（128 題） | E4 綁定後明示可靠度（1,344 題） |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 85.60% | 63.02% | 100.00% | 71.35% |
| Qwen3-VL-32B | 82.81% | 67.19% | 100.00% | 66.82% |
| InternVL3.5-8B | 87.50% | 68.75% | 100.00% | 75.37% |

以 80% hard accuracy 為門檻，三個模型的 E1 明示可靠度均通過，圖表讀取與綁定後明示可靠度均未通過。E4 為探索性開發集結果。

## 檔案與重現

- 原始工作目錄保留完成標記、模型與資料指紋、執行 metadata、launcher log 和逐筆結果。[runs_mi300x](../../runs_mi300x)
- 指標檔包含工作清單、資料 SHA-256、逐模型及逐條件分數、bootstrap 設定和對照門檻。[metrics.json](metrics.json)
- `analyze_main.py` 以 formal 完整性模式分析 76,800 筆 choice 紀錄，episode bootstrap 10,000 次（seed 0）。已核對資料指紋、孿生配對、重複列和原始 logits。E1 與 E4 使用 `analyze_controls.py`。
- 開發資料 `episodes_v3/episodes.jsonl` 的 SHA-256 為 `e00b8793d93182f09e39a4028d4da63bf9b1f531190142513569a2c55d6d2f1c`。可依原版本的 `portable/setup_data.sh` 重建資料。
