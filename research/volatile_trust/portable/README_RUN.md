# 在 MI300X（或任何 GPU）伺服器上執行

## 1. 取得程式碼
```bash
git clone <repo URL> vt && cd vt          # 以下指令都在 repo 根目錄執行
```

## 2. 建立環境（ROCm）
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2   # 依伺服器的 ROCm 版本調整
pip install -r research/volatile_trust/portable/requirements-rocm.txt
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```
- 如果用 docker 的 `rocm/pytorch` 映像檔，可以跳過安裝 torch 那一行。
- 最後一行應該顯示 HIP 版本，以及 MI300X 的數量。

## 3. 重新生成資料（固定亂數種子，生成後自動驗證 sha256）
```bash
bash research/volatile_trust/portable/setup_data.sh
```
- 會下載以下資料，再生成所有 episode：
  - Oxford-IIIT Pet，約 800 MB
  - Psych-101 xiong2023：以串流方式讀取 860 MB，只保留需要的 30 行
- 最後一行要出現 `Data ready and identical to the reference machine.`。如果出現 FAIL，**不要繼續**，先把輸出傳回來。

## 4. 下載模型（版本已固定）
```bash
export HF_TOKEN=<你的 HF read token>   # 或先執行 `hf auth login`
python research/volatile_trust/portable/download_models.py --models qwen3vl8b internvl8b qwen3vl32b internvl38b gemma3_27b qwen25vl72b qwen3vl8b_thinking qwen3vl32b_thinking
```
- **Gemma-3 是 gated 模型**：先用同一個 HF 帳號在 https://huggingface.co/google/gemma-3-27b-it 同意授權，再設定 `HF_TOKEN`（read 權限即可）。沒有授權或沒有 token 時，`download_models.py` 會停下來並提示。授權沒通過就先回報，替代方案（Mistral-Small-3.2-24B）見 `EXPERIMENTS.md` §7。
- 合計約 470 GB。另外兩個選配模型 `qwen3vl235b` 和 `internvl241b` 各約 470 GB，每個需要 4 張 MI300X。

## 5. 執行
```bash
python research/volatile_trust/portable/make_jobs.py      # 產生工作清單：266 個工作（tier 1：69、tier 2：99、tier 3：98）
```
每個工作都有 `tier` 欄位。`run_jobs.py --tier 1`（可給多個，例如 `--tier 2 3`）只跑指定 tier 的工作，可以和 `--models`、`--only`（工作 id 的子字串）合併使用。加 `--dry-run` 會列出要執行的指令，但不真的執行。

**建議順序（W02；不要一開始就全量執行）：**
1. **tier 1 的 dev 和跨平台檢查**，結果傳回來核對：
   ```bash
   python research/volatile_trust/portable/run_jobs.py --gpus 0 1 --tier 1 --only xcheck        # 跨平台檢查，先傳回比對
   nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 --tier 1 --only dev_ > launcher_t1_dev.out 2>&1 &
   ```
2. **確認 dev 的核心鏈可用之後**（E1 對照、A／A+／B／C 的 dev 結果都能被分析程式讀入並通過完整性檢查，而且規格已打 tag `prereg-v3`），才跑 tier 1 的 held-out 和 robust：
   ```bash
   nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 --tier 1 --only ho_ rob_ > launcher_t1_ho.out 2>&1 &
   ```
3. 之後再跑 **tier 2**，最後是 **tier 3**：
   ```bash
   nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 --tier 2 > launcher_t2.out 2>&1 &
   nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 --tier 3 > launcher_t3.out 2>&1 &
   ```

- **E9 推理工作很長**（tier 2，24 個 `*__e9_*` 工作）：每題自由生成，`--max-new-tokens 16384`，每題可能產生上萬個 token，比單一 token 讀出的工作慢好幾個數量級。每個模型 × 歷史分成 4 個 shard，可以中斷後續跑；建議放在 tier 2 的最後，或單獨用 `--only e9_` 執行。
- 每個 runner 跑完會先核對列數，正確才寫入 `_COMPLETE*` 標記。沒有標記的工作，重新啟動時會從中斷處續跑。
- 正在執行中的工作有 `_LOCK` 檔，啟動器不會重複啟動它。

## 5b. icon 更新後的重跑（偏離 D1，2026-11-12）
medical／sports 的 icon 換成 v3.1。**先停掉 run_jobs.py**，再 `git pull`，然後：
```bash
python research/volatile_trust/portable/supersede_icon_v3.py            # 先看會搬哪些（dry run）
python research/volatile_trust/portable/supersede_icon_v3.py --apply    # 搬到 runs_mi300x/_superseded_icons_v3/
```
之後照原本的指令重新啟動 run_jobs.py，被搬走的工作會自動用新 icon 重跑；其他已完成的工作會被跳過。
只影響這 8 種工作：`dev_controls`、`dev_binding`、`dev_full_image`、`ho_full_image`、`dev_full_coded`、`ho_full_coded`、`dev_sum_cp`、`ho_sum_cp`。
如果某個含圖片的工作沒有先搬走就續跑，runner 會拒絕（訊息含 `icons changed`），照上面搬走再跑即可。

## 6. 傳回結果
```bash
tar czf results_mi300x.tar.gz research/volatile_trust/runs_mi300x
```

## 注意
- 同一個模型的所有條件，都要在同一個平台上跑完。
- `episodes_v3` 是正式資料；`episodes_v1` 只用於跨平台檢查。
- 如果環境預先設了 `ROCR_VISIBLE_DEVICES`（例如在 SLURM 或 docker 裡），啟動器會自動把它清掉，以免 GPU 編號錯位。
