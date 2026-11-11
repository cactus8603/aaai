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
python research/volatile_trust/portable/download_models.py --models qwen3vl8b internvl8b qwen3vl32b internvl38b qwen25vl72b
```
合計約 325 GB。另外兩個選配模型 `qwen3vl235b` 和 `internvl241b` 各約 470 GB，每個需要 4 張 MI300X。

## 5. 執行
```bash
python research/volatile_trust/portable/make_jobs.py                       # 產生工作清單（177 個工作）
python research/volatile_trust/portable/run_jobs.py --gpus 0 1 --only xcheck    # 先跑跨平台檢查，結果傳回來比對
nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 > launcher.out 2>&1 &
```
- 只跑部分工作時，可以加 `--models qwen3vl32b` 或 `--only dev_`。加 `--dry-run` 會列出要執行的指令，但不真的執行。
- 每個 runner 跑完會先核對列數，正確才寫入 `_COMPLETE*` 標記。沒有標記的工作，重新啟動時會從中斷處續跑。
- 正在執行中的工作有 `_LOCK` 檔，啟動器不會重複啟動它。

## 6. 傳回結果
```bash
tar czf results_mi300x.tar.gz research/volatile_trust/runs_mi300x
```

## 注意
- 同一個模型的所有條件，都要在同一個平台上跑完。
- `episodes_v2` 是正式資料；`episodes_v1` 只用於跨平台檢查。
- 如果環境預先設了 `ROCR_VISIBLE_DEVICES`（例如在 SLURM 或 docker 裡），啟動器會自動把它清掉，以免 GPU 編號錯位。
