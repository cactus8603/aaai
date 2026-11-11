# 在 MI300X 伺服器上執行（可攜套件）

## 0. 這個套件包含什麼
- 實驗程式：`research/volatile_trust/*.py`，以及共用的 backend `research/source_trust/model_backends.py`。
- 已凍結的資料：`episodes_v1`（dev 和 held-out）、`episodes_robust`（穩健性），還有實際用到的寵物圖片。
- 工作清單：`portable/jobs.jsonl`，由 `make_jobs.py` 產生。每一行是一個工作。
- 啟動器：`portable/run_jobs.py`，會自動把工作分配到各張 GPU，可以中斷後續跑。

## 1. 解壓縮
```bash
mkdir -p ~/vt && cd ~/vt
tar xzf volatile_trust_portable.tar.gz      # 以下指令都在 ~/vt 底下執行
```

## 2. 建立環境（ROCm）
```bash
python3 -m venv .venv && source .venv/bin/activate
# 依伺服器的 ROCm 版本安裝 PyTorch，例如 ROCm 6.2：
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r research/volatile_trust/portable/requirements-rocm.txt
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```
最後一行應該顯示 HIP 版本和 MI300X。如果用 docker 的 `rocm/pytorch` 映像檔，可以跳過安裝 torch 那一行。

## 3. 下載模型（版本已固定）
```bash
python research/volatile_trust/portable/download_models.py --models qwen3vl32b internvl38b qwen25vl72b qwen3vl8b internvl8b
```
各模型大小約為：
| 模型 | 大小 |
|---|---|
| 32B | 67 GB |
| 38B | 77 GB |
| 72B | 147 GB |
| 8B | 各約 16 GB |

以上總共約 325 GB。另外兩個選配模型 `qwen3vl235b` 和 `internvl241b` 各約 470 GB，每個需要 4 張 MI300X。

## 4. 執行
```bash
# 先跑跨平台檢查（約 10 分鐘），把結果傳回來比對
python research/volatile_trust/portable/run_jobs.py --gpus 0 1 --only xcheck
# 然後跑全部工作，例如 8 張卡：
nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 > launcher.out 2>&1 &
# 只跑特定模型，或只跑 dev：
python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 --models qwen3vl32b internvl38b --only dev_
```
- 每個工作完成後，會在它的輸出資料夾寫一個 `_DONE`。重跑時會自動跳過已完成的工作，未完成的會從中斷處續跑。
- 先用 `--dry-run` 可以看到會執行哪些指令，但不會真的執行。

## 5. 傳回結果
```bash
tar czf results_mi300x.tar.gz research/volatile_trust/runs_mi300x
```
把這個檔案傳回原本的機器，解壓縮到同一個位置即可。分析程式會讀 `runs_mi300x/<模型>/<工作>/`。

## 注意
- **同一個模型的所有條件都要在 MI300X 上跑完**，不要一部分在 3090、一部分在 MI300X。
- 3090 上的結果只當作前導研究（pilot），用來決定設計。
- 不要修改 `episodes_*` 裡的檔案。它們已經凍結，並記錄了 sha256。
