#!/bin/bash -l
# Slurm 範本，來源是 ICML2027/experiments/slurm/run.slurm（H200 節點 hgx-cn01）。
#
# 用法（在 repo 根目錄送出；輸出 log 路徑相對於送出時的目錄）：
#   sbatch slurm/run.slurm --env <conda-env-prefix> --check
#   sbatch slurm/run.slurm --env <conda-env-prefix> -- python path/to/script.py --args
# 覆寫資源時把 sbatch 參數放在腳本前面，例如：
#   sbatch --job-name=sd-gate0 --time=00:30:00 --gres=gpu:2 --cpus-per-task=32 --mem=340G \
#          slurm/run.slurm --env ... -- ...
# h200q 每張 GPU 約配 16 CPU、170G 記憶體；多卡時請等比例放大。
#
# 每次執行的輸出都在 $RUN_ROOT/<job-name>/<job-id>/，指令可以讀環境變數 $RUN_DIR 當輸出位置。
# 可用的環境變數：
#   RUN_ROOT  預設 /work/ChihChungHsu/cactus8603/icml/runs（放在 git 外）
#   MODULES   額外要載入的 module，空白分隔，例如 "cuda12.8/toolkit/12.8.1 gcc/14.2.0"
#
#SBATCH --job-name=icml
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

usage() {
    echo 'Usage: sbatch run.slurm --env <conda-env-prefix> (--check | -- <command> [args...])' >&2
    exit 2
}

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

env_dir=""
check=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --env) [[ $# -ge 2 ]] || usage; env_dir=$2; shift 2 ;;
        --check) check=1; shift ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[[ -n $env_dir ]] || usage
if [[ $check -eq 0 && $# -eq 0 ]]; then usage; fi
if [[ ! -x $env_dir/bin/python ]]; then
    echo "Conda env not found: $env_dir" >&2
    exit 1
fi

repo_dir=$PWD
work_home=/work/ChihChungHsu/cactus8603
export RUN_ROOT=${RUN_ROOT:-$work_home/aaai/runs_slurm}
export RUN_DIR=$RUN_ROOT/${SLURM_JOB_NAME}/${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# /tmp 只有約 20G，所有快取與暫存都放到 /work（配額 1.5T）。
export HF_HOME=${HF_HOME:-$work_home/.cache/huggingface}
export PIP_CACHE_DIR=$work_home/.cache/pip
export TRITON_CACHE_DIR=$work_home/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$work_home/.cache/torchinductor
export VLLM_CACHE_ROOT=$work_home/.cache/vllm
export TMPDIR=$work_home/tmp/$SLURM_JOB_ID
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

module load miniconda3
for m in ${MODULES:-}; do module load "$m"; done

py() { conda run --no-capture-output --prefix "$env_dir" python "$@"; }

# ---- 重現資訊 ----
{
    echo "job_id: $SLURM_JOB_ID"
    echo "job_name: $SLURM_JOB_NAME"
    echo "node: $(hostname)"
    echo "start: $(date -Is)"
    echo "gpus: ${CUDA_VISIBLE_DEVICES:-}"
    echo "cpus: ${SLURM_CPUS_PER_TASK:-}"
    echo "env: $env_dir"
    echo "modules: ${MODULES:-}"
    echo "repo_commit: $(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo none)"
    echo "repo_dirty: $(git -C "$repo_dir" status --porcelain 2>/dev/null | wc -l)"
    printf 'command:'; printf ' %q' "$@"; echo
} > "$RUN_DIR/meta.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap --format=csv > "$RUN_DIR/gpu.csv"
conda list --prefix "$env_dir" --export > "$RUN_DIR/conda-list.txt" 2>/dev/null || true
py -m pip freeze > "$RUN_DIR/pip-freeze.txt" 2>/dev/null || true
cat "$RUN_DIR/meta.txt"
cat "$RUN_DIR/gpu.csv"

# ---- 環境檢查：每次都跑；--check 只跑這段 ----
py - <<'PYTHON'
import importlib, os, shutil
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU in this Slurm job.")
torch.zeros(1, device="cuda").item()
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} sm{p.major}{p.minor} {p.total_memory/2**30:.0f} GiB")
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
for name in ("vllm", "transformers", "flash_attn", "flashinfer", "torchao"):
    try:
        print(f"{name} {importlib.import_module(name).__version__}")
    except Exception:
        pass
hf_home = os.environ["HF_HOME"]
print(f"HF_HOME {hf_home} free {shutil.disk_usage(hf_home).free/1e12:.1f} TB (filesystem; user quota is 1.5 TB)")
token_file = os.path.join(hf_home, "token")
if not (os.environ.get("HF_TOKEN") or os.path.exists(token_file)):
    print("WARNING: no HF token; gated models (Llama) will fail and downloads may be rate-limited")
PYTHON

if [[ $check -eq 1 ]]; then
    echo "check: OK"
    exit 0
fi

# ---- 執行實驗 ----
start=$(date +%s)
set +e
conda run --no-capture-output --prefix "$env_dir" "$@" 2>&1 | tee "$RUN_DIR/stdout.log"
status=${PIPESTATUS[0]}
set -e
{
    echo "end: $(date -Is)"
    echo "elapsed_s: $(( $(date +%s) - start ))"
    echo "exit_code: $status"
} | tee -a "$RUN_DIR/meta.txt"
exit "$status"
