#!/bin/bash
# 在 aaai repo 根目錄執行：MODELS="qwen3vl32b" [DEP=<jobid>] bash research/volatile_trust/portable/h200/submit.sh
# 每個模型一個 Slurm 工作對（72B 用 2 張卡）：
#   第一階段：tier 1 的 dev 與跨平台檢查。結果先和 MI300X 對照，不一致就 scancel 第二階段。
#   第二階段（相依第一階段成功）：其餘全部，依 tier 1 → 2 → 3 → 4。
# DEP：第一階段要等的工作（例如同一張卡上前一個模型的第二階段），用 afterany。
set -e
A=/work/ChihChungHsu/cactus8603/aaai
J=research/volatile_trust/portable/h200/jobs_h200.jsonl
for m in $MODELS; do
  if [ "$m" = qwen25vl72b ]; then G=2; GPUS="0 1"; else G=1; GPUS=0; fi
  RJ="python research/volatile_trust/portable/run_jobs.py --jobs $J --gpus $GPUS --models $m"
  R="--gres=gpu:$G --cpus-per-task=$((16*G)) --mem=$((170*G))G"
  d=${DEP:+--dependency=afterany:$DEP}
  s1=$(sbatch --parsable $d $R --job-name=aaai-$m-s1 --time=3-00:00:00 research/volatile_trust/portable/h200/run.slurm --env $A/env -- \
       bash -c "$RJ --tier 1 --only dev_ xcheck")
  s2=$(sbatch --parsable --dependency=afterok:$s1 $R --job-name=aaai-$m-s2 --time=21-00:00:00 research/volatile_trust/portable/h200/run.slurm --env $A/env -- \
       bash -c "$RJ --tier 1 && $RJ --tier 2 && $RJ --tier 3 && $RJ --tier 4")
  echo "$m stage1=$s1 stage2=$s2"
done
