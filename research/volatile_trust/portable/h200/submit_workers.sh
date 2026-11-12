#!/bin/bash
# 在 aaai repo 根目錄執行：[NWORK=4] [NCHAIN=3] [TIER="1 2 3"] bash research/volatile_trust/portable/h200/submit_workers.sh
# 送出 NWORK 個工作者（各 1 張卡），每個工作者連排 NCHAIN 段（QOS 每段上限 2 天，afterany 接續）。
# 工作者依 jobs_h200.jsonl 的順序（= 優先序，跨模型）挑工作；實驗室有幾張空卡就有幾個在跑，其餘排隊。
set -e
A=/work/ChihChungHsu/cactus8603/aaai
T=${TIER:+--tier $TIER}
for w in $(seq 1 ${NWORK:-4}); do
  prev=""; ids=""
  for i in $(seq 1 ${NCHAIN:-3}); do
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --gres=gpu:1 --job-name=aaai-w$w --time=2-00:00:00 \
           research/volatile_trust/portable/h200/run.slurm --env $A/env -- \
           python research/volatile_trust/portable/h200/worker.py $T)
    ids="$ids $prev"
  done
  echo "worker $w:$ids"
done
