# Volatile Trust (anonymous submission code)

Do frozen vision-language models adapt their trust to sources whose reliability changes?
An in-context adaptation of the Behrens et al. (2008) volatile-advisor paradigm, with a
change-point (BOCPD) controller and an external Psych-101 task.

## Quick start (any GPU server; ROCm or CUDA)
```bash
python3 -m venv .venv && source .venv/bin/activate
# install torch for your platform (ROCm/CUDA), then:
pip install -r research/volatile_trust/portable/requirements-rocm.txt
bash research/volatile_trust/portable/setup_data.sh          # regenerates data, verifies sha256
python research/volatile_trust/portable/download_models.py --models qwen3vl32b internvl38b qwen25vl72b qwen3vl8b internvl8b
python research/volatile_trust/portable/make_jobs.py
python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 --only xcheck   # sanity check first
nohup python research/volatile_trust/portable/run_jobs.py --gpus 0 1 2 3 4 5 6 7 > launcher.out 2>&1 &
```
See `research/volatile_trust/portable/README_RUN.md` for details. Results go to `research/volatile_trust/runs_mi300x/`.
