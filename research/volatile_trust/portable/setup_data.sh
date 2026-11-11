#!/bin/bash
# Regenerate all data deterministically on a fresh machine, then verify sha256 against portable/checksums.json.
# Usage (from the repo root): bash research/volatile_trust/portable/setup_data.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
echo "[1/4] Oxford-IIIT Pet: official download (MD5-checked) + seeded selection + letterbox 224/448"
$PY research/source_trust/prepare_pet_data.py --cache .cache/datasets/oxford-iiit-pet --output research/source_trust/natural_images_v1 > /dev/null
echo "[2/4] Main-task episodes (dev 16 + held-out 48, with twins)"
$PY research/volatile_trust/generate_episodes.py --out research/volatile_trust/episodes_v1
echo "[3/4] Robustness episodes (IID, irregular dwell)"
$PY research/volatile_trust/generate_robust.py research/volatile_trust/episodes_robust
echo "[4/4] Verify checksums"
$PY - <<'EOF'
import hashlib, json, sys
ref = json.load(open('research/volatile_trust/portable/checksums.json'))
bad = 0
for path, want in ref.items():
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    ok = got == want
    bad += not ok
    print(('OK   ' if ok else 'FAIL ') + path)
idx = json.load(open('research/volatile_trust/episodes_v1/image_index.json'))
miss = 0
for iid, r in idx.items():
    for k in ('224', '448'):
        p = r['path' + k]
        if hashlib.sha256(open(p, 'rb').read()).hexdigest() != r['sha' + k]:
            miss += 1
print(f'pet image hash mismatches: {miss}')
sys.exit(1 if bad or miss else 0)
EOF
echo "Data ready and identical to the reference machine."
