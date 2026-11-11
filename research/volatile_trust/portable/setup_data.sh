#!/bin/bash
# Regenerate all data deterministically on a fresh clone and verify sha256 against portable/checksums.json.
# Usage (from anywhere): bash research/volatile_trust/portable/setup_data.sh   [PY=/path/to/python]
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
VT=research/volatile_trust
echo "[1/6] Oxford-IIIT Pet: official download (MD5-checked), seeded selection, letterbox 224/448"
$PY research/source_trust/prepare_pet_data.py --cache .cache/datasets/oxford-iiit-pet --output research/source_trust/natural_images_v1 > /dev/null
echo "[2/6] Main-task episodes v3 (canonical: dev 16 + held-out 48, with twins, scenarios)"
$PY $VT/generate_episodes.py --out $VT/episodes_v3
echo "[3/6] Pilot episodes v1 (only for the cross-platform check against 3090 pilot rows)"
$PY $VT/generate_episodes.py --legacy-v1 --out $VT/episodes_v1
echo "[4/6] Robustness episodes v3 (IID correctness, irregular dwell 6-16)"
$PY $VT/generate_robust.py $VT/episodes_robust_v3
echo "[5/6] Psych-101 xiong2023: pinned download (sha256-checked) + build"
$PY -I $VT/psych101/fetch_xiong.py
$PY -I $VT/psych101/build_xiong.py > /dev/null
echo "[6/6] Verify checksums"
$PY - <<'EOF'
import hashlib, json, sys
ref = json.load(open('research/volatile_trust/portable/checksums.json'))
bad = 0
for path, want in ref.items():
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    bad += got != want
    print(('OK   ' if got == want else 'FAIL ') + path)
miss = 0
for d in ("episodes_v1", "episodes_v3"):
    idx = json.load(open(f'research/volatile_trust/{d}/image_index.json'))
    for r in idx.values():
        for k in ('224', '448'):
            if hashlib.sha256(open(r['path' + k], 'rb').read()).hexdigest() != r['sha' + k]:
                miss += 1
print(f'pet image hash mismatches: {miss}')
sys.exit(1 if bad or miss else 0)
EOF
echo "Data ready and identical to the reference machine."
