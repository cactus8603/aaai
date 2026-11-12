"""Move outputs made with the v3 medical/sports icons aside, so run_jobs.py reruns them with the v3.1 icons.

Deviation D1 (EXPERIMENTS.md): the v3 medical and sports icons were illegible to the models, so every job that
shows icon reports is rerun. Text-only jobs, chart-reading, pet-photo-only, Psych-101, robustness, reasoning and
cross-platform jobs are unaffected and are left in place.

Affected job ids (per model): dev_controls, dev_binding, {dev,ho}_full_image, {dev,ho}_full_coded, {dev,ho}_sum_cp.

Usage (from the repository root; dry run by default):
  python research/volatile_trust/portable/supersede_icon_v3.py            # list what would move
  python research/volatile_trust/portable/supersede_icon_v3.py --apply    # move to runs_mi300x/_superseded_icons_v3/
Never run this while run_jobs.py is running (a job with a live _LOCK is refused).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs_mi300x'
AFFECTED = ('dev_controls', 'dev_binding', 'dev_full_image', 'ho_full_image', 'dev_full_coded', 'ho_full_coded',
            'dev_sum_cp', 'ho_sum_cp')


def icon_version(d):
    """'v3.1' if the output was made with the new icons, else 'v3' (key absent in v3 fingerprints)."""
    fps = list(d.glob('_FINGERPRINT*.json'))
    if not fps:
        return 'v3'
    vs = {json.loads(p.read_text()).get('icon_version', 'v3') for p in fps}
    return vs.pop() if len(vs) == 1 else 'mixed'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    dest_root = RUNS / '_superseded_icons_v3'
    moved = 0
    for model_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith('_')):
        for job in AFFECTED:
            d = model_dir / job
            if not d.exists():
                continue
            v = icon_version(d)
            if v == 'v3.1':
                print(f'keep   {d.relative_to(RUNS)} (already v3.1)')
                continue
            lock = d / '_LOCK'
            if lock.exists() and alive(int(lock.read_text().strip() or 0)):
                print(f'SKIP   {d.relative_to(RUNS)}: a runner is still writing it (stop run_jobs.py first)')
                continue
            dest = dest_root / model_dir.name / job
            print(f"{'move ' if a.apply else 'would move'} {d.relative_to(RUNS)} -> {dest.relative_to(RUNS)}  [{v}]")
            if a.apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(dest))
            moved += 1
    print(f"{moved} job output(s) {'moved' if a.apply else 'to move (dry run; add --apply)'}")


if __name__ == '__main__':
    main()
