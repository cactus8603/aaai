"""Exploratory: re-run image-report control conditions for medical/sports with alternative icons.

Patches run_volatile.icon with variants from icon_probe.VARIANTS, then runs run_controls.py unchanged.
Usage: python icon_probe_controls.py --medical <variant> --sports <variant> [--script volatile] -- <run_controls.py or run_volatile.py arguments>
       (variant names as in icon_probe.VARIANTS; 'v3_strip' / 'v3_trophy_x' are the original icons)
       Legacy: --icons new (= plus_minus + faces) | v3
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'source_trust'))
import icon_probe  # noqa: E402
import run_volatile  # noqa: E402


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:sys.argv.index('--')] else default


legacy = opt('--icons', None)
SCRIPT = opt('--script', 'controls')
med = opt('--medical', 'plus_minus' if legacy == 'new' else 'v3_strip')
spo = opt('--sports', 'faces' if legacy == 'new' else 'v3_trophy_x')
sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index('--') + 1:]
NEW = {'medical': icon_probe.VARIANTS['medical'][med], 'sports': icon_probe.VARIANTS['sports'][spo]}
orig = run_volatile.icon
run_volatile.icon = lambda dom, lab, P: NEW[dom](lab, P) if dom in NEW else orig(dom, lab, P)
print(f'icons: medical={med} sports={spo}', flush=True)
if SCRIPT == 'volatile':
    run_volatile.main()
else:
    import run_controls  # noqa: E402
    run_controls.main()
