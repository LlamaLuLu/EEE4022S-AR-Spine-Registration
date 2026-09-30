"""
List screenshot poses with quality flags, so you can see which shots are
usable before measuring them.

Each screenshot saved with 's' has a _pose.json (state, markers, RMS, pose,
envelope tier). A shot is flagged if the pose can't be trusted for
measurement.

Usage (from repo root):
    python scripts/atp/list_shots.py "data/screenshots/atp2_tiers/shot_*_pose.json"
    python scripts/atp/list_shots.py "data/screenshots/atp2_tiers/shot_20260929_08*_pose.json"
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "registration"))
try:
    from ar_theme import envelope_tier
except Exception:
    envelope_tier = None

MIN_MARKERS = 4
MAX_RMS_PX = 3.0


def flags(p):
    f = []
    if not p.get("pnp_ok", False):
        f.append("PnP failed")
    if p.get("state") not in ("NAVIGATING", "VERIFYING"):
        f.append(f"state {p.get('state')}")
    if (p.get("n_markers") or 0) < MIN_MARKERS:
        f.append(f"{p.get('n_markers')} markers")
    r = p.get("rms_px")
    if r is None:
        f.append("no RMS")
    elif r > MAX_RMS_PX:
        f.append(f"RMS {r:.1f}px")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", help="Glob of *_pose.json files")
    args = ap.parse_args()

    files = sorted(glob.glob(args.shots))
    if not files:
        print(f"No files match {args.shots}")
        return

    print(f"{'shot':<24s}{'state':<11s}{'mk':>3s}{'rms':>6s}{'dist':>6s}"
          f"{'elev':>6s}{'az':>6s}  {'tier':<8s} flags")
    ok_n = 0
    for f in files:
        p = json.load(open(f))
        name = re.sub(r"^shot_|_pose\.json$", "", Path(f).name)
        tier = p.get("env_tier")
        if tier is None and envelope_tier is not None:
            tier = envelope_tier(p.get("elevation_deg", 0), p.get("azimuth_deg", 0),
                                 p.get("distance_mm"))
        fl = flags(p)
        ok_n += not fl
        rms = p.get("rms_px")
        print(f"{name:<24s}{str(p.get('state')):<11s}{p.get('n_markers', 0):>3d}"
              f"{('%.1f' % rms) if rms is not None else '--':>6s}"
              f"{p.get('distance_mm', 0) / 10:>6.0f}{p.get('elevation_deg', 0):>6.0f}"
              f"{p.get('azimuth_deg', 0):>6.0f}  {str(tier):<8s} "
              f"{'; '.join(fl) if fl else 'ok'}")
    print(f"\n{len(files)} shots, {ok_n} without flags")


if __name__ == "__main__":
    main()