"""
ATP-8: registration repeatability across cycles.

Each step of the log is one registration cycle (launch or press 'r', wait for
NAVIGATING, hold still ~5 s, press 'n'). The camera does not move, so any
difference in the estimated pose between cycles is registration scatter.

Per cycle (NAVIGATING frames only, first --trim seconds dropped):
  RMS, predicted TRE for the chosen target, camera position and angles.
Across cycles: mean, SD, coefficient of variation (CV), plus the pose
repeatability (3D camera position SD in mm, angle SDs in degrees).

Pass criterion (ATP-8): predicted TRE < 2 mm in every cycle.

Usage (from repo root):
    python scripts/atp/analyse_repeatability.py data/atp/atp8_repeat.csv
    python scripts/atp/analyse_repeatability.py data/atp/atp8_repeat.csv --target L3_E_L --trim 1
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def fnum(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def circ_mean_sd(deg):
    """Mean and SD of angles in degrees, safe across the 0/360 wrap."""
    a = np.radians(np.asarray(deg, float))
    m = np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
    d = (np.degrees(a - m) + 180.0) % 360.0 - 180.0
    return (np.degrees(m) % 360.0), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--target", default="L3_E_L", help="Target label for TRE (default L3_E_L)")
    ap.add_argument("--trim", type=float, default=1.0, help="Seconds to drop at the start of each cycle")
    ap.add_argument("--min-frames", type=int, default=30, help="Ignore cycles with fewer NAVIGATING frames")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.log)) if r.get("state") != "SCREENSHOT"]
    if not rows:
        print("No data.")
        return
    tre_cols = [k for k in rows[0] if k.startswith("TRE_")]
    col = f"TRE_{args.target}"
    if col not in tre_cols:
        alt = [k for k in tre_cols if args.target.split("_")[0] in k]
        if not alt:
            print(f"No TRE column for {args.target}. Available: {tre_cols}")
            return
        col = alt[0]
        print(f"Target {args.target} not found, using {col}")

    steps = sorted({int(r["step"]) for r in rows})
    cyc = []
    for st in steps:
        sub = [r for r in rows if int(r["step"]) == st and r["state"] == "NAVIGATING"
               and r["pnp_ok"] == "True"]
        if not sub:
            continue
        t0 = fnum(sub[0]["timestamp"])
        sub = [r for r in sub if fnum(r["timestamp"]) - t0 >= args.trim]
        if len(sub) < args.min_frames:
            continue
        g = lambda k: np.array([fnum(r[k]) for r in sub])
        cyc.append({
            "step": st, "n": len(sub),
            "rms": np.nanmean(g("rms_px")), "tre": np.nanmean(g(col)), "tre_max": np.nanmax(g(col)),
            "markers": np.nanmean(g("n_markers")),
            "x": np.nanmean(g("cam_x")), "y": np.nanmean(g("cam_y")), "z": np.nanmean(g("cam_z")),
            "dist": np.nanmean(g("distance_mm")), "elev": np.nanmean(g("elevation_deg")),
            "az": circ_mean_sd(g("azimuth_deg"))[0],
        })
    if len(cyc) < 2:
        print("Need at least 2 complete cycles (NAVIGATING frames per step). "
              "Check that you pressed 'n' after each cycle.")
        return

    print(f"\n=== {Path(args.log).name}: {len(cyc)} cycles, target {col[4:]} ===")
    print(f"{'cycle':>5s}{'step':>6s}{'frames':>8s}{'markers':>9s}{'RMS px':>8s}{'TRE mm':>8s}"
          f"{'dist':>7s}{'elev':>6s}{'az':>7s}")
    for i, c in enumerate(cyc, 1):
        print(f"{i:>5d}{c['step']:>6d}{c['n']:>8d}{c['markers']:>9.1f}{c['rms']:>8.2f}{c['tre']:>8.2f}"
              f"{c['dist']:>7.0f}{c['elev']:>6.1f}{c['az']:>7.1f}")

    def stat(name, vals, unit=""):
        v = np.array(vals, float)
        m, sd = v.mean(), v.std(ddof=1)
        cv = 100 * sd / m if m else float("nan")
        print(f"  {name:<14s} mean {m:8.3f}   SD {sd:7.3f}   CV {cv:5.1f}%   {unit}")

    print("\nAcross cycles:")
    stat("RMS", [c["rms"] for c in cyc], "px")
    stat("TRE", [c["tre"] for c in cyc], "mm (predicted, depends only on the visible marker set)")

    P = np.array([[c["x"], c["y"], c["z"]] for c in cyc])
    sd_xyz = P.std(axis=0, ddof=1)
    sd3d = float(np.sqrt((sd_xyz ** 2).sum()))
    d_el = np.array([c["elev"] for c in cyc]).std(ddof=1)
    _, d_az = circ_mean_sd([c["az"] for c in cyc])
    d_dist = np.array([c["dist"] for c in cyc]).std(ddof=1)
    print("\nPose repeatability (camera held still, so this is registration scatter):")
    print(f"  camera position SD   x {sd_xyz[0]:.2f}  y {sd_xyz[1]:.2f}  z {sd_xyz[2]:.2f} mm   (3D {sd3d:.2f} mm)")
    print(f"  distance SD {d_dist:.2f} mm, azimuth SD {d_az:.2f} deg, elevation SD {d_el:.2f} deg")
    worst = max(c["tre_max"] for c in cyc)
    print(f"\nWorst predicted TRE in any cycle: {worst:.2f} mm")
    ok = all(c["tre"] < 2.0 for c in cyc) and worst < 2.0
    print("PASS: predicted TRE < 2 mm in every cycle." if ok else "FAIL: a cycle reached TRE >= 2 mm.")


if __name__ == "__main__":
    main()