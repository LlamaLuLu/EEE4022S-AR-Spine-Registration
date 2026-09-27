"""
Summarise a CSV log from live_ar_overlay_qt.py --log into ATP numbers.

Usage (from repo root):
    python scripts/atp/analyse_atp_log.py data/atp/atp1_distance_35cm.csv
    python scripts/atp/analyse_atp_log.py data/atp/*.csv          # batch
    python scripts/atp/analyse_atp_log.py data/atp/*.csv --out data/atp/summary.csv
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def load_log(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("state") == "SCREENSHOT":
                continue
            rows.append(r)
    return rows


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def summarise(path):
    rows = load_log(path)
    if not rows:
        return None

    t = np.array([fnum(r["timestamp"]) for r in rows])
    ok = np.array([r["pnp_ok"] == "True" for r in rows])
    rms = np.array([fnum(r["rms_px"]) for r in rows])
    nm = np.array([fnum(r["n_markers"]) for r in rows])
    dist = np.array([fnum(r["distance_mm"]) for r in rows])
    elev = np.array([fnum(r["elevation_deg"]) for r in rows])
    az = np.array([fnum(r["azimuth_deg"]) for r in rows])
    total_ms = sum(np.array([fnum(r[k]) for r in rows])
                   for k in ("grab_ms", "detect_pnp_ms", "overlay_ms", "qt_ms"))

    tre_cols = [k for k in rows[0].keys() if k.startswith("TRE_")]
    tre = np.array([[fnum(r[k]) for k in tre_cols] for r in rows])

    dur = t[-1] - t[0] if len(t) > 1 else 0
    good = ok & ~np.isnan(rms)

    def stats(a):
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return (np.nan,) * 4
        return a.mean(), a.std(), np.percentile(a, 95), a.max()

    rms_m, rms_s, rms_95, rms_max = stats(rms[good])
    ft_m, ft_s, ft_95, _ = stats(total_ms)
    tre_mean = np.nanmean(tre[good]) if good.any() else np.nan
    tre_max = np.nanmax(tre[good]) if good.any() else np.nan

    return {
        "file": Path(path).name,
        "frames": len(rows),
        "duration_s": round(dur, 1),
        "fps": round(len(rows) / dur, 1) if dur > 0 else np.nan,
        "pnp_success_pct": round(100 * ok.mean(), 1),
        "markers_mean": round(np.nanmean(nm), 2),
        "rms_mean": round(rms_m, 2),
        "rms_std": round(rms_s, 2),
        "rms_p95": round(rms_95, 2),
        "rms_max": round(rms_max, 2),
        "rms_cv_pct": round(100 * rms_s / rms_m, 1) if rms_m else np.nan,
        "dist_mm": round(np.nanmean(dist[good]), 0) if good.any() else np.nan,
        "elev_deg": round(np.nanmean(elev[good]), 1) if good.any() else np.nan,
        "az_deg": round(np.nanmean(az[good]), 1) if good.any() else np.nan,
        "frame_ms_mean": round(ft_m, 1),
        "frame_ms_p95": round(ft_95, 1),
        "tre_mean_mm": round(tre_mean, 3),
        "tre_max_mm": round(tre_max, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None, help="Write combined summary CSV")
    args = ap.parse_args()

    paths = []
    for p in args.logs:
        paths.extend(sorted(glob.glob(p)))

    results = [s for s in (summarise(p) for p in paths) if s]
    if not results:
        print("No data.")
        return

    for r in results:
        print(f"\n=== {r['file']} ===")
        for k, v in r.items():
            if k != "file":
                print(f"  {k:18s} {v}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nSummary written: {args.out}")


if __name__ == "__main__":
    main()