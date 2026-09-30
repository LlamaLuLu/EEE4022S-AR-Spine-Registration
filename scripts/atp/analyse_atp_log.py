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
import sys

_REG = str(Path(__file__).resolve().parents[1] / "registration")
if _REG not in sys.path:
    sys.path.insert(0, _REG)
try:
    from ar_theme import envelope_tier
except Exception:
    envelope_tier = None


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
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def summarise(path, rows=None, name=None):
    if rows is None:
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

    # Dropouts: consecutive PnP failures. Gaps < 300 ms are bridged by the
    # STALE hold, so the user only sees tracking lost in SEARCHING frames.
    gaps, i = [], 0
    while i < len(ok):
        if not ok[i]:
            j = i
            while j < len(ok) and not ok[j]:
                j += 1
            t_end = t[j] if j < len(t) else t[-1]
            gaps.append(t_end - t[i])
            i = j
        else:
            i += 1
    max_gap_ms = 1000 * max(gaps) if gaps else 0.0
    states = [r.get("state", "") for r in rows]
    lost_pct = 100 * sum(st == "SEARCHING" for st in states) / max(len(states), 1)
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
    if good.any():
        worst = np.nanmax(tre[good], axis=1)          # worst target per frame
        tre_worst_p95 = np.nanpercentile(worst, 95)
    else:
        tre_worst_p95 = np.nan

    return {
        "file": name or Path(path).name,
        "frames": len(rows),
        "duration_s": round(dur, 1),
        "fps": round(len(rows) / dur, 1) if dur > 0 else np.nan,
        "pnp_success_pct": round(100 * ok.mean(), 1),
        "lost_pct": round(lost_pct, 1),
        "max_gap_ms": round(max_gap_ms, 0),
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
        "tre_worst_p95_mm": round(tre_worst_p95, 3),
    }


def summarise_by_step(path, trim_s=0.0, trim_end_s=0.0):
    """One summary per step value within a single log.
    trim_s drops the first seconds of each step (movement / settling)."""
    rows = load_log(path)
    if not rows or "step" not in rows[0]:
        print(f"{path}: no 'step' column, rerun with the updated overlay")
        return []
    out = []
    for st in sorted({int(r["step"]) for r in rows}):
        sub = [r for r in rows if int(r["step"]) == st]
        if trim_s > 0 and sub:
            t0 = fnum(sub[0]["timestamp"])
            sub = [r for r in sub if fnum(r["timestamp"]) - t0 >= trim_s]
        if trim_end_s > 0 and sub:
            t1 = fnum(sub[-1]["timestamp"])
            sub = [r for r in sub if t1 - fnum(r["timestamp"]) >= trim_end_s]
        res = summarise(path, sub, name=f"{Path(path).stem} step {st}")
        if res:
            res = {"step": st, **res}
            out.append(res)
    return out


def print_step_table(results):
    cols = [("step", 5), ("frames", 7), ("pnp_success_pct", 7), ("lost_pct", 7),
            ("max_gap_ms", 8), ("markers_mean", 8),
            ("rms_mean", 7), ("rms_p95", 7), ("dist_mm", 8), ("elev_deg", 7),
            ("az_deg", 7), ("tre_mean_mm", 10), ("tre_worst_p95_mm", 11)]
    hdr = ["step", "frames", "pnp%", "lost%", "maxgap", "markers", "rms", "rms95", "dist", "elev",
           "az", "tre_mean", "worst_p95"]
    print("".join(f"{h:>{w}s}" for h, (_, w) in zip(hdr, cols)))
    for r in results:
        line = ""
        for k, w in cols:
            v = r.get(k)
            if isinstance(v, float):
                line += f"{v:>{w}.2f}" if k.startswith(("rms", "tre")) else f"{v:>{w}.1f}"
            else:
                line += f"{str(v):>{w}s}"
        print(line)


def step_tier(r):
    """Tier of a step from its mean pose. Steps with no pose count as outside."""
    if envelope_tier is None:
        return "unknown"
    e, a, d = r.get("elev_deg"), r.get("az_deg"), r.get("dist_mm")
    if any(v is None or not np.isfinite(v) for v in (e, a, d)):
        return "outside"
    return envelope_tier(e, a, d)


def summarise_by_tier(step_results):
    """Aggregate step summaries per envelope tier (frame-weighted where sensible)."""
    out = []
    for tier in ("full", "reduced", "outside"):
        sub = [r for r in step_results if r["tier"] == tier]
        if not sub:
            continue
        fr = np.array([r["frames"] for r in sub], float)

        def wmean(k):
            v = np.array([r.get(k, np.nan) for r in sub], float)
            m = np.isfinite(v)
            return float(np.average(v[m], weights=fr[m])) if m.any() else np.nan

        def vmax(k):
            v = np.array([r.get(k, np.nan) for r in sub], float)
            return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan

        out.append({
            "tier": tier, "steps": len(sub), "frames": int(fr.sum()),
            "pnp_success_pct": round(wmean("pnp_success_pct"), 1),
            "lost_pct": round(wmean("lost_pct"), 1),
            "max_gap_ms": round(vmax("max_gap_ms"), 0),
            "markers_mean": round(wmean("markers_mean"), 1),
            "rms_mean": round(wmean("rms_mean"), 2),
            "tre_mean_mm": round(wmean("tre_mean_mm"), 3),
            "tre_worst_p95_mm": round(vmax("tre_worst_p95_mm"), 3),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None, help="Write combined summary CSV")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="With --by-step: ignore the first N seconds of each step")
    ap.add_argument("--trim-end", type=float, default=0.0,
                    help="With --by-step: ignore the last N seconds of each step")
    ap.add_argument("--by-tier", action="store_true",
                    help="Classify each step by envelope tier and summarise per tier "
                         "(implies --by-step; accepts several logs)")
    ap.add_argument("--by-step", action="store_true",
                    help="Split each log by the step counter ('n' key) and summarise each step")
    args = ap.parse_args()

    paths = []
    for p in args.logs:
        paths.extend(sorted(glob.glob(p)))

    if args.by_tier:
        allsteps = []
        for p in paths:
            for r in summarise_by_step(p, args.trim, args.trim_end):
                r["tier"] = step_tier(r)
                allsteps.append(r)
        tiers = summarise_by_tier(allsteps)
        print(f"\n=== By envelope tier ({len(allsteps)} steps from {len(paths)} log(s)) ===")
        hdr = f"{'tier':<9s}{'steps':>6s}{'frames':>8s}{'pnp%':>7s}{'lost%':>7s}{'maxgap':>8s}" \
              f"{'markers':>8s}{'rms':>6s}{'tre_mean':>9s}{'worst_p95':>10s}"
        print(hdr)
        for t in tiers:
            print(f"{t['tier']:<9s}{t['steps']:>6d}{t['frames']:>8d}{t['pnp_success_pct']:>7.1f}"
                  f"{t['lost_pct']:>7.1f}{t['max_gap_ms']:>8.0f}{t['markers_mean']:>8.1f}"
                  f"{t['rms_mean']:>6.2f}{t['tre_mean_mm']:>9.2f}{t['tre_worst_p95_mm']:>10.2f}")
        if args.out and tiers:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(tiers[0].keys()))
                w.writeheader()
                w.writerows(tiers)
            steps_out = args.out.replace(".csv", "_steps.csv")
            with open(steps_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(allsteps[0].keys()))
                w.writeheader()
                w.writerows(allsteps)
            print(f"\nSummary written: {args.out} (+ per-step tiers: {steps_out})")
        return

    if args.by_step:
        results = []
        for p in paths:
            rs = summarise_by_step(p, args.trim, args.trim_end)
            if rs:
                print(f"\n=== {Path(p).name} ===")
                print_step_table(rs)
                results.extend(rs)
        if args.out and results:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                w.writerows(results)
            print(f"\nSummary written: {args.out}")
        return

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