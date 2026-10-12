"""
ATP-3 summary: one table per test (distance, azimuth, elevation), measured PnP success and markers
per pose for each focus setting, next to the simulator's prediction (V3-fitted visibility model, not refitted).

A pose is one dwell step: with the capture protocol (n, hold, n, move) the dwell steps are the ODD step numbers.

Usage (from repo root):
  python scripts/atp/atp3_summary.py
  python scripts/atp/atp3_summary.py --f35-prefix data/atp_v2/atp3 --f50-prefix data/atp_v2/atp3
Files read: <prefix>_1_distance_f35.csv, <prefix>_1_distance_f50.csv, ... (a missing file is skipped).
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tre_analysis"))
import mc_validation as mv                                          # noqa: E402

TESTS = ("1_distance", "2_azimuth", "3_elevation")


def dwell_stats(path):
    """list of dicts, one per odd (dwell) step, in step order."""
    if not Path(path).exists():
        return None
    steps = {}
    for r in csv.DictReader(open(path)):
        steps.setdefault(int(r["step"]), []).append(r)
    out = []
    for s in sorted(steps):
        if s % 2 == 0:
            continue
        rs = steps[s]
        ok = [r for r in rs if r["pnp_ok"] == "True"]
        med = lambda k: float(np.median([float(r[k]) for r in ok])) if ok else np.nan
        ids = {}
        for r in ok:
            for i in r["visible_ids"].split("|"):
                ids[int(i)] = ids.get(int(i), 0) + 1
        out.append(dict(step=s, frames=len(rs), pnp=100.0 * len(ok) / len(rs),
                        mk=float(np.mean([int(r["n_markers"]) for r in rs])),
                        d=med("distance_mm"), el=med("elevation_deg"), az=med("azimuth_deg"),
                        ids=" ".join(str(k) for k, v in sorted(ids.items()) if v >= 0.5 * len(ok))))
    return out


def window(xs, ys, thr=50.0):
    """ranges of consecutive x with y >= thr, as text."""
    runs, cur = [], []
    for x, y in zip(xs, ys):
        if y >= thr:
            cur.append(x)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return ", ".join(f"{r[0]:g} to {r[-1]:g}" for r in runs) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f35-prefix", default="data/atp_v2/atp3")
    ap.add_argument("--f50-prefix", default="data/atp_v2/atp3")
    ap.add_argument("--marker-map", default="data/dome_markers/marker_map.json")
    ap.add_argument("--calibration", default="data/calibration/s22_ultra_v1/camera_calibration.npz")
    ap.add_argument("--elev-dist", type=float, default=500.0, help="distance used for the elevation sweep (mm)")
    ap.add_argument("--az-dist", type=float, default=600.0)
    ap.add_argument("--az-skip", type=int, default=12,
                    help="azimuth file: index of the extra dwell between the el 45 and el 30 sweeps (return to 0)")
    ap.add_argument("--out", default="data/atp_v2/atp3_summary.csv")
    a = ap.parse_args()

    K, dist = mv.load_cal(a.calibration)
    ori, ids_all, size, gate = mv.oriented_from_map(a.marker_map)
    sv = mv.SoftVis()
    pred = lambda d, e, z: (mv.p_at_least(mv.marker_probs(ori, mv.make_camera_pose(d, e, z), K, dist, sv, size_mm=size), gate) * 100,
                            mv.marker_probs(ori, mv.make_camera_pose(d, e, z), K, dist, sv, size_mm=size).sum())
    azs = list(range(0, 360, 30))
    plan = {
        "1_distance": [("distance", [(d, 45, 270) for d in range(450, 1001, 50)], 0)],
        "2_azimuth": [("azimuth el 45", [(a.az_dist, 45, z) for z in azs], 2),
                      ("azimuth el 30", [(a.az_dist, 30, z) for z in azs], 2)],
        "3_elevation": [("elevation", [(a.elev_dist, e, 270) for e in range(10, 81, 10)], 1)],
    }
    rows = []
    print(f"gate {gate} markers, {len(ori)} markers {size:.0f} mm; PnP% = share of frames with a valid pose; "
          f"pred = simulator P(>= gate), not refitted\n")
    for test in TESTS:
        runs = {tag: dwell_stats(f"{pre}_{test}_{tag}.csv")
                for tag, pre in (("f35", a.f35_prefix), ("f50", a.f50_prefix))}
        for tag in list(runs):
            if runs[tag] is None:
                runs.pop(tag)
        if not runs:
            continue
        subs = plan[test]
        n_need = sum(len(p) for _, p, _ in subs)
        for name, poses, k in subs:
            print(f"== {name} ==")
            hdr = f"{'pose':>6s}" + "".join(f"{tag + ' PnP%':>10s}{tag + ' mk':>8s}" for tag in runs) + f"{'pred%':>7s}{'pred mk':>8s}   ids at 50% of valid frames ({' / '.join(runs)})"
            print(hdr)
            series = {tag: [] for tag in runs}; pser = []
            for idx, pz in enumerate(poses):
                line = f"{pz[k]:>6g}"
                idl = []
                for tag, dw in runs.items():
                    if test == "2_azimuth":
                        dw2 = [x for i, x in enumerate(dw) if i != a.az_skip]
                        j = idx + (0 if name.endswith("45") else len(azs))
                    else:
                        dw2, j = dw, idx
                    if j < len(dw2):
                        x = dw2[j]; series[tag].append((pz[k], x["pnp"]))
                        line += f"{x['pnp']:>10.0f}{x['mk']:>8.1f}"; idl.append(x["ids"])
                        rows.append([test, name, tag, pz[0], pz[1], pz[2], x["step"], x["frames"], f"{x['pnp']:.1f}",
                                     f"{x['mk']:.2f}", f"{x['d']:.0f}", f"{x['el']:.1f}", f"{x['az']:.1f}", x["ids"]])
                    else:
                        line += f"{'-':>10s}{'-':>8s}"; idl.append("-")
                p, e = pred(*pz); pser.append((pz[k], p))
                print(line + f"{p:>7.0f}{e:>8.1f}   " + " / ".join(idl))
            for tag, s in series.items():
                if s:
                    print(f"  {tag}: PnP >= 50% at {window(*zip(*s))}")
            print(f"  sim: PnP >= 50% at {window(*zip(*pser))}\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "sweep", "focus", "dist_mm", "elev_deg", "az_deg", "step", "frames", "pnp_pct", "mean_markers",
                    "meas_dist", "meas_el", "meas_az", "ids_majority"])
        w.writerows(rows)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()