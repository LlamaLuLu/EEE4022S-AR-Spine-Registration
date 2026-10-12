"""
ATP-4 summary (partial occlusion). Logs were captured with the gate-4 marker map, so one run gives both gates:
  gate 4 = frames with a valid pose (PnP ok);  gate 5 = frames with >= 5 markers detected.
Dwell steps are the ODD step numbers; step 0 is the setup before the first cover or hand.

  4.1 covering  : dwell step k = k markers covered (A: top first, B: ring first, top last)
  4.2 hands     : dwell step 1..19 = dots L5_R, L5_L, L4_R, L4_L, L3_R, L3_L, L2_R, L2_L, L1_R, L1_L
"Tracked" = valid pose in at least --thr % of the frames of that step (default 90).

Usage (from repo root):
  python scripts/atp/atp4_summary.py
"""
import argparse, csv, collections
from pathlib import Path
import numpy as np

COVER = {"A (top first)": "atp4_1_seqA.csv", "B (ring first)": "atp4_1_seqB.csv"}
HANDS = {"surgeon right (az 270, el 60)": "atp4_2_right.csv", "surgeon left (az 90, el 45)": "atp4_2_left.csv"}
DOTS = ["L5_R", "L5_L", "L4_R", "L4_L", "L3_R", "L3_L", "L2_R", "L2_L", "L1_R", "L1_L"]


def steps_of(path):
    st = collections.OrderedDict()
    for r in csv.DictReader(open(path)):
        st.setdefault(int(r["step"]), []).append(r)
    return st


def stat(rs):
    ok = [r for r in rs if r["pnp_ok"] == "True"]
    mk = [int(r["n_markers"]) for r in rs]
    tre = []
    for r in ok:
        v = [float(r[k]) for k in r if k.startswith("TRE_") and "_E_" in k and r[k] not in ("", "nan", "inf")]
        if v:
            tre.append(np.mean(v))
    ids = collections.Counter(i for r in ok for i in r["visible_ids"].split("|"))
    return dict(n=len(rs), g4=100.0 * len(ok) / len(rs), g5=100.0 * np.mean([m >= 5 for m in mk]),
                mk=float(np.mean(mk)), tre=float(np.mean(tre)) if tre else np.nan,
                rms=float(np.median([float(r["rms_px"]) for r in ok])) if ok else np.nan,
                ids=" ".join(k for k, v in sorted(ids.items(), key=lambda x: int(x[0])) if v >= 0.5 * max(len(ok), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/atp_v2")
    ap.add_argument("--thr", type=float, default=90.0)
    ap.add_argument("--out", default="data/atp_v2/atp4_summary.csv")
    a = ap.parse_args()
    rows = []
    print(f"tracked = valid pose in >= {a.thr:.0f}% of frames. g4 = PnP% (gate 4); g5 = % frames with >= 5 markers (gate 5)\n")
    for name, f in COVER.items():
        p = Path(a.dir) / f
        if not p.exists():
            continue
        st = steps_of(p)
        print(f"== 4.1 covering, sequence {name} ==")
        print(f"{'covered':>8s}{'frames':>8s}{'markers':>9s}{'g4 PnP%':>9s}{'g5 %':>6s}{'RMS px':>8s}{'pred TRE':>10s}   ids")
        for s in sorted(st):
            if s % 2 == 0 and s != 0:
                continue
            x = stat(st[s]); k = (s + 1) // 2 if s else 0
            if x["n"] < 30:
                continue
            print(f"{k:>8d}{x['n']:>8d}{x['mk']:>9.1f}{x['g4']:>9.0f}{x['g5']:>6.0f}{x['rms']:>8.2f}{x['tre']:>10.2f}   {x['ids']}")
            rows.append(["cover " + name, k, x["n"], f"{x['mk']:.2f}", f"{x['g4']:.1f}", f"{x['g5']:.1f}", f"{x['rms']:.2f}", f"{x['tre']:.2f}", x["ids"]])
        print()
    for name, f in HANDS.items():
        p = Path(a.dir) / f
        if not p.exists():
            continue
        st = steps_of(p)
        dw = [s for s in sorted(st) if s % 2 == 1]
        print(f"== 4.2 hands, camera {name} ==")
        print(f"{'dot':>6s}{'frames':>8s}{'markers':>9s}{'g4 PnP%':>9s}{'g5 %':>6s}{'pred TRE':>10s}   ids at 50% of valid frames")
        t4 = t5 = 0
        for dot, s in zip(DOTS, dw):
            x = stat(st[s])
            t4 += x["g4"] >= a.thr; t5 += x["g5"] >= a.thr
            print(f"{dot:>6s}{x['n']:>8d}{x['mk']:>9.1f}{x['g4']:>9.0f}{x['g5']:>6.0f}{x['tre']:>10.2f}   {x['ids']}")
            rows.append(["hands " + name, dot, x["n"], f"{x['mk']:.2f}", f"{x['g4']:.1f}", f"{x['g5']:.1f}", "", f"{x['tre']:.2f}", x["ids"]])
        print(f"  tracked at {a.thr:.0f}%: gate 4 {t4}/10, gate 5 {t5}/10  (dwell steps found: {len(dw)}, expected 10)\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "covered_or_dot", "frames", "mean_markers", "g4_pnp_pct", "g5_pct", "rms_px", "pred_tre_mm", "ids"])
        w.writerows(rows)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()