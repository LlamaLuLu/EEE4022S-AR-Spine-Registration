"""
Geometry sweeps for the dome designs.

  python scripts/tre_analysis/ring_sweep.py ring     ring radius sweep (D1, D2, D4)
  python scripts/tre_analysis/ring_sweep.py toph     top-marker height sweep (D1_final),
                                                     ITERATIVE vs SQPNP solver, at 600 and 900 mm
"""
import sys
import types

import cv2
import mc_validation as mv
from design_variants_oriented import build_dome

K, dist = mv.load_cal("data/calibration/s22_ultra_v1/camera_calibration.npz")
T, _ = mv.load_pedicles_from_json("annotations/pedicle_entry_targets.json")
a = types.SimpleNamespace(distance=600, gate=5, trials=200,
                          elevs=[10, 20, 30, 40, 50, 60, 70], az_step=45)


def run(ori):
    poses = mv.pose_grid(a)
    recs = mv.score_variant(ori, a, K, dist, T, mv.Vis(), mv.SoftVis(), poses, 20.0, {"hi": 0.91})
    ok = [x for x in recs if x["ok"]]
    return (f"tracked {len(ok)}/{len(recs)}  non-copl {sum(x['noncop'] for x in ok):>2d}  "
            f"MC live {mv._m(ok, 'mc_hi'):6.3f}  MC worst {mv._m(ok, 'mc_worst'):6.3f}  "
            f"Fitz worst {mv._m(ok, 'fitz_max'):.3f}")


def ring_sweep(flag=cv2.SOLVEPNP_SQPNP):
    mv.SOLVER_FLAG = flag
    a.distance = 600
    for cfg in ("D1_left_L3", "D2_caudal", "D4_split_half"):
        for r in (25, 30, 40, 50, 60):
            print(f"{cfg:<15s} r={r:>2d}  {run(build_dome(cfg, ring_r=r))}")
        print()


def toph_sweep():
    for flag, fname in ((cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"), (cv2.SOLVEPNP_SQPNP, "SQPNP")):
        mv.SOLVER_FLAG = flag
        for d in (600, 900):
            a.distance = d
            for h in (0, 5, 10, 15, 20, 25, 30, 40):
                print(f"{fname:<9s} {d} mm  top_h={h:>2d}  {run(build_dome('D1_final', top_h=h))}")
            print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "toph"
    if which == "ring":
        for flag, name in ((cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"), (cv2.SOLVEPNP_SQPNP, "SQPNP")):
            print(f"== {name} ==")
            ring_sweep(flag)
    else:
        toph_sweep()