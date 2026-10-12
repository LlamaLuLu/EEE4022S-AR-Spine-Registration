"""
Monte Carlo validation against the ATP measurements.

Builds on monte_carlo_tre.py (same geometry, camera model and solvePnP path)
and adds what is needed to compare the simulation with the physical tests:

  * overlay error: the on-screen (in-plane) target error in mm, which is what
    the ATP-2 click tool measures, alongside the 3D TRE the MC already gives
  * a visibility model with the live pipeline's 4-marker gate, a maximum
    view angle, a minimum apparent marker width and an in-frame test, fitted
    to the ATP-3 sweeps instead of assumed
  * noise set from the measured live reprojection RMS, not only the static test

Subcommands
  fit      fit the visibility model to the ATP-3 distance/azimuth/elevation sweeps
  design   compare every variant in design_variants_oriented (V0-V3 and the domes)
  tilt     sweep the dome tilt angle for each DOME_CONFIGS entry
  atp4     marker-set study: MC vs Fitzpatrick vs the ATP-4 measurements
  atp2     error budget: MC overlay error at the ATP-2 poses vs measured error
  fle      check the MC noise path against the static single-marker FLE
  predict  expected markers and P(PnP) for the planned ATP-3 poses of any variant (no refit)

Usage (from repo root; paths shown are the defaults):
  python scripts/tre_analysis/mc_validation.py fit
  python scripts/tre_analysis/mc_validation.py design --gate 5 --variants V3 D
  python scripts/tre_analysis/mc_validation.py design --gate 5 --variants D --hands both
  python scripts/tre_analysis/mc_validation.py tilt --gate 5 --tilts 15 20 25 30 35 40 45
  python scripts/tre_analysis/mc_validation.py atp4
  python scripts/tre_analysis/mc_validation.py atp2 --poses "data/screenshots/atp2_crisp/shot_*_pose.json" \
        --measured data/atp/atp2_crisp.csv

Other marker arrays (atp2, atp4, fit, predict). The default is the V3 L-bracket. For the as-built dome:
  python scripts/tre_analysis/mc_validation.py atp2 --marker-map data/dome_markers/marker_map.json \
        --poses "data/screenshots/atp2_v2/shot_*_pose.json" --measured data/atp_v2/atp2_dome.csv
  python scripts/tre_analysis/mc_validation.py predict --marker-map data/dome_markers/marker_map.json
--marker-map reads the marker size, the gate and the real marker poses (including the hand-calibrated offsets)
from the JSON the live overlay uses. --variant NAME picks a design_variants_oriented entry instead.
Visibility parameters stay at the V3 fit: apply them to the dome without refitting, then compare.
"""

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "registration"))

from monte_carlo_tre import (                                    # noqa: E402
    make_camera_pose, corners_from_oriented_marker, ray_intersects_cylinder,
    PHANTOM_CYLINDER_CENTER_XY, PHANTOM_CYLINDER_RADIUS_MM,
    PHANTOM_CYLINDER_Z_MIN, PHANTOM_CYLINDER_Z_MAX,
)
from design_variants import load_pedicles_from_json              # noqa: E402
from design_variants_oriented import (                          # noqa: E402
    VARIANTS_ORIENTED, V3_ORIENTED, DOME_CONFIGS, build_dome)

try:
    from fitzpatrick_tre import predict_tre                      # noqa: E402
except ImportError:
    predict_tre = None

IMG_W, IMG_H = 1920, 1080
MARKER_MM = 20.0
GATE = 4                      # live pipeline: MIN_MARKERS_PNP
FLE_FITZ_MM = 0.8             # Fitzpatrick input used in the pipeline


# ---------------------------------------------------------------------------
# Visibility model
# ---------------------------------------------------------------------------
@dataclass
class Vis:
    max_angle_deg: float = 76.0     # outward normal vs line of sight: the 50% point of the fitted soft model
    min_width_px: float = 13.0      # apparent marker width across the foreshortened axis: 50% point of the soft model
    margin_px: float = 10.0         # all four corners must lie this far inside the frame
    occlusion: bool = True


def visible_ids(oriented, pose, K, dist, vis, size_mm=MARKER_MM, occluders=()):
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    cam = -R.T @ tvec.ravel()
    f = 0.5 * (K[0, 0] + K[1, 1])
    out = []
    for i, (c, Rm) in enumerate(oriented):
        to_cam = cam - c
        u = to_cam / np.linalg.norm(to_cam)
        n = Rm[:, 2]
        cosang = float(n @ u)
        if cosang < np.cos(np.radians(vis.max_angle_deg)):
            continue
        z = float((R @ c + tvec.ravel())[2])
        if z <= 0 or f * size_mm / z * cosang < vis.min_width_px:
            continue
        if vis.occlusion and ray_intersects_cylinder(
                c + n * 0.5, to_cam, PHANTOM_CYLINDER_CENTER_XY, PHANTOM_CYLINDER_RADIUS_MM,
                PHANTOM_CYLINDER_Z_MIN, PHANTOM_CYLINDER_Z_MAX):
            continue
        if occluders and blocked(c + n * 0.5, to_cam, occluders):
            continue
        px, _ = cv2.projectPoints(corners_from_oriented_marker(c, Rm, size_mm), rvec, tvec, K, dist)
        px = px.reshape(-1, 2)
        m = vis.margin_px
        if (px[:, 0] < m).any() or (px[:, 0] > IMG_W - m).any() or \
           (px[:, 1] < m).any() or (px[:, 1] > IMG_H - m).any():
            continue
        out.append(i)
    return out


# ---------------------------------------------------------------------------
# Hand stand-in occluders and coplanarity
# ---------------------------------------------------------------------------
def _seg_seg_dist(p0, p1, q0, q1):
    """Shortest distance between segments p0-p1 and q0-q1."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0.0:
        t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def _seg_hits_box(p0, d, lo, hi):
    """Slab test: does the segment p0 + t*d, t in [0, 1], enter the box [lo, hi]?"""
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if p0[k] < lo[k] or p0[k] > hi[k]:
                return False
            continue
        ta, tb = sorted(((lo[k] - p0[k]) / d[k], (hi[k] - p0[k]) / d[k]))
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def blocked(origin, direction, occluders):
    """True if the sight line origin -> origin + direction passes through any occluder.
    Occluders: ("capsule", p0, p1, radius) or ("box", lo, hi)."""
    end = origin + direction
    for occ in occluders:
        if occ[0] == "capsule":
            if _seg_seg_dist(origin, end, occ[1], occ[2]) < occ[3]:
                return True
        elif _seg_hits_box(origin, direction, occ[1], occ[2]):
            return True
    return False


def hand_occluders(target, side):
    """Crude surgeon's hand and forearm working at one target.
    side = "left" (surgeon stands at +X) or "right" (-X).
    Hand: 90 x 30 x 100 mm box hovering beside and above the target.
    Forearm: 35 mm radius capsule from the wrist out to the elbow."""
    s = 1.0 if side == "left" else -1.0
    t = np.asarray(target, float)
    hc = t + np.array([s * 30.0, 50.0, 0.0])
    half = np.array([45.0, 15.0, 50.0])
    wrist = t + np.array([s * 75.0, 55.0, 0.0])
    elbow = t + np.array([s * 330.0, 200.0, -40.0])
    return [("box", hc - half, hc + half), ("capsule", wrist, elbow, 35.0)]


def noncoplanar(oriented, ids, size_mm=MARKER_MM, tol_mm=2.0):
    """True if the visible markers' corners do not all lie in one plane
    (RMS out-of-plane distance above tol_mm). A single flat face fails; two faces,
    or a raised top marker with a ring, pass."""
    if len(ids) < 2:
        return False
    pts = np.vstack([corners_from_oriented_marker(oriented[i][0], oriented[i][1], size_mm) for i in ids])
    pts = pts - pts.mean(0)
    sv = np.linalg.svd(pts, compute_uv=False)
    return float(sv[-1] / np.sqrt(len(pts))) > tol_mm


# ---------------------------------------------------------------------------
# Soft visibility: per-marker detection probability
# ---------------------------------------------------------------------------
@dataclass
class SoftVis:
    """Logistic detection probability per marker. The 50% points are the equivalent hard limits."""
    th50: float = 75.9      # view angle at which a marker is detected half the time (deg)   [fitted to ATP-3]
    s_th: float = 1.5       # logistic scale (deg); smaller is a sharper cutoff               [fitted to ATP-3]
    w50: float = 12.6       # apparent width at which detection is 50% (px)                    [fitted to ATP-3]
    s_w: float = 6.3        # logistic scale (px)                                              [fitted to ATP-3]

    def vec(self):
        return np.array([self.th50, self.s_th, self.w50, self.s_w])

    @staticmethod
    def from_vec(v):
        return SoftVis(*[float(x) for x in v])


def marker_probs(oriented, pose, K, dist, sv, occlusion=True, size_mm=MARKER_MM, occluders=()):
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    cam = -R.T @ tvec.ravel()
    f = 0.5 * (K[0, 0] + K[1, 1])
    out = np.zeros(len(oriented))
    for i, (c, Rm) in enumerate(oriented):
        to_cam = cam - c
        u = to_cam / np.linalg.norm(to_cam)
        n = Rm[:, 2]
        cosang = float(n @ u)
        z = float((R @ c + tvec.ravel())[2])
        if cosang <= 0.0 or z <= 0:
            continue
        if occlusion and ray_intersects_cylinder(
                c + n * 0.5, to_cam, PHANTOM_CYLINDER_CENTER_XY, PHANTOM_CYLINDER_RADIUS_MM,
                PHANTOM_CYLINDER_Z_MIN, PHANTOM_CYLINDER_Z_MAX):
            continue
        if occluders and blocked(c + n * 0.5, to_cam, occluders):
            continue
        theta = np.degrees(np.arccos(min(cosang, 1.0)))
        w = f * size_mm / z * cosang
        pa = 1.0 / (1.0 + np.exp((theta - sv.th50) / sv.s_th))
        pw = 1.0 / (1.0 + np.exp(-(w - sv.w50) / sv.s_w))
        out[i] = pa * pw
    return out


def p_at_least(p, k):
    """P(at least k of the independent Bernoulli(p_i) markers are detected)."""
    dp = np.zeros(len(p) + 1)
    dp[0] = 1.0
    for pi in p:
        new = dp * (1.0 - pi)
        new[1:] += dp[:-1] * pi
        dp = new
    return float(dp[k:].sum())


# ---------------------------------------------------------------------------
# Monte Carlo with overlay error
# ---------------------------------------------------------------------------
def simulate(oriented, ids, targets, pose, K, dist, sigma_px, n_trials=400, seed=0, size_mm=MARKER_MM):
    rvec, tvec = pose
    corners3d = np.vstack([corners_from_oriented_marker(oriented[i][0], oriented[i][1], size_mm)
                           for i in ids]).astype(np.float32)
    clean, _ = cv2.projectPoints(corners3d, rvec, tvec, K, dist)
    clean = clean.reshape(-1, 2)
    tgt = targets.astype(np.float64)
    p_true, _ = cv2.projectPoints(tgt, rvec, tvec, K, dist)
    p_true = p_true.reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    tc_true = (R @ tgt.T).T + tvec.T
    z_true = tc_true[:, 2]
    f = 0.5 * (K[0, 0] + K[1, 1])
    rng = np.random.default_rng(seed)
    e3, ei, rms = [], [], []
    for _ in range(n_trials):
        noisy = (clean + rng.normal(0.0, sigma_px, clean.shape)).astype(np.float32)
        ok, rv, tv = cv2.solvePnP(corners3d, noisy, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        Re, _ = cv2.Rodrigues(rv)
        e3.append(np.linalg.norm(tgt - (tc_true - tv.T) @ Re, axis=1))
        p_est, _ = cv2.projectPoints(tgt, rv, tv, K, dist)
        ei.append(np.linalg.norm(p_est.reshape(-1, 2) - p_true, axis=1) * z_true / f)
        proj, _ = cv2.projectPoints(corners3d, rv, tv, K, dist)
        rms.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - noisy) ** 2))))
    if not e3:
        return None
    e3, ei = np.array(e3), np.array(ei)
    return {"tre3_rms": np.sqrt((e3 ** 2).mean(0)), "ov_rms": np.sqrt((ei ** 2).mean(0)),
            "ov_mean": ei.mean(0), "reproj_rms": float(np.mean(rms)), "n_ok": len(e3)}


def sigma_for_rms(rms_px, n_markers):
    """Corner noise (px per axis) that gives a reprojection RMS of rms_px with n markers."""
    n_res = 2 * 4 * n_markers
    return rms_px / np.sqrt(max(1.0 - 6.0 / n_res, 0.05))


def fitz(oriented, ids, targets, fle=FLE_FITZ_MM):
    if predict_tre is None or len(ids) < 3:
        return None
    return predict_tre(np.array([oriented[i][0] for i in ids]), targets, fle)


def oriented_from_map(path):
    """Markers of a live marker_map.json as (centre, R) tuples, with the map's own corner order.
    Returns (oriented, aruco_ids, marker_mm, gate)."""
    m = json.load(open(path))
    ori, ids = [], []
    for sl in m["slots"]:
        c = np.asarray(sl["corners_mm"], float)            # TL, TR, BR, BL
        x = c[1] - c[0]; y = c[0] - c[3]
        x /= np.linalg.norm(x); y /= np.linalg.norm(y)
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        n = np.asarray(sl.get("normal", z), float)
        if float(n @ z) < 0.99:
            sys.exit(f"marker {sl['aruco_id']}: corner order gives a normal that disagrees with the map's normal")
        ori.append((c.mean(0), R)); ids.append(int(sl["aruco_id"]))
    size = float(m.get("marker_size_mm", MARKER_MM))
    for (cen, R), sl in zip(ori, m["slots"]):             # the rebuilt corners must reproduce the map
        err = np.abs(corners_from_oriented_marker(cen, R, size) - np.asarray(sl["corners_mm"], float)).max()
        if err > 0.05:
            sys.exit(f"marker {sl['aruco_id']}: rebuilt corners differ from the map by {err:.2f} mm")
    return ori, ids, size, int(m.get("min_markers_pnp", GATE))


def resolve_variant(a):
    """Set a.oriented, a.size_mm, a.gate_eff, a.ids, a.vname from --marker-map / --variant (default V3)."""
    a.is_v3 = False
    if a.marker_map:
        a.oriented, a.ids, a.size_mm, g = oriented_from_map(a.marker_map)
        a.vname = Path(a.marker_map).parent.name or "marker map"
        a.gate_eff = a.gate if a.gate is not None else g
    else:
        name = a.variant or "V3_8marker"
        if name not in VARIANTS_ORIENTED:
            sys.exit(f"unknown variant {name}; choose from {', '.join(VARIANTS_ORIENTED)} or use --marker-map")
        a.oriented, a.vname = VARIANTS_ORIENTED[name], name
        a.ids = list(range(len(a.oriented)))
        a.size_mm = a.marker_mm
        a.gate_eff = a.gate if a.gate is not None else GATE
        a.is_v3 = a.oriented is V3_ORIENTED
    return a


def load_cal(path):
    d = np.load(path)
    return d["camera_matrix"], d["dist_coeffs"]


def read_csv(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Missing {p}")
    return list(csv.DictReader(open(p)))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# fit: visibility model vs ATP-3
# ---------------------------------------------------------------------------
def load_atp3(paths, elev_drop=()):
    """Rows of (sweep, dist, elev, az, markers_mean, pnp%). Fills the pose of dead-zone steps.
    elev_drop: ATP-3.3 step numbers to leave out (default 9, the movement at the end of the run)."""
    rows = []
    for tag, p in zip(("distance", "azimuth", "elevation"), paths):
        rs = read_csv(p)
        if tag == "elevation" and elev_drop:
            rs = [r for r in rs if int(float(r.get("step", -1))) not in set(elev_drop)]
        d = np.array([fnum(r["dist_mm"]) for r in rs]); e = np.array([fnum(r["elev_deg"]) for r in rs])
        a = np.array([fnum(r["az_deg"]) for r in rs]); idx = np.arange(len(rs))
        for arr in (d, e):                      # nominal values carry through dead steps
            ok = np.isfinite(arr)
            arr[~ok] = np.interp(idx[~ok], idx[ok], arr[ok])
        ok = np.isfinite(a)
        if tag == "azimuth" and (~ok).any():    # uniform rotation through the dead zone
            a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
        for i, r in enumerate(rs):
            if np.isfinite(a[i]):
                rows.append((tag, d[i], e[i], a[i], fnum(r["markers_mean"]), fnum(r["pnp_success_pct"])))
    return rows


def n_visible_at(oriented, d, e, a, K, dist, vis, size_mm=MARKER_MM):
    pose = make_camera_pose(d, e, a)
    return len(visible_ids(oriented, pose, K, dist, vis, size_mm=size_mm))


def cross50(x, y, lo, hi, falling, level=50.0):
    """Linear-interpolated crossing of y through `level`, searched for x in [lo, hi]."""
    o = np.argsort(x); x, y = np.asarray(x)[o], np.asarray(y)[o]
    for i in range(len(x) - 1):
        if not (lo <= x[i] <= hi or lo <= x[i + 1] <= hi):
            continue
        y0, y1 = y[i], y[i + 1]
        if falling and y0 >= level > y1 or (not falling) and y0 < level <= y1:
            return x[i] + (level - y0) / (y1 - y0) * (x[i + 1] - x[i])
    return np.nan


def measured_boundaries(rows):
    az = [(r[3], r[5], r[4]) for r in rows if r[0] == "azimuth"]
    el = [(r[2], r[5], r[4]) for r in rows if r[0] == "elevation"]
    di = [(r[1], r[5], r[4]) for r in rows if r[0] == "distance"]
    a, p, m = (np.array(c) for c in zip(*az))
    out = {
        "dead zone starts (deg azimuth)": cross50(a, p, 90, 200, True),
        "dead zone ends (deg azimuth)": cross50(a, p, 250, 345, False),
        "both-face window opens (>=5 markers, deg)": cross50(a, m, 0, 60, False, 5.0),
        "both-face window closes (>=5 markers, deg)": cross50(a, m, 30, 100, True, 5.0),
    }
    e, p, _ = (np.array(c) for c in zip(*el))
    out["elevation limit (deg)"] = cross50(e, p, 30, 80, True)
    d, p, _ = (np.array(c) for c in zip(*di))
    out["far distance limit (mm)"] = cross50(d, p, 900, 2000, True)
    return out


def simulated_boundaries(K, dist, vis, ori=V3_ORIENTED, gate=GATE, size_mm=MARKER_MM):
    GATE = gate
    nv = lambda d, e, a: n_visible_at(ori, d, e, a, K, dist, vis, size_mm)
    azs = np.arange(0, 360, 1.0)
    n = np.array([nv(450, 33, x) for x in azs])
    out = {
        "dead zone starts (deg azimuth)": azs[(azs > 90) & (n < GATE)].min(),
        "dead zone ends (deg azimuth)": azs[(azs > 180) & (n >= GATE)].min(),
    }
    both = azs[(azs > 5) & (azs < 90) & (n >= 5)]
    out["both-face window opens (>=5 markers, deg)"] = both.min()
    out["both-face window closes (>=5 markers, deg)"] = both.max()
    els = np.arange(20, 80, 0.5)
    ne = np.array([nv(460, e, 45) for e in els])
    out["elevation limit (deg)"] = els[ne < GATE].min() if (ne < GATE).any() else np.nan
    ds = np.arange(500, 1800, 10)
    nd = np.array([nv(d, 35, 45) for d in ds])
    out["far distance limit (mm)"] = ds[nd < GATE].min() if (nd < GATE).any() else np.nan
    return out


def curve_cross(xs, ys, level, falling):
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (falling and y0 >= level > y1) or ((not falling) and y0 < level <= y1):
            return xs[i] + (level - y0) / (y1 - y0) * (xs[i + 1] - xs[i])
    return np.nan


def soft_boundaries(K, dist, sv, ori=V3_ORIENTED, gate=GATE, size_mm=MARKER_MM):
    pt = lambda d, e, a: p_at_least(marker_probs(ori, make_camera_pose(d, e, a), K, dist, sv, size_mm=size_mm), gate)
    en = lambda d, e, a: marker_probs(ori, make_camera_pose(d, e, a), K, dist, sv, size_mm=size_mm).sum()
    azs = np.arange(0, 360, 1.0)
    pa = np.array([pt(450, 33, x) for x in azs]); na = np.array([en(450, 33, x) for x in azs])
    out = {
        "dead zone starts (deg azimuth)": curve_cross(azs[azs >= 90], pa[azs >= 90], 0.5, True),
        "dead zone ends (deg azimuth)": curve_cross(azs[azs >= 250], pa[azs >= 250], 0.5, False),
        "both-face window opens (>=5 markers, deg)": curve_cross(azs[azs <= 60], na[azs <= 60], 5.0, False),
        "both-face window closes (>=5 markers, deg)": curve_cross(azs[(azs >= 30) & (azs <= 100)], na[(azs >= 30) & (azs <= 100)], 5.0, True),
    }
    els = np.arange(20, 80, 0.5)
    out["elevation limit (deg)"] = curve_cross(els, np.array([pt(460, e, 45) for e in els]), 0.5, True)
    ds = np.arange(700, 2000, 10.0)
    out["far distance limit (mm)"] = curve_cross(ds, np.array([pt(d, 35, 45) for d in ds]), 0.5, True)
    return out


def soft_from_args(a):
    d = SoftVis()
    return SoftVis(a.th50 if a.th50 is not None else d.th50, a.s_th if a.s_th is not None else d.s_th,
                   a.w50 if a.w50 is not None else d.w50, a.s_w if a.s_w is not None else d.s_w)


def cmd_fit(a):
    if not a.is_v3:
        sys.exit("fit refits the visibility model on the V3 ATP-3 data, so it only runs for V3. "
                 "For another array use `predict` (no refit), then compare with its own ATP-3 summaries.")
    try:
        from scipy.optimize import least_squares
    except ImportError:
        sys.exit("fit needs scipy:  pip install scipy")
    K, dist = load_cal(a.calibration)
    # close-range distance steps are left out: the real aim point was not logged. Azimuth and
    # elevation steps (taken at about 45 cm) stay in.
    rows = [r for r in load_atp3([a.atp31, a.atp32, a.atp33], a.elev_drop) if not (r[0] == "distance" and r[1] < a.min_dist)]
    meas = measured_boundaries(load_atp3([a.atp31, a.atp32, a.atp33], a.elev_drop))

    def predict(sv, d, e, az):
        p = marker_probs(V3_ORIENTED, make_camera_pose(d, e, az), K, dist, sv)
        return p_at_least(p, GATE), p.sum()

    def resid(v):
        sv = SoftVis.from_vec(v)
        out = []
        for r in rows:
            pt, en = predict(sv, r[1], r[2], r[3])
            out += [pt - r[5] / 100.0, (en - r[4]) / 8.0]
        return np.array(out)

    x0 = SoftVis().vec()
    fit = least_squares(resid, x0, bounds=([55, 0.5, 4, 0.3], [85, 10, 25, 8]), diff_step=0.05)
    sv = SoftVis.from_vec(fit.x)
    print(f"Per-marker detection model fitted to {len(rows)} ATP-3 steps (V3, gate {GATE} markers; distance steps under {a.min_dist:.0f} mm "
          f"and ATP-3.3 step(s) {sorted(a.elev_drop)} excluded)\n")
    print(f"  view angle:      50% detection at {sv.th50:.1f} deg (logistic scale {sv.s_th:.1f} deg)")
    print(f"  apparent width:  50% detection at {sv.w50:.1f} px   (logistic scale {sv.s_w:.1f} px)")
    P = np.array([predict(sv, r[1], r[2], r[3]) for r in rows])
    pm = np.array([r[5] for r in rows]); mm = np.array([r[4] for r in rows])
    print(f"\nIn-sample fit: PnP success RMS error {np.sqrt(np.mean((P[:, 0] * 100 - pm) ** 2)):.1f} percentage points; "
          f"detected markers RMS error {np.sqrt(np.mean((P[:, 1] - mm) ** 2)):.2f}")
    print(f"  (measured PnP% ranges {pm.min():.0f} to {pm.max():.0f}; correlation of predicted and measured PnP%: "
          f"{np.corrcoef(P[:, 0] * 100, pm)[0, 1]:.2f})")
    sb = soft_boundaries(K, dist, sv)
    print(f"\n{'boundary (50% points)':<46s}{'measured':>10s}{'simulated':>11s}{'difference':>12s}")
    for k in meas:
        mm_row = "mm" in k
        fmt, unit = (".0f", " mm") if mm_row else (".1f", " deg")      # one decimal for degrees, so rounding cannot mislead
        d = sb[k] - meas[k]
        print(f"{k:<46s}{format(meas[k], fmt):>10s}{format(sb[k], fmt):>11s}{format(d, '+' + fmt):>10s}{unit}")
    print(f"\nTo reuse these on other commands: --th50 {sv.th50:.1f} --s-th {sv.s_th:.1f} --w50 {sv.w50:.1f} --s-w {sv.s_w:.1f} "
          f"--max-angle {sv.th50:.0f} --min-width {sv.w50:.0f}")

    # held-out test on ATP-4: same parameters, never fitted to this data
    try:
        r4 = {int(float(r["step"])): r for r in read_csv(a.atp4)}
    except SystemExit:
        r4 = {}
    if r4:
        valid = [r for r in r4.values() if np.isfinite(fnum(r["dist_mm"]))]
        d = np.median([fnum(r["dist_mm"]) for r in valid]); e = np.median([fnum(r["elev_deg"]) for r in valid])
        az = np.median([fnum(r["az_deg"]) for r in valid])
        pose = make_camera_pose(d, e, az)
        p_phys = marker_probs(V3_ORIENTED, pose, K, dist, sv)
        print(f"\nHeld-out test on ATP-4 ({d:.0f} mm, elevation {e:.1f}, azimuth {az:.1f}); parameters are NOT refitted")
        print(f"{'step':>5s}{'markers':>9s}{'meas PnP%':>11s}{'pred PnP%':>11s}{'meas mk':>9s}{'pred mk':>9s}")
        dp, dm = [], []
        for step in sorted(ATP4_SETS):
            if step not in r4:
                continue
            cov = ATP4_SETS[step][1]
            p = np.array([0.0 if i in cov else p_phys[i] for i in range(8)])
            pt, en = p_at_least(p, GATE) * 100, p.sum()
            mp, mk = fnum(r4[step]["pnp_success_pct"]), fnum(r4[step]["markers_mean"])
            dp.append(pt - mp); dm.append(en - mk)
            print(f"{step:>5d}{8 - len(cov):>9d}{mp:>11.0f}{pt:>11.0f}{mk:>9.1f}{en:>9.1f}")
        print(f"\nHeld-out error: PnP success RMS {np.sqrt(np.mean(np.square(dp))):.1f} percentage points, "
              f"detected markers RMS {np.sqrt(np.mean(np.square(dm))):.2f} (n = {len(dp)} conditions)")
    return sv


# ---------------------------------------------------------------------------
# design / tilt: variant comparison
# ---------------------------------------------------------------------------
def pose_grid(a):
    return [(e, az) for e in a.elevs for az in np.arange(0, 360, a.az_step)]


def select_variants(a):
    """Variants whose name contains any of the --variants substrings (all if none given)."""
    if not a.variants:
        return dict(VARIANTS_ORIENTED)
    return {n: v for n, v in VARIANTS_ORIENTED.items() if any(s in n for s in a.variants)}


def score_variant(ori, a, K, dist, targets, vis, sv, poses, size_mm, sigmas):
    """One record per camera pose: visible markers, gate, coplanarity, MC and Fitzpatrick TRE."""
    recs = []
    for e, az in poses:
        pose = make_camera_pose(a.distance, e, az)
        ids = visible_ids(ori, pose, K, dist, vis, size_mm=size_mm)
        rec = {"el": e, "az": float(az), "n": len(ids), "ok": len(ids) >= a.gate,
               "noncop": noncoplanar(ori, ids, size_mm),
               "p": marker_probs(ori, pose, K, dist, sv, size_mm=size_mm)}
        if rec["ok"]:
            for key, sig in sigmas.items():
                r = simulate(ori, ids, targets, pose, K, dist, sig, a.trials, size_mm=size_mm)
                rec["mc_" + key] = r["tre3_rms"].mean()
                if key == "hi":
                    rec["ov_hi"] = r["ov_rms"].mean()
                    rec["mc_worst"] = r["tre3_rms"].max()
            fz = fitz(ori, ids, targets)
            rec["fitz"] = None if fz is None else fz.mean()
            rec["fitz_max"] = None if fz is None else fz.max()
        recs.append(rec)
    return recs


def hands_check(ori, a, K, dist, targets, vis, poses, size_mm, side):
    """Surgeon on `side`, camera on the same side. For every (pose, target) pair, put the hand
    stand-in over that target and count markers. Returns (cases, ok without hands, ok with hands,
    mean markers lost)."""
    sgn = 1.0 if side == "left" else -1.0
    own = [(e, az) for e, az in poses if sgn * np.sin(np.radians(az)) > 0.2]
    n_case = ok0 = ok1 = 0
    lost = []
    for e, az in own:
        pose = make_camera_pose(a.distance, e, az)
        base = visible_ids(ori, pose, K, dist, vis, size_mm=size_mm)
        for t in targets:
            occ = hand_occluders(t, side)
            ids = visible_ids(ori, pose, K, dist, vis, size_mm=size_mm, occluders=occ)
            n_case += 1
            ok0 += len(base) >= a.gate
            ok1 += len(ids) >= a.gate
            lost.append(len(base) - len(ids))
    return n_case, ok0, ok1, (float(np.mean(lost)) if lost else float("nan"))


def _m(recs, k):
    v = [r[k] for r in recs if r.get(k) is not None]
    return float(np.mean(v)) if v else float("nan")


def cmd_design(a):
    K, dist = load_cal(a.calibration)
    targets, _ = load_pedicles_from_json(a.targets)
    vis = Vis(a.max_angle, a.min_width, a.margin)
    sv = soft_from_args(a)
    poses = pose_grid(a)
    variants = select_variants(a)
    print(f"{len(poses)}-pose sweep at {a.distance:.0f} mm (elevations {a.elevs}, azimuth step {a.az_step} deg), "
          f"{a.marker_mm:.0f} mm markers, entry targets ({len(targets)}), gate {a.gate} markers, "
          f"visibility {vis.max_angle_deg:.0f} deg / {vis.min_width_px:.0f} px / margin {vis.margin_px:.0f}")
    print(f"noise: static {a.sigma_static} px and live {a.sigma_live} px per axis; {a.trials} trials per pose\n")
    res = {n: score_variant(ori, a, K, dist, targets, vis, sv, poses, a.marker_mm,
                            {"lo": a.sigma_static, "hi": a.sigma_live})
           for n, ori in variants.items()}

    print(f"{'variant':<19s}{'N':>3s}{'poses ok':>10s}{'MC static':>11s}{'MC live':>9s}{'overlay live':>14s}{'Fitzpatrick':>13s}"
          f"   (mean TRE over successful poses, mm)")
    for name, recs in res.items():
        ok = [r for r in recs if r["ok"]]
        nm = len(variants[name])
        if not ok:
            print(f"{name:<19s}{nm:>3d}{0:>7d}/{len(recs)}{'-':>11s}{'-':>9s}{'-':>14s}{'-':>13s}")
            continue
        print(f"{name:<19s}{nm:>3d}{len(ok):>7d}/{len(recs)}{_m(ok, 'mc_lo'):>11.3f}{_m(ok, 'mc_hi'):>9.3f}"
              f"{_m(ok, 'ov_hi'):>14.3f}{_m(ok, 'fitz'):>13.3f}")

    print(f"\nPoses tracked (out of {len(poses)}) as a function of the minimum-marker gate (hard limits | expected value, soft model):")
    print(f"{'variant':<19s}" + "".join(f"{'gate ' + str(g):>9s}" for g in (2, 3, 4, 5, 6)) + "   |"
          + "".join(f"{'gate ' + str(g):>9s}" for g in (2, 3, 4, 5, 6)))
    for name, recs in res.items():
        print(f"{name:<19s}" + "".join(f"{sum(r['n'] >= g for r in recs):>9d}" for g in (2, 3, 4, 5, 6)) + "   |"
              + "".join(f"{sum(p_at_least(r['p'], g) for r in recs):>9.1f}" for g in (2, 3, 4, 5, 6)))

    print(f"\nWhere each variant tracks (gate {a.gate}): non-coplanar marker set vs a single plane, "
          f"and poses whose worst target is predicted under 2 mm")
    for name, recs in res.items():
        ok = [r for r in recs if r["ok"]]
        if not ok:
            continue
        nb = sum(r["noncop"] for r in ok)
        acc = [r for r in ok if r.get("fitz_max") is not None and r["fitz_max"] < 2.0]
        hi_el = [r for r in recs if r["el"] >= 60]
        print(f"  {name:<19s} non-coplanar {nb:>3d}, coplanar {len(ok) - nb:>3d}  |  worst < 2 mm: {len(acc):>3d}  |  "
              f"tracked at elevation >= 60: {sum(r['ok'] for r in hi_el)}/{len(hi_el)}")

    if "V1_primary" in res and "V3_8marker" in res:
        c = [(r1, r3) for r1, r3 in zip(res["V1_primary"], res["V3_8marker"]) if r1["ok"] and r3["ok"]]
        if c:
            lower = sum(r3["mc_hi"] < r1["mc_hi"] for r1, r3 in c)
            rel = np.array([(r1["mc_hi"] - r3["mc_hi"]) / r1["mc_hi"] for r1, r3 in c]) * 100
            print(f"\nV3 vs V1 on the {len(c)} poses where both track: V3 lower at {lower}/{len(c)}, "
                  f"{np.percentile(rel, 10):.0f} to {np.percentile(rel, 90):.0f}% lower (10th to 90th percentile)")

    if a.hands != "none":
        sides = ["left", "right"] if a.hands == "both" else [a.hands]
        print(f"\nHand stand-in (box hand plus capsule forearm) over each target in turn; camera on the surgeon's side; gate {a.gate}")
        print(f"{'variant':<19s}{'surgeon':>9s}{'cases':>7s}{'ok, no hands':>14s}{'ok, hands':>11s}{'markers lost':>14s}")
        for name in res:
            for side in sides:
                n_case, ok0, ok1, lost = hands_check(variants[name], a, K, dist, targets, vis, poses, a.marker_mm, side)
                if n_case:
                    print(f"{name:<19s}{side:>9s}{n_case:>7d}{100 * ok0 / n_case:>13.0f}%{100 * ok1 / n_case:>10.0f}%{lost:>14.2f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "n_markers", "elev", "az", "n_visible", "tracked", "noncoplanar",
                    "mc_static_mm", "mc_live_mm", "overlay_live_mm", "fitzpatrick_mm", "fitzpatrick_worst_mm"])
        for name, recs in res.items():
            for r in recs:
                w.writerow([name, len(variants[name]), r["el"], r["az"], r["n"], int(r["ok"]), int(r["noncop"]),
                            *(f"{r[k]:.4f}" if r.get(k) is not None else ""
                              for k in ("mc_lo", "mc_hi", "ov_hi", "fitz", "fitz_max"))])
    print(f"Saved {out}")


def cmd_tilt(a):
    K, dist = load_cal(a.calibration)
    targets, _ = load_pedicles_from_json(a.targets)
    vis = Vis(a.max_angle, a.min_width, a.margin)
    sv = soft_from_args(a)
    poses = pose_grid(a)
    configs = [c for c in DOME_CONFIGS if not a.configs or any(s in c for s in a.configs)]
    print(f"Tilt sweep: {len(poses)} poses at {a.distance:.0f} mm, {a.marker_mm:.0f} mm markers, gate {a.gate}, "
          f"live noise {a.sigma_live} px, {a.trials} trials per pose\n")
    print(f"{'config':<19s}{'tilt':>5s}{'N':>4s}{'tracked':>9s}{'non-copl':>10s}{'worst<2':>9s}{'el>=60':>8s}"
          f"{'expected@gate':>15s}{'MC live':>9s}{'Fitz worst':>12s}")
    rows = []
    for cfg in configs:
        for t in a.tilts:
            ori = build_dome(cfg, tilt_deg=t)
            recs = score_variant(ori, a, K, dist, targets, vis, sv, poses, a.marker_mm, {"hi": a.sigma_live})
            ok = [r for r in recs if r["ok"]]
            acc = sum(1 for r in ok if r.get("fitz_max") is not None and r["fitz_max"] < 2.0)
            hi_el = [r for r in recs if r["el"] >= 60]
            row = {"config": cfg, "tilt": t, "N": len(ori), "tracked": len(ok),
                   "noncop": sum(r["noncop"] for r in ok), "worst_lt2": acc,
                   "el60": sum(r["ok"] for r in hi_el), "el60_n": len(hi_el),
                   "expected": sum(p_at_least(r["p"], a.gate) for r in recs),
                   "mc_live": _m(ok, "mc_hi"), "fitz_worst": _m(ok, "fitz_max")}
            rows.append(row)
            print(f"{cfg:<19s}{t:>5.0f}{row['N']:>4d}{len(ok):>5d}/{len(recs):<3d}{row['noncop']:>10d}{acc:>9d}"
                  f"{row['el60']:>4d}/{row['el60_n']:<3d}{row['expected']:>15.1f}{row['mc_live']:>9.3f}{row['fitz_worst']:>12.3f}")
        print()
    out = Path(a.out_tilt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# atp4: marker-set study
# ---------------------------------------------------------------------------
ATP4_SETS = {   # step -> (sequence, covered-markers set after that step)
    18: ("A/B", set()),
    4: ("A", {1}), 6: ("A", {1, 6}), 8: ("A", {1, 6, 3}), 10: ("A", {1, 6, 3, 4}),
    12: ("A", {1, 6, 3, 4, 2}), 14: ("A", {1, 6, 3, 4, 2, 7}), 16: ("A", {1, 6, 3, 4, 2, 7, 5}),
    20: ("B", {4}), 22: ("B", {4, 5}), 24: ("B", {4, 5, 6}), 26: ("B", {4, 5, 6, 7}),
    28: ("B", {4, 5, 6, 7, 3}), 30: ("B", {4, 5, 6, 7, 3, 1}), 32: ("B", {4, 5, 6, 7, 3, 1, 2}),
}


def cmd_atp4(a):
    if not a.is_v3:
        sys.exit("atp4 uses the V3 covered-marker sequences (ATP4_SETS), so it only runs for V3.")
    K, dist = load_cal(a.calibration)
    targets, labels = load_pedicles_from_json(a.targets)
    rows = {int(float(r["step"])): r for r in read_csv(a.atp4)}
    vis = Vis(a.max_angle, a.min_width, a.margin)
    valid = [r for r in rows.values() if np.isfinite(fnum(r["dist_mm"]))]
    d = np.median([fnum(r["dist_mm"]) for r in valid]); e = np.median([fnum(r["elev_deg"]) for r in valid])
    az = np.median([fnum(r["az_deg"]) for r in valid])
    pose = make_camera_pose(d, e, az)
    phys = set(visible_ids(V3_ORIENTED, pose, K, dist, vis))
    print(f"ATP-4 pose: {d:.0f} mm, elevation {e:.1f}, azimuth {az:.1f}; markers physically visible to the model: {sorted(phys)}\n")
    sig = a.sigma_live
    sv = soft_from_args(a)
    p_phys = marker_probs(V3_ORIENTED, pose, K, dist, sv)
    print(f"{'step':>4s}{'seq':>5s}{'covered':>9s}{'sim vis':>8s}{'meas mk':>8s}{'PnP%':>6s}{'pred%':>6s}  {'gate':>5s}"
          f"{'Fitz mean':>10s}{'log TRE':>8s}{'MC 3D':>7s}{'MC overlay':>11s}")
    for step in sorted(ATP4_SETS):
        if step not in rows:
            continue
        seq, cov = ATP4_SETS[step]
        ids = sorted(phys - cov)
        r = rows[step]
        ok = len(ids) >= GATE
        fz = fitz(V3_ORIENTED, ids, targets) if len(ids) >= 3 else None
        mc = simulate(V3_ORIENTED, ids, targets, pose, K, dist, sig, a.trials) if ok else None
        pred = p_at_least(np.array([0.0 if i in cov else p_phys[i] for i in range(8)]), GATE) * 100
        print(f"{step:>4d}{seq:>5s}{len(cov):>9d}{len(ids):>8d}{fnum(r['markers_mean']):>8.1f}{fnum(r['pnp_success_pct']):>6.0f}{pred:>6.0f}  "
              f"{'pass' if ok else 'FAIL':>5s}"
              f"{(fz.mean() if fz is not None else float('nan')):>10.2f}{fnum(r.get('tre_mean_mm')):>8.2f}"
              f"{(mc['tre3_rms'].mean() if mc else float('nan')):>7.2f}{(mc['ov_rms'].mean() if mc else float('nan')):>11.2f}")
    print("\n'pred%' is the soft-model probability of reaching 4 markers. 'log TRE' is the Fitzpatrick mean the overlay logged over the frames "
          "actually tracked, so it sits above 'Fitz mean' (nominal set) wherever fewer markers than 'sim vis' were really detected.")


# ---------------------------------------------------------------------------
# atp2: error budget
# ---------------------------------------------------------------------------
def budget(a, K, dist, targets, labels, poses_glob, measured_csv, live_rms):
    """Per-measurement (level, measured error, MC static, MC live) overlay errors, in mm."""
    lab_idx = {l: i for i, l in enumerate(labels)}
    meas = read_csv(measured_csv)
    poses = {Path(f).name.replace("_pose.json", ""): json.load(open(f)) for f in glob.glob(poses_glob)}
    ori, size, gate = a.oriented, a.size_mm, a.gate_eff
    sig_live = a.sigma if a.sigma else sigma_for_rms(live_rms, len(ori))
    id2idx = {v: i for i, v in enumerate(a.ids)}
    cache, out, sig_used = {}, [], []
    for r in meas:
        shot = r["shot"]
        if shot not in poses or r["label"] not in lab_idx:
            continue
        p = poses[shot]
        if shot not in cache:
            pose = make_camera_pose(p["distance_mm"], p["elevation_deg"], p["azimuth_deg"])
            ids = [id2idx[i] for i in p.get("visible_ids", []) if i in id2idx] or \
                visible_ids(ori, pose, K, dist, Vis(a.max_angle, a.min_width, a.margin), size_mm=size)
            if len(ids) < gate:
                cache[shot] = None
            else:
                # live-equivalent noise from this shot's own logged reprojection RMS and marker count
                rms_s = fnum(p.get("rms_px"))
                sig_s = sig_live if (a.sigma or not a.per_shot_rms or not np.isfinite(rms_s)) \
                    else sigma_for_rms(rms_s, len(ids))
                sig_used.append(sig_s)
                cache[shot] = (simulate(ori, ids, targets, pose, K, dist, a.sigma_static, a.trials, size_mm=size),
                               simulate(ori, ids, targets, pose, K, dist, sig_s, a.trials, size_mm=size))
        if cache[shot] is None:
            continue
        lo, hi = cache[shot]
        i = lab_idx[r["label"]]
        out.append((r["level"], float(r["err_mm"]), lo["ov_rms"][i], hi["ov_rms"][i]))
    return out, len(poses), len(meas), (float(np.mean(sig_used)) if sig_used else sig_live)


def cmd_atp2(a):
    K, dist = load_cal(a.calibration)
    targets, labels = load_pedicles_from_json(a.targets)
    out, n_poses, n_meas, sig_live = budget(a, K, dist, targets, labels, a.poses, a.measured, a.live_rms)
    sig_static = a.sigma_static
    print(f"array: {a.vname} ({len(a.oriented)} markers, {a.size_mm:.0f} mm, gate {a.gate_eff})")
    print(f"{n_poses} pose files, {n_meas} measurements")
    how = "from each shot's own logged RMS and marker count" if (a.per_shot_rms and not a.sigma) else f"from live RMS {a.live_rms} px, {len(a.oriented)} markers"
    print(f"corner noise: static test {sig_static:.3f} px; live-equivalent {sig_live:.2f} px on average ({how})\n")
    if not out:
        sys.exit(f"No measurements matched a pose file with at least {a.gate_eff} markers.")
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    sysm = lambda m, k: float(np.sqrt(max(m * m - k * k, 0.0)))
    print(f"{'level':>6s}{'n':>4s}{'measured':>10s}{'MC static':>11s}{'MC live':>9s}{'systematic':>18s}")
    print(f"{'':>6s}{'':>4s}{'RMS mm':>10s}{'RMS mm':>11s}{'RMS mm':>9s}{'(live .. static)':>18s}")
    for lv in ("L1", "L2", "L3", "L4", "L5", "all"):
        sub = out if lv == "all" else [o for o in out if o[0] == lv]
        if not sub:
            continue
        m, k0, k1 = rms([o[1] for o in sub]), rms([o[2] for o in sub]), rms([o[3] for o in sub])
        print(f"{lv:>6s}{len(sub):>4d}{m:>10.2f}{k0:>11.2f}{k1:>9.2f}{sysm(m, k1):>11.2f} ..{sysm(m, k0):>5.2f}")
    m, k1 = rms([o[1] for o in out]), rms([o[3] for o in out])
    print(f"\nRandom noise at the live level explains {100 * k1 * k1 / (m * m):.0f}% of the squared measured error. "
          f"The systematic remainder is {sysm(m, k1):.2f} mm RMS "
          f"({sysm(m, rms([o[2] for o in out])):.2f} mm if only the static-test noise is assumed).")
    print("MC overlay error is the on-screen error at the target depth, the same quantity the click tool measures. "
          f"Poses with fewer than {a.gate_eff} markers are skipped (the live pipeline would not track them).")


# ---------------------------------------------------------------------------
# predict: planned ATP-3 poses for any array, with the V3-fitted visibility model
# ---------------------------------------------------------------------------
def cmd_predict(a):
    K, dist = load_cal(a.calibration)
    sv = soft_from_args(a)
    ori, size, gate = a.oriented, a.size_mm, a.gate_eff
    print(f"Prediction for {a.vname}: {len(ori)} markers, {size:.0f} mm, gate {gate}. Visibility model: 50% at "
          f"{sv.th50:.1f} deg and {sv.w50:.1f} px (V3 fit, NOT refitted). Write these down before measuring.\n")
    sweeps = [("distance (az 270, el 45)", [(d, 45, 270) for d in range(450, 1001, 50)], "dist mm", 0),
              ("azimuth at 600 mm, el 45", [(600, 45, z) for z in range(0, 360, 30)], "az deg", 2),
              ("azimuth at 600 mm, el 30", [(600, 30, z) for z in range(0, 360, 30)], "az deg", 2),
              ("elevation (600 mm, az 270)", [(600, e, 270) for e in range(10, 81, 10)], "el deg", 1)]
    rows = []
    for title, poses, lab, k in sweeps:
        print(f"{title}\n{lab:>9s}{'exp mk':>8s}{'hard mk':>9s}{'P(>=gate)%':>12s}   per-marker P")
        for d, e, z in poses:
            pose = make_camera_pose(d, e, z)
            p = marker_probs(ori, pose, K, dist, sv, size_mm=size)
            hard = len(visible_ids(ori, pose, K, dist, Vis(sv.th50, sv.w50, a.margin), size_mm=size))
            pg = p_at_least(p, gate) * 100
            print(f"{(d, e, z)[k]:>9d}{p.sum():>8.1f}{hard:>9d}{pg:>12.0f}   " + " ".join(f"{x:.2f}" for x in p))
            rows.append([title, d, e, z, f"{p.sum():.3f}", hard, f"{pg:.1f}"] + [f"{x:.3f}" for x in p])
        print()
    out = Path(a.out_predict); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sweep", "dist_mm", "elev_deg", "az_deg", "expected_markers", "hard_markers", "p_pnp_pct"]
                   + [f"p_id{i}" for i in a.ids])
        w.writerows(rows)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# figs: figures for the Results chapter
# ---------------------------------------------------------------------------
def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 7.5, "savefig.dpi": 300, "pdf.fonttype": 42})
    BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#7f7f7f"
    K, dist = load_cal(a.calibration)
    sv = soft_from_args(a)
    outdir = Path(a.figdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- tracking: model vs measurement -------------------------------------------------
    rows = load_atp3([a.atp31, a.atp32, a.atp33], a.elev_drop)
    pt = lambda d, e, az: p_at_least(marker_probs(V3_ORIENTED, make_camera_pose(d, e, az), K, dist, sv), GATE) * 100
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    ds = np.arange(300, 1400, 20.0); axs[0].plot(ds / 10, [pt(d, 35, 45) for d in ds], c=BLUE, lw=1.2, label="Simulated")
    azs = np.arange(0, 361, 3.0); axs[1].plot(azs, [pt(450, 33, x) for x in azs], c=BLUE, lw=1.2)
    els = np.arange(5, 71, 1.0); axs[2].plot(els, [pt(460, e, 45) for e in els], c=BLUE, lw=1.2)
    for ax, tag, xi, sc in ((axs[0], "distance", 1, 0.1), (axs[1], "azimuth", 3, 1.0), (axs[2], "elevation", 2, 1.0)):
        sel = [r for r in rows if r[0] == tag]
        ax.plot([r[xi] * sc for r in sel], [r[5] for r in sel], "o", c=ORANGE, ms=3, label="Measured (ATP-3)")
    axs[0].axvspan(0, 55, color=GREY, alpha=0.12, lw=0)
    axs[0].set_xlabel("Distance (cm)"); axs[1].set_xlabel("Azimuth (°)"); axs[2].set_xlabel("Elevation (°)")
    axs[1].set_xticks(range(0, 361, 90)); axs[0].set_ylabel("PnP success (% of frames)")
    axs[0].text(27, 62, "close range\nexcluded:\naim point\nnot logged", ha="center", va="center", fontsize=6.5, color=GREY)
    axs[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(outdir / "fig_mc_tracking_validation.pdf", bbox_inches="tight"); fig.savefig(outdir / "fig_mc_tracking_validation.png", bbox_inches="tight")
    plt.close(fig); print(f"wrote {outdir / 'fig_mc_tracking_validation.pdf'}")

    # --- error budget by level -----------------------------------------------------------
    targets, labels = load_pedicles_from_json(a.targets)
    sets = [("Crisp focus", a.poses, a.measured, a.live_rms), ("Earlier sessions", a.poses_early, a.measured_early, a.live_rms_early)]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    lv_all = ["L5", "L4", "L3", "L2", "L1"]
    for ax, (name, pg, mc, lr) in zip(axs, sets):
        try:
            out, *_ = budget(a, K, dist, targets, labels, pg, mc, lr)
        except SystemExit:
            out = []
        if not out:
            ax.axis("off"); print(f"  {name}: no matched data, panel left empty"); continue
        x = np.arange(len(lv_all)); w = 0.27
        m = [rms([o[1] for o in out if o[0] == lv]) if any(o[0] == lv for o in out) else np.nan for lv in lv_all]
        k = [rms([o[3] for o in out if o[0] == lv]) if any(o[0] == lv for o in out) else np.nan for lv in lv_all]
        sy = [np.sqrt(max(mm * mm - kk * kk, 0.0)) if np.isfinite(mm) else np.nan for mm, kk in zip(m, k)]
        ax.bar(x - w, m, w, color=GREY, label="Measured")
        ax.bar(x, k, w, color=BLUE, label="Random (simulated)")
        ax.bar(x + w, sy, w, color=ORANGE, label="Systematic remainder")
        ax.axhline(2.0, c=GREY, ls="--", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(lv_all); ax.set_title(name)
        ax.set_xlabel("Pedicle level (L5 nearest the markers)")
    axs[0].set_ylabel("Overlay error, RMS (mm)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(outdir / "fig_error_budget.pdf", bbox_inches="tight"); fig.savefig(outdir / "fig_error_budget.png", bbox_inches="tight")
    plt.close(fig); print(f"wrote {outdir / 'fig_error_budget.pdf'}")


# ---------------------------------------------------------------------------
# fle: noise-path check
# ---------------------------------------------------------------------------
def cmd_fle(a):
    K, dist = load_cal(a.calibration)
    s = MARKER_MM / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    rng = np.random.default_rng(0)
    print("single-marker PnP (IPPE_SQUARE): centroid error vs corner noise, marker tilted 30 deg")
    for dmm in (350, 450, 600):
        rv0 = np.array([np.radians(30), 0, 0.0]).reshape(3, 1); tv0 = np.array([0, 0, float(dmm)]).reshape(3, 1)
        clean, _ = cv2.projectPoints(obj, rv0, tv0, K, dist); clean = clean.reshape(-1, 2)
        for sig in (a.sigma_static, a.sigma_live):
            e = []
            for _ in range(1500):
                ok, rv, tv = cv2.solvePnP(obj, (clean + rng.normal(0, sig, clean.shape)).astype(np.float32), K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    e.append((tv - tv0).ravel())
            e = np.array(e)
            print(f"  {dmm} mm, {sig:5.3f} px: 3D {np.sqrt((e ** 2).sum(1).mean()):5.2f} mm "
                  f"(depth {np.sqrt((e[:, 2] ** 2).mean()):5.2f}, lateral {np.sqrt((e[:, :2] ** 2).sum(1).mean()):4.2f})")
    print("measured static FLE at 350 mm: 0.39 to 1.19 mm per marker, median 0.815 mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "design", "tilt", "atp4", "atp2", "fle", "figs", "predict"])
    d = "data/atp/"
    ap.add_argument("--calibration", default="data/calibration/s22_ultra_v1/camera_calibration.npz")
    ap.add_argument("--targets", default="annotations/pedicle_entry_targets.json")
    ap.add_argument("--atp31", default=d + "atp3_1_summary.csv")
    ap.add_argument("--atp32", default=d + "atp3_2_summary.csv")
    ap.add_argument("--atp33", default=d + "atp3_3_summary.csv")
    ap.add_argument("--atp4", default=d + "atp4_summary.csv")
    ap.add_argument("--poses", default="data/screenshots/atp2_crisp/shot_*_pose.json")
    ap.add_argument("--measured", default=d + "atp2_crisp.csv")
    ap.add_argument("--poses-early", default="data/screenshots/atp2_tiers/shot_*_pose.json", help="figs: earlier-focus pose files")
    ap.add_argument("--measured-early", default=d + "atp2_tiers_capture2.csv", help="figs: earlier-focus measurements")
    ap.add_argument("--live-rms-early", type=float, default=1.1, help="figs: live RMS in the earlier-focus sessions (px)")
    ap.add_argument("--figdir", default="docs/figures")
    ap.add_argument("--max-angle", type=float, default=76.0, help="hard limit used for marker sets (from `fit`)")
    ap.add_argument("--min-width", type=float, default=13.0, help="hard limit used for marker sets (from `fit`)")
    ap.add_argument("--th50", type=float, default=None, help="soft model override (from `fit`)")
    ap.add_argument("--s-th", type=float, default=None)
    ap.add_argument("--w50", type=float, default=None)
    ap.add_argument("--s-w", type=float, default=None)
    ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--elev-drop", type=int, nargs="*", default=[9], dest="elev_drop",
                    help="ATP-3.3 step numbers to leave out (default 9: movement at the end of the run)")
    ap.add_argument("--min-dist", type=float, default=550.0, help="fit: ignore ATP-3 steps closer than this (aim point not logged)")
    ap.add_argument("--gate", type=int, default=None, help="minimum markers (default 4, or the marker map's min_markers_pnp)")
    ap.add_argument("--marker-map", default=None, help="atp2/predict: use the real marker poses, size and gate from a live marker_map.json")
    ap.add_argument("--variant", default=None, help="atp2/predict: a design_variants_oriented name (default V3_8marker)")
    ap.add_argument("--out-predict", default="results/tre_analysis/atp3_prediction.csv")
    ap.add_argument("--distance", type=float, default=450.0)
    ap.add_argument("--sigma-static", type=float, default=0.121)
    ap.add_argument("--sigma-live", type=float, default=0.91)
    ap.add_argument("--sigma", type=float, default=None, help="atp2: override the corner noise")
    ap.add_argument("--live-rms", type=float, default=0.87, help="atp2: live reprojection RMS (px), used when a pose file has no rms_px")
    ap.add_argument("--no-per-shot-rms", dest="per_shot_rms", action="store_false",
                    help="atp2: use the single --live-rms for every shot instead of each shot's own RMS")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--out", default="results/tre_analysis/design_sweep.csv")
    ap.add_argument("--out-tilt", default="results/tre_analysis/tilt_sweep.csv")
    ap.add_argument("--marker-mm", type=float, default=MARKER_MM, help="design/tilt: marker edge length (mm)")
    ap.add_argument("--elevs", type=float, nargs="+", default=[10, 20, 30, 40, 50, 60, 70],
                    help="design/tilt: camera elevations (deg)")
    ap.add_argument("--az-step", type=float, default=45.0, help="design/tilt: azimuth step (deg)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="design: only variants whose name contains one of these strings (e.g. V3 D)")
    ap.add_argument("--configs", nargs="*", default=None, help="tilt: only these DOME_CONFIGS (substrings)")
    ap.add_argument("--tilts", type=float, nargs="+", default=[15, 20, 25, 30, 35, 40, 45], help="tilt: angles (deg)")
    ap.add_argument("--hands", choices=["none", "left", "right", "both"], default="none",
                    help="design: hand stand-in occlusion check for a surgeon on this side")
    a = ap.parse_args()
    resolve_variant(a)
    a.gate_cli = a.gate
    if a.gate is None:
        a.gate = a.gate_eff            # design/tilt/figs read a.gate
    {"fit": cmd_fit, "design": cmd_design, "tilt": cmd_tilt, "atp4": cmd_atp4, "atp2": cmd_atp2, "fle": cmd_fle, "figs": cmd_figs, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()