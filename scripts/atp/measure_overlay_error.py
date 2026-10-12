"""
ATP-2: measure overlay error between projected pedicle targets and the
physical ground-truth dots painted on the phantom.

Each screenshot taken with 's' in live_ar_overlay_qt.py saves:
    shot_..._clean.png   camera frame with no overlays
    shot_..._pose.json   camera pose (rvec, tvec), intrinsics, marker info

Because the pose is saved, the target's projected position is computed
exactly. You only click the centre of each silver dot. The tool shows a
zoomed crop around each target; the cyan cross is where the overlay
places the target, you click where the dot really is.

Outputs:
  <out>.csv      one row per (screenshot, target): in-plane error in px
                 and mm at the target's depth, error direction in the
                 phantom frame, distance from marker centroid, predicted TRE
  <out>_3d.csv   one row per target seen in 2+ screenshots: dot position
                 triangulated from all views, full 3D error vs target

Controls (in the zoom window):
    left click   mark the dot centre (click again to adjust)
    space/enter  accept and go to the next target
    s            skip (dot not visible / occluded)
    q            quit and save everything measured so far

Usage (from repo root):
    python scripts/atp/measure_overlay_error.py \
        --shots "data/screenshots/shot_*_pose.json" \
        --out data/atp/atp2_overlay_error
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for sub in ("registration", "tre_analysis"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from ar_targets import load_targets            # noqa: E402
from ar_board import load_corner_lut   # noqa: E402
from ar_theme import envelope_tier                       # noqa: E402

try:
    from ar_tre import compute_tre_scalar, DEFAULT_FLE_MM  # noqa: E402
except Exception:
    compute_tre_scalar, DEFAULT_FLE_MM = None, 0.8

CROP = 120      # half-width of crop in source pixels
SCALE = 4       # zoom factor, gives 0.25px click resolution
WIN = "ATP-2 measure"


# --------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------

def cam_ray(px, K, D, R, t):
    """Ray (origin, unit direction) in phantom frame through pixel px."""
    n = cv2.undistortPoints(np.array([[px]], np.float64), K, D).reshape(2)
    d_cam = np.array([n[0], n[1], 1.0])
    C = (-R.T @ t).ravel()
    d = R.T @ d_cam
    return C, d / np.linalg.norm(d), n


def triangulate(rays):
    """Least-squares closest point to a set of rays."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ C
    X = np.linalg.solve(A, b)
    resid = [np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C)) for C, d in rays]
    return X, float(np.sqrt(np.mean(np.square(resid))))


def max_ray_angle_deg(rays):
    ang = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            c = np.clip(np.dot(rays[i][1], rays[j][1]), -1, 1)
            ang = max(ang, np.degrees(np.arccos(abs(c))))
    return ang


# --------------------------------------------------------------------
# Click UI
# --------------------------------------------------------------------

class Clicker:
    def __init__(self):
        self.pt = None
        self.last_hidden = False
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self._cb)

    def _cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt = (x, y)

    def ask(self, img, target_px, title, require_hidden=False):
        """
        Returns clicked full-image pixel (float), 'skip', or 'quit'.
        Shows a bold red cross at the projected target. Press h to hide it.
        With require_hidden=True a click is only accepted while the cross is
        hidden, so you never click with the target marked on screen.
        self.last_hidden records whether the cross was hidden at accept time.
        """
        zoom, x0, y0 = make_view(img, target_px)
        self.pt = None
        cross_visible = True
        msg = ""

        while True:
            disp = draw_view(zoom, x0, y0, target_px, self.pt, cross_visible,
                             title, require_hidden, msg)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(20) & 0xFF
            if k == ord("h"):
                cross_visible = not cross_visible
                msg = ""
            elif k in (ord(" "), 13) and self.pt is not None:
                if require_hidden and cross_visible:
                    msg = "hide the cross first (h), then click the dot"
                    continue
                self.last_hidden = not cross_visible
                return zoom_to_full(self.pt[0], self.pt[1], x0, y0)
            elif k == ord("s"):
                return "skip"
            elif k == ord("q"):
                return "quit"


def draw_view(zoom, x0, y0, target_px, pt, cross_visible, title,
              require_hidden=False, msg=""):
    """Compose the display: bold red reticle at the target (clear centre so the
    dot stays visible), your click, and the header text. Pure function."""
    disp = zoom.copy()
    if cross_visible:
        hx, hy = (int(round(t)) for t in full_to_zoom(target_px[0], target_px[1], x0, y0))
        gap, arm, thick = 12, 48, 3
        red = (0, 0, 255)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(disp, (hx + dx * gap, hy + dy * gap),
                     (hx + dx * (gap + arm), hy + dy * (gap + arm)),
                     red, thick, cv2.LINE_AA)
    if pt is not None:
        cv2.circle(disp, pt, 6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(disp, pt, (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(disp, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(disp, msg, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
    else:
        help_txt = ("h=hide cross, click dot centre, space=accept  s=skip  q=quit"
                    if not require_hidden else
                    "h=hide cross FIRST, click dot centre, space=accept  s=skip  q=quit")
        cv2.putText(disp, help_txt, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
    return disp


def zoom_to_full(zx, zy, x0, y0):
    """Zoomed-view pixel -> full-image pixel (OpenCV pixel-centre convention)."""
    return x0 + (zx + 0.5) / SCALE - 0.5, y0 + (zy + 0.5) / SCALE - 0.5


def full_to_zoom(u, v, x0, y0):
    return (u - x0 + 0.5) * SCALE - 0.5, (v - y0 + 0.5) * SCALE - 0.5


def make_view(img, centre):
    """Zoomed crop around centre. Returns (zoom_img, x0, y0) where x0,y0 is
    the crop origin in full-image pixels. Full-image px = x0 + zoom_px / SCALE."""
    pad = cv2.copyMakeBorder(img, CROP, CROP, CROP, CROP,
                             cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    crop = pad[cy:cy + 2 * CROP, cx:cx + 2 * CROP]
    zoom = cv2.resize(crop, None, fx=SCALE, fy=SCALE,
                      interpolation=cv2.INTER_CUBIC)
    return zoom, cx - CROP, cy - CROP


def compare_sessions(prev_csv, rows):
    """Paired comparison with an earlier measurement of the same screenshots."""
    if not Path(prev_csv).exists():
        print(f"\nCompare: {prev_csv} not found, skipping the comparison "
              f"(your measurements are saved).")
        return
    prev = {(r["shot"], r["label"]): float(r["err_mm"])
            for r in csv.DictReader(open(prev_csv))}
    a, b = [], []
    for r in rows:
        k = (r["shot"], r["label"])
        if k in prev:
            a.append(prev[k])
            b.append(r["err_mm"])
    if len(a) < 3:
        print(f"\nCompare: only {len(a)} matching (shot, target) pairs in {prev_csv}")
        return
    a, b = np.array(a), np.array(b)
    d = b - a
    print(f"\n=== Session comparison vs {Path(prev_csv).name} ({len(a)} pairs) ===")
    print(f"mean error  previous {a.mean():.2f}  this {b.mean():.2f}  "
          f"(change {d.mean():+.2f} mm, lower in {(d < 0).sum()}/{len(d)})")
    print(f"paired difference SD {d.std(ddof=1):.2f} mm, mean |diff| {np.abs(d).mean():.2f}, "
          f"max {np.abs(d).max():.2f}")
    print(f"click repeatability (one session) ~ {d.std(ddof=1) / np.sqrt(2):.2f} mm; "
          f"session-to-session shift {d.mean():+.2f} mm")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True,
                    help='Glob of pose JSON files, e.g. "data/screenshots/shot_*_pose.json"')
    ap.add_argument("--isthmus", default="annotations/pedicle_isthmus_targets.json")
    ap.add_argument("--entry", default="annotations/pedicle_entry_targets.json")
    ap.add_argument("--marker-map", default="data/bracket_markers/marker_map.json")
    ap.add_argument("--labels", default="_E_",
                    help="Substring filter for target labels (default: entry points)")
    ap.add_argument("--fle", type=float, default=DEFAULT_FLE_MM)
    ap.add_argument("--out", default="data/atp/atp2_overlay_error")
    ap.add_argument("--require-hidden", action="store_true",
                    help="Only accept a click while the target cross is hidden (press h). "
                         "Removes anchoring on the marker; use for the numbers you report.")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Skip shots whose file name contains any of these strings "
                         "(e.g. --skip 093431_001 093445_003)")
    ap.add_argument("--compare", default=None,
                    help="Earlier per-view CSV of the same screenshots, for a paired comparison")
    args = ap.parse_args()

    shots = sorted(glob.glob(args.shots))
    if not shots:
        print(f"No pose files match {args.shots}")
        return

    if args.skip:
        kept = [f for f in shots if not any(k in Path(f).name for k in args.skip)]
        print(f"Skipping {len(shots) - len(kept)} shot(s) matching {args.skip}")
        shots = kept
        if not shots:
            print("No shots left after --skip.")
            return

    pts, labels, _ = load_targets(args.isthmus, args.entry)
    sel = [i for i, l in enumerate(labels) if args.labels in l]
    print(f"{len(shots)} screenshot(s), {len(sel)} target(s) matching '{args.labels}'")

    lut = load_corner_lut(args.marker_map if Path(args.marker_map).exists() else None)
    marker_centroid = np.mean([c.mean(axis=0) for c in lut.values()], axis=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows, rays = [], {}
    ui = Clicker()
    quit_all = False

    for shot in shots:
        if quit_all:
            break
        pose = json.load(open(shot))
        img_path = shot.replace("_pose.json", "_clean.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Missing {img_path}, skipping")
            continue
        K = np.array(pose["camera_matrix"], float)
        D = np.array(pose["dist_coeffs"], float)
        rvec = np.array(pose["rvec"], float).reshape(3, 1)
        tvec = np.array(pose["tvec"], float).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        f = 0.5 * (K[0, 0] + K[1, 1])
        h, w = img.shape[:2]

        tier = pose.get("env_tier") or envelope_tier(
            pose.get("elevation_deg", np.nan), pose.get("azimuth_deg", np.nan),
            pose.get("distance_mm"))

        tre = None
        if compute_tre_scalar is not None and pose.get("visible_ids"):
            try:
                tre = compute_tre_scalar(pose["visible_ids"], lut, pts, args.fle)
            except Exception:
                tre = None

        proj, _ = cv2.projectPoints(pts[sel].astype(np.float64), rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)

        for k, i in enumerate(sel):
            p = proj[k]
            if not (0 <= p[0] < w and 0 <= p[1] < h):
                continue
            title = f"{Path(shot).stem.replace('_pose', '')}   {labels[i]}"
            res = ui.ask(img, p, title, require_hidden=args.require_hidden)
            if res == "quit":
                quit_all = True
                break
            if res == "skip":
                continue

            click = np.array(res)
            err_px = float(np.linalg.norm(click - p))
            X_cam = (R @ pts[i].reshape(3, 1) + tvec).ravel()
            depth = float(X_cam[2])

            # Back-project click to the target's depth plane (in camera frame)
            C, d, n = cam_ray(click, K, D, R, tvec)
            Q_cam = np.array([n[0] * depth, n[1] * depth, depth])
            Q = (R.T @ (Q_cam.reshape(3, 1) - tvec)).ravel()
            dv = Q - pts[i]
            rays.setdefault(labels[i], []).append((C, d))

            rows.append({
                "shot": Path(shot).stem.replace("_pose", ""),
                "label": labels[i],
                "level": labels[i].split("_")[0],
                "side": labels[i].split("_")[-1],
                "distance_mm": round(pose.get("distance_mm", np.nan), 1),
                "elev_deg": round(pose.get("elevation_deg", np.nan), 1),
                "az_deg": round(pose.get("azimuth_deg", np.nan), 1),
                "n_markers": pose.get("n_markers"),
                "tier": tier,
                "rms_px": pose.get("rms_px"),
                "target_u": round(p[0], 2), "target_v": round(p[1], 2),
                "dot_u": round(click[0], 2), "dot_v": round(click[1], 2),
                "err_px": round(err_px, 2),
                "cross_hidden": bool(ui.last_hidden),
                "depth_mm": round(depth, 1),
                "err_mm": round(float(np.linalg.norm(dv)), 3),
                "dx_mm": round(dv[0], 3), "dy_mm": round(dv[1], 3), "dz_mm": round(dv[2], 3),
                "dist_from_markers_mm": round(float(np.linalg.norm(pts[i] - marker_centroid)), 1),
                "pred_tre_mm": None if tre is None else round(float(tre[i]), 3),
            })
            print(f"  {labels[i]:8s} err {err_px:6.2f}px = {np.linalg.norm(dv):5.2f}mm "
                  f"(dx {dv[0]:+.2f}, dy {dv[1]:+.2f}, dz {dv[2]:+.2f})")

    cv2.destroyAllWindows()

    if not rows:
        print("Nothing measured.")
        return

    # --- per-view CSV ---
    out2d = f"{args.out}.csv"
    with open(out2d, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    if args.compare:
        compare_sessions(args.compare, rows)

    # --- triangulated 3D CSV ---
    rows3d = []
    for lab, rs in sorted(rays.items()):
        if len(rs) < 2:
            continue
        X, resid = triangulate(rs)
        i = labels.index(lab)
        e = X - pts[i]
        rows3d.append({
            "label": lab, "n_views": len(rs),
            "max_ray_angle_deg": round(max_ray_angle_deg(rs), 1),
            "ray_residual_mm": round(resid, 3),
            "err3d_mm": round(float(np.linalg.norm(e)), 3),
            "dx_mm": round(e[0], 3), "dy_mm": round(e[1], 3), "dz_mm": round(e[2], 3),
            "dist_from_markers_mm": round(float(np.linalg.norm(pts[i] - marker_centroid)), 1),
        })
    if rows3d:
        with open(f"{args.out}_3d.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows3d[0].keys()))
            wr.writeheader()
            wr.writerows(rows3d)

    # --- summary ---
    e = np.array([r["err_mm"] for r in rows])
    print(f"\n=== In-plane overlay error ({len(rows)} measurements) ===")
    print(f"mean {e.mean():.2f} mm | median {np.median(e):.2f} | "
          f"max {e.max():.2f} | >2mm: {(e > 2).sum()}/{len(e)}")
    print(f"\n{'level':<6s}{'n':>4s}{'mean mm':>10s}{'max mm':>9s}{'dist mm':>9s}{'pred TRE':>10s}")
    for lv in sorted({r['level'] for r in rows}):
        sub = [r for r in rows if r["level"] == lv]
        ee = np.array([r["err_mm"] for r in sub])
        pt = [r["pred_tre_mm"] for r in sub if r["pred_tre_mm"] is not None]
        print(f"{lv:<6s}{len(sub):>4d}{ee.mean():>10.2f}{ee.max():>9.2f}"
              f"{np.mean([r['dist_from_markers_mm'] for r in sub]):>9.0f}"
              f"{(np.mean(pt) if pt else float('nan')):>10.2f}")
    print(f"\n{'tier':<9s}{'n':>4s}{'mean mm':>10s}{'median':>8s}{'max mm':>9s}{'>2mm':>7s}")
    for tier in ("full", "reduced", "outside"):
        ee = np.array([r["err_mm"] for r in rows if r["tier"] == tier])
        if len(ee):
            print(f"{tier:<9s}{len(ee):>4d}{ee.mean():>10.2f}{np.median(ee):>8.2f}"
                  f"{ee.max():>9.2f}{(ee > 2).sum():>4d}/{len(ee)}")
    if rows3d:
        e3 = np.array([r["err3d_mm"] for r in rows3d])
        print(f"\n=== Triangulated 3D error ({len(rows3d)} targets) ===")
        print(f"mean {e3.mean():.2f} mm | max {e3.max():.2f}")
        weak = [r["label"] for r in rows3d if r["max_ray_angle_deg"] < 15]
        if weak:
            print(f"Note: views of {', '.join(weak)} span < 15 deg, depth is poorly constrained.")
    print(f"\nSaved: {out2d}" + (f", {args.out}_3d.csv" if rows3d else ""))


if __name__ == "__main__":
    main()