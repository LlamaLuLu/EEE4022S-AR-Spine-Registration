"""
ATP-1.2: single-face registration check.

With only one bracket face visible, the markers are coplanar and PnP has a
two-fold ambiguity: it can return a mirror pose with the camera on the wrong
side of the face. The overlay guards against this (a side check that rejects
such solutions). This script reads an overlay CSV log and reports:

  - RMS for frames where only Face A, only Face B, or both faces were used
  - WRONG-SIDE ACCEPTED: accepted single-face frames whose camera position is
    on the wrong side of the face plane. This must be 0 (the pass criterion).
  - how many frames the safeguard rejected (needs the 'amb_rejected' column,
    logged by the current overlay; older logs show n/a)
  - fresh acquisitions (entering VERIFYING, e.g. after pressing 'r', when
    there is no previous pose to start from) and whether the very first
    accepted pose of each was on the correct side
  - the camera elevation range the single-face frames were taken at
  - POSE CONSISTENCY across each acquisition: the last accepted pose before
    the re-lock vs the first pose after it. The camera has not moved in that
    instant, so they must agree. A tilt-flipped (mirror) solution leaves the
    camera in front of the face, so the wrong-side test cannot see it, but it
    shows up here as a jump of several degrees / tens of mm.

Face planes (phantom frame, mm):
  Face A (caudal wall)  outer surface Z = -103.7, camera must have Z < -103.7
  Face B (lateral wall) outer surface X = +70.1,  camera must have X > +70.1

Usage (from repo root):
    python scripts/atp/check_single_face.py data/atp/atp1_2_single_face.csv
    python scripts/atp/check_single_face.py data/atp/atp3_2_azimuth.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

FACE_A_Z = -103.7
FACE_B_X = 70.1


def fnum(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def face_of(ids):
    if not ids:
        return None
    if all(i < 4 for i in ids):
        return "A"
    if all(i >= 4 for i in ids):
        return "B"
    return "both"


def stats(x):
    x = np.array([v for v in x if not np.isnan(v)])
    if len(x) == 0:
        return "--", "--", "--"
    return f"{x.mean():.2f}", f"{x.std():.2f}", f"{np.percentile(x, 95):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.log)) if r.get("state") != "SCREENSHOT"]
    if not rows:
        print("No data.")
        return
    has_amb = "amb_rejected" in rows[0]

    groups = {"A": [], "B": [], "both": []}
    elev = {"A": [], "B": [], "both": []}
    wrong = {"A": 0, "B": 0}
    acq = {"A": 0, "B": 0, "both": 0, "other": 0}
    acq_wrong = 0
    prev_state = None
    last_pose = None            # (t, az, el, x, y, z) of the last accepted pose
    jumps = []                  # (dt_s, d_az, d_el, d_pos_mm) per acquisition
    for idx, r in enumerate(rows):
        ids = [int(i) for i in r["visible_ids"].split("|") if i != ""]
        g = face_of(ids)
        cx, cz = fnum(r["cam_x"]), fnum(r["cam_z"])

        def wrong_side(g, cx, cz):
            return (g == "A" and cz > FACE_A_Z) or (g == "B" and cx < FACE_B_X)

        # Fresh acquisition = entering VERIFYING from NAVIGATING or SEARCHING.
        # (Pressing 'r' sets SEARCHING and the next frame re-locks within the
        # same frame, so the log usually shows NAVIGATING -> VERIFYING directly.
        # STALE -> VERIFYING is a resume after a brief dropout, not an acquisition.)
        if r["state"] == "VERIFYING" and prev_state in ("NAVIGATING", "SEARCHING"):
            if r.get("pnp_ok") == "True" and g is not None:
                acq[g] += 1
                acq_wrong += int(wrong_side(g, cx, cz))
                # compare with the last accepted pose just before the re-lock
                t_now = fnum(r["timestamp"])
                if last_pose is not None and t_now - last_pose[0] <= 2.0:
                    d_az = (fnum(r["azimuth_deg"]) - last_pose[1] + 180.0) % 360.0 - 180.0
                    d_el = fnum(r["elevation_deg"]) - last_pose[2]
                    d_pos = float(np.linalg.norm(
                        [fnum(r["cam_x"]) - last_pose[3], fnum(r["cam_y"]) - last_pose[4],
                         cz - last_pose[5]]))
                    jumps.append((t_now - last_pose[0], d_az, d_el, d_pos))
            else:
                acq["other"] += 1
        prev_state = r["state"]

        if r.get("pnp_ok") == "True":
            last_pose = (fnum(r["timestamp"]), fnum(r["azimuth_deg"]), fnum(r["elevation_deg"]),
                         fnum(r["cam_x"]), fnum(r["cam_y"]), cz)
        if r.get("pnp_ok") != "True" or g is None:
            continue
        groups[g].append(fnum(r["rms_px"]))
        elev[g].append(fnum(r["elevation_deg"]))
        if wrong_side(g, cx, cz):
            wrong[g] += 1

    print(f"\n=== {Path(args.log).name}: {len(rows)} frames ===")
    print(f"{'faces used':<14s}{'accepted':>9s}{'RMS mean':>10s}{'RMS SD':>8s}{'RMS p95':>9s}{'wrong side':>12s}")
    for g, name in (("A", "Face A only"), ("B", "Face B only"), ("both", "both faces")):
        m, sd, p95 = stats(groups[g])
        ws = wrong.get(g, "--")
        print(f"{name:<14s}{len(groups[g]):>9d}{m:>10s}{sd:>8s}{p95:>9s}{str(ws):>12s}")

    for g, name in (("A", "Face A only"), ("B", "Face B only")):
        e = np.array([v for v in elev[g] if not np.isnan(v)])
        if len(e):
            print(f"  {name}: camera elevation {np.percentile(e, 5):.0f} to "
                  f"{np.percentile(e, 95):.0f} deg (5th to 95th percentile)")

    n_acq = sum(acq.values())
    print(f"\nFresh acquisitions (entering VERIFYING): {n_acq}  "
          f"[Face A only {acq['A']}, Face B only {acq['B']}, both faces {acq['both']}, "
          f"no valid pose {acq['other']}]")
    print(f"Wrong side at the first accepted pose of an acquisition: {acq_wrong}")

    AZ_TOL_DEG, POS_TOL_MM = 3.0, 30.0
    flipped = 0
    if jumps:
        J = np.array(jumps)
        daz, dpos = np.abs(J[:, 1]), J[:, 3]
        flipped = int(np.sum((daz > AZ_TOL_DEG) | (dpos > POS_TOL_MM)))
        print(f"\nPose consistency across {len(J)} acquisitions "
              f"(last pose before the re-lock vs first pose after):")
        print(f"  azimuth change   median {np.median(daz):.2f} deg, max {daz.max():.2f} deg")
        print(f"  elevation change median {np.median(np.abs(J[:, 2])):.2f} deg, max {np.abs(J[:, 2]).max():.2f} deg")
        print(f"  camera position  median {np.median(dpos):.1f} mm, max {dpos.max():.1f} mm")
        print(f"  acquisitions beyond {AZ_TOL_DEG:.0f} deg or {POS_TOL_MM:.0f} mm: {flipped}"
              f"{'   <-- possible tilt flip' if flipped else ''}")
    else:
        print("\nPose consistency: no acquisitions with a pose just before the re-lock to compare")

    if has_amb:
        rej = sum(int(r["amb_rejected"] or 0) for r in rows)
        print(f"\nSafeguard rejected {rej} frame(s) ({100 * rej / len(rows):.2f}% of all frames)")
    else:
        print("\nSafeguard rejections: n/a (log has no 'amb_rejected' column)")

    tot = wrong["A"] + wrong["B"] + acq_wrong + flipped
    n_single = len(groups["A"]) + len(groups["B"])
    if n_single == 0:
        print("\nNo single-face frames found. Cover one face and view the other.")
    elif tot == 0:
        print(f"\nPASS: {n_single} accepted single-face frames, none on the wrong side of the face.")
    else:
        print(f"\nFAIL: {tot} wrong-side frame(s) / acquisition jump(s), see above.")


if __name__ == "__main__":
    main()