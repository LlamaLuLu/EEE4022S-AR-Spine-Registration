"""
Live AR overlay for fiducial-based spine surgical navigation.

Thin orchestrator: imports board geometry, TRE computation, target
loading, overlay drawing, and STL wireframe from sibling modules.
Runs the detection-PnP-render loop with a three-state machine.

Usage (from repo root):
    python scripts/registration/live_ar_overlay.py \
        --source http://192.168.1.42:8080/video \
        --calib data/calibration/s22_ultra/camera_calibration.npz \
        --isthmus annotations/pedical_isthmus_targets.json \
        --entry annotations/pedicle_entry_targets.json \
        --marker-map data/bracket_markers/marker_map.json

Keyboard:
    c  confirm registration      r  reject / re-register
    t  toggle trajectories       e  toggle TRE ellipses
    w  toggle wireframe          d  toggle detach offset
    h  toggle HUD                ?  toggle shortcut help
    q  quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ar_board import (
    ARUCO_DICT, build_corner_lut, load_marker_map,
    multi_marker_pnp, decompose_pose,
    DEFAULT_MARKER_SIZE_MM, MIN_MARKERS_PNP,
)
from ar_tre import compute_tre_scalar, DEFAULT_FLE_MM
from ar_targets import load_targets
from ar_overlay import (
    TextRenderer, draw_targets, draw_trajectories,
    draw_wireframe, draw_hud, is_in_envelope,
)
from ar_stl import load_stl_for_wireframe


STALE_HOLD_SEC = 0.3
VERIFY_COUNTDOWN_SEC = 3.0
DETACH_OFFSET_PX = 60
ROTATE_MAP = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
              180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def open_source(source):
    try:
        return cv2.VideoCapture(int(source))
    except ValueError:
        return cv2.VideoCapture(source)


def main():
    ap = argparse.ArgumentParser(description="Live AR overlay for spine navigation")
    ap.add_argument("--source", default="0")
    ap.add_argument("--calib", default="data/calibration/s22_ultra/camera_calibration.npz")
    ap.add_argument("--isthmus", default=None, help="Slicer mrk.json (isthmus targets)")
    ap.add_argument("--entry", default=None, help="Slicer mrk.json (entry targets)")
    ap.add_argument("--stl", default=None, help="Phantom STL for wireframe verify")
    ap.add_argument("--marker-map", default=None, help="marker_map.json from generate_bracket_markers.py")
    ap.add_argument("--fle", type=float, default=DEFAULT_FLE_MM)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    args = ap.parse_args()

    # --- Calibration ---
    calib = np.load(args.calib)
    K = calib["camera_matrix"]
    D = calib["dist_coeffs"]

    # --- Board geometry ---
    if args.marker_map:
        slot_to_id = load_marker_map(args.marker_map)
        print(f"Loaded marker map: {slot_to_id}")
    else:
        slot_to_id = list(range(8))
    corner_lut = build_corner_lut(slot_to_id=slot_to_id)
    n_total = len(slot_to_id)

    # --- Targets ---
    target_pts, target_labels, traj_pairs = load_targets(args.isthmus, args.entry)
    print(f"Targets: {len(target_pts)}, trajectory pairs: {len(traj_pairs)}")

    # --- Wireframe ---
    mesh_verts, mesh_edges = None, None
    if args.stl:
        mesh_verts, mesh_edges = load_stl_for_wireframe(args.stl)
        if mesh_verts is not None:
            print(f"STL wireframe: {len(mesh_verts)} verts, {len(mesh_edges)} edges")

    # --- Video ---
    rotate_code = ROTATE_MAP[args.rotate]
    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    # --- State ---
    state = "SEARCHING"
    prev_rvec = prev_tvec = None
    last_good_time = verify_start_time = 0.0
    show_traj = show_ellipses = show_wire = show_hud = True
    show_help = detach = False
    tr = None  # TextRenderer, initialised on first frame

    print("Live AR overlay running. Press '?' for shortcuts, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        h, w = frame.shape[:2]
        if tr is None or tr.h != h:
            tr = TextRenderer(h, w)

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # --- Multi-marker PnP ---
        pnp_ok = False
        n_markers = 0
        reproj_rms = float("inf")
        distance_mm = elev_deg = az_deg = 0.0
        cam_pos = np.zeros(3)
        in_env = False
        tre_values = None
        visible_ids = []

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids,
                                          borderColor=(60, 160, 60))
            pnp_ok, rvec, tvec, n_markers, reproj_rms = multi_marker_pnp(
                ids, corners, corner_lut, K, D, prev_rvec, prev_tvec,
            )
            if pnp_ok:
                prev_rvec, prev_tvec = rvec.copy(), tvec.copy()
                last_good_time = now
                cam_pos, distance_mm, elev_deg, az_deg = decompose_pose(rvec, tvec)
                in_env = is_in_envelope(elev_deg, az_deg)
                visible_ids = [m for m in ids.flatten() if m in corner_lut]
                tre_values = compute_tre_scalar(visible_ids, corner_lut,
                                                target_pts, args.fle)

                if state in ("SEARCHING", "STALE"):
                    state = "VERIFYING"
                    verify_start_time = now

        # --- Stale / lost ---
        if not pnp_ok:
            if prev_rvec is not None and (now - last_good_time) < STALE_HOLD_SEC:
                if state not in ("SEARCHING",):
                    state = "STALE"
                rvec, tvec = prev_rvec, prev_tvec
                cam_pos, distance_mm, elev_deg, az_deg = decompose_pose(rvec, tvec)
                in_env = is_in_envelope(elev_deg, az_deg)
                tre_values = compute_tre_scalar(visible_ids, corner_lut,
                                                target_pts, args.fle)
            else:
                state = "SEARCHING"
                prev_rvec = prev_tvec = None
                visible_ids = []

        # --- Verify countdown ---
        countdown = 0.0
        if state == "VERIFYING":
            countdown = VERIFY_COUNTDOWN_SEC - (now - verify_start_time)
            if countdown <= 0:
                state = "NAVIGATING"
                print("Registration confirmed (auto).")

        # --- Draw ---
        doff = DETACH_OFFSET_PX if detach else 0

        if state in ("VERIFYING", "NAVIGATING", "STALE"):
            if show_wire and mesh_verts is not None and state != "STALE":
                draw_wireframe(frame, mesh_verts, mesh_edges,
                               rvec, tvec, K, D, cam_pos)

            if state == "NAVIGATING":
                if show_traj:
                    draw_trajectories(frame, target_pts, traj_pairs,
                                      rvec, tvec, K, D, doff)
                draw_targets(frame, target_pts, target_labels, tre_values,
                             rvec, tvec, K, D, visible_ids, corner_lut,
                             args.fle, in_env, tr, show_ellipses, doff)

        if show_hud:
            draw_hud(frame, tr, n_markers, n_total, pnp_ok, reproj_rms,
                     distance_mm, elev_deg, az_deg, in_env,
                     state, tre_values, target_labels, countdown, show_help)

        cv2.imshow("AR Spine Navigation", frame)

        # --- Keys ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and state == "VERIFYING":
            state = "NAVIGATING"
            print("Registration confirmed (manual).")
        elif key == ord("r"):
            state = "SEARCHING"
            prev_rvec = prev_tvec = None
        elif key == ord("t"):
            show_traj = not show_traj
        elif key == ord("e"):
            show_ellipses = not show_ellipses
        elif key == ord("w"):
            show_wire = not show_wire
        elif key == ord("d"):
            detach = not detach
        elif key == ord("h"):
            show_hud = not show_hud
        elif key == ord("?"):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()