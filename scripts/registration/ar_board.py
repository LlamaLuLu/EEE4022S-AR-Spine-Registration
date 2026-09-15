"""
Board geometry and multi-marker solvePnP for the V3 L-bracket.

Builds a lookup table mapping ArUco dictionary IDs to their 3D corner
positions in the phantom LPS frame, and provides a single-call
multi-marker PnP solver that aggregates all visible markers.
"""

import json
from pathlib import Path
import cv2
import numpy as np

# Repo-internal imports
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRE_DIR = _REPO_ROOT / "scripts" / "tre_analysis"
if str(_TRE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRE_DIR))

from design_variants_oriented import V3_ORIENTED

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

# Default slot-to-ID mapping (overridden by --marker-map JSON at runtime)
DEFAULT_SLOT_TO_ID = list(range(8))
DEFAULT_MARKER_SIZE_MM = 20.0
MIN_MARKERS_PNP = 4


def build_corner_lut(oriented_markers=V3_ORIENTED,
                     slot_to_id=DEFAULT_SLOT_TO_ID,
                     marker_size_mm=DEFAULT_MARKER_SIZE_MM):
    """
    Returns dict: aruco_id -> (4, 3) float32 corner positions in
    phantom LPS frame (mm).  Corner order matches OpenCV convention:
    top-left, top-right, bottom-right, bottom-left in marker-local
    coordinates.
    """
    lut = {}
    for slot_idx, (centroid, R_m2p) in enumerate(oriented_markers):
        aruco_id = slot_to_id[slot_idx]
        s = marker_size_mm / 2.0
        local = np.array([[-s, +s, 0], [+s, +s, 0],
                          [+s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        lut[aruco_id] = (centroid + local @ R_m2p.T).astype(np.float32)
    return lut


def load_marker_map(json_path):
    """Load a marker map JSON (from generate_bracket_markers.py)."""
    with open(json_path) as f:
        data = json.load(f)
    return [entry["aruco_id"] for entry in data["slots"]]


def multi_marker_pnp(detected_ids, detected_corners, corner_lut,
                     camera_matrix, dist_coeffs,
                     prev_rvec=None, prev_tvec=None):
    """
    Aggregate all detected board markers into one solvePnP call.

    Returns (success, rvec, tvec, n_used, reproj_rms).
    """
    obj_pts, img_pts = [], []

    for marker_id, corners_2d in zip(detected_ids.flatten(), detected_corners):
        if marker_id in corner_lut:
            obj_pts.append(corner_lut[marker_id])
            img_pts.append(corners_2d[0])

    n_used = len(obj_pts)
    if n_used < MIN_MARKERS_PNP:
        return False, None, None, n_used, float("inf")

    obj_pts = np.vstack(obj_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)

    if prev_rvec is not None and prev_tvec is not None:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not ok:
        return False, None, None, n_used, float("inf")

    reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    err = (reproj.reshape(-1, 2) - img_pts).ravel()
    reproj_rms = float(np.sqrt(np.mean(err ** 2)))

    return True, rvec, tvec, n_used, reproj_rms


def decompose_pose(rvec, tvec):
    """
    Camera position in phantom LPS + spherical coordinates.

    Returns (cam_pos, distance_mm, elevation_deg, azimuth_deg).
    Azimuth: 0 = caudal (-Z), 90 = patient left (+X).
    """
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).flatten()

    distance_mm = float(np.linalg.norm(cam_pos))
    elevation_deg = float(np.degrees(np.arcsin(
        np.clip(cam_pos[1] / max(distance_mm, 1e-6), -1, 1)
    )))
    azimuth_deg = float(np.degrees(np.arctan2(cam_pos[0], -cam_pos[2])))
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    return cam_pos, distance_mm, elevation_deg, azimuth_deg