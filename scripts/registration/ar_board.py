"""
Board geometry and multi-marker solvePnP.

Builds a lookup table mapping ArUco dictionary IDs to their 3D corner
positions in the phantom LPS frame (mm), and provides a single-call
multi-marker PnP solver that aggregates all visible markers.

Two marker map formats are supported by load_corner_lut():
  * bracket maps (generate_bracket_markers.py): slots carry only aruco_id;
    geometry comes from V3_ORIENTED in design_variants_oriented.py
  * explicit-corner maps (generate_dome_markers.py): every slot carries
    corners_mm (TL, TR, BR, BL) in the phantom frame; used as-is
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRE_DIR = _REPO_ROOT / "scripts" / "tre_analysis"
if str(_TRE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRE_DIR))

from design_variants_oriented import V3_ORIENTED     # noqa: E402

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

# Default slot-to-ID mapping (overridden by --marker-map JSON at runtime)
DEFAULT_SLOT_TO_ID = list(range(8))
DEFAULT_MARKER_SIZE_MM = 20.0
MIN_MARKERS_PNP = 4             # default gate; a marker map can set its own (dome maps use 5)

# Solver. SQPNP is globally optimal and needs no starting guess; ITERATIVE can
# converge to a wrong pose on near-planar marker sets (see Simulator update).
PNP_FLAG = cv2.SOLVEPNP_SQPNP
PNP_REFINE_LM = False           # optional Levenberg-Marquardt polish on reprojection error


class CornerLUT(dict):
    """dict aruco_id -> (4, 3) float32 corners, plus the board's gate and marker size."""

    def __init__(self, *args, min_markers=MIN_MARKERS_PNP, marker_size_mm=None, source="", **kw):
        super().__init__(*args, **kw)
        self.min_markers = min_markers
        self.marker_size_mm = marker_size_mm
        self.source = source


def build_corner_lut(oriented_markers=V3_ORIENTED,
                     slot_to_id=DEFAULT_SLOT_TO_ID,
                     marker_size_mm=DEFAULT_MARKER_SIZE_MM):
    """
    Returns dict: aruco_id -> (4, 3) float32 corner positions in
    phantom LPS frame (mm).  Corner order matches OpenCV convention:
    top-left, top-right, bottom-right, bottom-left in marker-local
    coordinates.
    """
    lut = CornerLUT(min_markers=MIN_MARKERS_PNP, marker_size_mm=marker_size_mm, source="V3_ORIENTED")
    for slot_idx, (centroid, R_m2p) in enumerate(oriented_markers):
        aruco_id = slot_to_id[slot_idx]
        s = marker_size_mm / 2.0
        local = np.array([[-s, +s, 0], [+s, +s, 0],
                          [+s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        lut[aruco_id] = (centroid + local @ R_m2p.T).astype(np.float32)
    return lut


def load_marker_map(json_path):
    """Slot-to-ID list from a bracket marker map (kept for older callers)."""
    with open(json_path) as f:
        data = json.load(f)
    return [entry["aruco_id"] for entry in data["slots"]]


def load_corner_lut(json_path=None, marker_size_mm=None):
    """
    Corner LUT from a marker map JSON, or the default V3 bracket if json_path is None.
    marker_size_mm overrides the map's size (bracket maps only; explicit-corner maps
    already encode the size in their corners).
    """
    if json_path is None:
        return build_corner_lut(marker_size_mm=marker_size_mm or DEFAULT_MARKER_SIZE_MM)

    with open(json_path) as f:
        data = json.load(f)
    slots = data["slots"]
    size = float(data.get("marker_size_mm", DEFAULT_MARKER_SIZE_MM))

    if all("corners_mm" in s for s in slots):
        lut = CornerLUT(min_markers=int(data.get("min_markers_pnp", MIN_MARKERS_PNP)),
                        marker_size_mm=size, source=str(json_path))
        for s in slots:
            c = np.asarray(s["corners_mm"], dtype=np.float32)
            if c.shape != (4, 3):
                raise ValueError(f"ID {s['aruco_id']}: corners_mm must be 4 x 3, got {c.shape}")
            lut[int(s["aruco_id"])] = c
        return lut

    slot_to_id = [s["aruco_id"] for s in slots]
    lut = build_corner_lut(slot_to_id=slot_to_id, marker_size_mm=marker_size_mm or size)
    lut.min_markers = int(data.get("min_markers_pnp", MIN_MARKERS_PNP))
    lut.source = str(json_path)
    return lut


def multi_marker_pnp(detected_ids, detected_corners, corner_lut,
                     camera_matrix, dist_coeffs,
                     prev_rvec=None, prev_tvec=None):
    """
    Aggregate all detected board markers into one solvePnP call.
    The gate is the board's own (corner_lut.min_markers) when available.

    Returns (success, rvec, tvec, n_used, reproj_rms).
    """
    obj_pts, img_pts = [], []

    for marker_id, corners_2d in zip(detected_ids.flatten(), detected_corners):
        if marker_id in corner_lut:
            obj_pts.append(corner_lut[marker_id])
            img_pts.append(corners_2d[0])

    n_used = len(obj_pts)
    if n_used < getattr(corner_lut, "min_markers", MIN_MARKERS_PNP):
        return False, None, None, n_used, float("inf")

    obj_pts = np.vstack(obj_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)

    if PNP_FLAG == cv2.SOLVEPNP_ITERATIVE and prev_rvec is not None and prev_tvec is not None:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=PNP_FLAG)

    if not ok:
        return False, None, None, n_used, float("inf")

    if PNP_REFINE_LM:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, camera_matrix, dist_coeffs, rvec, tvec)

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