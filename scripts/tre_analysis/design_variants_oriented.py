"""
Variant definitions with per-marker orientation, used by Monte Carlo.

Each marker is a tuple: (centroid_mm, R_marker_to_phantom).
R_marker_to_phantom is a 3x3 rotation matrix whose columns are the
marker's local X, Y, Z axes expressed in phantom (LPS) coordinates. The
marker's own XY plane holds the 4 corners; local Z is the outward normal.

Kept separate from design_variants.py so Fitzpatrick (which only needs
centroids) stays uncluttered.
"""

import numpy as np


# ---------- Orientation matrices (LPS convention) ----------

# Base plate marker: face-up, marker plane parallel to phantom XZ plane.
#   local X (in-plane) -> +X_phantom (patient left)
#   local Y (in-plane) -> +Z_phantom (superior/cranial)
#   local Z (normal)   -> +Y_phantom (up, out of the plate)
R_FLAT_UP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
]).T

# Face A marker: perpendicular to Z axis (caudal wall at Z = -95, normal -Z).
#   local X (in-plane) -> +X_phantom (marker spreads laterally)
#   local Y (in-plane) -> +Y_phantom (marker spreads vertically)
#   local Z (normal)   -> -Z_phantom (outward, toward caudal camera)
R_FACE_A = np.array([
    [1, 0,  0],
    [0, 1,  0],
    [0, 0, -1],
]).T

# Face B marker: perpendicular to X axis (lateral wall at X = +30, normal +X).
#   local X (in-plane) -> +Z_phantom (marker spreads along spine axis)
#   local Y (in-plane) -> +Y_phantom (marker spreads vertically)
#   local Z (normal)   -> +X_phantom (outward, toward camera on patient's left)
R_FACE_B = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
]).T


# ---------- Face-local helpers ----------

def _fA(h, v):
    """Face A marker at face-local (h, v) offset from top-left corner."""
    return np.array([-40.0 + h, 80.0 - v, -95.0])

def _fB(h, v):
    """Face B marker at face-local (h, v) offset from top-left corner."""
    return np.array([+30.0, 80.0 - v, -95.0 - h])


# ---------- V0: 4 coplanar markers on base plate ----------

V0_ORIENTED = [
    (np.array([   0.0,   0.5,  +75.0]), R_FLAT_UP),
    (np.array([   0.0,   0.5,  -75.0]), R_FLAT_UP),
    (np.array([ +40.0,   0.5,    0.0]), R_FLAT_UP),
    (np.array([ -40.0,   0.5,    0.0]), R_FLAT_UP),
]


# ---------- V1: L-bracket, 6 markers spread ----------
# Face-local pattern: (15,15), (65,15), (40,65)

V1_ORIENTED = [
    (_fA(15, 15), R_FACE_A),
    (_fA(65, 15), R_FACE_A),
    (_fA(40, 65), R_FACE_A),
    (_fB(15, 15), R_FACE_B),
    (_fB(65, 15), R_FACE_B),
    (_fB(40, 65), R_FACE_B),
]


# ---------- V2: same bracket, clustered ----------
# Face-local pattern: (27,27), (53,27), (40,53)

V2_ORIENTED = [
    (_fA(27, 27), R_FACE_A),
    (_fA(53, 27), R_FACE_A),
    (_fA(40, 53), R_FACE_A),
    (_fB(27, 27), R_FACE_B),
    (_fB(53, 27), R_FACE_B),
    (_fB(40, 53), R_FACE_B),
]


# ---------- V3: same bracket, 8 markers ----------
# Face-local pattern: (15,15), (65,15), (15,65), (65,65)

V3_ORIENTED = [
    (_fA(15, 15), R_FACE_A),
    (_fA(65, 15), R_FACE_A),
    (_fA(15, 65), R_FACE_A),
    (_fA(65, 65), R_FACE_A),
    (_fB(15, 15), R_FACE_B),
    (_fB(65, 15), R_FACE_B),
    (_fB(15, 65), R_FACE_B),
    (_fB(65, 65), R_FACE_B),
]


VARIANTS_ORIENTED = {
    "V0_baseline":  V0_ORIENTED,
    "V1_primary":   V1_ORIENTED,
    "V2_clustered": V2_ORIENTED,
    "V3_8marker":   V3_ORIENTED,
}


if __name__ == "__main__":
    # Sanity: for each face orientation, verify corners land on the face plane.
    from monte_carlo_tre import corners_from_oriented_marker
    print("Face A marker corner test (should have Z = -95 for all 4 corners):")
    print(corners_from_oriented_marker(np.array([0, 40, -95]), R_FACE_A, 20.0))
    print("\nFace B marker corner test (should have X = +30 for all 4 corners):")
    print(corners_from_oriented_marker(np.array([30, 40, -115]), R_FACE_B, 20.0))
    print("\nBase plate marker (should have Y = 0.5 for all 4 corners):")
    print(corners_from_oriented_marker(np.array([0, 0.5, 0]), R_FLAT_UP, 20.0))