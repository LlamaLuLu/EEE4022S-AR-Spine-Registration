"""
Variant definitions with per-marker orientation, used by Monte Carlo.

Each marker is a tuple: (centroid_mm, R_marker_to_phantom).
R_marker_to_phantom is a 3x3 rotation matrix whose columns are the
marker's local X, Y, Z axes expressed in phantom coordinates. The
marker's own XY plane is where the 4 corners live; local Z is the
outward-facing normal.

Kept separate from design_variants.py so Fitzpatrick (which only needs
centroids) stays uncluttered.
"""

import numpy as np


# ---------- Common orientation matrices ----------

# Base plate marker: face-up, marker plane parallel to phantom XY.
# Marker local X = phantom X, local Y = phantom Y, local Z (normal) = +phantom Z.
R_FLAT_UP = np.eye(3)

# Rotation matrix convention:
#   Columns are the marker's local axes expressed in phantom-frame coords.
#   Row-based construction with .T lets each row spell out one axis clearly.

# Face A marker: perpendicular to phantom Y-axis, wall stands at Y = -95.
#   local X (in-plane) → +X_phantom (marker spreads laterally)
#   local Y (in-plane) → +Z_phantom (marker spreads vertically)
#   local Z (normal)   → -Y_phantom (outward, facing caudal camera)
R_FACE_A = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
]).T

# Face B marker: perpendicular to phantom X-axis, wall stands at X = -30.
#   local X (in-plane) → +Y_phantom (marker spreads along spine axis)
#   local Y (in-plane) → +Z_phantom (marker spreads vertically)
#   local Z (normal)   → -X_phantom (outward, facing lateral camera)
R_FACE_B = np.array([
    [ 0, 1, 0],
    [ 0, 0, 1],
    [-1, 0, 0],
]).T


# ---------- V0: 4 coplanar markers on the base plate ----------

V0_ORIENTED = [
    (np.array([  0.0,  +80.0,  0.5]), R_FLAT_UP),
    (np.array([  0.0,  -80.0,  0.5]), R_FLAT_UP),
    (np.array([+22.0,    0.0,  0.5]), R_FLAT_UP),
    (np.array([-22.0,    0.0,  0.5]), R_FLAT_UP),
]

# ---------- V1: L-bracket, 6 markers spread ----------

V1_ORIENTED = [
    # Face A markers (perpendicular to Y-axis, normal facing -Y)
    (np.array([-20.0, -95.0,  65.0]), R_FACE_A),
    (np.array([+20.0, -95.0,  65.0]), R_FACE_A),
    (np.array([  0.0, -95.0,  15.0]), R_FACE_A),
    # Face B markers (perpendicular to X-axis, normal facing -X)
    (np.array([-30.0, -105.0,  65.0]), R_FACE_B),
    (np.array([-30.0, -128.0,  65.0]), R_FACE_B),
    (np.array([-30.0, -116.0,  15.0]), R_FACE_B),
]

# ---------- V2: same L-bracket, clustered markers ----------

V2_ORIENTED = [
    (np.array([-8.0, -95.0, 45.0]), R_FACE_A),
    (np.array([+8.0, -95.0, 45.0]), R_FACE_A),
    (np.array([ 0.0, -95.0, 30.0]), R_FACE_A),
    (np.array([-30.0, -112.0, 45.0]), R_FACE_B),
    (np.array([-30.0, -118.0, 45.0]), R_FACE_B),
    (np.array([-30.0, -115.0, 30.0]), R_FACE_B),
]

# ---------- V3: same L-bracket, 8 markers ----------

V3_ORIENTED = [
    (np.array([-22.0, -95.0, 65.0]), R_FACE_A),
    (np.array([+22.0, -95.0, 65.0]), R_FACE_A),
    (np.array([-22.0, -95.0, 15.0]), R_FACE_A),
    (np.array([+22.0, -95.0, 15.0]), R_FACE_A),
    (np.array([-30.0, -103.0, 65.0]), R_FACE_B),
    (np.array([-30.0, -128.0, 65.0]), R_FACE_B),
    (np.array([-30.0, -103.0, 15.0]), R_FACE_B),
    (np.array([-30.0, -128.0, 15.0]), R_FACE_B),
]


VARIANTS_ORIENTED = {
    "V0_baseline":  V0_ORIENTED,
    "V1_primary":   V1_ORIENTED,
    "V2_clustered": V2_ORIENTED,
    "V3_8marker":   V3_ORIENTED,
}


if __name__ == "__main__":
    # Quick sanity: for each face orientation, verify the corners land on
    # the face plane they should.
    from monte_carlo_tre import corners_from_oriented_marker
    print("Face A marker corner test (should have Y = -95 for all 4 corners):")
    c = corners_from_oriented_marker(np.array([0, -95, 40]), R_FACE_A, 20.0)
    print(c)
    print("\nFace B marker corner test (should have X = -30 for all 4 corners):")
    c = corners_from_oriented_marker(np.array([-30, -115, 40]), R_FACE_B, 20.0)
    print(c)
    print("\nBase plate marker (should have Z = 0.5 for all 4 corners):")
    c = corners_from_oriented_marker(np.array([0, 0, 0.5]), R_FLAT_UP, 20.0)
    print(c)