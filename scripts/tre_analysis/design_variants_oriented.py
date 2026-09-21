"""
Variant definitions with per-marker orientation, used by Monte Carlo.

Each marker is a tuple: (centroid_mm, R_marker_to_phantom).
R_marker_to_phantom is a 3x3 rotation matrix whose columns are the
marker's local X, Y, Z axes expressed in phantom (LPS) coordinates.

Physical bracket (L-bracket v1.3 STL, Blender coords x1000 = mm):
  Face A (caudal wall): outer surface Z = -103.7, X from -10 to 70
  Face B (lateral wall): outer surface X = 70.1,  Z from -103.7 to -23.7
  Junction corner: X = 70.1, Z = -103.7

Physical face block placement (confirmed by observation):
  Face A: left column (ID 0,2) near junction (+X), right column (ID 1,3) far
  Face B: right column (ID 5,7) near junction (-Z), left column (ID 4,6) far
"""

import numpy as np


# ---------- Orientation matrices (LPS convention) ----------

R_FLAT_UP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
]).T

# Face A: caudal wall, outer surface at Z = -103.7, normal points -Z.
# Viewing from outside (-Z direction, looking toward +Z):
#   left = junction (+X), right = far (-X).
#   TL-to-TR (local X) goes from junction toward far = -X direction.
R_FACE_A = np.array([
    [-1, 0,  0],    # local X -> -X (left to right = junction to far)
    [ 0, 1,  0],    # local Y -> +Y (up)
    [ 0, 0, -1],    # local Z -> -Z (outward normal)
]).T

# Face B: lateral wall, outer surface at X = 70.1, normal points +X.
# Viewing from outside (+X direction, looking toward -X):
#   left = cranial/far (Z ~ -24), right = junction (Z ~ -104).
#   TL-to-TR (local X) goes from far toward junction = -Z direction.
R_FACE_B = np.array([
    [ 0, 0, -1],    # local X -> -Z (left to right = far to junction)
    [ 0, 1,  0],    # local Y -> +Y (up)
    [ 1, 0,  0],    # local Z -> +X (outward normal)
]).T


# ---------- Face-local helpers ----------

def _fA(h, v):
    """Face A: h=0 at junction edge (X~70), h increases toward far (-X).
    Face block left column (h=15) is near junction, right (h=65) is far."""
    return np.array([70.0 - h, 80.0 - v, -103.7])

def _fB(h, v):
    """Face B: h=0 at cranial/far edge (Z~-24), h increases toward junction (-Z).
    Face block left column (h=15) is far, right (h=65) is near junction."""
    return np.array([70.1, 80.0 - v, -23.7 - h])


# ---------- V0: 4 coplanar markers on base plate ----------

V0_ORIENTED = [
    (np.array([   0.0,   0.5,  +75.0]), R_FLAT_UP),
    (np.array([   0.0,   0.5,  -75.0]), R_FLAT_UP),
    (np.array([ +40.0,   0.5,    0.0]), R_FLAT_UP),
    (np.array([ -40.0,   0.5,    0.0]), R_FLAT_UP),
]

# ---------- V1: L-bracket, 6 markers spread ----------
V1_ORIENTED = [
    (_fA(15, 15), R_FACE_A),
    (_fA(65, 15), R_FACE_A),
    (_fA(40, 65), R_FACE_A),
    (_fB(15, 15), R_FACE_B),
    (_fB(65, 15), R_FACE_B),
    (_fB(40, 65), R_FACE_B),
]

# ---------- V2: same bracket, clustered ----------
V2_ORIENTED = [
    (_fA(27, 27), R_FACE_A),
    (_fA(53, 27), R_FACE_A),
    (_fA(40, 53), R_FACE_A),
    (_fB(27, 27), R_FACE_B),
    (_fB(53, 27), R_FACE_B),
    (_fB(40, 53), R_FACE_B),
]

# ---------- V3: same bracket, 8 markers ----------
V3_ORIENTED = [
    (_fA(15, 15), R_FACE_A),   # slot 0: near junction, top
    (_fA(65, 15), R_FACE_A),   # slot 1: far, top
    (_fA(15, 65), R_FACE_A),   # slot 2: near junction, bottom
    (_fA(65, 65), R_FACE_A),   # slot 3: far, bottom
    (_fB(15, 15), R_FACE_B),   # slot 4: far/cranial, top
    (_fB(65, 15), R_FACE_B),   # slot 5: near junction, top
    (_fB(15, 65), R_FACE_B),   # slot 6: far/cranial, bottom
    (_fB(65, 65), R_FACE_B),   # slot 7: near junction, bottom
]

VARIANTS_ORIENTED = {
    "V0_baseline":  V0_ORIENTED,
    "V1_primary":   V1_ORIENTED,
    "V2_clustered": V2_ORIENTED,
    "V3_8marker":   V3_ORIENTED,
}


if __name__ == "__main__":
    from monte_carlo_tre import corners_from_oriented_marker
    print("Face A corner test (all Z should = -103.7):")
    print(corners_from_oriented_marker(np.array([30, 40, -103.7]), R_FACE_A, 20.0))
    print("\nFace B corner test (all X should = 70.1):")
    print(corners_from_oriented_marker(np.array([70.1, 40, -60]), R_FACE_B, 20.0))

    print("\n=== V3 marker centroids ===")
    for i, (c, R) in enumerate(V3_ORIENTED):
        face = 'A' if i < 4 else 'B'
        junc = "near" if i in (0,2,5,7) else "far"
        print(f"Slot {i} (Face {face}, {junc} junction): "
              f"[{c[0]:+7.1f}, {c[1]:+7.1f}, {c[2]:+7.1f}]")