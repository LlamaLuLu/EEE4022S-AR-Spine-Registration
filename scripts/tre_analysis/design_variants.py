"""
Marker layouts (V0-V3) and pedicle targets in the LPS phantom frame.

LPS convention (matches Slicer default and DICOM):
  origin  : centre of base plate top surface
  +X      : patient left  (lateral)
  +Y      : posterior     (vertical, up when phantom is prone on plate)
  +Z      : superior      (spine axis, cranial is +Z, caudal is -Z)
  units   : millimetres

Physical bracket (L-bracket v1.3 STL, Blender coords x1000 = mm):
  Face A (caudal wall): outer surface Z = -103.7, X from -10 to 70
  Face B (lateral wall): outer surface X = 70.1,  Z from -103.7 to -23.7
  Junction corner: X = 70.1, Z = -103.7

Face block placement (confirmed by observation):
  Face A: left column (ID 0,2) near junction, right (ID 1,3) far
  Face B: left column (ID 4,6) far, right (ID 5,7) near junction
"""

import json
from pathlib import Path
import numpy as np


# ---------- Bracket geometry ----------

BRACKET = {
    "face_A_z_mm":     -103.7,
    "face_A_x_min":     -10.0,
    "face_A_x_max":     +70.0,
    "face_A_height":     80.0,
    "face_B_x_mm":      +70.1,
    "face_B_z_near":   -103.7,
    "face_B_z_far":     -23.7,
    "face_B_height":     80.0,
}


def _face_A(h_mm, v_mm):
    """h=0 at junction edge (X~70), h increases toward far (-X)."""
    return [70.0 - h_mm, 80.0 - v_mm, -103.7]

def _face_B(h_mm, v_mm):
    """h=0 at cranial/far edge (Z~-24), h increases toward junction (-Z)."""
    return [+70.1, 80.0 - v_mm, -23.7 - h_mm]


# ---------- V0: 4 coplanar markers on base plate ----------

V0_MARKERS = np.array([
    [   0.0,   0.5,  +75.0],
    [   0.0,   0.5,  -75.0],
    [ +40.0,   0.5,    0.0],
    [ -40.0,   0.5,    0.0],
])

# ---------- V1: L-bracket, 6 markers spread ----------
V1_MARKERS = np.array([
    _face_A(15, 15), _face_A(65, 15), _face_A(40, 65),
    _face_B(15, 15), _face_B(65, 15), _face_B(40, 65),
])

# ---------- V2: same bracket, clustered ----------
V2_MARKERS = np.array([
    _face_A(27, 27), _face_A(53, 27), _face_A(40, 53),
    _face_B(27, 27), _face_B(53, 27), _face_B(40, 53),
])

# ---------- V3: same bracket, 8 markers ----------
V3_MARKERS = np.array([
    _face_A(15, 15), _face_A(65, 15), _face_A(15, 65), _face_A(65, 65),
    _face_B(15, 15), _face_B(65, 15), _face_B(15, 65), _face_B(65, 65),
])

VARIANTS = {
    "V0_baseline":  V0_MARKERS,
    "V1_primary":   V1_MARKERS,
    "V2_clustered": V2_MARKERS,
    "V3_8marker":   V3_MARKERS,
}


# ---------- Pedicle targets ----------

_LEVELS_Z = {"L1": +65.0, "L2": +32.5, "L3": 0.0, "L4": -32.5, "L5": -65.0}
_PEDICLE_HEIGHT_MM = 50.0
_PEDICLE_HALF_LATERAL_MM = 25.0

PLACEHOLDER_PEDICLE_TARGETS = np.array(
    [[+_PEDICLE_HALF_LATERAL_MM, _PEDICLE_HEIGHT_MM, z] for z in _LEVELS_Z.values()]
    + [[-_PEDICLE_HALF_LATERAL_MM, _PEDICLE_HEIGHT_MM, z] for z in _LEVELS_Z.values()]
)

PLACEHOLDER_PEDICLE_LABELS = (
    [f"{level}_L" for level in _LEVELS_Z]
    + [f"{level}_R" for level in _LEVELS_Z]
)


# ---------- Pedicle loader ----------

LPS_TO_PHANTOM_TRANSLATION_MM = np.array([-8.0, +155.0, +579.0])


def load_pedicles_from_json(json_path: str | Path,
                            translation_mm: np.ndarray = LPS_TO_PHANTOM_TRANSLATION_MM):
    with open(json_path) as f:
        data = json.load(f)
    positions = []
    labels = []
    for cp in data["markups"][0]["controlPoints"]:
        raw = np.array(cp["position"])
        positions.append(raw + translation_mm)
        labels.append(cp["label"])
    return np.array(positions), labels


if __name__ == "__main__":
    print("Design variant summary (LPS phantom frame):")
    for name, markers in VARIANTS.items():
        centroid = markers.mean(axis=0)
        span = markers.max(axis=0) - markers.min(axis=0)
        print(f"  {name:14s} N={len(markers)}  "
              f"centroid=({centroid[0]:+6.1f},{centroid[1]:+6.1f},{centroid[2]:+6.1f})  "
              f"span=({span[0]:5.1f},{span[1]:5.1f},{span[2]:5.1f})")