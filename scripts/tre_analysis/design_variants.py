"""
Marker layouts (V0-V3) and pedicle targets in the LPS phantom frame.

LPS convention (matches Slicer default and DICOM):
  origin  : centre of base plate top surface
  +X      : patient left  (lateral)
  +Y      : posterior     (vertical, up when phantom is prone on plate)
  +Z      : superior      (spine axis, cranial is +Z, caudal is -Z)
  units   : millimetres

Base plate is 100mm x 170mm x 5mm, top surface at Y = 0.
L-bracket sits at the caudal end (Z_min) of the base plate, on the
patient's LEFT side (+X). If your bracket is on patient's right instead,
flip the sign of face_B_x_mm and all Face B marker X values.

Since Slicer exports pedicle targets in LPS too, only a translation is
needed to load them into the phantom frame (no rotation).
"""

import json
from pathlib import Path
import numpy as np


# ---------- Bracket geometry ----------

BRACKET = {
    "face_A_z_mm":     -95.0,   # caudal wall at Z = -95
    "face_A_x_half":    40.0,   # spans X in [-40, +40] (80mm wide)
    "face_A_height":    80.0,   # spans Y in [0, 80]
    "face_B_x_mm":     +30.0,   # lateral wall at X = +30 (patient's left)
    "face_B_z_far":   -175.0,   # spans Z in [-95, -175] (80mm deep, extends caudally)
    "face_B_height":    80.0,
}


# Convert face-local (h, v) offset-from-top-left coordinates to LPS phantom frame.
#   Face A top-left corner: (X=-40, Y=+80, Z=-95). h increases along +X, v increases along -Y.
#   Face B top-left corner: (X=+30, Y=+80, Z=-95). h increases along -Z (caudal), v along -Y.

def _face_A(h_mm, v_mm):
    return [-40.0 + h_mm, 80.0 - v_mm, -95.0]

def _face_B(h_mm, v_mm):
    return [+30.0, 80.0 - v_mm, -95.0 - h_mm]


# ---------- V0: 4 coplanar markers on base plate ----------
# Base plate top surface at Y = 0. Markers slightly above at Y = 0.5.

V0_MARKERS = np.array([
    [   0.0,   0.5,  +75.0],   # cranial (near L1)
    [   0.0,   0.5,  -75.0],   # caudal  (near L5)
    [ +40.0,   0.5,    0.0],   # patient left
    [ -40.0,   0.5,    0.0],   # patient right
])


# ---------- V1: L-bracket, 6 markers spread ----------
# Face-local pattern: (15,15), (65,15), (40,65) on each 80x80 face.

V1_MARKERS = np.array([
    _face_A(15, 15), _face_A(65, 15), _face_A(40, 65),
    _face_B(15, 15), _face_B(65, 15), _face_B(40, 65),
])


# ---------- V2: same bracket, markers clustered ----------
# Face-local pattern: (27,27), (53,27), (40,53).

V2_MARKERS = np.array([
    _face_A(27, 27), _face_A(53, 27), _face_A(40, 53),
    _face_B(27, 27), _face_B(53, 27), _face_B(40, 53),
])


# ---------- V3: same bracket, 8 markers at face corners ----------
# Face-local pattern: (15,15), (65,15), (15,65), (65,65).

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


# ---------- Pedicle target placeholders ----------

# 5 vertebrae along Z (spine axis), bilateral pairs at +/-25mm in X,
# all at Y = +50 (elevated above base plate). Replace via load_pedicles_from_json.
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

# Translation from original CT-frame LPS to phantom-mounted LPS frame.
# Determine empirically: measure where an original LPS landmark (e.g. L3
# pedicle centre) lands vs where the corresponding vertebra sits in Blender
# after positioning the phantom on the base plate.
#
# Example: if original L3 pedicle centre is at (+8, -107, -580) in CT LPS,
# and your L3 vertebra sits at (-8, +50, 0) in phantom LPS, then
# translation = (-16, +157, +580).
LPS_TO_PHANTOM_TRANSLATION_MM = np.array([-8.0, +155.0, +579.0])


def load_pedicles_from_json(json_path: str | Path,
                            translation_mm: np.ndarray = LPS_TO_PHANTOM_TRANSLATION_MM):
    """
    Load Slicer mrk.json pedicle targets and translate to phantom frame.

    Both source and phantom frames are LPS, so only translation is required
    (no rotation). Translation accounts for moving the CT-frame phantom onto
    the base plate at origin.
    """
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

    print(f"\nPedicle targets (placeholder): {len(PLACEHOLDER_PEDICLE_TARGETS)}")
    for axis_name, i in [("X", 0), ("Y", 1), ("Z", 2)]:
        vals = PLACEHOLDER_PEDICLE_TARGETS[:, i]
        print(f"  {axis_name}: {vals.min():+.1f} to {vals.max():+.1f} mm")