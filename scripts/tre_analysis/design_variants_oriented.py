"""
Variant definitions with per-marker orientation, used by Monte Carlo.

Each marker is a tuple: (centroid_mm, R_marker_to_phantom).
R_marker_to_phantom is a 3x3 rotation matrix whose columns are the
marker's local X, Y, Z axes expressed in phantom (LPS) coordinates:
  local X = marker "right"  (top-left corner -> top-right corner)
  local Y = marker "up"     (bottom edge -> top edge)
  local Z = outward normal  (towards a camera that can read it)
Every R must be a proper rotation (det = +1): X x Y = Z. A det = -1 frame is a
mirror image; the simulator does not notice, but a live marker map built from
it lists the corners in the wrong order.

Physical bracket (L-bracket v1.3 STL, Blender coords x1000 = mm):
  Face A (caudal wall): outer surface Z = -103.7, X from -10 to 70
  Face B (lateral wall): outer surface X = 70.1,  Z from -103.7 to -23.7
  Junction corner: X = 70.1, Z = -103.7

Physical face block placement (confirmed by observation):
  Face A: left column (ID 0,2) near junction (+X), right column (ID 1,3) far
  Face B: right column (ID 5,7) near junction (-Z), left column (ID 4,6) far

Domes (V4/V5 family): see dome_oriented() and DOME_CONFIGS below.
"""

import numpy as np


# ---------- Orientation matrices (LPS convention) ----------

# Marker lying flat, read from above with cranial (+Z) at the top of the image.
# Seen from above, the image's right-hand side is patient RIGHT (-X), so local X = -X.
# (The previous version used +X here, which made this a mirrored frame, det = -1.
#  V0 results are unaffected because the simulator is self-consistent.)
R_FLAT_UP = np.array([
    [-1, 0, 0],    # local X -> -X
    [ 0, 0, 1],    # local Y -> +Z (cranial = up in the image)
    [ 0, 1, 0],    # local Z -> +Y (normal points up, posterior)
]).T

# Face A: caudal wall, outer surface at Z = -103.7, normal points -Z.
R_FACE_A = np.array([
    [-1, 0,  0],    # local X -> -X (left to right = junction to far)
    [ 0, 1,  0],    # local Y -> +Y (up)
    [ 0, 0, -1],    # local Z -> -Z (outward normal)
]).T

# Face B: lateral wall, outer surface at X = 70.1, normal points +X.
R_FACE_B = np.array([
    [ 0, 0, -1],    # local X -> -Z (left to right = far to junction)
    [ 0, 1,  0],    # local Y -> +Y (up)
    [ 1, 0,  0],    # local Z -> +X (outward normal)
]).T

for _R in (R_FLAT_UP, R_FACE_A, R_FACE_B):
    assert np.isclose(np.linalg.det(_R), 1.0)


# ---------- Face-local helpers ----------

def _fA(h, v):
    """Face A: h=0 at junction edge (X~70), h increases toward far (-X)."""
    return np.array([70.0 - h, 80.0 - v, -103.7])


def _fB(h, v):
    """Face B: h=0 at cranial/far edge (Z~-24), h increases toward junction (-Z)."""
    return np.array([70.1, 80.0 - v, -23.7 - h])


# ---------- Dome generator ----------

def dome_oriented(centre, ring_r=45.0, n_ring=6, tilt_deg=15.0,
                  ring_h=0.0, top_h=25.0, arc_deg=360.0, arc_mid_deg=0.0,
                  top=True):
    """One dome module: a flat top marker raised top_h above the module centre,
    plus n_ring markers on a ring of radius ring_r, each tilted outward by tilt_deg.

    centre      : (X, Y, Z) of the ring centre in phantom mm. Y = height above plate.
    arc_deg     : 360 for a full ring. Less than 360 gives a partial ring; its
                  markers are spread evenly from arc_mid - arc/2 to arc_mid + arc/2.
    arc_mid_deg : direction the partial ring faces, as an angle in the XZ plane
                  measured from +X towards +Z (0 = patient left, 90 = cranial,
                  -90 = caudal, 180 = patient right).

    Ring marker frame: local Z = tilted normal, local X = horizontal tangent,
    local Y = Z x X, which points down the slope (away from the dome centre).
    So each printed ring marker's top edge faces OUTWARD.
    """
    c = np.asarray(centre, float)
    up = np.array([0.0, 1.0, 0.0])
    out = [(c + top_h * up, R_FLAT_UP)] if top else []
    t = np.radians(tilt_deg)
    for i in range(n_ring):
        if arc_deg >= 360:
            f = i / n_ring
            phi = np.radians(arc_mid_deg) + f * 2 * np.pi
        else:
            f = 0.0 if n_ring == 1 else i / (n_ring - 1) - 0.5
            phi = np.radians(arc_mid_deg) + f * np.radians(arc_deg)
        u = np.array([np.cos(phi), 0.0, np.sin(phi)])
        n = np.cos(t) * up + np.sin(t) * u
        x = np.array([-np.sin(phi), 0.0, np.cos(phi)])
        R = np.column_stack([x, np.cross(n, x), n])
        assert np.isclose(np.linalg.det(R), 1.0)
        out.append((c + ring_r * u + ring_h * up, R))
    return out


# Each config is a list of modules (keyword arguments for dome_oriented).
# Y is the module centre height above the plate top. The printed spine reaches
# about Y = 100 (entry targets sit at Y 90 to 100), so modules at Y = 90 can be
# seen across the spine; at Y = 40 the spine hides them from the far side.
# Half domes face away from the spine, so their flat side can sit right at the
# plate end; full rings need ring_r of clearance on the inward side.
DOME_CONFIGS = {
    "D1_left_L3":        [dict(centre=(100, 90, 0),   ring_r=40, n_ring=6)],
    "D2_caudal":  [dict(centre=(10, 90, -140), ring_r=40, n_ring=6)],
    "D3_split_full":     [dict(centre=(10, 90, -125), ring_r=30, n_ring=4),
                          dict(centre=(10, 90, +115), ring_r=30, n_ring=4)],
    "D4_split_half":     [dict(centre=(10, 90, -95), ring_r=40, n_ring=4, arc_deg=180, arc_mid_deg=-90),
                          dict(centre=(10, 90, +90), ring_r=40, n_ring=4, arc_deg=180, arc_mid_deg=+90)],
    "D5_split_half_min": [dict(centre=(10, 90, -95), ring_r=40, n_ring=3, arc_deg=180, arc_mid_deg=-90),
                          dict(centre=(10, 90, +90), ring_r=40, n_ring=3, arc_deg=180, arc_mid_deg=+90)],
    "D6_split_half_low": [dict(centre=(10, 40, -95), ring_r=40, n_ring=4, arc_deg=180, arc_mid_deg=-90),
                          dict(centre=(10, 40, +90), ring_r=40, n_ring=4, arc_deg=180, arc_mid_deg=+90)],
    "D1_final":   [dict(centre=(110, 90, 0), ring_r=50, n_ring=6, tilt_deg=5)]
}
DOME_TILT_DEG = 5.0


def build_dome(name, tilt_deg=None, **overrides):
    """All markers of a DOME_CONFIGS entry, optionally with a different tilt
    (or any other dome_oriented keyword) applied to every module."""
    out = []
    for mod in DOME_CONFIGS[name]:
        kw = dict(mod)
        kw.setdefault("tilt_deg", DOME_TILT_DEG)
        if tilt_deg is not None:
            kw["tilt_deg"] = tilt_deg
        kw.update(overrides)
        out += dome_oriented(**kw)
    return out


# ---------- V0: 4 coplanar markers on base plate ----------

V0_ORIENTED = [
    (np.array([   0.0,   0.5,  +75.0]), R_FLAT_UP),
    (np.array([   0.0,   0.5,  -75.0]), R_FLAT_UP),
    (np.array([ +40.0,   0.5,    0.0]), R_FLAT_UP),
    (np.array([ -40.0,   0.5,    0.0]), R_FLAT_UP),
]

# ---------- V1: L-bracket, 6 markers spread ----------
V1_ORIENTED = [
    (_fA(15, 15), R_FACE_A), (_fA(65, 15), R_FACE_A), (_fA(40, 65), R_FACE_A),
    (_fB(15, 15), R_FACE_B), (_fB(65, 15), R_FACE_B), (_fB(40, 65), R_FACE_B),
]

# ---------- V2: same bracket, clustered ----------
V2_ORIENTED = [
    (_fA(27, 27), R_FACE_A), (_fA(53, 27), R_FACE_A), (_fA(40, 53), R_FACE_A),
    (_fB(27, 27), R_FACE_B), (_fB(53, 27), R_FACE_B), (_fB(40, 53), R_FACE_B),
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
VARIANTS_ORIENTED.update({name: build_dome(name) for name in DOME_CONFIGS})


if __name__ == "__main__":
    for name, ori in VARIANTS_ORIENTED.items():
        dets = [np.linalg.det(R) for _, R in ori]
        c = np.array([m[0] for m in ori])
        print(f"{name:<15s} N={len(ori):2d}  all det=+1: {np.allclose(dets, 1.0)}  "
              f"X {c[:, 0].min():+6.1f}..{c[:, 0].max():+6.1f}  Y {c[:, 1].min():+5.1f}..{c[:, 1].max():+5.1f}  "
              f"Z {c[:, 2].min():+6.1f}..{c[:, 2].max():+6.1f}")