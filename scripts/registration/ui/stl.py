"""
STL loading and wireframe / outline rendering.

Two APIs live here:

1. load_stl_for_wireframe(path)      the original single-mesh loader. UNCHANGED, so ATP builds still run.
2. load_vertebra_meshes(dir, ...)    per-vertebra meshes (phantom_l1.stl ... phantom_l5.stl), fast and decimated,
   draw_vertebra_layers(frame, ...)  and a layered renderer: selected level bold, neighbours ghosted, rest hidden.

Typical use (UI v2):

    meshes = load_vertebra_meshes("phantom/meshes/v3_overlay", cell_mm=1.0)       # once, at startup
    ...
    # VERIFYING: every level as a thin outline
    draw_vertebra_layers(frame, meshes, rvec, tvec, K, D, selected=None)
    # NAVIGATING: level 3 bold, levels 2 and 4 ghosted, others hidden
    draw_vertebra_layers(frame, meshes, rvec, tvec, K, D, selected=3)

Meshes are exported from Blender in metres in the phantom frame, so scale=1000 gives millimetres with no offset
(checked: every entry point lies 0.25 to 0.91 mm from its own vertebra mesh).
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

try:
    import cv2
except ImportError:                                   # the loader works without OpenCV; only drawing needs it
    cv2 = None

# BGR (OpenCV). Teal matches ar_theme / the HUD accent.
TEAL = (195, 195, 60)
WHITE = (235, 235, 235)


# ============================================================================ 1. original API (unchanged)
def load_stl_for_wireframe(stl_path):
    """
    Load an STL file and return geometry for wireframe rendering.

    Returns (verts: Nx3 float32, edges: list of (i0, i1, normal_vec))
    or (None, None) if the file doesn't exist.
    """
    path = Path(stl_path)
    if not path.exists():
        print(f"STL not found: {stl_path}")
        return None, None

    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"\n" in raw[:100]:
        return _load_ascii(path)
    return _load_binary(raw)


def _load_binary(raw):
    n_tri = int.from_bytes(raw[80:84], "little")
    verts, normals = [], []
    offset = 84
    for _ in range(n_tri):
        normal = np.frombuffer(raw, np.float32, 3, offset).copy()
        v0 = np.frombuffer(raw, np.float32, 3, offset + 12).copy()
        v1 = np.frombuffer(raw, np.float32, 3, offset + 24).copy()
        v2 = np.frombuffer(raw, np.float32, 3, offset + 36).copy()
        verts.extend([v0, v1, v2])
        normals.extend([normal] * 3)
        offset += 50
    return _dedup(np.array(verts), normals, n_tri)


def _load_ascii(path):
    verts, normals = [], []
    current_normal = np.zeros(3)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            elif line.startswith("vertex"):
                parts = line.split()
                verts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                normals.append(current_normal.copy())
    n_tri = len(verts) // 3
    return _dedup(np.array(verts), normals, n_tri)


def _dedup(verts_raw, normals_raw, n_tri):
    """Deduplicate vertices, build edge list with face normals."""
    rounded = np.round(verts_raw, decimals=3)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)

    edges = set()
    edge_normals = {}
    for tri in range(n_tri):
        i0, i1, i2 = inverse[tri * 3], inverse[tri * 3 + 1], inverse[tri * 3 + 2]
        normal = normals_raw[tri * 3]
        for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
            key = (min(a, b), max(a, b))
            edges.add(key)
            edge_normals[key] = normal

    edge_list = [(a, b, edge_normals[(a, b)]) for a, b in edges]
    return unique.astype(np.float32), edge_list


# ============================================================================ 2. per-vertebra meshes
@dataclass
class VertebraMesh:
    level: int                  # 1 to 5
    verts: np.ndarray           # Nx3 float32, millimetres, phantom frame
    tris: np.ndarray            # Mx3 int32
    tri_normals: np.ndarray     # Mx3 float32, unit, outward
    tri_centres: np.ndarray     # Mx3 float32
    edges: np.ndarray           # Ex2 int32
    edge_tri_a: np.ndarray      # E int32: adjacent triangle A
    edge_tri_b: np.ndarray      # E int32: adjacent triangle B (== A on a boundary edge)
    edge_crease: np.ndarray     # E bool: sharp edge (dihedral angle above the threshold)
    centre: np.ndarray          # 3 float32, bounding-box centre (mm)


def _read_stl_triangles(path):
    """Return (triangles Mx3x3 float64 in file units). Binary fast path; ASCII fallback."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = int.from_bytes(raw[80:84], "little")
        if len(raw) == 84 + 50 * n:
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(raw, dt, count=n, offset=84)["v"].astype(np.float64)
    text = raw.decode("utf-8", errors="ignore")
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    return np.array(nums, dtype=np.float64).reshape(-1, 3, 3)


def _cluster_decimate(verts, tris, cell):
    """Vertex-clustering decimation: merge vertices that fall in the same cube of side `cell` (mm)."""
    key = np.floor(verts / cell).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    # representative = mean of the cluster (keeps shape better than picking a corner)
    sums = np.zeros((len(first), 3)); cnt = np.zeros(len(first))
    np.add.at(sums, inv, verts); np.add.at(cnt, inv, 1)
    new_verts = sums / cnt[:, None]
    t = inv[tris]
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t = t[keep]
    # drop duplicate triangles (same vertex set)
    s = np.sort(t, axis=1)
    _, ui = np.unique(s, axis=0, return_index=True)
    return new_verts, t[np.sort(ui)]


def _build_mesh(level, tri_xyz, scale, offset, cell_mm, crease_deg):
    pts = tri_xyz.reshape(-1, 3) * scale + np.asarray(offset, float)
    key = np.round(pts, 4)
    verts, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    tris = inv.reshape(-1, 3)
    if cell_mm and cell_mm > 0:
        verts, tris = _cluster_decimate(verts, tris, cell_mm)

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    tris, nrm, ln = tris[ok], nrm[ok], ln[ok]
    nrm = nrm / ln[:, None]
    centres = (verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]) / 3.0

    # make normals point away from the mesh centre (robust to inconsistent winding after decimation)
    cen = verts.mean(axis=0)
    flip = np.einsum("ij,ij->i", nrm, centres - cen) < 0
    # winding is only reliable locally, so use the outward test only to detect a globally inverted mesh
    if flip.mean() > 0.5:
        nrm = -nrm

    # edges with their (up to two) adjacent triangles
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    tri_id = np.concatenate([np.arange(len(tris))] * 3)
    es = np.sort(e, axis=1)
    order = np.lexsort((es[:, 1], es[:, 0]))
    es, tri_id = es[order], tri_id[order]
    new = np.ones(len(es), bool); new[1:] = np.any(es[1:] != es[:-1], axis=1)
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(es))
    edges = es[starts]
    ta = tri_id[starts]
    tb = np.where(ends - starts > 1, tri_id[np.minimum(starts + 1, len(es) - 1)], ta)
    dots = np.einsum("ij,ij->i", nrm[ta], nrm[tb])
    crease = dots < np.cos(np.deg2rad(crease_deg))
    lo, hi = verts.min(0), verts.max(0)
    return VertebraMesh(level, verts.astype(np.float32), tris.astype(np.int32), nrm.astype(np.float32),
                        centres.astype(np.float32), edges.astype(np.int32), ta.astype(np.int32), tb.astype(np.int32),
                        crease, ((lo + hi) / 2).astype(np.float32))


def load_vertebra_meshes(folder, prefix="phantom_l", levels=(1, 2, 3, 4, 5), scale=1000.0, offset=(0.0, 0.0, 0.0),
                         cell_mm=1.0, crease_deg=35.0):
    """
    Load phantom_l1.stl ... phantom_l5.stl. Returns {level: VertebraMesh}.

    scale    1000 for Blender metres -> mm (no offset needed: the exports are already in the phantom frame)
    cell_mm  decimation cell in mm (1.0 is about 5 to 8 times fewer triangles; 0 disables)
    """
    folder = Path(folder)
    meshes = {}
    for lv in levels:
        p = folder / f"{prefix}{lv}.stl"
        if not p.exists():
            print(f"vertebra STL not found: {p}")
            continue
        meshes[lv] = _build_mesh(lv, _read_stl_triangles(p), scale, offset, cell_mm, crease_deg)
    return meshes


def level_from_label(label):
    """'L3_E_R' -> (3, 'R').  Returns (None, None) if the label does not parse."""
    m = re.match(r"L(\d)_(?:E_)?([LR])$", str(label))
    return (int(m.group(1)), m.group(2)) if m else (None, None)


# ============================================================================ 3. projection and drawing
def outline_segments(mesh, rvec, tvec, K, D, mode="outline"):
    """
    Project one mesh and return line segments as an int32 array (S, 2, 2) in pixels.

    mode "outline": silhouette edges plus visible crease edges (clean technical-drawing look)
    mode "silhouette": silhouette only
    mode "wire": every front-facing edge
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    cam = (-R.T @ np.asarray(tvec, float).reshape(3)).astype(np.float32)           # camera centre, phantom frame
    front = np.einsum("ij,ij->i", mesh.tri_normals, cam - mesh.tri_centres) > 0
    fa, fb = front[mesh.edge_tri_a], front[mesh.edge_tri_b]
    if mode == "wire":
        keep = fa | fb
    elif mode == "silhouette":
        keep = fa != fb
    else:
        keep = (fa != fb) | (mesh.edge_crease & (fa | fb))
    if not keep.any():
        return np.zeros((0, 2, 2), np.int32)
    e = mesh.edges[keep]
    used, inv = np.unique(e.reshape(-1), return_inverse=True)
    pts, _ = cv2.projectPoints(mesh.verts[used].astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, D)
    pts = pts.reshape(-1, 2)[inv].reshape(-1, 2, 2)
    return np.round(pts).astype(np.int32)


def _blend_lines(frame, segs, color, alpha, thickness):
    """Draw all segments in one polylines call and alpha-blend only the pixels the lines touch."""
    if len(segs) == 0:
        return
    h, w = frame.shape[:2]
    segs = np.clip(segs, -20000, 20000)
    x0, y0 = max(int(segs[..., 0].min()) - thickness - 1, 0), max(int(segs[..., 1].min()) - thickness - 1, 0)
    x1, y1 = min(int(segs[..., 0].max()) + thickness + 2, w), min(int(segs[..., 1].max()) + thickness + 2, h)
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.polylines(layer, (segs - np.array([x0, y0], np.int32)).astype(np.int32), False, 255, thickness, cv2.LINE_AA)
    idx = layer > 0                                   # lines cover few pixels: blend just those
    a = layer[idx].astype(np.float32)[:, None] * (alpha / 255.0)
    roi = frame[y0:y1, x0:x1]
    px = roi[idx].astype(np.float32)
    roi[idx] = (px * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


STYLE = {
    "selected": dict(color=TEAL,  alpha=0.95, thickness=1, mode="outline", fill_alpha=0.12),
    "ghost":    dict(color=WHITE, alpha=0.16, thickness=1, mode="silhouette"),
    # All-levels view: cleaner silhouette lines plus a very faint tint (fill_alpha) so each
    # vertebra reads as a region. Set fill_alpha to 0 to switch the tint off.
    "verify":   dict(color=TEAL,  alpha=0.85, thickness=1, mode="silhouette", fill_alpha=0.14),
}

FILL_DOWNSCALE = 4              # tint grid is 1/4 resolution: faster; 2 is tighter, 6 is softer
_FILL_K3 = np.ones((3, 3), np.uint8)

def _fill_mesh(frame, mesh, rvec, tvec, K, D, color, alpha, exact=False):
    """Faint tint over the visible surface of one vertebra. Splats the visible vertices on a
    coarse grid, closes the gaps and scales up: ~10x faster than filling every triangle."""
    if alpha <= 0:
        return
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    cam = (-R.T @ np.asarray(tvec, float).reshape(3)).astype(np.float32)
    front = np.einsum("ij,ij->i", mesh.tri_normals, cam - mesh.tri_centres) > 0
    if not front.any():
        return
    if exact:
        _fill_mesh_exact(frame, mesh, front, rvec, tvec, K, D, color, alpha)
        return
    used = np.zeros(len(mesh.verts), bool)
    used[mesh.tris[front].ravel()] = True
    pts, _ = cv2.projectPoints(mesh.verts[used].astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, D)
    pts = np.round(np.clip(pts.reshape(-1, 2), -20000, 20000)).astype(np.int32)
    h, w = frame.shape[:2]
    x0, y0 = max(int(pts[:, 0].min()), 0), max(int(pts[:, 1].min()), 0)
    x1, y1 = min(int(pts[:, 0].max()) + 1, w), min(int(pts[:, 1].max()) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    s = FILL_DOWNSCALE
    ok = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    q = (pts[ok] - np.array([x0, y0], np.int32)) // s
    small = np.zeros(((y1 - y0) // s + 2, (x1 - x0) // s + 2), np.uint8)
    small[q[:, 1], q[:, 0]] = 255
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, _FILL_K3)
    mask = cv2.resize(small, (small.shape[1] * s, small.shape[0] * s),
                      interpolation=cv2.INTER_LINEAR)[:y1 - y0, :x1 - x0]
    a = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def _fill_mesh_exact(frame, mesh, front, rvec, tvec, K, D, color, alpha):
    """Tint every visible triangle. Slower than the coarse splat (one fillPoly over all triangles)
    but exact at any magnification, so it is used in the zoom view, where the splat leaves holes."""
    pts, _ = cv2.projectPoints(mesh.verts.astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, D)
    pts = np.round(np.clip(pts.reshape(-1, 2), -20000, 20000)).astype(np.int32)
    tri = pts[mesh.tris[front]]
    h, w = frame.shape[:2]
    x0, y0 = max(int(tri[..., 0].min()), 0), max(int(tri[..., 1].min()), 0)
    x1, y1 = min(int(tri[..., 0].max()) + 1, w), min(int(tri[..., 1].max()) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, tri - np.array([x0, y0], np.int32), 255)
    roi = frame[y0:y1, x0:x1]
    tinted = cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0)
    cv2.copyTo(tinted, mask, roi)                      # only where the vertebra is


def draw_vertebra_layers(frame, meshes, rvec, tvec, K, D, selected=None, ghost_neighbours=True, style=None,
                         exact_fill=False):
    """
    Draw the vertebrae onto `frame` (BGR uint8, modified in place). Returns the list of levels drawn.

    selected=None   every level as a thin outline (VERIFYING)
    selected=3      level 3 bold, its neighbours (2 and 4) ghosted, others hidden (NAVIGATING focus mode)
    """
    st = {**STYLE, **(style or {})}
    drawn = []
    if selected is None:
        for lv, m in meshes.items():
            s = st["verify"]
            _fill_mesh(frame, m, rvec, tvec, K, D, s["color"], s.get("fill_alpha", 0.0), exact_fill)
            _blend_lines(frame, outline_segments(m, rvec, tvec, K, D, s["mode"]), s["color"], s["alpha"], s["thickness"])
            drawn.append(lv)
        return drawn
    if ghost_neighbours:
        for lv in (selected - 1, selected + 1):
            if lv in meshes:
                s = st["ghost"]
                _blend_lines(frame, outline_segments(meshes[lv], rvec, tvec, K, D, s["mode"]), s["color"], s["alpha"], s["thickness"])
                drawn.append(lv)
    if selected in meshes:
        s = st["selected"]
        _fill_mesh(frame, meshes[selected], rvec, tvec, K, D, s["color"], s.get("fill_alpha", 0.0), exact_fill)
        _blend_lines(frame, outline_segments(meshes[selected], rvec, tvec, K, D, s["mode"]), s["color"], s["alpha"], s["thickness"])
        drawn.append(selected)
    return drawn