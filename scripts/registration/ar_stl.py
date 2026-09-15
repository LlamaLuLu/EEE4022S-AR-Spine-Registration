"""
Lightweight STL loader for wireframe overlay.

Loads binary or ASCII STL, deduplicates vertices, and returns an edge
list with per-edge face normals for backface culling during projection.
"""

from pathlib import Path
import numpy as np


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