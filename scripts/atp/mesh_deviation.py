#!/usr/bin/env python3
"""How far does the Blender-decimated mesh set deviate from the full mesh set?

python scripts/atp/mesh_deviation.py
python scripts/atp/mesh_deviation.py --full-prefix full_phantom_l --samples 400000

For each level (L1..L5) it samples points on both surfaces and reports, in mm, the mean,
95th percentile and maximum distance from one surface to the other, in both directions
(the larger of the two maxima is the Hausdorff distance).
Needs numpy, pandas and scipy only (pip install scipy if missing). No trimesh.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree


def read_stl(path):
    """Return triangles as an (n, 3, 3) float array. Handles binary and ASCII STL."""
    raw = open(path, "rb").read()
    if len(raw) >= 84:
        n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        if 84 + 50 * n == len(raw):  # binary
            rec = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            return rec["v"].astype(np.float64)
    verts = [list(map(float, ln.split()[1:4])) for ln in raw.decode("ascii", "ignore").splitlines()
             if ln.strip().startswith("vertex")]
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


def sample_surface(tris, n, rng):
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return tris[idx, 0] + u[:, None] * e1[idx] + v[:, None] * e2[idx], float(area.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", default="phantom/meshes")
    ap.add_argument("--full-prefix", default="full_phantom_l")
    ap.add_argument("--blend-dir", default="phantom/meshes/v3_overlay")
    ap.add_argument("--blend-prefix", default="phantom_l")
    ap.add_argument("--scale", type=float, default=None, help="to mm; default auto (metres -> 1000)")
    ap.add_argument("--samples", type=int, default=1000000, help="points per mesh per level")
    ap.add_argument("--out", default="data/atp_v2/mesh_deviation.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for k in range(1, 6):
        fp = f"{a.full_dir}/{a.full_prefix}{k}.stl"
        bp = f"{a.blend_dir}/{a.blend_prefix}{k}.stl"
        if not (os.path.exists(fp) and os.path.exists(bp)):
            print(f"L{k}: missing {fp} or {bp}, skipped"); continue
        tf, tb = read_stl(fp), read_stl(bp)
        sc = a.scale or (1000.0 if float(np.ptp(tf.reshape(-1, 3), axis=0).max()) < 1.0 else 1.0)
        tf, tb = tf * sc, tb * sc
        pf, area_f = sample_surface(tf, a.samples, rng)
        pb, _ = sample_surface(tb, a.samples, rng)
        d_b2f = cKDTree(pf).query(pb)[0]   # blender surface to full surface
        d_f2b = cKDTree(pb).query(pf)[0]   # full surface to blender surface
        both = np.r_[d_b2f, d_f2b]
        rows.append({"level": f"L{k}", "full_tris": len(tf), "blend_tris": len(tb),
                     "reduction_x": round(len(tf) / len(tb), 1),
                     "mean_mm": round(float(both.mean()), 3), "p95_mm": round(float(np.percentile(both, 95)), 3),
                     "max_blend_to_full_mm": round(float(d_b2f.max()), 3),
                     "max_full_to_blend_mm": round(float(d_f2b.max()), 3),
                     "hausdorff_mm": round(float(max(d_b2f.max(), d_f2b.max())), 3),
                     "sample_spacing_mm": round(float(np.sqrt(area_f / a.samples)), 3), "scale_to_mm": sc})
    if not rows:
        raise SystemExit("No level pairs found. Check --full-prefix / --blend-dir.")
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.to_string(index=False))
    print(f"\nOverall: mean {t.mean_mm.mean():.3f} mm, worst Hausdorff {t.hausdorff_mm.max():.3f} mm. "
          f"Values near or below the sample spacing are at the noise floor of this method; real deviation is at most about the value shown minus that (~{t.sample_spacing_mm.max():.2f} mm).")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out, index=False)
    print("Saved", a.out)


if __name__ == "__main__":
    main()