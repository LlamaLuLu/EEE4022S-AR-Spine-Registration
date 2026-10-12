import csv, sys, numpy as np
sys.path[:0] = ["scripts/registration", "scripts/tre_analysis"]
from ar_targets import load_targets
pts, labels, _ = load_targets("annotations/pedicle_isthmus_targets.json",
                              "annotations/pedicle_entry_targets.json")
rows = list(csv.DictReader(open("data/atp_v2/atp2_dome_3d.csv")))
P = np.array([pts[labels.index(r["label"])] for r in rows])
E = np.array([[float(r["dx_mm"]), float(r["dy_mm"]), float(r["dz_mm"])] for r in rows])
Q = P + E
rms = lambda v: np.sqrt((v ** 2).sum(1).mean())
print(f"n={len(P)}  3D error RMS before: {rms(E):.2f} mm")
mp, mq = P.mean(0), Q.mean(0)
A, B = P - mp, Q - mq
U, S, Vt = np.linalg.svd(A.T @ B)
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1, 1, d]) @ U.T
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
res = B - A @ R.T
print(f"rigid: rotation {ang:.2f} deg, translation {np.round(mq - mp @ R.T, 2)} mm, residual RMS {rms(res):.2f} mm")
s = S.sum() / (A ** 2).sum()
print(f"scaled: scale {s:.4f} ({100 * (s - 1):+.1f}%), residual RMS {rms(B - s * A @ R.T):.2f} mm")
print("translation only: residual RMS", f"{rms(E - E.mean(0)):.2f}", "mm, mean offset", np.round(E.mean(0), 2))

c = np.array([120.0, 105.0, 0.0])                      # dome centre
t2 = mq - c - (mp - c) @ R.T                           # residual translation if pivoting about the dome centre
w, v = np.linalg.eig(R); ax = np.real(v[:, np.argmin(abs(w - 1))]); ax /= np.linalg.norm(ax)
print(f"axis {np.round(ax, 2)} (X=left, Y=up, Z=cranial)")
print(f"translation about dome centre: {np.round(t2, 2)} mm")