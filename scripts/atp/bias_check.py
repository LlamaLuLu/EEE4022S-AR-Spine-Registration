import csv, numpy as np
rows = list(csv.DictReader(open("data/atp_v2/atp2_dome.csv")))
for lab in sorted({r["label"] for r in rows}):
    s = [r for r in rows if r["label"] == lab]
    m = lambda k: np.mean([float(r[k]) for r in s]); sd = lambda k: np.std([float(r[k]) for r in s])
    print(f"{lab:8} n={len(s):2}  dx {m('dx_mm'):+.2f}±{sd('dx_mm'):.2f}  dy {m('dy_mm'):+.2f}±{sd('dy_mm'):.2f}  dz {m('dz_mm'):+.2f}±{sd('dz_mm'):.2f}")