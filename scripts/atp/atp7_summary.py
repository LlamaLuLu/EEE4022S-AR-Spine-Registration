"""ATP-7 summary (dome): lighting + glare.
python scripts/atp/atp7_summary.py --lighting data/atp_v2/atp7_lighting.csv --glare data/atp_v2/atp7_glare.csv
Hold steps default to the ones used in the 12 Oct run (see --light-steps / --glare-steps).
"""
import argparse, collections
import pandas as pd

def load(f):
    d = pd.read_csv(f)
    for c in ["n_markers", "rms_px", "distance_mm", "amb_rejected"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["pnp_ok"] = d.pnp_ok.astype(str) == "True"
    d["t"] = d.timestamp - d.timestamp.iloc[0]
    return d

def summ(d, steps, labels, trim, ids=range(10, 17)):
    rows = []
    for s, lab in zip(steps, labels):
        g = d[d.step == s]
        if g.empty:
            print(f"step {s} not found"); continue
        g = g[(g.t >= g.t.min() + trim) & (g.t <= g.t.max() - trim)]
        ok = g[g.pnp_ok]; n = len(g)
        c = collections.Counter(i for v in ok.visible_ids.dropna() for i in str(v).split("|"))
        r = {"test": lab, "step": s, "frames": n, "pnp_pct": round(100 * len(ok) / n, 1),
             "mean_markers": round(g.n_markers.mean(), 2), "min_markers": int(g.n_markers.min()),
             "rms_mean": round(ok.rms_px.mean(), 2), "rms_p95": round(ok.rms_px.quantile(.95), 2),
             "rms_max": round(ok.rms_px.max(), 2)}
        for i in ids:
            r[f"id{i}_pct"] = round(100 * c.get(str(i), 0) / n)
        rows.append(r)
    return pd.DataFrame(rows)

ap = argparse.ArgumentParser()
ap.add_argument("--lighting", default="data/atp_v2/atp7_lighting.csv"); ap.add_argument("--glare", default="data/atp_v2/atp7_glare.csv")
ap.add_argument("--light-steps", type=int, nargs="+", default=[2, 4, 6, 8])
ap.add_argument("--glare-steps", type=int, nargs="+", default=[2, 5, 7, 9, 11])
ap.add_argument("--trim", type=float, default=3.0)
ap.add_argument("--out", default="data/atp_v2/atp7_summary.csv")
a = ap.parse_args()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
out = []
if a.lighting:
    t = summ(load(a.lighting), a.light_steps, ["Normal", "Dim", "Bright direct", "Partial shadow"], a.trim)
    t.insert(0, "part", "lighting"); print(t.to_string(index=False)); out.append(t)
if a.glare:
    t = summ(load(a.glare), a.glare_steps, ["Baseline", "Torch ID10", "Torch ID13", "Torch ID11", "Torch ID16"], a.trim)
    t.insert(0, "part", "glare"); print(t.to_string(index=False)); out.append(t)
if out:
    pd.concat(out).to_csv(a.out, index=False); print("saved", a.out)