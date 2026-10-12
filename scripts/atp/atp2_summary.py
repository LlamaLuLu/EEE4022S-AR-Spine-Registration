import csv
import numpy as np

rows = list(csv.DictReader(open("data/atp_v2/atp2_dome.csv")))
f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else np.nan
e = np.array([f(r, "err_mm") for r in rows])
d = np.array([f(r, "dist_from_markers_mm") for r in rows])
t = np.array([f(r, "pred_tre_mm") for r in rows])
lv = np.array([r["level"] for r in rows])
rms = lambda v: np.sqrt(np.mean(v ** 2))

print(f"ALL  n={len(e)}  mean {e.mean():.2f}  median {np.median(e):.2f}  RMS {rms(e):.2f}  "
      f"max {e.max():.2f}  >2mm {(e > 2).sum()}/{len(e)} ({100 * (e > 2).mean():.0f}%)\n")
print(f"{'level':<6}{'n':>3}{'mean':>7}{'median':>8}{'RMS':>6}{'max':>6}{'>2mm':>7}{'dist mm':>9}{'pred TRE':>10}")
for l in sorted(set(lv)):
    m = lv == l
    print(f"{l:<6}{m.sum():>3}{e[m].mean():>7.2f}{np.median(e[m]):>8.2f}{rms(e[m]):>6.2f}{e[m].max():>6.2f}"
          f"{(e[m] > 2).sum():>4}/{m.sum():<2}{d[m].mean():>8.0f}{np.nanmean(t[m]):>10.2f}")

print(f"\ncorr(error, distance from dome) = {np.corrcoef(e, d)[0, 1]:+.2f}")
ok = np.isfinite(t)
print(f"corr(error, predicted TRE)      = {np.corrcoef(e[ok], t[ok])[0, 1]:+.2f}")
for s in ("L", "R"):
    m = np.array([r["side"] == s for r in rows])
    print(f"side {s}: n={m.sum()}  mean {e[m].mean():.2f}  RMS {rms(e[m]):.2f}")
for lo, hi, name in ((0, 40, "el < 40"), (40, 55, "el 40-55"), (55, 90, "el > 55")):
    m = np.array([lo <= f(r, "elev_deg") < hi for r in rows])
    if m.any():
        print(f"{name}: n={m.sum()}  mean {e[m].mean():.2f}  >2mm {(e[m] > 2).sum()}")