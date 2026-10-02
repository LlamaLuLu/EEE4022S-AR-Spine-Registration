"""
Report figures for the ATP results.

Generates vector PDFs (plus 300 dpi PNGs) from the CSVs the ATP scripts already
write. A figure whose input file is missing is skipped with a message, so you
can run this at any point.

Figures
  envelope     working-range diagram: azimuth x distance (plan) and
               elevation x distance (side). Read from ar_theme.py, so it can
               never disagree with the HUD.                       (no data file)
  tracking     ATP-3: PnP success and lost % vs distance, azimuth, elevation
  overlay      ATP-2: overlay error vs distance from markers, by focus state
  occlusion    ATP-4: predicted TRE and PnP success vs visible markers
  tiers        ATP-3: tracking and predicted TRE per envelope tier
  calibration  camera calibration v0 vs v1 and ruler validation
  timing       ATP-6: per-stage processing time and glass-to-glass latency

Usage (from repo root):
    python scripts/atp/make_report_figures.py
    python scripts/atp/make_report_figures.py --only envelope tracking
    python scripts/atp/make_report_figures.py --out docs/figures

Writes figures_latex_snippets.tex alongside the figures (captions to edit).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Rectangle
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "registration"))

# ---------------------------------------------------------------------------
# Envelope limits: from ar_theme.py (single source of truth), with a fallback
# ---------------------------------------------------------------------------
try:
    import ar_theme as T
    FULL_DIST, FULL_ELEV, FULL_AZ = T.ENV_FULL_DIST_MM, T.ENV_FULL_ELEV, T.ENV_FULL_AZ
    RED_DIST, RED_ELEV, RED_AZ = T.ENV_REDUCED_DIST_MM, T.ENV_REDUCED_ELEV, T.ENV_REDUCED_AZ
    ENV_SOURCE = "ar_theme.py"
except Exception:
    FULL_DIST, FULL_ELEV, FULL_AZ = (550.0, 1000.0), (10.0, 35.0), (30.0, 55.0)
    RED_DIST, RED_ELEV, RED_AZ = (330.0, 1240.0), (0.0, 52.0), (325.0, 120.0)
    ENV_SOURCE = "built-in fallback (ar_theme.py not found)"

# Colour-blind-safe palette (Okabe-Ito)
C_FULL, C_RED, C_OUT = "#009E73", "#E69F00", "#D55E00"
C_BLUE, C_GREY, C_BLACK, C_SKY = "#0072B2", "#7f7f7f", "#000000", "#56B4E9"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
})
W = 6.3  # text width in inches (A4, normal margins)

LATEX = []


def save(fig, out, name, caption, label):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    LATEX.append(
        "\\begin{figure}[htbp]\n  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{figures/{name}.pdf}}\n"
        f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n\\end{{figure}}\n")
    print(f"  wrote {name}.pdf / .png")


def read(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return rows or None


def col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            out.append(np.nan)
    return np.array(out)


def band(ax, lo, hi, color, alpha, label=None, wrap=False):
    if wrap and lo > hi:                       # azimuth range that wraps through 0/360
        ax.axvspan(lo, 360, color=color, alpha=alpha, lw=0, label=label)
        ax.axvspan(0, hi, color=color, alpha=alpha, lw=0)
    else:
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, label=label)


# ---------------------------------------------------------------------------
# 1. Working-range diagram
# ---------------------------------------------------------------------------
def fig_envelope(out, a):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.5))
    cm = lambda mm: mm / 10.0

    # --- (a) plan view: azimuth x distance. 0 deg = caudal (bottom), 90 = patient left (left) ---
    ax = axs[0]
    r_lo, r_hi = cm(RED_DIST[0]), cm(RED_DIST[1])

    def az_wedge(az_a, az_b, r0, r1, **kw):
        """Annular sector from azimuth az_a to az_b (clockwise from caudal), radii in cm."""
        th1, th2 = 270 - az_b, 270 - az_a          # matplotlib: degrees counter-clockwise from +x
        ax.add_patch(Wedge((0, 0), r1, th1, th2, width=r1 - r0, **kw))

    # reduced tier spans RED_AZ[0] -> 360 and 0 -> RED_AZ[1]
    az_wedge(RED_AZ[0] - 360, RED_AZ[1], r_lo, r_hi, fc=C_RED, ec="none", alpha=0.35)
    # no tracking = the rest of the circle
    az_wedge(RED_AZ[1], RED_AZ[0], r_lo, r_hi, fc=C_OUT, ec=C_OUT, alpha=0.10, hatch="////", lw=0)
    az_wedge(FULL_AZ[0], FULL_AZ[1], cm(FULL_DIST[0]), cm(FULL_DIST[1]), fc=C_FULL, ec="none", alpha=0.7)

    # phantom footprint (to scale) and L-bracket. page_x = -X (patient left to the left), page_y = Z
    ax.add_patch(Rectangle((-5.02, -8.37), 11.03, 16.97, fc="#d9d9d9", ec="k", lw=0.5, zorder=3))
    ax.plot([-7.01, 0.99], [-10.37, -10.37], c="k", lw=1.8, zorder=4)     # Face A (caudal wall)
    ax.plot([-7.01, -7.01], [-10.37, -2.37], c="k", lw=1.8, zorder=4)     # Face B (lateral wall)

    R = r_hi * 1.13
    for az, lab in ((0, "0°\ncaudal"), (90, "90°\npatient left"), (180, "180°\ncranial"), (270, "270°\npatient right")):
        t = np.radians(270 - az)
        ax.text(R * np.cos(t), R * np.sin(t), lab, ha="center", va="center", fontsize=7)
    for az in sorted({FULL_AZ[0], FULL_AZ[1], RED_AZ[0], RED_AZ[1]}):
        t = np.radians(270 - az)
        ax.plot([0, r_hi * 1.04 * np.cos(t)], [0, r_hi * 1.04 * np.sin(t)], c=C_GREY, lw=0.4, ls=":")
        ax.text(r_hi * 1.09 * np.cos(t), r_hi * 1.09 * np.sin(t), f"{az:.0f}°", ha="center", va="center",
                fontsize=6.5, color=C_GREY)
    for r in (50, 100):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=C_GREY, lw=0.3, ls="--"))
        ax.text(0.7, -r - 4, f"{r} cm", fontsize=6, color=C_GREY, ha="left")
    ax.set_xlim(-R * 1.25, R * 1.25); ax.set_ylim(-R * 1.25, R * 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) Plan view: azimuth and distance", pad=2)

    # --- (b) side view: elevation x distance ---
    ax = axs[1]

    def el_wedge(e0, e1, r0, r1, **kw):
        ax.add_patch(Wedge((0, 0), r1, e0, e1, width=r1 - r0, **kw))

    el_wedge(RED_ELEV[1], 90, r_lo, r_hi, fc=C_OUT, ec=C_OUT, alpha=0.10, hatch="////", lw=0)
    el_wedge(RED_ELEV[0], RED_ELEV[1], r_lo, r_hi, fc=C_RED, ec="none", alpha=0.35)
    el_wedge(FULL_ELEV[0], FULL_ELEV[1], cm(FULL_DIST[0]), cm(FULL_DIST[1]), fc=C_FULL, ec="none", alpha=0.7)
    ax.add_patch(Rectangle((-8.5, -0.5), 17.0, 12.7, fc="#d9d9d9", ec="k", lw=0.5, zorder=3))   # phantom
    ax.plot([-8.5, 8.5], [-0.5, -0.5], c="k", lw=1.0, zorder=4)
    for e in sorted({FULL_ELEV[0], FULL_ELEV[1], RED_ELEV[1], 90}):
        t = np.radians(e)
        ax.plot([0, r_hi * 1.04 * np.cos(t)], [0, r_hi * 1.04 * np.sin(t)], c=C_GREY, lw=0.4, ls=":")
        ax.text(r_hi * 1.09 * np.cos(t), r_hi * 1.09 * np.sin(t), f"{e:.0f}°", ha="center", va="center",
                fontsize=6.5, color=C_GREY)
    ax.text(r_hi * 1.09, 0, "0°", ha="center", va="center", fontsize=6.5, color=C_GREY)
    for r in (50, 100):
        ax.add_patch(Wedge((0, 0), r, 0, 90, fill=False, ec=C_GREY, lw=0.3, ls="--"))
        ax.text(r - 1, -5, f"{r} cm", fontsize=6, color=C_GREY, ha="right")
    ax.set_xlim(-12, R * 1.12); ax.set_ylim(-14, R * 1.12)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) Side view: elevation and distance", pad=2)

    handles = [
        Rectangle((0, 0), 1, 1, fc=C_FULL, alpha=0.7),
        Rectangle((0, 0), 1, 1, fc=C_RED, alpha=0.35),
        Rectangle((0, 0), 1, 1, fc=C_OUT, alpha=0.10, hatch="////", ec=C_OUT),
    ]
    labels = [
        f"Full accuracy: both faces, {FULL_DIST[0]/10:.0f}-{FULL_DIST[1]/10:.0f} cm, "
        f"azimuth {FULL_AZ[0]:.0f}-{FULL_AZ[1]:.0f}°, elevation {FULL_ELEV[0]:.0f}-{FULL_ELEV[1]:.0f}°",
        f"Reduced: tracking holds, one face or steeper view (to {RED_DIST[1]/10:.0f} cm, "
        f"elevation up to {RED_ELEV[1]:.0f}°)",
        "Outside: tracking unreliable or lost",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(wspace=0.02, bottom=0.12)
    save(fig, out, "fig_envelope_diagram",
         "Measured working range of the L-bracket system, relative to the phantom origin (centre of the base plate "
         "top face). The L-bracket (heavy lines) sits at the caudal-left corner. Each limit was measured one "
         "parameter at a time, with the other two near the centre of the full-accuracy range, so combined extremes "
         "are not characterised.",
         "fig:results:envelope")


# ---------------------------------------------------------------------------
# 2. ATP-3 tracking vs distance / azimuth / elevation
# ---------------------------------------------------------------------------
def fill_az(az):
    """Interpolate azimuth for steps with no accepted pose (dead zone), assuming uniform rotation."""
    az = az.copy()
    idx = np.arange(len(az))
    good = np.isfinite(az)
    if good.sum() >= 2:
        az[~good] = np.interp(idx[~good], idx[good], az[good], left=np.nan, right=np.nan)
    return az, ~good


def fig_tracking(out, a):
    r31, r32, r33 = read(a.atp31), read(a.atp32), read(a.atp33)
    if not (r31 or r32 or r33):
        print("  tracking: skipped (no ATP-3 summary CSVs found)")
        return
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True)

    def lines(ax, x, rows, interpolated=None):
        pnp, lost = col(rows, "pnp_success_pct"), col(rows, "lost_pct")
        o = np.argsort(x)
        ax.plot(x[o], pnp[o], "-", c=C_BLUE, lw=1.0)
        ax.plot(x[o], lost[o], "--", c=C_OUT, lw=0.9)
        meas = np.ones(len(x), bool) if interpolated is None else ~interpolated
        ax.plot(x[meas], pnp[meas], "o", c=C_BLUE, ms=2.5, lw=0, label="PnP success")
        ax.plot(x[meas], lost[meas], "s", c=C_OUT, ms=2.2, lw=0, label="Overlay lost (SEARCHING)")
        if interpolated is not None and interpolated.any():
            ax.plot(x[interpolated], pnp[interpolated], "o", mfc="white", mec=C_BLUE, ms=3, lw=0,
                    label="Interpolated azimuth (no pose)")
            ax.plot(x[interpolated], lost[interpolated], "s", mfc="white", mec=C_OUT, ms=2.6, lw=0)

    if r31:
        ax = axs[0]
        x = col(r31, "dist_mm") / 10
        band(ax, RED_DIST[0] / 10, RED_DIST[1] / 10, C_RED, 0.25, "Reduced tier")
        band(ax, FULL_DIST[0] / 10, FULL_DIST[1] / 10, C_FULL, 0.30, "Full tier")
        lines(ax, x, r31)
        ax.set_xlabel("Distance to phantom origin (cm)"); ax.set_title("(a) Distance")
    if r32:
        ax = axs[1]
        az, interp = fill_az(col(r32, "az_deg"))
        band(ax, RED_AZ[0], RED_AZ[1], C_RED, 0.25, wrap=True)
        band(ax, FULL_AZ[0], FULL_AZ[1], C_FULL, 0.30)
        lines(ax, az, r32, interp)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
        ax.set_xlabel("Azimuth (°)"); ax.set_title("(b) Azimuth")
    if r33:
        ax = axs[2]
        drop = set(a.elev_drop)
        rows = [r for r in r33 if int(float(r.get("step", -1))) not in drop]
        if len(rows) < len(r33):
            print(f"  tracking: dropped elevation step(s) {sorted(drop)} (movement at end of run)")
        x = col(rows, "elev_deg")
        keep = np.isfinite(x)
        rows = [r for r, k in zip(rows, keep) if k]
        band(ax, RED_ELEV[0], RED_ELEV[1], C_RED, 0.25)
        band(ax, FULL_ELEV[0], FULL_ELEV[1], C_FULL, 0.30)
        lines(ax, x[keep], rows)
        ax.set_xlabel("Elevation (°)"); ax.set_title("(c) Elevation")
    axs[0].set_ylabel("Share of frames (%)")
    axs[0].set_ylim(-3, 103)
    h, l = axs[0].get_legend_handles_labels()
    hh, ll = axs[1].get_legend_handles_labels()
    keep = [(x, y) for x, y in zip(h + hh, l + ll)]
    seen, hs, ls = set(), [], []
    for x, y in keep:
        if y not in seen:
            seen.add(y); hs.append(x); ls.append(y)
    fig.legend(hs, ls, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_atp3_tracking",
         "ATP-3 sweeps. Share of frames with a valid pose (PnP success) and share spent in the SEARCHING state "
         "(overlay lost) against (a) distance, (b) azimuth and (c) elevation. Shading shows the envelope tiers used "
         "by the HUD. Open circles in (b) are interpolated azimuths for positions with no valid pose.",
         "fig:results:atp3")


# ---------------------------------------------------------------------------
# 3. ATP-2 overlay error vs distance from markers
# ---------------------------------------------------------------------------
def cluster_ci(vals, clusters, n=2000, seed=0):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    ids = np.unique(clusters)
    if len(ids) < 2:
        return np.nan, np.nan
    groups = [vals[clusters == c] for c in ids]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def fig_overlay(out, a):
    early = []
    for p in a.atp2_early:
        r = read(p)
        if r:
            early += r
    crisp = read(a.atp2_crisp) or []
    if not early and not crisp:
        print("  overlay: skipped (no ATP-2 CSVs found)")
        return
    fig, ax = plt.subplots(figsize=(W * 0.8, 3.0))
    groups = [("Earlier sessions (focus unverified)", early, C_OUT, -0.25),
              ("Crisp focus (matches calibration)", crisp, C_BLUE, 0.25)]
    rng = np.random.default_rng(1)
    levels = ["L5", "L4", "L3", "L2", "L1"]
    xs = {}
    for _, rows, _, _ in groups:
        for lv in levels:
            d = [float(r["dist_from_markers_mm"]) / 10 for r in rows if r["level"] == lv]
            if d:
                xs[lv] = np.mean(d)
    for name, rows, color, off in groups:
        if not rows:
            continue
        err_all = np.array([float(r["err_mm"]) for r in rows])
        lab = f"{name}: mean {err_all.mean():.2f} mm (n = {len(rows)})"
        first = True
        for lv in levels:
            sel = [r for r in rows if r["level"] == lv]
            if not sel or lv not in xs:
                continue
            e = np.array([float(r["err_mm"]) for r in sel])
            ax.scatter(xs[lv] + off + rng.normal(0, 0.12, len(e)), e, s=6, c=color, alpha=0.30, lw=0)
            lo, hi = cluster_ci(e, [r["shot"] for r in sel])
            m = e.mean()
            yerr = None if np.isnan(lo) else [[m - lo], [hi - m]]
            ax.errorbar(xs[lv] + off, m, yerr=yerr, fmt="o", c=color, ms=4.5, capsize=2.5, lw=1.0,
                        label=lab if first else None)
            first = False
    pred = []
    for lv in levels:
        v = [float(r["pred_tre_mm"]) for r in (early + crisp) if r["level"] == lv and r.get("pred_tre_mm") not in ("", None)]
        if v and lv in xs:
            pred.append((xs[lv], np.mean(v)))
    if pred:
        pred = np.array(pred)
        ax.plot(pred[:, 0], pred[:, 1], "-s", c=C_BLACK, ms=3.5, lw=0.9, label="Predicted TRE (Fitzpatrick)")
    ax.axhline(2.0, c=C_GREY, ls="--", lw=0.8)
    allerr = [float(r["err_mm"]) for r in (early + crisp)]
    ax.set_ylim(0, max(allerr) * 1.40)
    ax.text(ax.get_xlim()[0] + 0.1, 2.04, "2 mm criterion", ha="left", va="bottom", fontsize=7, color=C_GREY)
    ax.set_xlabel("Distance of target from marker centroid (cm)")
    ax.set_ylabel("Overlay error (mm)")
    top = ax.secondary_xaxis("top")
    top.set_xticks([xs[l] for l in levels if l in xs]); top.set_xticklabels([l for l in levels if l in xs])
    top.spines["top"].set_visible(True)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_atp2_overlay_error",
         "Overlay error at the entry-point targets against the targets' distance from the marker centroid, "
         "measured against painted ground-truth dots with the target marker hidden while clicking. Points are "
         "individual measurements; markers show the mean with a 95\\% cluster-bootstrap interval over screenshots.",
         "fig:results:atp2")


# ---------------------------------------------------------------------------
# 4. ATP-4 occlusion
# ---------------------------------------------------------------------------
# Step numbers follow the ATP-4 run sheet (even steps are the holds). Step 18 is the
# 8-marker recheck at the end of sequence A and serves as the baseline for both sequences.
ATP4_A = {18: 8, 4: 7, 6: 6, 8: 5, 10: 4, 12: 3, 14: 2, 16: 1}
ATP4_B = {18: 8, 20: 7, 22: 6, 24: 5, 26: 4, 28: 3, 30: 2, 32: 1}
MIN_PNP_FOR_TRE = 50.0     # TRE is only drawn where at least this share of frames tracked


def fig_occlusion(out, a):
    rows = read(a.atp4)
    if not rows:
        print("  occlusion: skipped (no ATP-4 summary CSV found)")
        return
    by = {int(float(r["step"])): r for r in rows}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharex=True)
    for name, mp, c, mk in (("A: both faces kept", ATP4_A, C_BLUE, "o"),
                            ("B: Face B stripped first", ATP4_B, C_OUT, "s")):
        pts = [(n, by[s]) for s, n in mp.items() if s in by]
        pts.sort()
        n = np.array([p[0] for p in pts])
        pnp = np.array([float(p[1]["pnp_success_pct"]) for p in pts])
        tre = np.array([float(p[1]["tre_mean_mm"]) if p[1]["tre_mean_mm"] not in ("", "nan") else np.nan for p in pts])
        w95 = np.array([float(p[1]["tre_worst_p95_mm"]) if p[1]["tre_worst_p95_mm"] not in ("", "nan") else np.nan for p in pts])
        tre[pnp < MIN_PNP_FOR_TRE] = np.nan
        w95[pnp < MIN_PNP_FOR_TRE] = np.nan
        axs[0].plot(n, tre, "-" + mk, c=c, ms=3.5, lw=1.0, label=f"{name}: mean target")
        axs[0].plot(n, w95, "--" + mk, c=c, ms=3, lw=0.8, mfc="none", label=f"{name}: worst target (p95)")
        axs[1].plot(n, pnp, "-" + mk, c=c, ms=3.5, lw=1.0, label=name)
    axs[0].axhline(2.0, c=C_GREY, ls="--", lw=0.8)
    axs[0].text(1.1, 2.06, "2 mm", fontsize=7, color=C_GREY)
    for ax in axs:
        ax.axvspan(0.5, 3.5, color=C_OUT, alpha=0.10, lw=0)
        ax.set_xlim(8.4, 0.6)
        ax.set_xlabel("Markers visible (of 8)")
    ylo, yhi = axs[0].get_ylim()
    axs[0].text(2.0, ylo + (yhi - ylo) * 0.04, "below the\n4-marker gate", ha="center", va="bottom", fontsize=6.5, color=C_OUT)
    axs[0].set_ylabel("Predicted TRE (mm)"); axs[0].set_title("(a) Predicted TRE")
    axs[1].set_ylabel("PnP success (% of frames)"); axs[1].set_title("(b) Tracking"); axs[1].set_ylim(-3, 103)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=6.8)
    fig.tight_layout()
    save(fig, out, "fig_atp4_occlusion",
         "ATP-4 occlusion. (a) Predicted TRE (mean over targets, and 95th percentile of the worst target) and "
         "(b) PnP success against the number of visible markers, covering markers progressively. In sequence A the "
         "last four markers are the near-junction cluster (IDs 0, 2, 5, 7); in sequence B they are all on Face A "
         "(TRE is not drawn where fewer than half the frames tracked). Both sequences share the 8-marker baseline "
         "recorded at the end of sequence A. Tracking is not attempted below four markers.",
         "fig:results:atp4")


# ---------------------------------------------------------------------------
# 5. Tier comparison
# ---------------------------------------------------------------------------
def fig_tiers(out, a):
    rows = read(a.tiers)
    if not rows:
        print("  tiers: skipped (no atp3_tier_summary.csv found)")
        return
    order = {"full": 0, "reduced": 1, "outside": 2}
    rows = sorted(rows, key=lambda r: order.get(r["tier"], 9))
    names = [r["tier"].capitalize() for r in rows]
    x = np.arange(len(rows)); w = 0.36
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.85, 2.5))
    axs[0].bar(x - w / 2, col(rows, "pnp_success_pct"), w, color=C_BLUE, label="PnP success")
    axs[0].bar(x + w / 2, col(rows, "lost_pct"), w, color=C_OUT, label="Overlay lost")
    axs[0].set_ylabel("Share of frames (%)"); axs[0].set_title("(a) Tracking")
    axs[1].bar(x - w / 2, col(rows, "tre_mean_mm"), w, color=C_FULL, label="Mean over targets")
    axs[1].bar(x + w / 2, col(rows, "tre_worst_p95_mm"), w, color=C_RED, label="Worst target, p95")
    axs[1].axhline(2.0, c=C_GREY, ls="--", lw=0.8)
    axs[1].set_ylabel("Predicted TRE (mm)"); axs[1].set_title("(b) Predicted TRE")
    labs = [f"{n}\n({float(r['markers_mean']):.1f})" for n, r in zip(names, rows)]
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(labs)
    axs[0].set_ylim(0, 125); axs[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=7)
    axs[1].set_ylim(0, max(col(rows, "tre_worst_p95_mm")) * 1.38); axs[1].legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, out, "fig_tier_comparison",
         "Tracking and predicted TRE per envelope tier, pooled over the ATP-3 sweeps (frame-weighted). "
         "Mean detected markers per tier are in brackets under each label. Predicted TRE depends only on the visible marker set.",
         "fig:results:tiers")


# ---------------------------------------------------------------------------
# 6. Calibration
# ---------------------------------------------------------------------------
# Values from the calibration sessions; edit here if you recalibrate.
CAL_V0_RMS = 1.24
CAL_V1_TRAIN, CAL_V1_HOLD = 0.51, 0.52
RULER_V1 = [(25, 24.9), (35, 34.7), (45, 44.9)]            # (true cm, measured z cm), right after calibration
RULER_RECHECK = [(35, 34.57), (45, 45.28)]                 # re-check in the crisp focus state


def fig_calibration(out, a):
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.9, 2.5))
    ax = axs[0]
    vals = [CAL_V0_RMS, CAL_V1_TRAIN, CAL_V1_HOLD]
    ax.bar([0, 1, 2], vals, color=[C_GREY, C_BLUE, C_SKY], width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["v0", "v1\ntraining", "v1\nheld-out"])
    ax.set_ylabel("Reprojection RMS (px)"); ax.set_title("(a) Calibration residual")
    ax.set_ylim(0, 1.45)
    ax = axs[1]
    ax.axhspan(-5, 5, color=C_FULL, alpha=0.12, lw=0)
    ax.axhline(0, c=C_GREY, lw=0.6)
    for data, lab, c, mk in ((RULER_V1, "Just after calibration", C_BLUE, "o"),
                             (RULER_RECHECK, "Re-check, crisp focus", C_OUT, "s")):
        ax.plot([d[0] for d in data], [(d[1] - d[0]) * 10 for d in data], "-" + mk, c=c, ms=4, lw=0.9, label=lab)
    ax.set_xlabel("True distance (cm)"); ax.set_ylabel("Depth error (mm)")
    ax.set_title("(b) Ruler validation"); ax.set_xticks([25, 35, 45])
    ax.text(45.6, 4.3, "±5 mm", fontsize=6.5, color=C_FULL, ha="right")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    save(fig, out, "fig_calibration",
         "Camera calibration. (a) ChArUco reprojection RMS for the first (v0) and final (v1) calibrations; v1 is "
         "reported on the training and held-out image sets. (b) Depth error of the pose estimate against a ruler "
         "at 25 to 45 cm, immediately after calibration and in a later re-check.",
         "fig:results:calib")


# ---------------------------------------------------------------------------
# 7. Timing and latency
# ---------------------------------------------------------------------------
LATENCY_MS = [186, 187, 151, 249, 245, 115, 139, 180, 187, 214, 147]   # stopwatch test, 2026-10-01


def fig_timing(out, a):
    rows = read(a.atp6)
    lat = np.array(a.latency if a.latency else LATENCY_MS, float)
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.9, 2.6), gridspec_kw={"width_ratios": [1, 1]})
    if rows:
        by = {int(float(r["step"])): r for r in rows}
        stages = [("grab_ms", "Frame grab", C_SKY), ("detect_pnp_ms", "Detect + PnP", C_BLUE),
                  ("overlay_ms", "Overlay", C_OUT), ("qt_ms", "Qt display", C_GREY)]
        ax = axs[0]
        for xi, (st, lab) in enumerate(((a.wire_on_step, "Wireframe on"), (a.wire_off_step, "Wireframe off"))):
            if st not in by:
                continue
            bottom = 0.0
            for key, name, c in stages:
                v = float(by[st][key])
                ax.bar(xi, v, 0.55, bottom=bottom, color=c, label=name if xi == 0 else None)
                bottom += v
            tot = float(by[st]["frame_ms_mean"]) if by[st].get("frame_ms_mean") not in (None, "") else bottom
            ax.text(xi, bottom + 0.8, f"{tot:.1f} ms\n{float(by[st]['fps']):.1f} FPS", ha="center", fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Wireframe on", "Wireframe off"])
        ax.set_ylabel("Processing time per frame (ms)"); ax.set_title("(a) Pipeline timing")
        ax.set_ylim(0, 42); ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    else:
        axs[0].axis("off"); print("  timing: no atp6_summary.csv, drawing latency panel only")
    ax = axs[1]
    jit = np.random.default_rng(2).normal(0, 0.04, len(lat))
    ax.boxplot([lat], positions=[0], widths=0.4, showfliers=False, medianprops={"color": C_BLACK})
    ax.scatter(jit, lat, s=10, c=C_BLUE, alpha=0.7, zorder=3)
    ax.axhline(200, c=C_OUT, ls="--", lw=0.9)
    ax.text(0.32, 203, "200 ms", fontsize=7, color=C_OUT, ha="right")
    ax.text(0, lat.max() + 12, f"mean {lat.mean():.0f} ms, SD {lat.std(ddof=1):.0f} ms (n = {len(lat)})",
            ha="center", fontsize=7)
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([]); ax.set_ylim(0, lat.max() + 40)
    ax.set_ylabel("Camera-to-screen latency (ms)"); ax.set_title("(b) End-to-end latency")
    fig.tight_layout()
    save(fig, out, "fig_atp6_timing",
         "ATP-6. (a) Mean per-frame processing time by pipeline stage with the wireframe overlay on and off. "
         "(b) End-to-end camera-to-screen latency measured with an on-screen stopwatch photographed through the "
         "camera; each point is one reading.",
         "fig:results:atp6")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    d = "data/atp/"
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["envelope", "tracking", "overlay", "occlusion", "tiers", "calibration", "timing"])
    ap.add_argument("--atp31", default=d + "atp3_1_summary.csv")
    ap.add_argument("--atp32", default=d + "atp3_2_summary.csv")
    ap.add_argument("--atp33", default=d + "atp3_3_summary.csv")
    ap.add_argument("--elev-drop", type=int, nargs="*", default=[9], dest="elev_drop",
                    help="ATP-3.3 step numbers to leave out (default 9: movement at the end of the run)")
    ap.add_argument("--atp2-early", nargs="*", dest="atp2_early",
                    default=[d + "atp2_tiers_hidden.csv", d + "atp2_tiers_capture2.csv"])
    ap.add_argument("--atp2-crisp", default=d + "atp2_crisp.csv", dest="atp2_crisp")
    ap.add_argument("--atp4", default=d + "atp4_summary.csv")
    ap.add_argument("--tiers", default=d + "atp3_tier_summary.csv")
    ap.add_argument("--atp6", default=d + "atp6_summary.csv")
    ap.add_argument("--wire-on-step", type=int, default=1)
    ap.add_argument("--wire-off-step", type=int, default=2)
    ap.add_argument("--latency", type=float, nargs="*", default=None,
                    help="Latency readings in ms (default: the 11 stopwatch readings)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = {"envelope": fig_envelope, "tracking": fig_tracking, "overlay": fig_overlay,
            "occlusion": fig_occlusion, "tiers": fig_tiers, "calibration": fig_calibration, "timing": fig_timing}
    todo = a.only or list(figs)
    print(f"Envelope limits read from {ENV_SOURCE}")
    for name in todo:
        print(f"[{name}]")
        figs[name](out, a)
    if LATEX:
        (out / "figures_latex_snippets.tex").write_text(
            "% Edit the captions to say what YOU want the reader to see.\n\n" + "\n".join(LATEX))
        print(f"wrote {out / 'figures_latex_snippets.tex'}")


if __name__ == "__main__":
    main()