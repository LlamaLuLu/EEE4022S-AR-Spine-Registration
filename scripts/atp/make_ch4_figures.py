"""
Chapter 4 figures in the shared biomedical style (see scripts/fig_style.py).

Run from the repo root:
    python scripts/atp/make_ch4_figures.py                 # everything
    python scripts/atp/make_ch4_figures.py --only calib atp2

Outputs (PDF + 300 dpi PNG) go to docs/figures:
    fig_calib_depth      fig:results:calib       depth check, grouped % error bars
    fig_atp2_by_level    fig:results:atp2        overlay error by level, P1 and P2
    fig_sweep_grids      fig:results:sweepgrids  design-sweep camera poses
    fig_envelope_p1p2    fig:results:envelope    P1 and P2 working envelopes
    fig_p2_range         fig:results:p2range     P2 sweeps (replaces dome_operating_range.png)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fig_style as FS

FS.apply()
W = FS.W
MINUS = "−"


def read(path):
    p = Path(path)
    if not p.exists():
        print(f"  missing {p}")
        return []
    return list(csv.DictReader(open(p)))


def flt(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def cluster_ci(vals, clusters, n=4000, seed=0):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    ids = np.unique(clusters)
    if len(ids) < 2:
        return np.nan, np.nan
    groups = [vals[clusters == c] for c in ids]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------------------
# 1. Calibration depth check
# ---------------------------------------------------------------------------------------
RULER_V1 = [(25, 24.9), (35, 34.7), (45, 44.9)]            # (true cm, recovered depth cm), just after calibration
RULER_RECHECK = [(35, 34.57), (45, 45.28)]                 # final focus state


def fig_calib(out, a):
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.9))
    dists = [25, 35, 45]
    w = 0.30
    sets = [("Just after calibration", RULER_V1, FS.BLUE, -w / 2 - 0.02), ("Final focus state", RULER_RECHECK, FS.TEAL, w / 2 + 0.02)]
    for lab, data, c, off in sets:
        d = dict(data)
        xs, ys, zs = [], [], []
        for i, t in enumerate(dists):
            if t in d:
                xs.append(i + off); ys.append((d[t] - t) / t * 100); zs.append(d[t])
        ax.bar(xs, ys, w, color=c, label=lab, zorder=3)
        for x, y, z in zip(xs, ys, zs):
            s = f"{y:+.1f}%".replace("-", MINUS)
            ax.text(x, y + (0.12 if y >= 0 else -0.12), s, ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=9, fontweight="bold", color=FS.NAVY if c == FS.BLUE else FS.TEAL)
            ax.text(x, -2.18, f"{z:.1f}", ha="center", va="bottom", fontsize=8.5, color=FS.SLATE)
    ax.text(0 + w / 2 + 0.02, -0.12, "not repeated", ha="center", va="top", fontsize=7.5, color=FS.GREY, rotation=90)
    ax.axhline(0, c=FS.BLACK, lw=0.9, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{d} cm" for d in dists])
    ax.set_xlabel("Measured distance (ruler). Grey numbers: recovered depth (cm)")
    ax.set_ylabel("Depth error (% of distance)")
    ax.set_ylim(-2.4, 1.4)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper left", ncol=1)
    ax.set_axisbelow(True)
    save(fig, out, "fig_calib_depth")


# ---------------------------------------------------------------------------------------
# 2. ATP-2 overlay error by level
# ---------------------------------------------------------------------------------------
def load_err(paths):
    rows = []
    for p in ([paths] if isinstance(paths, str) else paths):
        rows += read(p)
    return [r for r in rows if np.isfinite(flt(r.get("err_mm")))]


def fig_atp2(out, a):
    groups = [
        ("P1, drifted focus", load_err([a.p1_drift1, a.p1_drift2]), FS.P1_DRIFT, "o"),
        ("P1, calibrated focus", load_err(a.p1_cal), FS.P1_CAL, "o"),
        ("P2, f35", load_err(a.p2_f35), FS.P2_F35, "s"),
        ("P2, f50", load_err(a.p2_f50), FS.P2_F50, "s"),
    ]
    levels = ["L1", "L2", "L3", "L4", "L5"]
    fig, ax = plt.subplots(figsize=(W, 3.5))
    rng = np.random.default_rng(3)
    offs = np.array([-0.36, -0.12, 0.12, 0.36])
    SP = 1.35  # level spacing
    handles = []
    for (name, rows, c, mk), off in zip(groups, offs):
        e_all = np.array([float(r["err_mm"]) for r in rows])
        print(f"  {name}: n = {len(rows)}, shots = {len({r['shot'] for r in rows})}, mean = {e_all.mean():.2f}")
        for i, lv in enumerate(levels):
            sel = [r for r in rows if r["level"] == lv]
            if not sel:
                continue
            e = np.array([float(r["err_mm"]) for r in sel])
            ax.scatter(i * SP + off + rng.normal(0, 0.02, len(e)), e, s=9, c=c, alpha=0.45, lw=0, zorder=2)
            lo, hi = cluster_ci(e, [r["shot"] for r in sel])
            m = e.mean()
            yerr = None if np.isnan(lo) else [[m - lo], [hi - m]]
            ax.errorbar(i * SP + off, m, yerr=yerr, fmt=mk, c=c, mec=FS.BLACK if c in (FS.P1_DRIFT, FS.P2_F35) else "white",
                        mew=0.6, ms=7, capsize=3, lw=1.3, zorder=4, ecolor=FS.NAVY if c in (FS.P1_CAL, FS.P1_DRIFT) else "#065F52")
        handles.append(Line2D([], [], marker=mk, ls="", color=c, mec=FS.BLACK if c in (FS.P1_DRIFT, FS.P2_F35) else "white",
                              ms=8, label=f"{name} (mean {e_all.mean():.2f} mm, n = {len(rows)})"))
    ax.axhline(2.0, c=FS.BLACK, ls="--", lw=1.1, zorder=1)
    ax.text(4 * SP + 0.65, 2.06, "2 mm criterion", ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([i * SP for i in range(5)]); ax.set_xticklabels(levels)
    ax.set_xlim(-0.6, 4 * SP + 0.68)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel("Lumbar level (L1 furthest from the P1 bracket, L5 nearest)")
    ax.set_ylabel("Overlay error (mm)")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.2, handletextpad=0.3)
    save(fig, out, "fig_atp2_by_level")


# ---------------------------------------------------------------------------------------
# 3. Design-sweep pose grids (spherical frame)
# ---------------------------------------------------------------------------------------
def fig_grids(out, a):
    cmap = LinearSegmentedColormap.from_list("elev", ["#CFE6F1", FS.SKY, FS.BLUE, FS.NAVY])
    fig = plt.figure(figsize=(W, 3.05))
    specs = [("(a) P1: 40 poses, 450 mm", [10, 20, 30, 40, 50]),
             ("(b) P2: 56 poses at 600 and 900 mm", [10, 20, 30, 40, 50, 60, 70])]
    for k, (title, elevs) in enumerate(specs):
        ax = fig.add_subplot(1, 2, k + 1, projection="polar")
        ax.set_theta_zero_location("S"); ax.set_theta_direction(-1)
        for el in elevs:
            col = cmap((el - 10) / 60.0)
            for az in range(0, 360, 45):
                ax.bar(np.radians(az), 10, width=np.radians(45), bottom=90 - el - 5, color=col,
                       edgecolor="white", lw=1.0, zorder=2)
                ax.plot(np.radians(az), 90 - el, "o", c="white", ms=2.6, zorder=3)
        ax.set_ylim(0, 90)
        ax.set_yticks([])
        for el in elevs:
            ax.text(np.radians(22.5), 90 - el, f"{el}°", ha="center", va="center", fontsize=7, color=FS.NAVY,
                    bbox=dict(boxstyle="round,pad=0.03", fc="white", ec="none"), zorder=6)
        ax.set_xticks(np.radians(range(0, 360, 45)))
        ax.set_xticklabels(["0°\ncaudal", "45°", "90°\nleft", "135°", "180°\ncranial", "225°", "270°\nsurgeon", "315°"],
                           fontsize=8)
        ax.tick_params(axis="x", pad=2)
        ax.grid(False); ax.spines["polar"].set_visible(False)
        ax.set_title(title, fontsize=9.5, pad=16, loc="center")
    fig.text(0.5, 0.0, "Radius: elevation (centre = overhead, rim = horizon; labels give each ring). "
             "Angle: azimuth about the phantom origin.", ha="center", fontsize=8, color=FS.SLATE)
    fig.subplots_adjust(wspace=0.6, bottom=0.09, top=0.86)
    save(fig, out, "fig_sweep_grids")


# ---------------------------------------------------------------------------------------
# helpers for the envelope and P2 range figures
# ---------------------------------------------------------------------------------------
def load_p2_sweeps(path):
    rows = [r for r in read(path) if r["focus"] == "f50"]
    def pick(sweep):
        sel = [r for r in rows if r["sweep"] == sweep]
        return sel
    return pick


def cls(p):
    return FS.FULL if p >= 95 else (FS.PART if p >= 50 else FS.NONE)


B_FULL, B_PART = FS.BLUE, "#B7D3EA"          # P1: blue theme
G_FULL, G_PART = FS.GREEN, "#B4E3C9"         # P2: green theme
G_TEXT = "#1F7A52"


def edge(x_good, x_bad):
    return (x_good + x_bad) / 2.0


def range_panel(ax, xlim, full, part, none_from=None, none_to=None, unit="", tested=None, arrow_hi=None, label_ends=True, skip_lo=False, cf=None, cp=None, lab=None):
    """Horizontal range bar: light bar = tracking holds, dark bar = full, hatched grey = no tracking."""
    cf = cf or FS.FULL; cp = cp or FS.PART; lab = lab or FS.TEAL
    ax.set_xlim(*xlim); ax.set_ylim(-0.9, 0.9)
    for lo, hi in ((none_from, none_to),):
        pass
    if none_from is not None:
        ax.add_patch(Rectangle((none_from, -0.45), (none_to if none_to is not None else xlim[1]) - none_from, 0.9,
                               fc=FS.NONE, ec="white", hatch="///", lw=0, zorder=1))
    if none_to is not None and none_from is None:
        pass
    ax.add_patch(Rectangle((part[0], -0.45), part[1] - part[0], 0.9, fc=cp, ec="none", zorder=2))
    ax.add_patch(Rectangle((full[0], -0.45), full[1] - full[0], 0.9, fc=cf, ec="none", zorder=3))
    if label_ends:
        if not skip_lo: ax.text(full[0], -0.6, f"{full[0]:g}", ha="center", va="top", fontsize=10.5, fontweight="bold", color=lab)
        ax.text(full[1], -0.6, f"{full[1]:.0f}" + ("+" if arrow_hi else ""), ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=lab)
        if part[0] < full[0] - 1e-9:
            ax.text(part[0], 0.6, f"{part[0]:.0f}", ha="center", va="bottom", fontsize=10.5, color=FS.SLATE)
        if part[1] > full[1] + 1e-9:
            ax.text(part[1], 0.6, f"{part[1]:.0f}", ha="center", va="bottom", fontsize=10.5, color=FS.SLATE)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(unit, fontsize=10.5)
    ax.tick_params(axis="x", labelsize=10)


def ring(ax, lo, hi, r0, r1, color, hatch=None):
    """Annular sector from azimuth lo to hi (deg, may wrap)."""
    if hi < lo:
        hi += 360
    th = np.radians(np.linspace(lo, hi, max(int(hi - lo) // 3 + 2, 3)))
    ax.fill_between(th, r0, r1, color=color, lw=0, hatch=hatch, edgecolor="white" if hatch else color, zorder=2)


def polar_setup(ax):
    ax.set_theta_zero_location("S"); ax.set_theta_direction(-1)
    ax.set_ylim(0, 3.0)
    ax.set_yticks([]); ax.grid(False); ax.spines["polar"].set_visible(False)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=10.5)
    ax.tick_params(axis="x", pad=-1)


def fig_envelope(out, a):
    pick = load_p2_sweeps(a.p2_atp3)
    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.05, 1.25], hspace=0.38, wspace=0.25)

    # ---------------- P1 (values from tab:results:envelope) ----------------
    ax = fig.add_subplot(gs[0, 0])
    range_panel(ax, (0, 140), full=(55, 100), part=(33, 124), none_from=124, unit="Distance (cm)", cf=B_FULL, cp=B_PART, lab=B_FULL)
    ax.set_title("(a) Distance", fontsize=11.5); axd1 = ax
    ax = fig.add_subplot(gs[0, 2])
    range_panel(ax, (0, 90), full=(10, 35), part=(0, 52), none_from=56, unit="Elevation (°)", cf=B_FULL, cp=B_PART, lab=B_FULL)
    ax.set_title("(c) Elevation", fontsize=11.5)
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    polar_setup(ax)
    ring(ax, 150, 300, 0.7, 2.9, FS.NONE, hatch="///")
    ring(ax, 325, 120, 0.7, 2.9, B_PART)
    ring(ax, 30, 55, 0.7, 2.9, B_FULL)
    ax.set_title("(b) Azimuth", fontsize=11.5, pad=26, loc="center")

    # ---------------- P2 (classes from measured PnP success, f50) ----------------
    d = sorted(pick("distance"), key=lambda r: flt(r["dist_mm"]))
    dx = [flt(r["dist_mm"]) / 10 for r in d]; dp = [flt(r["pnp_pct"]) for r in d]
    full_hi = edge(max(x for x, p in zip(dx, dp) if p >= 95), min(x for x, p in zip(dx, dp) if p < 95 and x > 50))
    part_hi = edge(max(x for x, p in zip(dx, dp) if p >= 50), min(x for x, p in zip(dx, dp) if p < 50 and x > 55))
    ax = fig.add_subplot(gs[1, 0])
    range_panel(ax, (0, 140), full=(min(dx), full_hi), part=(min(dx), part_hi), none_from=part_hi, unit="Distance (cm)", skip_lo=True, cf=G_FULL, cp=G_PART, lab=G_TEXT)
    axd2 = ax
    ax.axvline(min(dx), c=FS.SLATE, lw=0.8, ls=":")
    ax.text(min(dx) - 1.5, 0.0, "closest\ntested", ha="right", va="center", fontsize=9.5, color=FS.SLATE)

    e = sorted(pick("elevation"), key=lambda r: flt(r["elev_deg"]))
    ex = [flt(r["elev_deg"]) for r in e]; ep = [flt(r["pnp_pct"]) for r in e]
    e_none = edge(max(x for x, p in zip(ex, ep) if p < 50 and x < 40), min(x for x, p in zip(ex, ep) if p >= 50))
    e_full = edge(max(x for x, p in zip(ex, ep) if p < 95 and x < 50), min(x for x, p in zip(ex, ep) if p >= 95))
    ax = fig.add_subplot(gs[1, 2])
    range_panel(ax, (0, 90), full=(e_full, max(ex)), part=(e_none, max(ex)), unit="Elevation (°)", arrow_hi=True, cf=G_FULL, cp=G_PART, lab=G_TEXT)
    ax.add_patch(Rectangle((0, -0.45), e_none, 0.9, fc=FS.NONE, ec="white", hatch="///", lw=0, zorder=1))
    ax.text(max(ex) + 1.5, 0.0, "tested\nto 80°", ha="left", va="center", fontsize=9.5, color=FS.SLATE)

    ax = fig.add_subplot(gs[1, 1], projection="polar")
    polar_setup(ax)
    for sweep, r0, r1 in (("azimuth el 30", 0.7, 1.75), ("azimuth el 45", 1.85, 2.9)):
        for r in pick(sweep):
            az = flt(r["az_deg"]); p = flt(r["pnp_pct"])
            c_ = cls(p); ring(ax, az - 15, az + 15, r0, r1, {FS.FULL: G_FULL, FS.PART: G_PART}.get(c_, c_), hatch="///" if c_ == FS.NONE else None)
    ax.text(np.radians(150), 1.3, "30°", color="white", fontsize=9.5, ha="center", va="center", fontweight="bold")
    ax.text(np.radians(150), 2.35, "45°", color="white", fontsize=9.5, ha="center", va="center", fontweight="bold")

    # row labels and legend
    p1 = axd1.get_position(); p2 = axd2.get_position()
    fig.text(0.015, p1.y0 + p1.height / 2 + 0.03, "P1\nbracket", rotation=90, ha="center", va="center", fontsize=13, fontweight="bold", color=B_FULL)
    fig.text(0.015, p2.y0 + p2.height / 2 + 0.03, "P2\ndome", rotation=90, ha="center", va="center", fontsize=13, fontweight="bold", color=G_TEXT)
    from matplotlib.legend_handler import HandlerTuple
    leg = [(Patch(fc=B_FULL), Patch(fc=G_FULL)),
           (Patch(fc=B_PART), Patch(fc=G_PART)),
           Patch(fc=FS.NONE, hatch="///", ec="white")]
    labs = ["Full: P1 full-accuracy tier, P2 PnP success of 95% or more",
            "Partial: P1 tracking holds, P2 PnP success of 50 to 95%",
            "None: P1 no tracking, P2 PnP success below 50%"]
    fig.legend(handles=leg, labels=labs, loc="lower center", ncol=1, bbox_to_anchor=(0.52, -0.14), fontsize=10,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=2.4)
    save(fig, out, "fig_envelope_p1p2")


# ---------------------------------------------------------------------------------------
# 3D working-envelope volumes (spherical-shell blocks, as in a dS diagram)
# ---------------------------------------------------------------------------------------
def _xyz(r, el, az):
    el, az = np.radians(el), np.radians(az)
    return -r * np.cos(el) * np.sin(az), -r * np.cos(el) * np.cos(az), r * np.sin(el)


def shell3d(ax, r0, r1, e0, e1, a0, a1, color, alpha=0.5, edge=None, lw=0.8, rays=False):
    """Block of a spherical shell: distance r0..r1, elevation e0..e1, azimuth a0..a1 (a1 < a0 wraps through 360)."""
    if a1 < a0:
        a1 += 360
    n = 24
    R = np.linspace(r0, r1, 2); E = np.linspace(e0, e1, n); A = np.linspace(a0, a1, max(int(a1 - a0) // 6 + 2, 4))
    faces = []
    e_, a_ = np.meshgrid(E, A)
    faces += [_xyz(r1, e_, a_)] + ([_xyz(r0, e_, a_)] if color == FS.FULL else [])   # outer (and, for solid blocks, inner) shell
    r_, a_ = np.meshgrid(R, A)
    faces += [_xyz(r_, e0, a_), _xyz(r_, e1, a_)]                    # elevation floor and ceiling
    r_, e_ = np.meshgrid(R, E)
    faces += [_xyz(r_, e_, a0), _xyz(r_, e_, a1)]                    # azimuth side walls
    for X, Y, Z in faces:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False, antialiased=True, rcount=24, ccount=40)
    ec = edge or (FS.NAVY if color == FS.FULL else "#5E9E94")
    lw = lw if color == FS.FULL else 0.7
    for r in (r0, r1):
        for e in (e0, e1):
            ax.plot(*_xyz(r, e, np.linspace(a0, a1, 40)), c=ec, lw=lw)
        for a in (a0, a1):
            ax.plot(*_xyz(r, np.linspace(e0, e1, 20), a), c=ec, lw=lw)
    for a in (a0, a1):
        for e in (e0, e1):
            ax.plot(*_xyz(np.array([r0, r1]), e, a), c=ec, lw=lw)
    if rays:                                                          # rays from the phantom origin to the block corners
        for a in (a0, a1 % 360):
            for e in (e0, e1):
                ax.plot(*_xyz(np.array([0, r0]), e, a), c=ec, lw=0.6, ls="--")


def dead_wedge(ax, rmax, a0, a1, color=FS.GREY):
    if a1 < a0:
        a1 += 360
    A = np.linspace(a0, a1, 30)
    X, Y, Z = _xyz(np.array([[0.0], [rmax]]), 0, A[None, :])
    ax.plot_surface(X, Y, np.zeros_like(X), color=color, alpha=0.5, linewidth=0, shade=False)
    ax.plot(*_xyz(rmax, 0, A), c=FS.SLATE, lw=0.8, ls="--")


def env3d_panel(ax, title, rmax, blocks, dead, labels, view=(24, -48)):
    ax.set_proj_type("ortho")
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot_surface(np.outer([0, 1], rmax * 1.05 * np.cos(th)), np.outer([0, 1], rmax * 1.05 * np.sin(th)),
                    np.zeros((2, 120)), color=FS.LIGHT, alpha=0.35, linewidth=0, shade=False)
    ax.plot(rmax * 1.05 * np.cos(th), rmax * 1.05 * np.sin(th), 0, c=FS.GREY, lw=0.6)
    for d in dead:
        dead_wedge(ax, rmax * 1.05, *d)
    for b in blocks:
        shell3d(ax, *b[:6], color=b[6], alpha=b[7], rays=False)
    L = rmax * 1.2
    ax.plot([0, 0], [0, 0], [0, L * 0.9], c=FS.BLACK, lw=1.3, zorder=1)
    labs = []
    for az, name in ((0, "0°\ncaudal"), (90, "90°\nleft"), (180, "180°\ncranial"), (270, "270°\nsurgeon")):
        ax.plot(*_xyz(np.array([0, rmax * (1.5 if az in (0, 180) else 1.3)]), 0, az), c=FS.BLACK, lw=1.0, zorder=2)      # horizontal axis line
        labs.append((_xyz(rmax * (2.0 if az in (0, 180) else 1.62), 0, az), name))
    ax.text(0, 0, L, "up", fontsize=8, ha="center", color=FS.SLATE)
    ax.plot([0], [0], [0], "o", c=FS.BLACK, ms=5)                     # phantom origin
    for txt, (r, e, a) in labels:
        x, y, z = _xyz(r, e, a)
        ax.text(x, y, z, txt, fontsize=8.5, fontweight="bold", color=FS.NAVY, ha="center", va="center", zorder=10)
    ax.set_xlim(-rmax * 1.25, rmax * 1.25); ax.set_ylim(-rmax * 1.25, rmax * 1.25); ax.set_zlim(0, rmax * 1.15)
    ax.set_box_aspect((1, 1, 0.5)); ax.set_axis_off()
    ax.view_init(*view)
    from mpl_toolkits.mplot3d import proj3d
    for (x, y, z), name in labs:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.text2D(x2, y2, name, transform=ax.transData, fontsize=8, ha="center", va="center", color=FS.SLATE,
                  zorder=50, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.figure.text(0.25 if title.startswith("(a)") else 0.75, 0.78, title, fontsize=10.5, fontweight="bold", ha="center")


def fig_envelope3d(out, a):
    fig = plt.figure(figsize=(W, 4.0))
    p1 = [(33, 124, 0, 52, 325, 120, FS.PART, 0.07), (55, 100, 10, 35, 30, 55, FS.FULL, 0.45)]
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    env3d_panel(ax, "(a) P1 bracket", 124, p1, [(150, 300)],
                [], view=(30, -62))
    p2 = [(45, 62.5, 25, 80, 315, 255, FS.PART, 0.07), (45, 57.5, 35, 80, 315, 255, FS.FULL, 0.45)]
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    env3d_panel(ax, "(b) P2 dome", 62.5, p2, [(255, 285)],
                [], view=(30, -62))
    leg = [Patch(fc=FS.FULL, label="Full (P1 full-accuracy tier, P2 PnP 95% or more)"),
           Patch(fc=FS.PART, label="Partial (P1 tracking holds, P2 PnP 50 to 95%)"),
           Patch(fc=FS.NONE, ec=FS.SLATE, label="No tracking (azimuth dead zone)")]
    fig.legend(handles=leg, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.075), fontsize=8.5)
    fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0.2, wspace=0)
    from matplotlib.transforms import Bbox
    Wf, Hf = fig.get_size_inches()
    box = Bbox.from_extents(0, 0.065 * Hf, Wf, Hf * 0.82)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_envelope_3d.{ext}", bbox_inches=box, pad_inches=0)
    plt.close(fig); print("  wrote fig_envelope_3d.pdf / .png")


def fig_p2range(out, a):
    pick = load_p2_sweeps(a.p2_atp3)
    fig = plt.figure(figsize=(W, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.15], hspace=0.62, wspace=0.28)

    # (a) azimuth grid
    ax = fig.add_subplot(gs[0, :])
    rows_by = {el: sorted(pick(f"azimuth el {el}"), key=lambda r: flt(r["az_deg"])) for el in (45, 30)}
    for yi, el in enumerate((30, 45)):
        for r in rows_by[el]:
            az = flt(r["az_deg"]); p = flt(r["pnp_pct"]); c = cls(p)
            ax.add_patch(Rectangle((az - 15, yi - 0.45), 30, 0.9, fc=c, ec="white", lw=1.5,
                                   hatch="///" if c == FS.NONE else None))
            ax.text(az, yi, f"{p:.0f}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if c == FS.FULL else FS.BLACK)
    ax.add_patch(Rectangle((255, -0.5), 30, 2.0, fc="none", ec=FS.BLACK, lw=1.6, zorder=5))
    ax.text(270, 1.62, "surgeon's right", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-15, 345); ax.set_ylim(-0.55, 1.95)
    ax.set_xticks(range(0, 360, 30)); ax.set_yticks([0, 1]); ax.set_yticklabels(["el 30°", "el 45°"])
    ax.set_xlabel("Azimuth (°)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) PnP success (%) around the dome at 600 mm", fontsize=10)

    def sweep_panel(ax, xs, ps, xlabel, title, fill_to=None):
        o = np.argsort(xs); xs = np.array(xs)[o]; ps = np.array(ps)[o]
        ax.axhspan(95, 103, color=FS.FULL, alpha=0.12, lw=0)
        ax.axhline(95, c=FS.BLACK, ls="--", lw=1.0)
        ax.plot(xs, ps, "-", c=FS.GREY, lw=1.0, zorder=2)
        ax.scatter(xs, ps, c=[cls(p) for p in ps], s=48, edgecolor=FS.SLATE, lw=0.7, zorder=3)
        ax.set_ylim(-4, 103); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
        ax.text(xs.max(), 90, "95%", ha="right", va="top", fontsize=8.5, fontweight="bold")
        ax.yaxis.grid(True, zorder=0)

    d = pick("distance")
    ax = fig.add_subplot(gs[1, 0])
    sweep_panel(ax, [flt(r["dist_mm"]) / 10 for r in d], [flt(r["pnp_pct"]) for r in d], "Distance (cm)",
                "(b) Distance, el 45°, az 270°")
    ax.set_ylabel("PnP success (%)")
    e = pick("elevation")
    ax = fig.add_subplot(gs[1, 1])
    sweep_panel(ax, [flt(r["elev_deg"]) for r in e], [flt(r["pnp_pct"]) for r in e], "Elevation (°)",
                "(c) Elevation, 500 mm, az 270°")
    leg = [Patch(fc=FS.FULL, label="95% or more"), Patch(fc=FS.PART, label="50 to 95%"),
           Patch(fc=FS.NONE, hatch="///", ec="white", label="below 50%")]
    fig.legend(handles=leg, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    save(fig, out, "fig_p2_range")


def fig_p2range_row(out, a):
    pick = load_p2_sweeps(a.p2_atp3)
    fig = plt.figure(figsize=(W * 1.55, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 1, 1], wspace=0.3, left=0.04, right=0.995, bottom=0.27, top=0.84)

    # (a) azimuth grid
    ax = fig.add_subplot(gs[0, 0])
    rows_by = {el: sorted(pick(f"azimuth el {el}"), key=lambda r: flt(r["az_deg"])) for el in (45, 30)}
    for yi, el in enumerate((30, 45)):
        for r in rows_by[el]:
            az = flt(r["az_deg"]); p = flt(r["pnp_pct"]); c = cls(p)
            ax.add_patch(Rectangle((az - 15, yi - 0.45), 30, 0.9, fc=c, ec="white", lw=1.5,
                                   hatch="///" if c == FS.NONE else None))
            ax.text(az, yi, f"{p:.0f}", ha="center", va="center", fontsize=8.8, fontweight="bold",
                    color="white" if c == FS.FULL else FS.BLACK)
    ax.add_patch(Rectangle((255, -0.5), 30, 2.0, fc="none", ec=FS.BLACK, lw=1.6, zorder=5))
    ax.text(270, 1.62, "surgeon's right", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-15, 345); ax.set_ylim(-0.55, 1.95)
    ax.set_xticks(range(0, 360, 30)); ax.set_yticks([0, 1]); ax.set_yticklabels(["el 30°", "el 45°"])
    ax.set_xlabel("Azimuth (°)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) Around the dome at 600 mm", fontsize=10)

    def sweep_panel(ax, xs, ps, xlabel, title, fill_to=None):
        o = np.argsort(xs); xs = np.array(xs)[o]; ps = np.array(ps)[o]
        ax.axhspan(95, 103, color=FS.FULL, alpha=0.12, lw=0)
        ax.axhline(95, c=FS.BLACK, ls="--", lw=1.0)
        ax.plot(xs, ps, "-", c=FS.GREY, lw=1.0, zorder=2)
        ax.scatter(xs, ps, c=[cls(p) for p in ps], s=48, edgecolor=FS.SLATE, lw=0.7, zorder=3)
        ax.set_ylim(-4, 103); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
        ax.text(xs.max(), 90, "95%", ha="right", va="top", fontsize=8.5, fontweight="bold")
        ax.yaxis.grid(True, zorder=0)

    d = pick("distance")
    ax = fig.add_subplot(gs[0, 1])
    sweep_panel(ax, [flt(r["dist_mm"]) / 10 for r in d], [flt(r["pnp_pct"]) for r in d], "Distance (cm)",
                "(b) Distance")
    ax.set_ylabel("PnP success (%)")
    e = pick("elevation")
    ax = fig.add_subplot(gs[0, 2])
    sweep_panel(ax, [flt(r["elev_deg"]) for r in e], [flt(r["pnp_pct"]) for r in e], "Elevation (°)",
                "(c) Elevation")
    leg = [Patch(fc=FS.FULL, label="95% or more"), Patch(fc=FS.PART, label="50 to 95%"),
           Patch(fc=FS.NONE, hatch="///", ec="white", label="below 50%")]
    fig.legend(handles=leg, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    save(fig, out, "fig_p2_range_row")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", nargs="*", default=None, choices=["calib", "atp2", "grids", "envelope", "env3d", "p2range", "p2range_row"])
    ap.add_argument("--p1-cal", default="data/atp/atp2_crisp.csv", dest="p1_cal")
    ap.add_argument("--p1-drift1", default="data/atp/atp2_tiers_hidden.csv", dest="p1_drift1")
    ap.add_argument("--p1-drift2", default="data/atp/atp2_tiers_capture2.csv", dest="p1_drift2")
    ap.add_argument("--p2-f35", default="data/atp_v2/atp2_dome.csv", dest="p2_f35")
    ap.add_argument("--p2-f50", default="data/atp_v2/atp2_dome_f50.csv", dest="p2_f50")
    ap.add_argument("--p2-atp3", default="data/atp_v2/atp3_summary.csv", dest="p2_atp3")
    a = ap.parse_args()
    a.p1_cal = [a.p1_cal]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    figs = {"calib": fig_calib, "atp2": fig_atp2, "grids": fig_grids, "envelope": fig_envelope, "env3d": fig_envelope3d, "p2range": fig_p2range, "p2range_row": fig_p2range_row}
    for k in (a.only or list(figs)):
        print(f"[{k}]")
        figs[k](out, a)


if __name__ == "__main__":
    main()
