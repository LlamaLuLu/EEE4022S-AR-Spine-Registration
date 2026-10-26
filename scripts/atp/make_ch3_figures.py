#!/usr/bin/env python3
"""
Chapter 3 figures in the same look as the Chapter 4 figures (shared palette in fig_style.py).

  python scripts/atp/make_ch3_figures.py                       # all
  python scripts/atp/make_ch3_figures.py --only cycles states  # some
  python scripts/atp/make_ch3_figures.py --out docs/figures

  cycles        fig:method:cycles       method_cycles.pdf
  architecture  fig:method:arch         method_architecture.pdf
  states        fig:method:states       fig_method_states.pdf     (checked against ar_tracker.py)
  visibility    fig:method:visibility   fig_method_visibility.pdf (replaces the version in make_method_figures.py)
  ui            fig:method:hud          ui_v2_annotated.png       (numbered circles; needs --ui-src)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse, Arc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as FS  # noqa: E402

FS.apply()
W = FS.W
INK = FS.BLACK
# tints used behind the two prototypes (P1 blue, P2 green) and the design/runtime halves
P1_TINT, P1_EDGE = "#E3EEF8", FS.BLUE
P2_TINT, P2_EDGE = "#E1F3EA", FS.GREEN
P2_TEXT = "#1F7A52"
FIT = dict(th50=75.9, s_th=1.5, w50=12.6, s_w=6.3)      # mc_validation.py fit
FOCAL_PX = 0.5 * (1305.88 + 1302.46)                     # camera_calibration.npz
MARKER_MM = 20.0


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------------------
# drawing helpers (canvas units: 100 per inch, y up)
# ---------------------------------------------------------------------------------------
def canvas(w_units, h_units):
    fig = plt.figure(figsize=(w_units / 100, h_units / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_units); ax.set_ylim(0, h_units); ax.axis("off")
    return fig, ax


def rbox(ax, cx, cy, w, h, fc, ec, lw=1.3, r=7, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def arr(ax, p0, p1, color=FS.SLATE, lw=1.3, style="-|>", ls="-", ms=9, z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 zorder=z, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def poly(ax, pts, color=FS.SLATE, lw=1.3, z=4, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="butt", solid_joinstyle="round")


def tag(ax, x, y, text, color=FS.SLATE, fs=8.5, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    return ax.text(x, y, text, fontsize=fs, color=color, zorder=6, linespacing=1.25, **kw)


# ---------------------------------------------------------------------------------------
# 1. Two design cycles (swim lanes)
# ---------------------------------------------------------------------------------------
def fig_cycles(out):
    Wu, Hu = 630, 290
    fig, ax = canvas(Wu, Hu)
    lanes = [
        dict(y0=164, y1=286, edge=P1_EDGE, tint=P1_TINT, name="Cycle 1: L-bracket",
             hyp="H1: a spread, eight-marker bracket reaches a mean overlay error below 2 mm",
             steps=["Select layout\n(Fitzpatrick TRE,\nV0 to V3)", "Build P1\n(V3 bracket)",
                    "Test P1\n(ATP-1 to 8)", "Limits found:\n5 of 56 poses,\ndead zone"]),
        dict(y0=4, y1=126, edge=P2_EDGE, tint=P2_TINT, name="Cycle 2: dome",
             hyp="H2: a dome widens coverage and survives hands, at the same accuracy",
             steps=["Fit simulator\nto P1 data", "Design dome\n(D1 to D4)",
                    "Build P2\n(D1_final)", "Test P2\n(predictions,\nATP-1 to 8)"]),
    ]
    bw, bh, gap = 120, 74, 24
    x_first = 50 + (Wu - 6 - 14 - 50 - (4 * bw + 3 * gap)) / 2
    centres = []
    for L in lanes:
        # lane body and left label strip
        ax.add_patch(FancyBboxPatch((6, L["y0"]), Wu - 12, L["y1"] - L["y0"], boxstyle="round,pad=0,rounding_size=8",
                                    fc=L["tint"], ec=L["edge"], lw=1.2, zorder=1))
        ax.add_patch(FancyBboxPatch((6, L["y0"]), 28, L["y1"] - L["y0"], boxstyle="round,pad=0,rounding_size=8",
                                    fc=L["edge"], ec=L["edge"], lw=1.2, zorder=2))
        ax.add_patch(Rectangle((22, L["y0"]), 12, L["y1"] - L["y0"], fc=L["edge"], ec="none", zorder=2))
        ax.text(20, (L["y0"] + L["y1"]) / 2, L["name"], rotation=90, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color="white", zorder=5)
        tag(ax, 336, L["y1"] - 12, L["hyp"], color=INK, fs=8.6, style="italic")
        cy = L["y0"] + 54
        xs = [x_first + i * (bw + gap) + bw / 2 for i in range(4)]
        centres.append((xs, cy))
        for i, (x, t) in enumerate(zip(xs, L["steps"])):
            rbox(ax, x, cy, bw, bh, "white", L["edge"], lw=1.5)
            tag(ax, x, cy - 2, t, color=INK, fs=8.8)
            ax.add_patch(Circle((x - bw / 2 + 11, cy + bh / 2 - 11), 8, fc=L["edge"], ec="white", lw=1.0, zorder=5))
            tag(ax, x - bw / 2 + 11, cy + bh / 2 - 11, str(i + 1), color="white", fs=8.5, fontweight="bold")
            if i < 3:
                arr(ax, (x + bw / 2 + 2, cy), (xs[i + 1] - bw / 2 - 2, cy), color=L["edge"], lw=1.6)
    # hand-off from the last step of Cycle 1 to the first step of Cycle 2
    (xs1, cy1), (xs2, cy2) = centres
    ymid = 145
    poly(ax, [(xs1[3], cy1 - bh / 2 - 2), (xs1[3], ymid), (xs2[0], ymid)], color=FS.SLATE, lw=1.6)
    arr(ax, (xs2[0], ymid), (xs2[0], cy2 + bh / 2 + 2), color=FS.SLATE, lw=1.6)
    tag(ax, (xs1[3] + xs2[0]) / 2, ymid + 9, "limits of P1 set the design goals for P2", color=INK, fs=8.8,
        style="italic")
    save(fig, out, "method_cycles")


# ---------------------------------------------------------------------------------------
# 2. System architecture (grouped, with simple line icons)
# ---------------------------------------------------------------------------------------
def ic_ct(ax, x, y, c, a):
    for k in (-11, 0, 11):
        ax.add_patch(Polygon([(x - 16, y + k - 4), (x - 8, y + k + 4), (x + 16, y + k + 4), (x + 8, y + k - 4)],
                             fc="white", ec=c, lw=1.5, zorder=5, joinstyle="round"))
    ax.add_patch(Polygon([(x - 16, y - 4), (x - 8, y + 4), (x + 16, y + 4), (x + 8, y - 4)], fc=a, ec=c, lw=1.5,
                         alpha=0.55, zorder=6, joinstyle="round"))


def ic_segment(ax, x, y, c, a):
    ax.add_patch(FancyBboxPatch((x - 16, y - 14), 32, 28, boxstyle="round,pad=0,rounding_size=3", fc="white",
                                ec=c, lw=1.5, zorder=5))
    ax.add_patch(Ellipse((x - 1, y), 20, 14, angle=18, fc=a, ec=c, lw=1.3, alpha=0.75, zorder=6))
    ax.add_patch(Circle((x + 7, y + 4), 2.2, fc=c, ec="none", zorder=7))


def ic_cube(ax, x, y, c, a):
    top = [(x, y + 15), (x + 15, y + 7), (x, y - 1), (x - 15, y + 7)]
    left = [(x - 15, y + 7), (x, y - 1), (x, y - 17), (x - 15, y - 9)]
    right = [(x, y - 1), (x + 15, y + 7), (x + 15, y - 9), (x, y - 17)]
    for pts, fc, al in ((top, "white", 1.0), (left, a, 0.35), (right, a, 0.7)):
        ax.add_patch(Polygon(pts, fc=fc, ec=c, lw=1.5, alpha=1.0 if al == 1.0 else al, zorder=5, joinstyle="round"))
    for pts in (top, left, right):
        ax.add_patch(Polygon(pts, fc="none", ec=c, lw=1.5, zorder=6, joinstyle="round"))


def ic_board(ax, x, y, c, a):
    ax.add_patch(FancyBboxPatch((x - 16, y - 15), 32, 30, boxstyle="round,pad=0,rounding_size=3", fc="white",
                                ec=c, lw=1.5, zorder=5))
    for dx, dy in ((-9, 6), (3, 6), (-9, -6), (3, -6)):
        ax.add_patch(Rectangle((x + dx - 3, y + dy - 3), 6, 6, fc=c, ec="none", zorder=6))
    ax.add_patch(Circle((x + 12, y - 11), 6.5, fc=a, ec="white", lw=1.2, zorder=7))
    poly(ax, [(x + 9, y - 11), (x + 11.2, y - 13.5), (x + 15, y - 8.5)], color="white", lw=1.6, z=8)


def ic_camera(ax, x, y, c, a):
    ax.add_patch(FancyBboxPatch((x - 17, y - 12), 34, 24, boxstyle="round,pad=0,rounding_size=4", fc="white",
                                ec=c, lw=1.5, zorder=5))
    ax.add_patch(Polygon([(x - 7, y + 12), (x - 4, y + 17), (x + 4, y + 17), (x + 7, y + 12)], fc="white", ec=c,
                         lw=1.5, zorder=5))
    ax.add_patch(Circle((x, y), 8, fc="white", ec=c, lw=1.5, zorder=6))
    ax.add_patch(Circle((x, y), 3.6, fc=a, ec="none", zorder=7))


def ic_aruco(ax, x, y, c, a):
    n, s = 5, 5.6
    pat = ["11111", "10010", "11101", "10100", "11111"]
    x0, y0 = x - n * s / 2, y - n * s / 2
    ax.add_patch(Rectangle((x0, y0), n * s, n * s, fc=c, ec="none", zorder=5))
    for r, row in enumerate(pat[1:-1], start=1):
        for q, v in enumerate(row[1:-1], start=1):
            if v == "1":
                ax.add_patch(Rectangle((x0 + q * s, y0 + (n - 1 - r) * s), s, s, fc="white", ec="none", zorder=6))
    for cx_, cy_ in ((x0, y0), (x0 + n * s, y0), (x0, y0 + n * s), (x0 + n * s, y0 + n * s)):
        ax.add_patch(Circle((cx_, cy_), 2.4, fc=a, ec="white", lw=0.8, zorder=7))


def ic_pose(ax, x, y, c, a):
    o = (x - 7, y - 8)
    arr(ax, o, (o[0], o[1] + 24), color=FS.NAVY, lw=1.8, ms=9)
    arr(ax, o, (o[0] + 24, o[1]), color=a, lw=1.8, ms=9)
    arr(ax, o, (o[0] + 15, o[1] + 15), color=FS.BLUE, lw=1.8, ms=9)
    ax.add_patch(Circle(o, 2.8, fc=c, ec="none", zorder=7))


def ic_target(ax, x, y, c, a):
    ax.add_patch(Circle((x, y), 12, fc="white", ec=c, lw=1.5, zorder=5))
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        poly(ax, [(x + dx * 8, y + dy * 8), (x + dx * 18, y + dy * 18)], color=c, lw=1.5, z=5)
    poly(ax, [(x - 7, y - 6), (x + 3, y + 4)], color=a, lw=1.6, z=6)
    ax.add_patch(Circle((x + 3, y + 4), 3.0, fc=a, ec="none", zorder=7))


def ic_screen(ax, x, y, c, a):
    ax.add_patch(FancyBboxPatch((x - 18, y - 8), 36, 26, boxstyle="round,pad=0,rounding_size=3", fc="white",
                                ec=c, lw=1.5, zorder=5))
    poly(ax, [(x, y - 8), (x, y - 14)], color=c, lw=1.5)
    poly(ax, [(x - 8, y - 14), (x + 8, y - 14)], color=c, lw=1.5)
    for r, col in ((9, FS.SKY), (5.5, a), (2.2, c)):
        ax.add_patch(Circle((x, y + 5), r, fc="white" if r > 2.3 else c, ec=col, lw=1.5, zorder=6))


def fig_architecture(out):
    Wu, Hu = 630, 328
    fig, ax = canvas(Wu, Hu)
    xs = [69, 192, 315, 438, 561]
    bw, bh = 100, 96
    groups = [
        dict(name="Design time (once)", x0=8, x1=504, y0=176, y1=304, tint=P1_TINT, edge=FS.BLUE, ink=FS.NAVY,
             items=[("CT volume\n(VerSe2020)", ic_ct), ("Segment and\nannotate\n(Slicer)", ic_segment),
                    ("Phantom and\nmount (Blender,\n3D print)", ic_cube), ("Board selection\n(TRE and\nMonte Carlo)", ic_board)]),
        dict(name="Runtime (every frame)", x0=8, x1=622, y0=14, y1=142, tint=P2_TINT, edge=FS.GREEN, ink=P2_TEXT,
             items=[("Calibrated\ncamera frame", ic_camera), ("ArUco\ndetection", ic_aruco),
                    ("Multi-marker\nPnP pose", ic_pose), ("Targets into\ncamera frame", ic_target),
                    ("AR overlay and\nTRE indicator", ic_screen)]),
    ]
    tops = {}
    for g in groups:
        ax.add_patch(FancyBboxPatch((g["x0"], g["y0"]), g["x1"] - g["x0"], g["y1"] - g["y0"],
                                    boxstyle="round,pad=0,rounding_size=10", fc=g["tint"], ec=g["edge"], lw=1.2, zorder=1))
        pw = 8.4 * len(g["name"]) * 0.93 + 22
        rbox(ax, g["x0"] + 16 + pw / 2, g["y1"], pw, 20, g["edge"], g["edge"], lw=1, r=10, z=3)
        tag(ax, g["x0"] + 16 + pw / 2, g["y1"], g["name"], color="white", fs=9.2, fontweight="bold")
        cy = (g["y0"] + g["y1"]) / 2 - 7
        for i, (lab, icon) in enumerate(g["items"]):
            x = xs[i]
            rbox(ax, x, cy, bw, bh, "white", g["edge"], lw=1.4)
            icon(ax, x, cy + 18, g["ink"], g["edge"])
            tag(ax, x, cy - 26, lab, color=INK, fs=8.2)
            if i < len(g["items"]) - 1:
                arr(ax, (x + bw / 2 + 3, cy), (xs[i + 1] - bw / 2 - 3, cy), color=FS.SLATE, lw=1.5)
        tops[g["name"]] = cy
    # marker layout hand-off: Board selection (design) -> PnP pose (runtime)
    cy_d, cy_r = tops["Design time (once)"], tops["Runtime (every frame)"]
    ymid = 159
    poly(ax, [(xs[3], cy_d - bh / 2 - 2), (xs[3], ymid), (xs[2], ymid)], color=FS.SLATE, lw=1.5)
    arr(ax, (xs[2], ymid), (xs[2], cy_r + bh / 2 + 2), color=FS.SLATE, lw=1.5)
    tag(ax, xs[3] + 8, ymid + 5, "marker layout", color=INK, fs=8.4, style="italic", ha="left")
    save(fig, out, "method_architecture")


# ---------------------------------------------------------------------------------------
# 3. State machine of the overlay (checked against scripts/registration/ar_tracker.py)
# ---------------------------------------------------------------------------------------
def fig_states(out):
    Wu, Hu = 630, 300
    fig, ax = canvas(Wu, Hu)
    bw, bh = 128, 64
    cyr = 196
    st = {
        "Searching": dict(c=(85, cyr), fc=FS.PALE, ec=FS.SLATE, sub="no pose yet", ls="-"),
        "Verifying": dict(c=(315, cyr), fc="#DCEAF6", ec=FS.BLUE, sub="check the outline", ls="-"),
        "Navigating": dict(c=(545, cyr), fc="#D5EFE5", ec=FS.TEAL, sub="guidance shown", ls="-"),
        "Stale": dict(c=(430, 66), fc="white", ec=FS.GREY, sub="last pose held", ls=(0, (4, 2.5))),
    }
    for k, s in st.items():
        cx, cy = s["c"]
        rbox(ax, cx, cy, bw, bh, s["fc"], s["ec"], lw=1.7, r=8, ls=s["ls"])
        tag(ax, cx, cy + 9, k, color=INK, fs=10.5, fontweight="bold")
        tag(ax, cx, cy - 12, s["sub"], color=FS.SLATE, fs=8.4)
    # forward path
    arr(ax, (85 + bw / 2 + 3, cyr), (315 - bw / 2 - 3, cyr), color=FS.SLATE, lw=1.6)
    tag(ax, 200, cyr + 14, "pose recovered", color=INK, fs=8.6)
    arr(ax, (315 + bw / 2 + 3, cyr), (545 - bw / 2 - 3, cyr), color=FS.SLATE, lw=1.6)
    tag(ax, 430, cyr + 24, "1.0 s elapsed", color=INK, fs=8.6)
    tag(ax, 430, cyr + 12, "or key C", color=INK, fs=8.6)
    # pose lost (down) and pose back within 0.3 s (up): two separate arrows per side
    arr(ax, (296, cyr - bh / 2 - 2), (380, 66 + bh / 2 + 2), color=FS.SLATE, lw=1.5, ms=9)
    arr(ax, (412, 66 + bh / 2 + 2), (328, cyr - bh / 2 - 2), color=FS.SLATE, lw=1.5, ms=9)
    arr(ax, (564, cyr - bh / 2 - 2), (480, 66 + bh / 2 + 2), color=FS.SLATE, lw=1.5, ms=9)
    arr(ax, (448, 66 + bh / 2 + 2), (532, cyr - bh / 2 - 2), color=FS.SLATE, lw=1.5, ms=9)
    tag(ax, 312, 134, "pose lost", color=INK, fs=8.6, ha="right")
    tag(ax, 548, 134, "pose lost", color=INK, fs=8.6, ha="left")
    tag(ax, 430, 131, "pose back\nwithin 0.3 s", color=INK, fs=8.6)
    tag(ax, 430, 18, "a pose that returns resumes the state it left", color=FS.SLATE, fs=8.2)
    # Stale -> Searching
    poly(ax, [(430 - bw / 2 - 2, 66), (85, 66)], color=FS.SLATE, lw=1.5)
    arr(ax, (85, 66), (85, cyr - bh / 2 - 3), color=FS.SLATE, lw=1.5)
    tag(ax, 230, 80, "no pose for 0.3 s", color=INK, fs=8.6)
    # reset from any state
    poly(ax, [(545, cyr + bh / 2 + 2), (545, 268), (85, 268), (85, 246)], color=FS.GREY, lw=1.3, ls=(0, (4, 3)))
    arr(ax, (85, 247), (85, cyr + bh / 2 + 3), color=FS.GREY, lw=1.6, ms=10)
    rbox(ax, 315, 268, 300, 18, "white", "white", lw=0, r=4, z=5)
    tag(ax, 315, 268, "key R or board change: reset from any state", color=FS.SLATE, fs=8.6)
    save(fig, out, "fig_method_states")


# ---------------------------------------------------------------------------------------
# 4. Marker visibility model (decluttered)
# ---------------------------------------------------------------------------------------
def logistic_down(x, x50, sc):
    return 1.0 / (1.0 + np.exp((x - x50) / sc))


def logistic_up(x, x50, sc):
    return 1.0 / (1.0 + np.exp(-(x - x50) / sc))


def fig_visibility(out):
    fig = plt.figure(figsize=(W, 4.3))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.1, top=0.945, wspace=0.34, hspace=0.55)
    th50, s_th, w50, s_w = FIT["th50"], FIT["s_th"], FIT["w50"], FIT["s_w"]

    # (a) geometry: camera below, marker above, tilted by theta
    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.4, 3.0); ax.set_ylim(-3.9, 1.7)
    th = np.deg2rad(50)
    nvec = np.array([np.sin(th), -np.cos(th)]); tvec = np.array([np.cos(th), np.sin(th)])
    p0, p1 = -0.85 * tvec, 0.85 * tvec
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=FS.SLATE, lw=5, solid_capstyle="butt", zorder=4)
    ax.text(0.75, 1.0, "marker", fontsize=9, color=FS.SLATE, ha="left", va="center")
    ax.add_patch(FancyArrowPatch((0, 0), tuple(1.45 * nvec), arrowstyle="-|>", mutation_scale=10, color=FS.TEAL, lw=1.8, zorder=5))
    ax.text(1.45 * nvec[0] + 0.1, 1.45 * nvec[1] - 0.05, "normal", color=FS.TEAL, fontsize=9, ha="left", va="center")
    ax.plot([0, 0], [0, -2.0], color=FS.GREY, lw=1.1, ls=(0, (4, 3)))
    ax.add_patch(Arc((0, 0), 1.6, 1.6, theta1=-90, theta2=-40, color=FS.TEAL, lw=1.4))
    ax.text(0.24, -1.1, r"$\theta$", color=FS.TEAL, fontsize=12, ha="center", va="center")
    yp = -2.0
    ax.plot([-1.7, 1.7], [yp, yp], color=FS.BLUE, lw=2.0)
    for pt in (p0, p1):
        ax.plot([pt[0], pt[0]], [pt[1], yp], color=FS.BLUE, lw=0.9, ls=(0, (2, 2)))
    ax.annotate("", xy=(p0[0], yp - 0.28), xytext=(p1[0], yp - 0.28), arrowprops=dict(arrowstyle="<->", color=FS.BLUE, lw=1.2))
    ax.text(0, yp - 0.62, r"$w\propto\cos\theta$", color=FS.BLUE, fontsize=9.5, ha="center", va="center")
    ax.text(1.85, yp, "image", color=FS.BLUE, fontsize=9, ha="left", va="center")
    ax.add_patch(Polygon([[0, -3.45], [-0.3, -3.9], [0.3, -3.9]], fc=FS.BLUE, ec="none"))
    ax.text(0.5, -3.7, "camera", color=FS.BLUE, fontsize=9, ha="left", va="center")
    ax.set_title("(a) What the detector sees", loc="left", fontsize=10)

    # (b) view angle
    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 90, 400)
    ax.plot(x, logistic_down(x, th50, s_th), color=FS.TEAL, lw=2.2)
    ax.plot([th50], [0.5], "o", color=FS.TEAL, ms=6, zorder=5)
    ax.plot([th50, th50], [0, 0.5], color=FS.GREY, lw=0.9, ls=(0, (3, 3)))
    ax.text(th50 - 3, 0.5, f"50% at {th50:.1f}°", ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 90); ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel(r"view angle $\theta$ (degrees)"); ax.set_ylabel("detection probability")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b) Falls with view angle", loc="left", fontsize=10)

    # (c) apparent width
    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(0, 50, 400)
    ax.plot(x, logistic_up(x, w50, s_w), color=FS.BLUE, lw=2.2)
    ax.plot([w50], [0.5], "o", color=FS.BLUE, ms=6, zorder=5)
    ax.plot([w50, w50], [0, 0.5], color=FS.GREY, lw=0.9, ls=(0, (3, 3)))
    ax.text(w50 + 2.8, 0.5, f"50% at {w50:.1f} px", ha="left", va="center", fontsize=9, color=INK)
    for dist in (100, 60):
        wf = FOCAL_PX * MARKER_MM / (dist * 10)
        ax.plot([wf, wf], [0, 0.07], color=INK, lw=1.6)
        ax.text(wf, 0.1, f"marker at\n{dist} cm", ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.15)
    ax.set_xlim(0, 50); ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel(r"apparent marker width $w$ (px)"); ax.set_ylabel("detection probability")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(c) Rises with apparent width", loc="left", fontsize=10)

    # (d) combined map
    ax = fig.add_subplot(gs[1, 1])
    d_cm = np.linspace(30, 130, 300); th_deg = np.linspace(0, 89, 300)
    D, T = np.meshgrid(d_cm, th_deg)
    Wd = FOCAL_PX * MARKER_MM * np.cos(np.deg2rad(T)) / (D * 10.0)
    P = logistic_down(T, th50, s_th) * logistic_up(Wd, w50, s_w)
    im = ax.pcolormesh(D, T, P, cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
    cs = ax.contour(D, T, P, levels=[0.5], colors=[INK], linewidths=1.4)
    ax.clabel(cs, fmt={0.5: "50%"}, fontsize=9)
    ax.set_xlim(30, 130); ax.set_ylim(0, 89)
    ax.set_xlabel("distance (cm)"); ax.set_ylabel(r"view angle $\theta$ (degrees)")
    ax.set_title("(d) Both effects, one marker", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("detection probability", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    for a in fig.axes[1:]:
        a.tick_params(labelsize=9)
    save(fig, out, "fig_method_visibility")


def fig_visibility_row(out):
    fig = plt.figure(figsize=(10.0, 2.75))
    gs = fig.add_gridspec(1, 4, left=0.05, right=0.992, bottom=0.2, top=0.88, wspace=0.56)
    th50, s_th, w50, s_w = FIT["th50"], FIT["s_th"], FIT["w50"], FIT["s_w"]

    # (a) geometry: camera below, marker above, tilted by theta
    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.4, 3.0); ax.set_ylim(-3.9, 1.7)
    th = np.deg2rad(50)
    nvec = np.array([np.sin(th), -np.cos(th)]); tvec = np.array([np.cos(th), np.sin(th)])
    p0, p1 = -0.85 * tvec, 0.85 * tvec
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=FS.SLATE, lw=5, solid_capstyle="butt", zorder=4)
    ax.text(0.75, 1.0, "marker", fontsize=9, color=FS.SLATE, ha="left", va="center")
    ax.add_patch(FancyArrowPatch((0, 0), tuple(1.45 * nvec), arrowstyle="-|>", mutation_scale=10, color=FS.TEAL, lw=1.8, zorder=5))
    ax.text(1.45 * nvec[0] + 0.1, 1.45 * nvec[1] - 0.05, "normal", color=FS.TEAL, fontsize=9, ha="left", va="center")
    ax.plot([0, 0], [0, -2.0], color=FS.GREY, lw=1.1, ls=(0, (4, 3)))
    ax.add_patch(Arc((0, 0), 1.6, 1.6, theta1=-90, theta2=-40, color=FS.TEAL, lw=1.4))
    ax.text(0.24, -1.1, r"$\theta$", color=FS.TEAL, fontsize=12, ha="center", va="center")
    yp = -2.0
    ax.plot([-1.7, 1.7], [yp, yp], color=FS.BLUE, lw=2.0)
    for pt in (p0, p1):
        ax.plot([pt[0], pt[0]], [pt[1], yp], color=FS.BLUE, lw=0.9, ls=(0, (2, 2)))
    ax.annotate("", xy=(p0[0], yp - 0.28), xytext=(p1[0], yp - 0.28), arrowprops=dict(arrowstyle="<->", color=FS.BLUE, lw=1.2))
    ax.text(0, yp - 0.62, r"$w\propto\cos\theta$", color=FS.BLUE, fontsize=9.5, ha="center", va="center")
    ax.text(1.85, yp, "image", color=FS.BLUE, fontsize=9, ha="left", va="center")
    ax.add_patch(Polygon([[0, -3.45], [-0.3, -3.9], [0.3, -3.9]], fc=FS.BLUE, ec="none"))
    ax.text(0.5, -3.7, "camera", color=FS.BLUE, fontsize=9, ha="left", va="center")
    ax.set_title("(a) What the detector sees", loc="left", fontsize=10)

    # (b) view angle
    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 90, 400)
    ax.plot(x, logistic_down(x, th50, s_th), color=FS.TEAL, lw=2.2)
    ax.plot([th50], [0.5], "o", color=FS.TEAL, ms=6, zorder=5)
    ax.plot([th50, th50], [0, 0.5], color=FS.GREY, lw=0.9, ls=(0, (3, 3)))
    ax.text(th50 - 3, 0.5, f"50% at {th50:.1f}°", ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 90); ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel(r"view angle $\theta$ (degrees)"); ax.set_ylabel("detection probability")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b) Falls with view angle", loc="left", fontsize=10)

    # (c) apparent width
    ax = fig.add_subplot(gs[0, 2])
    x = np.linspace(0, 50, 400)
    ax.plot(x, logistic_up(x, w50, s_w), color=FS.BLUE, lw=2.2)
    ax.plot([w50], [0.5], "o", color=FS.BLUE, ms=6, zorder=5)
    ax.plot([w50, w50], [0, 0.5], color=FS.GREY, lw=0.9, ls=(0, (3, 3)))
    ax.text(w50 + 2.8, 0.5, f"50% at {w50:.1f} px", ha="left", va="center", fontsize=9, color=INK)
    for dist in (100, 60):
        wf = FOCAL_PX * MARKER_MM / (dist * 10)
        ax.plot([wf, wf], [0, 0.07], color=INK, lw=1.6)
        ax.text(wf, 0.1, f"marker\nat {dist} cm", ha="center", va="bottom", fontsize=8, color=INK, linespacing=1.15)
    ax.set_xlim(0, 50); ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel(r"apparent marker width $w$ (px)"); ax.set_ylabel("detection probability")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(c) Rises with width", loc="left", fontsize=10)

    # (d) combined map
    ax = fig.add_subplot(gs[0, 3])
    d_cm = np.linspace(30, 130, 300); th_deg = np.linspace(0, 89, 300)
    D, T = np.meshgrid(d_cm, th_deg)
    Wd = FOCAL_PX * MARKER_MM * np.cos(np.deg2rad(T)) / (D * 10.0)
    P = logistic_down(T, th50, s_th) * logistic_up(Wd, w50, s_w)
    im = ax.pcolormesh(D, T, P, cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
    cs = ax.contour(D, T, P, levels=[0.5], colors=[INK], linewidths=1.4)
    ax.clabel(cs, fmt={0.5: "50%"}, fontsize=9)
    ax.set_xlim(30, 130); ax.set_ylim(0, 89)
    ax.set_xlabel("distance (cm)"); ax.set_ylabel(r"view angle $\theta$ (degrees)")
    ax.set_title("(d) Both effects, one marker", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("detection probability", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    for a in fig.axes[1:]:
        a.tick_params(labelsize=9)
    save(fig, out, "fig_method_visibility_row")


# ---------------------------------------------------------------------------------------
# 5. UI screenshot: numbered callouts with bigger circles
# ---------------------------------------------------------------------------------------
def fig_ui(out, src):
    from PIL import Image, ImageDraw, ImageFont
    if not src or not Path(src).exists():
        print("  ui: skipped (pass --ui-src with the clean UI screenshot, 3024 x 1964)")
        return
    im = Image.open(src).convert("RGB"); d = ImageDraw.Draw(im)
    S = im.width / 2000.0
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            f = ImageFont.truetype(p, int(44 * S)); break
        except OSError:
            continue
    blue = (30, 90, 220); white = (255, 255, 255)
    items = [(1, (425, 175), (313, 175)), (2, (425, 338), (376, 338)), (3, (425, 555), (320, 555)),
             (4, (425, 745), (327, 745)), (5, (700, 300), (827, 300)), (6, (1198, 540), (1198, 615)),
             (7, (1800, 860), (1800, 935)), (8, (1400, 1150), (1503, 1150)), (9, (1700, 155), (1803, 155)),
             (10, (425, 925), (313, 925)), (11, (425, 1130), (340, 1130)), (12, (800, 1150), (800, 1235))]
    r = int(36 * S)
    for n, (x, y), (tx, ty) in items:
        x, y, tx, ty = [v * S for v in (x, y, tx, ty)]
        d.line([(x, y), (tx, ty)], fill=white, width=int(9 * S)); d.line([(x, y), (tx, ty)], fill=blue, width=int(5 * S))
        k = 9
        d.ellipse([tx - k - 3, ty - k - 3, tx + k + 3, ty + k + 3], fill=white); d.ellipse([tx - k, ty - k, tx + k, ty + k], fill=blue)
        d.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], fill=white); d.ellipse([x - r, y - r, x + r, y + r], fill=blue)
        s = str(n)
        m = Image.new("L", (int(120 * S), int(120 * S)), 0)
        ImageDraw.Draw(m).text((int(20 * S), int(20 * S)), s, fill=255, font=f)
        ib = m.getbbox()
        ox, oy = (ib[0] + ib[2]) / 2 - int(20 * S), (ib[1] + ib[3]) / 2 - int(20 * S)
        d.text((x - ox, y - oy), s, fill=white, font=f)
    im.save(out / "ui_v2_annotated.png")
    print("  wrote ui_v2_annotated.png")


# ---------------------------------------------------------------------------------------
# 5. Monte Carlo simulation flow (inputs -> simulated pose -> comparisons)
# ---------------------------------------------------------------------------------------
def fig_mcflow(out):
    Wu, Hu = 700, 500
    fig, ax = canvas(Wu, Hu)
    IN_F, IN_E = "#E3EEF8", FS.BLUE
    OUT_F, OUT_E = "#E1F3EA", FS.GREEN
    xi, xs, xo = 95, 350, 610
    wi, ws, wo = 170, 236, 176
    for x, t in ((xi, "Measured inputs"), (xs, "Simulation"), (xo, "Used for")):
        w = {xi: wi, xs: ws, xo: wo}[x] + 6
        ax.text(x, Hu - 16, t, fontsize=10.5, fontweight="bold", color=INK, ha="center", va="center")
        poly(ax, [(x - w / 2, Hu - 32), (x + w / 2, Hu - 32)], color=FS.LIGHT, lw=1.2)
    # simulation container
    ax.add_patch(FancyBboxPatch((xs - 134, 16), 268, Hu - 66, boxstyle="round,pad=0,rounding_size=9",
                                fc="#F7F8F9", ec=FS.LIGHT, lw=1.2, zorder=0))
    steps = [("Choose pose and markers", "visibility model decides\nwhich markers are seen"),
             ("Project the marker corners", "calibrated camera"),
             ("Add corner noise", "Gaussian, independent per axis"),
             ("Recover the pose", "solvePnP, as in the live pipeline"),
             ("Evaluate the targets", "3D TRE and overlay error"),
             ("RMS over 400 trials", "gives the error for this pose")]
    pitch, bh = 72, 58
    ys = [Hu - 82 - i * pitch for i in range(6)]
    for i, ((t, sub), y) in enumerate(zip(steps, ys)):
        rbox(ax, xs, y, ws, bh, "white", FS.BLUE, lw=1.6)
        ax.add_patch(Circle((xs - ws / 2 + 17, y), 9, fc=FS.BLUE, ec="white", lw=1.0, zorder=5))
        tag(ax, xs - ws / 2 + 17, y, str(i + 1), color="white", fs=8.8, fontweight="bold")
        tag(ax, xs + 12, y + 11, t, color=INK, fs=8.8, fontweight="bold")
        tag(ax, xs + 12, y - 11, sub, color=FS.SLATE, fs=8.0)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arr(ax, (xs, y0 - bh / 2), (xs, y1 + bh / 2 + 1), color=FS.BLUE, lw=1.6)
    # inputs feed steps 1, 2, 3 and 5
    ins = [(0, "Detection model", "fitted to the\nATP-3 sweeps"),
           (1, "Camera calibration", "v1: focal length,\ncentre, distortion"),
           (2, "Corner noise", "static 0.121 px\nlive 0.8 to 1.2 px"),
           (4, "Marker layouts, targets", "V0 to V3, ten\nentry points")]
    for k, t, sub in ins:
        y = ys[k]
        rbox(ax, xi, y, wi, bh + 4, IN_F, IN_E, lw=1.6)
        tag(ax, xi, y + 14, t, color=INK, fs=8.8, fontweight="bold")
        tag(ax, xi, y - 9, sub, color=FS.SLATE, fs=8.0)
        arr(ax, (xi + wi / 2 + 2, y), (xs - ws / 2 - 2, y), color=IN_E, lw=1.6)
    # outputs
    outs = [("Design comparison", "V0 to V3, 40 poses,\ngates of 2 to 5 markers"),
            ("Noise-path check", "single-marker error\nagainst the measured FLE"),
            ("Held-out occlusion test", "tracking and marker\ncounts against ATP-4"),
            ("Error budget", "random part against the\nmeasured ATP-2 overlay error")]
    oy = [ys[0], ys[1] - 4, ys[2] - 8, ys[3] - 12]
    oy = [Hu - 82 - i * 86 for i in range(4)]
    bus = 500
    poly(ax, [(xs + ws / 2 + 2, ys[5]), (bus, ys[5]), (bus, oy[0])], color=OUT_E, lw=1.6)
    for y, (t, sub) in zip(oy, outs):
        rbox(ax, xo, y, wo, 64, OUT_F, OUT_E, lw=1.6)
        tag(ax, xo, y + 16, t, color=INK, fs=8.8, fontweight="bold")
        tag(ax, xo, y - 9, sub, color=FS.SLATE, fs=8.0)
        arr(ax, (bus, y), (xo - wo / 2 - 2, y), color=OUT_E, lw=1.6)
    poly(ax, [(bus, oy[0]), (bus, oy[-1])], color=OUT_E, lw=1.6)
    save(fig, out, "fig_method_mcflow")


FIGS = {"cycles": fig_cycles, "architecture": fig_architecture, "states": fig_states, "visibility": fig_visibility,
        "mcflow": fig_mcflow, "visibility_row": fig_visibility_row}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", nargs="*", choices=list(FIGS) + ["ui"])
    ap.add_argument("--ui-src", default="docs/figures/ui_v2_clean.png")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in (a.only or list(FIGS) + ["ui"]):
        if k == "ui":
            fig_ui(out, a.ui_src)
        else:
            FIGS[k](out)


if __name__ == "__main__":
    main()