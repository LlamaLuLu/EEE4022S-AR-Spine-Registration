#!/usr/bin/env python3
"""
Methodology-chapter diagrams (no measured data needed).

    python scripts/atp/make_method_figures.py                 # all four -> docs/figures/
    python scripts/atp/make_method_figures.py --only states   # variants | states | mcflow | visibility

Outputs (PDF for LaTeX, PNG preview):
    fig_method_variants    marker layouts V0 to V3        (label fig:method:variants)
    fig_method_states      overlay state machine          (label fig:method:states)
    fig_method_mcflow      Monte Carlo simulation + checks (label fig:method:mcflow)
    fig_method_visibility  per-marker detection model     (label fig:method:visibility)

Numbers shown come from the project: marker layouts mirror scripts/tre_analysis/design_variants.py,
state timings mirror live_ar_overlay_qt.py, and the detection-model values are the fitted ones from
`mc_validation.py fit` (edit FIT below if you refit).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Arc

# ----------------------------------------------------------------------------- style
INK = "#1b1b1b"
GREY = "#6b6b6b"
LIGHT = "#d9d9d9"
BLUE, VERM, GREEN, AMBER, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"   # Okabe-Ito
A_TINT, A_EDGE = "#D9E9F5", BLUE        # Face A (caudal wall)
B_TINT, B_EDGE = "#FBE4D1", VERM        # Face B (patient-left wall)
HUD = {"SEARCHING": "#555F64", "VERIFYING": "#3CC3C3", "NAVIGATING": "#64D28C", "STALE": "#D75A5A"}  # ar_qt_styles.py

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42, "savefig.dpi": 300,
    "figure.dpi": 150, "axes.linewidth": 0.6,
})
W_IN = 6.3   # text width, inches

# Fitted detection model (mc_validation.py fit): 50% points and logistic scales
FIT = dict(th50=75.9, s_th=1.5, w50=12.6, s_w=6.3)
FOCAL_PX = 0.5 * (1305.88 + 1302.46)     # camera_calibration.npz
MARKER_MM = 20.0


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def box(ax, cx, cy, w, h, fc, ec, lw=1.0, r=6, z=2):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK, lw=1.0, style="-|>", rad=0.0, ls="-", z=3, ms=8):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


# ============================================================================ 1. variants
# (h, v) = distance along the face and down from its top edge, mm, as in design_variants.py
#   Face A: h from the junction edge outward.   Face B: h from the far (cranial) edge towards the junction.
V1 = dict(A=[(15, 15), (65, 15), (40, 65)], B=[(15, 15), (65, 15), (40, 65)])
V2 = dict(A=[(27, 27), (53, 27), (40, 53)], B=[(27, 27), (53, 27), (40, 53)])
V3 = dict(A=[(15, 15), (65, 15), (15, 65), (65, 65)], B=[(15, 15), (65, 15), (15, 65), (65, 65)])
V0 = [(0, 75), (0, -75), (40, 0), (-40, 0)]            # (X, Z) on the base plate


def fig_variants(out):
    Wm, Hm = 372.0, 360.0
    cw, ch = 180.0, 112.0
    s = W_IN / Wm
    fig = plt.figure(figsize=(Wm * s, Hm * s))
    ov = fig.add_axes([0, 0, 1, 1])                      # overlay: virtual-mm coordinates, y downwards
    ov.set_xlim(0, Wm); ov.set_ylim(Hm, 0); ov.axis("off")

    def cell(col, row, xlim, ylim):
        x0 = 3 + col * (cw + 6)
        y0 = 29 + row * (ch + 31)
        ax = fig.add_axes([x0 / Wm, 1 - (y0 + ch) / Hm, cw / Wm, ch / Hm])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        return ax, x0, y0

    def title(x0, y0, name, text, sub):
        ov.text(x0, y0 - 10.5, name, fontsize=10.5, fontweight="bold", color=INK, ha="left", va="bottom")
        ov.text(x0 + 17, y0 - 10.5, text, fontsize=8.8, color=INK, ha="left", va="bottom")
        ov.text(x0, y0 - 3.0, sub, fontsize=6.8, color=GREY, ha="left", va="bottom")

    def marker(ax, cx, cy, mid):
        ax.add_patch(Rectangle((cx - 10, cy - 10), 20, 20, fc=INK, ec="none", zorder=5))
        ax.text(cx, cy, str(mid), color="white", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=6)

    # ---- V0: plan view of the plate, cranial to the right, patient left at the top ----
    ax, x0, y0 = cell(0, 0, (-90, 90), (-56, 56))
    ax.add_patch(Rectangle((-83.6, -50.2), 163.0, 100.4, fc="#ececec", ec="#8a8a8a", lw=0.9, zorder=1))
    ax.add_patch(Rectangle((-77.5, -15.4), 155.4, 30.8, fc="#cfcfcf", ec="none", zorder=2))
    ax.text(0, -24, "spine (occludes the plate markers)", fontsize=6.6, color=GREY, ha="center", va="center", style="italic")
    for i, (X, Z) in enumerate(V0):
        marker(ax, Z, X, i)
    ax.text(0, 53.4, "patient left", fontsize=6.6, color=GREY, ha="center", va="center")
    ax.text(0, -53.4, "patient right", fontsize=6.6, color=GREY, ha="center", va="center")
    ax.text(-62, 37, "caudal end", fontsize=6.6, color=GREY, ha="center", va="center")
    ax.text(62, 37, "cranial end", fontsize=6.6, color=GREY, ha="center", va="center")
    title(x0, y0, "V0", "Flat baseline board", "4 coplanar markers on the base plate, seen from above")

    # ---- V1 to V3: unfolded L-bracket as seen from outside the corner ----
    def unfolded(col, row, spec, name, text, sub):
        ax, x0, y0 = cell(col, row, (-90, 90), (88, -24))
        ax.add_patch(Rectangle((-80, 0), 80, 80, fc=B_TINT, ec=B_EDGE, lw=1.0, zorder=1))
        ax.add_patch(Rectangle((0, 0), 80, 80, fc=A_TINT, ec=A_EDGE, lw=1.0, zorder=1))
        ax.plot([0, 0], [-3, 80], color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=2)
        ax.text(-40, -9, "Face B", fontsize=8, color=B_EDGE, fontweight="bold", ha="center", va="center")
        ax.text(40, -9, "Face A", fontsize=8, color=A_EDGE, fontweight="bold", ha="center", va="center")
        mid = 0
        for h, v in spec["A"]:
            marker(ax, h, v, mid); mid += 1
        for h, v in spec["B"]:
            marker(ax, -(80 - h), v, mid); mid += 1
        title(x0, y0, name, text, sub)
        return ax

    unfolded(1, 0, V1, "V1", "L-bracket, spread", "6 markers, centres 15 to 65 mm from the face edges")
    unfolded(0, 1, V2, "V2", "L-bracket, clustered", "6 markers, centres 27 to 53 mm from the face edges")
    unfolded(1, 1, V3, "V3", "L-bracket, 8 markers (fabricated)", "8 markers, centres 15 to 65 mm from the face edges")

    # ---- legend row ----
    ly = 29 + 2 * (ch + 31) - 22         # top of the legend row
    axi = fig.add_axes([8 / Wm, 1 - (ly + 62) / Hm, 104 / Wm, 62 / Hm])
    axi.set_xlim(-150, 92); axi.set_ylim(-56, 106); axi.set_aspect("equal", adjustable="box"); axi.axis("off")
    axi.add_patch(Rectangle((-83.6, -50.2), 163.0, 100.4, fc="#ececec", ec="#8a8a8a", lw=0.6))
    axi.add_patch(Rectangle((-77.5, -15.4), 155.4, 30.8, fc="#cfcfcf", ec="none"))
    axi.add_patch(Rectangle((-103.7, -10), 10, 80, fc=A_EDGE, ec="none"))             # Face A wall (caudal)
    axi.add_patch(Rectangle((-103.7, 60.1), 80, 10, fc=B_EDGE, ec="none"))            # Face B wall (patient left)
    axi.text(-114, 30, "A", color=A_EDGE, fontsize=8, fontweight="bold", ha="center", va="center")
    axi.text(-64, 82, "B", color=B_EDGE, fontsize=8, fontweight="bold", ha="center", va="center")
    axi.add_patch(FancyArrowPatch((-146, 104), (-112, 76), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1.4))
    kx = 124
    ov.text(kx, ly + 4, "Key", fontsize=7.6, fontweight="bold", color=INK, ha="left", va="center")
    ov.add_patch(Rectangle((kx, ly + 11), 9, 9, fc=A_TINT, ec=A_EDGE, lw=0.8)); ov.text(kx + 13, ly + 15.5, "Face A, caudal wall", fontsize=7, va="center")
    ov.add_patch(Rectangle((kx, ly + 24), 9, 9, fc=B_TINT, ec=B_EDGE, lw=0.8)); ov.text(kx + 13, ly + 28.5, "Face B, patient-left wall", fontsize=7, va="center")
    kx2 = 262
    ov.add_patch(Rectangle((kx2, ly + 10), 11, 11, fc=INK, ec="none")); ov.text(kx2 + 5.5, ly + 15.5, "n", color="white", fontsize=7, ha="center", va="center", fontweight="bold")
    ov.text(kx2 + 15, ly + 15.5, "20 mm marker, ID n", fontsize=7, va="center")
    ov.plot([kx2, kx2 + 20], [ly + 29, ly + 29], color=INK, lw=1.8, solid_capstyle="butt")
    ov.text(kx2 + 24, ly + 29, "20 mm, same scale", fontsize=7, va="center")
    ov.text(kx, ly + 43, "Plan view at left: where the bracket sits on the plate. V1 to V3 are\n"
            "unfolded as seen from outside the bracket corner (arrow), junction in the middle.",
            fontsize=6.8, color=GREY, ha="left", va="top", linespacing=1.4)
    save(fig, out, "fig_method_variants")


# ============================================================================ 2. state machine
def fig_states(out):
    Wu, Hu = 630.0, 340.0                     # units of 0.01 inch
    fig = plt.figure(figsize=(Wu / 100, Hu / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wu); ax.set_ylim(0, Hu); ax.axis("off")
    bw, bh = 128, 72
    pos = {"SEARCHING": (72, 215), "VERIFYING": (315, 215), "NAVIGATING": (558, 215), "STALE": (436, 80)}
    what = {"SEARCHING": "No pose.\nNothing is drawn.",
            "VERIFYING": "Wireframe shown,\n2 s countdown.",
            "NAVIGATING": "Targets, trajectories\nand TRE rings.",
            "STALE": "Last good pose held\nfor up to 0.3 s."}
    for k, (cx, cy) in pos.items():
        c = HUD[k]
        box(ax, cx, cy, bw, bh, fc=matplotlib.colors.to_rgba(c, 0.16), ec=c, lw=1.6)
        ax.add_patch(Rectangle((cx - bw / 2 + 6, cy + bh / 2 - 3.2), bw - 12, 3.2, fc=c, ec="none", zorder=4))
        ax.text(cx, cy + 14, k, fontsize=8.6, fontweight="bold", color=INK, ha="center", va="center", zorder=5)
        ax.text(cx, cy - 11, what[k], fontsize=6.9, color=INK, ha="center", va="center", linespacing=1.3, zorder=5)

    def L(x, y, t, color=INK, ha="center"):
        return ax.text(x, y, t, fontsize=7, color=color, ha=ha, va="center", linespacing=1.3)
    cy = 215
    # SEARCHING -> VERIFYING
    arrow(ax, (72 + bw / 2, cy), (315 - bw / 2, cy))
    L(193.5, cy + 13, "pose recovered")
    # VERIFYING -> NAVIGATING
    arrow(ax, (315 + bw / 2, cy), (558 - bw / 2, cy))
    L(436.5, cy + 13, "countdown ends"); L(436.5, cy - 13, "or key C", color=GREY)
    # pose lost and regained (double-headed): VERIFYING <-> STALE and NAVIGATING <-> STALE
    arrow(ax, (335, cy - bh / 2), (410, 80 + bh / 2), style="<|-|>")
    arrow(ax, (538, cy - bh / 2), (462, 80 + bh / 2), style="<|-|>")
    L(436, 148, "\u2193 pose lost\n\u2191 pose back\nwithin 0.3 s", color=GREY)
    # STALE -> SEARCHING
    arrow(ax, (436 - bw / 2, 66), (72, cy - bh / 2), rad=-0.28)
    L(140, 62, "no pose after 0.3 s", ha="center")
    # key R, from any state
    arrow(ax, (558, cy + bh / 2), (72, cy + bh / 2), ls=(0, (4, 3)), color=GREY, rad=0.15, lw=1.0)
    L(315, 308, "key R: reject and restart (from any state)", color=GREY)
    # definitions
    ax.text(8, 30, "Pose recovered: 4 or more markers detected and, when only one bracket\n"
            "face is visible, the camera is not on the wrong side of that face.",
            fontsize=6.6, color=GREY, ha="left", va="center", linespacing=1.4)
    save(fig, out, "fig_method_states")


# ============================================================================ 3. Monte Carlo flow
def fig_mcflow(out):
    Wu, Hu = 630.0, 470.0
    fig = plt.figure(figsize=(Wu / 100, Hu / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wu); ax.set_ylim(0, Hu); ax.axis("off")
    IN_F, IN_E = "#E7F1F8", BLUE
    OUT_F, OUT_E = "#E6F4EE", GREEN
    # column headers
    for x, t in ((95, "Measured inputs"), (315, "Simulation"), (535, "Used for")):
        ax.text(x, 448, t, fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center")
        ax.plot([x - 78, x + 78], [436, 436], color=LIGHT, lw=1)
    # simulation frame + steps
    ax.add_patch(FancyBboxPatch((218, 24), 194, 400, boxstyle="round,pad=0,rounding_size=8", fc="#fafafa", ec=LIGHT, lw=1.0, zorder=0))
    steps = [("1  Choose pose and markers", "visibility model decides\nwhich markers are seen"),
             ("2  Project the marker corners", "calibrated camera"),
             ("3  Add corner noise", "Gaussian, independent per axis"),
             ("4  Recover the pose", "solvePnP, as in the live pipeline"),
             ("5  Evaluate the targets", "3D TRE and overlay error"),
             ("6  RMS over 400 trials", "gives the error for this pose")]
    ys = [388, 322, 256, 190, 124, 58]
    bw, bh = 170, 50
    for (t, sub), y in zip(steps, ys):
        box(ax, 315, y, bw, bh, fc="white", ec=BLUE, lw=1.2)
        ax.text(315, y + 10, t, fontsize=7.4, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(315, y - 9, sub, fontsize=6.7, color=GREY, ha="center", va="center", linespacing=1.25)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (315, y0 - bh / 2), (315, y1 + bh / 2 + 1), color=BLUE, lw=1.2)
    # inputs
    ins = [(388, "Detection model", "fitted to the ATP-3 sweeps"),
           (322, "Camera calibration", "v1: f, centre, distortion"),
           (256, "Corner noise", "static 0.121 px\nlive 0.8 to 1.2 px"),
           (124, "Marker layouts, targets", "V0 to V3, ten entry points")]
    for y, t, sub in ins:
        box(ax, 95, y, 156, 54, fc=IN_F, ec=IN_E, lw=1.0)
        ax.text(95, y + 11, t, fontsize=7.6, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(95, y - 8, sub, fontsize=6.7, color=GREY, ha="center", va="center", linespacing=1.25)
        arrow(ax, (95 + 78, y), (315 - bw / 2 - 1, y), color=IN_E, lw=1.0)
    # outputs
    outs = [(388, "Design comparison", "V0 to V3, 40 poses,\nminimum-marker gates 2 to 5"),
            (306, "Noise-path check", "single-marker error against\nthe measured FLE"),
            (224, "Held-out occlusion test", "tracking and marker counts\nagainst ATP-4"),
            (142, "Error budget", "random part against the\nmeasured ATP-2 overlay error")]
    busx = 438
    ax.plot([busx, busx], [58, 388], color=OUT_E, lw=1.2, zorder=1)
    arrow(ax, (315 + bw / 2 + 1, 58), (busx, 58), color=OUT_E, lw=1.2, style="-")
    for y, t, sub in outs:
        box(ax, 535, y, 156, 56, fc=OUT_F, ec=OUT_E, lw=1.0)
        ax.text(535, y + 13, t, fontsize=7.6, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(535, y - 8, sub, fontsize=6.7, color=GREY, ha="center", va="center", linespacing=1.25)
        arrow(ax, (busx, y), (535 - 78 - 1, y), color=OUT_E, lw=1.2)
    save(fig, out, "fig_method_mcflow")


# ============================================================================ 4. visibility model
def logistic_down(x, x50, sc):      # falls with x
    return 1.0 / (1.0 + np.exp((x - x50) / sc))


def logistic_up(x, x50, sc):        # rises with x
    return 1.0 / (1.0 + np.exp(-(x - x50) / sc))


def fig_visibility(out):
    fig = plt.figure(figsize=(W_IN, 5.5))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, bottom=0.075, top=0.955, wspace=0.28, hspace=0.40, height_ratios=[1.12, 1.0])
    th50, s_th, w50, s_w = FIT["th50"], FIT["s_th"], FIT["w50"], FIT["s_w"]

    # (a) geometry: camera below, marker above, tilted by theta
    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal"); ax.axis("off"); ax.set_anchor("W")
    ax.set_xlim(-2.3, 3.0); ax.set_ylim(-4.3, 1.6)
    th = np.deg2rad(50)
    nvec = np.array([np.sin(th), -np.cos(th)])                   # outward normal, tilted from the line of sight
    tvec = np.array([np.cos(th), np.sin(th)])                    # along the marker
    p0, p1 = -0.85 * tvec, 0.85 * tvec
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=4.5, solid_capstyle="butt", zorder=4)
    ax.text(0.78, 0.92, "marker, side $s$ = 20 mm", fontsize=7.2, color=INK, ha="left", va="center")
    ax.add_patch(FancyArrowPatch((0, 0), tuple(1.35 * nvec), arrowstyle="-|>", mutation_scale=9, color=VERM, lw=1.6, zorder=5))
    ax.text(1.35 * nvec[0] + 0.1, 1.35 * nvec[1] - 0.05, "outward\nnormal", color=VERM, fontsize=7.2, ha="left", va="center", linespacing=1.15)
    ax.plot([0, 0], [0, -2.4], color=GREY, lw=1.0, ls=(0, (4, 3)))
    ax.text(-0.12, -1.35, "line of sight,\ndepth $z$", color=GREY, fontsize=7, ha="right", va="center", linespacing=1.15)
    ax.add_patch(Arc((0, 0), 1.7, 1.7, theta1=-90, theta2=-40, color=VERM, lw=1.3))
    ax.text(0.30, -1.08, r"$\theta$", color=VERM, fontsize=11, ha="center", va="center")
    # image plane with the projected marker width
    yp = -2.4
    ax.plot([-1.7, 1.7], [yp, yp], color=BLUE, lw=1.8)
    ax.text(1.78, yp, "image plane", color=BLUE, fontsize=7, ha="left", va="center")
    for pt in (p0, p1):
        ax.plot([pt[0], pt[0]], [pt[1], yp], color=BLUE, lw=0.8, ls=(0, (2, 2)))
    ax.annotate("", xy=(p0[0], yp - 0.22), xytext=(p1[0], yp - 0.22), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.0))
    ax.text(0, yp - 0.50, r"$s\cos\theta$", color=BLUE, fontsize=9, ha="center", va="center")
    ax.add_patch(plt.Polygon([[0, -3.25], [-0.34, -3.85], [0.34, -3.85]], fc=BLUE, ec="none"))
    ax.text(0.55, -3.58, "camera", color=BLUE, fontsize=7, ha="left", va="center")
    ax.text(0.35, -4.2, r"apparent width  $w=f\,s\cos\theta\,/\,z$  (px)", fontsize=8.2, ha="center", va="center", color=INK)
    ax.set_title("(a) What the detector sees", loc="left", fontsize=9, fontweight="bold")

    # (b) view angle
    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 90, 400)
    ax.plot(x, logistic_down(x, th50, s_th), color=VERM, lw=2)
    ax.axvline(th50, color=GREY, lw=0.8, ls=(0, (4, 3))); ax.axhline(0.5, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.plot([th50], [0.5], "o", color=VERM, ms=5, zorder=5)
    ax.text(th50 - 2, 0.58, f"50% at {th50:.1f}°", ha="right", fontsize=7.5, color=INK)
    ax.set_xlim(0, 90); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"view angle $\theta$ (degrees)"); ax.set_ylabel("detection probability")
    ax.set_title("(b) Falls with view angle", loc="left", fontsize=9, fontweight="bold")

    # (c) apparent width
    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(0, 50, 400)
    ax.plot(x, logistic_up(x, w50, s_w), color=BLUE, lw=2)
    ax.axvline(w50, color=GREY, lw=0.8, ls=(0, (4, 3))); ax.axhline(0.5, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.plot([w50], [0.5], "o", color=BLUE, ms=5, zorder=5)
    ax.text(w50 + 1.2, 0.34, f"50% at {w50:.1f} px", ha="left", fontsize=7.5, color=INK)
    for dist, lab in ((100, "100 cm"), (60, "60 cm")):
        wf = FOCAL_PX * MARKER_MM / (dist * 10)
        ax.plot([wf, wf], [0, 0.08], color=INK, lw=1.4)
        ax.text(wf, 0.12, f"{lab}\nface-on", ha="center", fontsize=6.4, color=INK, linespacing=1.15)
    ax.set_xlim(0, 50); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"apparent marker width $w$ (px)"); ax.set_ylabel("detection probability")
    ax.set_title("(c) Rises with apparent width", loc="left", fontsize=9, fontweight="bold")

    # (d) combined map
    ax = fig.add_subplot(gs[1, 1])
    d_cm = np.linspace(30, 130, 300)
    th_deg = np.linspace(0, 89, 300)
    D, T = np.meshgrid(d_cm, th_deg)
    Wd = FOCAL_PX * MARKER_MM * np.cos(np.deg2rad(T)) / (D * 10.0)
    P = logistic_down(T, th50, s_th) * logistic_up(Wd, w50, s_w)
    im = ax.pcolormesh(D, T, P, cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
    cs = ax.contour(D, T, P, levels=[0.5], colors=[INK], linewidths=1.3)
    ax.clabel(cs, fmt={0.5: "50%"}, fontsize=7)
    ax.axvspan(55, 100, color=GREEN, alpha=0.0)
    for xv in (55, 100):
        ax.axvline(xv, color=GREEN, lw=1.2, ls=(0, (4, 3)))
    ax.text(77.5, 4, "full-accuracy\ndistances", ha="center", va="bottom", fontsize=6.6, color=GREEN, linespacing=1.15)
    ax.set_xlim(30, 130); ax.set_ylim(0, 89)
    ax.set_xlabel("distance (cm)"); ax.set_ylabel(r"view angle $\theta$ (degrees)")
    ax.set_title("(d) One 20 mm marker, both effects", loc="left", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.set_label("detection probability", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    for a in fig.axes[1:]:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.tick_params(labelsize=7.5)
    save(fig, out, "fig_method_visibility")


# ============================================================================ main
FIGS = {"variants": fig_variants, "states": fig_states, "mcflow": fig_mcflow, "visibility": fig_visibility}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", choices=list(FIGS), nargs="*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in (a.only or FIGS):
        FIGS[k](out)


if __name__ == "__main__":
    main()