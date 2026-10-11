#!/usr/bin/env python3
"""
UI v2 wireframes, one image per state (low fidelity, vector scene).

    python scripts/ui/make_ui_wireframes.py --out docs/ui_wireframes

Writes: 1_searching, 2_verifying, 3_navigating_overview, 4_navigating_focus, 5_tracking_lost (png + pdf)
and ui_wireframes_sheet.pdf (all five).  Pure matplotlib, no data files.
Design tokens match ar_theme.py where they exist (teal, green, red, grey); amber is only used for 'near limit'.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Polygon, FancyArrowPatch, Arc
from matplotlib.backends.backend_pdf import PdfPages

W, H, CAP = 1600, 1000, 70          # window size and caption band (px at 100 dpi)
PANEL, TEXT, MUTED = "#0F1419", "#EAF0F2", "#93A0AD"
TEAL, GREEN, AMBER, RED, GREY = "#3CC3C3", "#64D28C", "#E8B04A", "#D75A5A", "#6B747A"
BONE, BONE_EDGE = "#C9C3B7", "#8F8A7F"
HALO = [pe.withStroke(linewidth=4.5, foreground=(0, 0, 0, 0.55))]

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


# ----------------------------------------------------------------------------- primitives
def new_canvas():
    fig = plt.figure(figsize=(W / 100, (H + CAP) / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H + CAP, 0); ax.axis("off")
    return fig, ax


def rr(ax, x, y, w, h, fc=PANEL, ec="none", alpha=0.9, r=14, lw=1.2, z=5):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, alpha=alpha, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def tx(ax, x, y, s, size=13, color=TEXT, weight="normal", ha="left", va="center", z=8, halo=False, family=None, alpha=1.0):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, alpha=alpha,
                family=family or "DejaVu Sans")
    if halo:
        t.set_path_effects(HALO)
    return t


def dot(ax, x, y, r, c, z=8):
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=z))


def badge(ax, x, y, n):
    """numbered callout"""
    ax.add_patch(Circle((x, y), 13, fc="#FFFFFF", ec="#111", lw=1.5, zorder=20))
    tx(ax, x, y + 0.5, str(n), size=11, color="#111", weight="bold", ha="center", z=21)


def keycap(ax, x, y, label, size=11):
    w = max(26, 11 * len(label) + 14)
    rr(ax, x, y - 14, w, 28, fc="#1C242C", ec="#3A4753", alpha=0.95, r=6, lw=1)
    tx(ax, x + w / 2, y, label, size=size, color=TEXT, weight="bold", ha="center")
    return w


# ----------------------------------------------------------------------------- scene
LEVELS = ["L1", "L2", "L3", "L4", "L5"]


def vert_pose(i):
    return 430 + 118 * i, 250 + 88 * i, 1.30 + 0.07 * i          # cx, cy, scale


def entry_xy(i, side):
    cx, cy, s = vert_pose(i)
    return cx + (-1 if side == "L" else 1) * 44 * s, cy - 8 * s


def draw_vertebra(ax, i, alpha=1.0, outline_only=False, outline_c=TEAL, lw=1.6, z=2):
    cx, cy, s = vert_pose(i)
    fc = "none" if outline_only else BONE
    ec = outline_c if outline_only else BONE_EDGE
    a = 1.0 if outline_only else alpha
    parts = [
        Ellipse((cx, cy + 6 * s), 96 * s, 58 * s, angle=-8, fc=fc, ec=ec, lw=lw, alpha=a, zorder=z),
        Ellipse((cx - 62 * s, cy - 4 * s), 62 * s, 17 * s, angle=-14, fc=fc, ec=ec, lw=lw, alpha=a, zorder=z),
        Ellipse((cx + 62 * s, cy - 4 * s), 62 * s, 17 * s, angle=10, fc=fc, ec=ec, lw=lw, alpha=a, zorder=z),
        FancyBboxPatch((cx - 7 * s, cy - 38 * s), 14 * s, 50 * s, boxstyle="round,pad=0,rounding_size=6",
                       fc=fc, ec=ec, lw=lw, alpha=a, zorder=z + 0.1),
    ]
    for p in parts:
        ax.add_patch(p)


def draw_scene(ax, ghost_except=None, ghost_alpha=0.28):
    """Table, phantom base, bracket hint and the five vertebrae."""
    n = 140
    for k in range(n):                                          # soft vertical gradient behind everything
        c = np.array([0x35, 0x2A, 0x23]) + (np.array([0x1B, 0x15, 0x12]) - np.array([0x35, 0x2A, 0x23])) * (k / n)
        ax.add_patch(plt.Rectangle((0, k * H / n), W, H / n + 1, fc=c / 255, ec="none", zorder=0))
    L1 = np.array(vert_pose(0)[:2], float); L5 = np.array(vert_pose(4)[:2], float)
    d = (L5 - L1) / np.linalg.norm(L5 - L1); nrm = np.array([-d[1], d[0]])
    shift = np.array([0, 34])
    P = [L1 - d * 130 + nrm * 190 + shift, L5 + d * 130 + nrm * 190 + shift, L5 + d * 130 - nrm * 190 + shift, L1 - d * 130 - nrm * 190 + shift]
    ax.add_patch(Polygon(P, fc="#4E5256", ec="#2E3134", lw=2, zorder=1, alpha=0.95))
    for k in range(3):                                                         # post tabs under the vertebrae
        pass
    for i in range(5):
        a = 1.0
        if ghost_except is not None and i != ghost_except:
            a = ghost_alpha
        draw_vertebra(ax, i, alpha=a, z=2)
    # marker bracket hint: two white faces with black squares, caudal end of the base plate
    ax.add_patch(Polygon([[960, 745], [1160, 770], [1160, 890], [960, 860]], fc="#EDEDED", ec="#999", lw=1.5, zorder=3))
    ax.add_patch(Polygon([[1160, 770], [1300, 742], [1300, 856], [1160, 890]], fc="#D9D9D9", ec="#999", lw=1.5, zorder=3))
    for (x, y) in [(990, 775), (1070, 785), (990, 820), (1070, 830)]:
        ax.add_patch(plt.Rectangle((x, y), 40, 40, fc="#111", ec="none", zorder=4, angle=4))
    for (x, y) in [(1185, 775), (1238, 765), (1185, 815), (1238, 805)]:
        ax.add_patch(plt.Rectangle((x, y), 34, 34, fc="#111", ec="none", zorder=4, angle=-8))


# ----------------------------------------------------------------------------- shared UI parts
def state_pill(ax, name, color, x=None):
    w = 62 + 17 * len(name)
    x = W / 2 - w / 2 if x is None else x
    rr(ax, x, 22, w, 50, r=25, alpha=0.92)
    dot(ax, x + 28, 47, 7, color)
    tx(ax, x + 46, 47, name, size=15, weight="bold")


def trust_badge(ax, level, reason):
    col = {"good": GREEN, "limited": GREY, "lost": RED, "search": GREY}[level]
    title = {"good": "Tracking good", "limited": "Tracking limited", "lost": "Tracking lost", "search": "Searching"}[level]
    h = 82 if reason else 54
    rr(ax, 28, 22, 360, h, r=14, alpha=0.92)
    ax.add_patch(plt.Rectangle((28, 22 + 10), 5, h - 20, fc=col, ec="none", zorder=7))
    dot(ax, 56, 48, 7, col)
    tx(ax, 74, 48, title, size=15, weight="bold")
    if reason:
        tx(ax, 56, 80, reason, size=11.5, color=MUTED)


def level_chips(ax, active=None, side="R", y0=170):
    x = W - 112
    rr(ax, x - 14, y0 - 22, 100, 392, r=16, alpha=0.82)
    for i, lv in enumerate(LEVELS):
        y = y0 + i * 58
        on = (active == i)
        rr(ax, x, y, 72, 44, fc=TEAL if on else "#1C242C", ec=TEAL if on else "#3A4753", alpha=1.0 if on else 0.95, r=10, lw=1.2)
        tx(ax, x + 36, y + 22, lv, size=15, weight="bold", ha="center", color="#06201F" if on else TEXT)
    y = y0 + 5 * 58 + 6
    for j, lab in enumerate(["L", "R"]):
        on = (side == lab and active is not None)
        rr(ax, x + j * 38, y, 34, 34, fc=TEAL if on else "#1C242C", ec="#3A4753", alpha=1.0 if on else 0.95, r=8, lw=1)
        tx(ax, x + j * 38 + 17, y + 17, lab, size=12, weight="bold", ha="center", color="#06201F" if on else TEXT)


def hint_bar(ax, items, y=H - 38):
    total = sum(11 * len(k) + 14 + 12 + 8.0 * len(t) + 18 for k, t in items)
    x = W / 2 - total / 2
    rr(ax, x - 14, y - 22, total + 22, 44, r=22, alpha=0.8)
    cx = x
    for k, t in items:
        w = keycap(ax, cx, y, k)
        tx(ax, cx + w + 8, y, t, size=11.5, color=MUTED)
        cx += w + 8 + 8.0 * len(t) + 18


def target_card(ax, title, lines, accent=GREEN):
    h = 40 + 30 * len(lines) + 16
    y = H - 40 - h - 56
    rr(ax, 28, y, 400, h, r=14, alpha=0.92)
    ax.add_patch(plt.Rectangle((28, y + 10), 5, h - 20, fc=accent, ec="none", zorder=7))
    tx(ax, 50, y + 30, title, size=17, weight="bold")
    for k, (lab, val) in enumerate(lines):
        yy = y + 66 + 30 * k
        tx(ax, 50, yy, lab, size=12.5, color=MUTED)
        tx(ax, 408, yy, val, size=14.5, weight="bold", ha="right")


def caption(ax, text_lines):
    ax.add_patch(plt.Rectangle((0, H), W, CAP, fc="#E9EDF0", ec="none", zorder=30))
    for k, line in enumerate(text_lines):
        tx(ax, 22, H + 22 + 26 * k, line, size=12.5, color="#1B2430", z=31)


# ----------------------------------------------------------------------------- glyphs
def ring(ax, x, y, r, c, lw=2.5, z=12, ls="-", alpha=1.0, halo=True):
    p = Circle((x, y), r, fc="none", ec=c, lw=lw, ls=ls, zorder=z, alpha=alpha)
    if halo:
        p.set_path_effects(HALO)
    ax.add_patch(p)


def entry_glyph(ax, x, y, status_c, label=None, cta=True):
    """Entry point: status ring (TRE), double 'act here' ring, centre dot."""
    ring(ax, x, y, 36, status_c, lw=2.2, ls=(0, (5, 4)), z=11, alpha=0.95)          # TRE uncertainty ring (dashed)
    if cta:
        ring(ax, x, y, 23, "#FFFFFF", lw=2.6, z=12)
        ring(ax, x, y, 13, "#FFFFFF", lw=1.6, z=12, alpha=0.7)
    dot(ax, x, y, 4.5, "#FFFFFF", z=13)
    ax.add_patch(Circle((x, y), 5.5, fc="none", ec="#000", lw=1.0, zorder=13, alpha=0.6))


def trajectory(ax, e, u, inward=150, outward=300, depth_ticks=True):
    """Solid arrow into the bone (with depth ticks) + dashed fading extension outward, arrowhead pointing in."""
    ex, ey = e
    u = np.array(u, float); u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    tip = np.array([ex, ey]) + u * inward
    # inward, solid
    ln = ax.plot([ex, tip[0]], [ey, tip[1]], color="#FFFFFF", lw=4.0, solid_capstyle="round", zorder=12)[0]
    ln.set_path_effects(HALO)
    ah = FancyArrowPatch((tip - u * 18)[:2], tip[:2], arrowstyle="-|>", mutation_scale=30, color="#FFFFFF", lw=0, zorder=13)
    ah.set_path_effects(HALO); ax.add_patch(ah)
    if depth_ticks:
        for k in range(1, 6):
            p = np.array([ex, ey]) + u * (inward * k / 6)
            ax.plot([p[0] - 8 * n[0], p[0] + 8 * n[0]], [p[1] - 8 * n[1], p[1] + 8 * n[1]], color="#FFFFFF", lw=2.2, zorder=13)
        tx(ax, *(tip + u * 26 + n * 6), "isthmus", size=12, weight="bold", halo=True, z=14)
    # outward extension: dashed, fading, arrowhead pointing back towards the entry
    steps = 26
    for k in range(steps):
        a = 1.0 - 0.62 * (k / steps)
        p0 = np.array([ex, ey]) - u * (34 + outward * k / steps)
        p1 = np.array([ex, ey]) - u * (34 + outward * (k + 0.55) / steps)
        seg = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#FFFFFF", lw=4.0, alpha=a, solid_capstyle="butt", zorder=11)[0]
        seg.set_path_effects([pe.withStroke(linewidth=5.5, foreground=(0, 0, 0, 0.35 * a))])
    ahead = np.array([ex, ey]) - u * 58
    ah2 = FancyArrowPatch((ahead - u * 70)[:2], ahead[:2], arrowstyle="-|>", mutation_scale=34, color="#FFFFFF", lw=0, zorder=13)
    ah2.set_path_effects(HALO); ax.add_patch(ah2)
    return tip


def action_label(ax, x, y, text, sub=None):
    w = 30 + 12.6 * len(text)
    rr(ax, x, y - 20, w, 40, fc="#FFFFFF", alpha=0.97, r=20, z=15)
    tx(ax, x + w / 2, y, text, size=13.5, color="#0F1419", weight="bold", ha="center", z=16)
    if sub:
        tx(ax, x + w / 2, y + 34, sub, size=11.5, color=TEXT, ha="center", halo=True, z=16)


# ----------------------------------------------------------------------------- states
def w1_searching():
    fig, ax = new_canvas(); draw_scene(ax)
    state_pill(ax, "SEARCHING", GREY)
    trust_badge(ax, "search", "")
    rr(ax, W / 2 - 300, 300, 600, 150, r=22, alpha=0.88)
    ax.add_patch(plt.Rectangle((W / 2 - 250, 335), 62, 62, fc="#111", ec="#fff", lw=2, zorder=9))
    ax.add_patch(plt.Rectangle((W / 2 - 238, 347), 38, 38, fc="#fff", ec="none", zorder=9))
    ax.add_patch(plt.Rectangle((W / 2 - 230, 355), 22, 22, fc="#111", ec="none", zorder=9))
    tx(ax, W / 2 - 160, 352, "Show the marker array to the camera", size=19, weight="bold")
    tx(ax, W / 2 - 160, 386, "Nothing is drawn until the pose is found.", size=13.5, color=MUTED)
    tx(ax, W / 2 - 160, 418, "0 of 8 markers in view", size=13.5, color=TEXT)
    hint_bar(ax, [("h", "details"), ("?", "help"), ("q", "quit")])
    for n, (x, y) in enumerate([(206, 94), (W / 2, 96 + 200), (W / 2 + 330, 300)], 1):
        pass
    badge(ax, 18, 22, 1); badge(ax, W / 2 + 150, 22, 2); badge(ax, W / 2 + 300, 318, 3); badge(ax, W / 2 - 330, H - 38, 4)
    caption(ax, ["1 Trust badge: one word + reason.   2 State pill.   3 One instruction, nothing else on screen (no stale projections).",
                 "4 Hint bar: only the keys that matter now. Full key list behind '?'. Engineer diagnostics behind 'h'."])
    return fig


def w2_verifying():
    fig, ax = new_canvas(); draw_scene(ax)
    for i in range(5):
        draw_vertebra(ax, i, outline_only=True, outline_c=TEAL, lw=1.8, z=9)                    # edge-only outline
    state_pill(ax, "VERIFYING", TEAL)
    trust_badge(ax, "good", "Registration found. Check it before trusting it.")
    rr(ax, W / 2 - 330, 110, 660, 62, r=18, alpha=0.9)
    tx(ax, W / 2, 141, "Does the outline sit on the bone?", size=21, weight="bold", ha="center")
    # countdown ring
    cx, cy = W / 2 - 230, H - 170
    rr(ax, cx - 250, cy - 70, 500, 140, r=24, alpha=0.9)
    ax.add_patch(Circle((cx - 170, cy), 40, fc="none", ec="#2B3640", lw=7, zorder=8))
    ax.add_patch(Arc((cx - 170, cy), 80, 80, theta1=-90, theta2=150, color=TEAL, lw=7, zorder=9))
    tx(ax, cx - 170, cy, "2", size=26, weight="bold", ha="center")
    tx(ax, cx - 110, cy - 16, "Auto-confirms in 2 s", size=15, weight="bold")
    wc = keycap(ax, cx - 110, cy + 22, "c"); tx(ax, cx - 110 + wc + 8, cy + 22, "confirm now", size=12.5, color=MUTED)
    wr = keycap(ax, cx + 60, cy + 22, "r"); tx(ax, cx + 60 + wr + 8, cy + 22, "reject", size=12.5, color=MUTED)
    ox4, oy4 = vert_pose(1)[0] + 60, vert_pose(1)[1] - 30
    badge(ax, 18, 22, 1); badge(ax, W / 2 + 330, 110, 2); badge(ax, cx - 250, cy - 70, 3); badge(ax, ox4, oy4, 4)
    caption(ax, ["1 Trust badge.   2 The question the user must answer.   3 Countdown = consent by inaction; tap or key to reject.",
                 "4 Edge-only outline (thin, one colour, low clutter, no fill) so the physical bone stays visible underneath. No targets yet."])
    return fig


def w3_overview():
    fig, ax = new_canvas(); draw_scene(ax)
    state_pill(ax, "NAVIGATING", GREEN)
    trust_badge(ax, "good", "")
    for i in range(5):
        for side in ("L", "R"):
            x, y = entry_xy(i, side)
            ring(ax, x, y, 11, "#FFFFFF", lw=2.2, z=12, alpha=0.9)
            dot(ax, x, y, 2.6, "#FFFFFF", z=13)
        cx, cy, s = vert_pose(i)
        tx(ax, cx + 108 * s, cy - 6 * s, LEVELS[i], size=15, weight="bold", halo=True, z=14)
    level_chips(ax, active=None)
    rr(ax, W / 2 - 260, 118, 520, 52, r=26, alpha=0.9)
    tx(ax, W / 2, 144, "Choose a level to see its screw paths", size=16.5, weight="bold", ha="center")
    hint_bar(ax, [("1-5", "level"), ("a", "all levels"), ("h", "details"), ("?", "help")])
    ex2, ey2 = entry_xy(1, 'R'); badge(ax, 18, 22, 1); badge(ax, ex2 + 6, ey2 - 26, 2); badge(ax, W - 118, 170, 3); badge(ax, W / 2 + 290, 118, 4)
    caption(ax, ["1 Trust badge (reason only when something is limited).   2 Entry points only: small rings, no trajectories, nothing to read.",
                 "3 Level selector (touch targets on the phone; keys 1-5 on the laptop).   4 Prompts the next action instead of showing everything."])
    return fig


def w4_focus():
    fig, ax = new_canvas(); draw_scene(ax, ghost_except=2, ghost_alpha=0.22)
    draw_vertebra(ax, 2, outline_only=True, outline_c=TEAL, lw=2.4, z=9)
    state_pill(ax, "NAVIGATING", GREEN)
    trust_badge(ax, "good", "")
    # other pedicle of the same vertebra: hollow ring only
    xl, yl = entry_xy(2, "L")
    ring(ax, xl, yl, 12, "#FFFFFF", lw=2.0, z=12, alpha=0.65)
    tx(ax, xl - 20, yl - 24, "L3 left", size=12, color=TEXT, ha="right", halo=True, z=14)
    # selected target: right pedicle of L3
    ex, ey = entry_xy(2, "R")
    u = (-0.50, 0.87)                                                            # into the bone: medial and anterior (on screen)
    entry_glyph(ax, ex, ey, GREEN)
    trajectory(ax, (ex, ey), u, inward=130, outward=330)
    action_label(ax, ex + 118, ey - 34, "ENTER HERE", sub="L3 right pedicle")
    ox = ex - u[0] * 190; oy = ey - u[1] * 190
    tx(ax, ox + 34, oy - 6, "aim along this line, then insert", size=13, weight="bold", halo=True, z=16)
    level_chips(ax, active=2, side="R")
    target_card(ax, "L3 right", [("Angle", "18° medial, 12° caudal"), ("Depth to isthmus", "38 mm"), ("Accuracy", "within 2 mm")], accent=GREEN)
    hint_bar(ax, [("[ ]", "side"), ("tab", "next level"), ("a", "all"), ("h", "details"), ("?", "help")])
    badge(ax, ex - 52, ey - 52, 1); badge(ax, ex + 296, ey - 52, 2); badge(ax, ox - 24, oy - 6, 3)
    badge(ax, ex - 70, ey + 70, 4); badge(ax, 18, H - 235, 5); badge(ax, W - 128, 170, 6)
    caption(ax, ["1 Entry: dashed ring = accuracy, solid double ring = 'act here'.   2 Call to action + level.   3 Extension beyond the entry shows the direction of approach.",
                 "4 Solid arrow into the bone with depth ticks every 10 mm.   5 Target card: angle, depth, accuracy.   6 Other levels fade; the other side shows as a ring only."])
    return fig


def w5_lost():
    fig, ax = new_canvas(); draw_scene(ax, ghost_except=None)
    ex, ey = entry_xy(2, "R")
    ring(ax, ex, ey, 24, "#9AA3AB", lw=2.2, z=11, alpha=0.6)
    ax.plot([ex, ex - 65], [ey, ey + 113], color="#9AA3AB", lw=3, alpha=0.5, zorder=11)
    tx(ax, ex + 34, ey - 40, "last known position", size=12.5, color="#C9D0D6", halo=True, z=14)
    rr(ax, 0, 0, W, 108, fc="#4A1A1A", alpha=0.92, r=0, z=6)
    ax.add_patch(plt.Rectangle((0, 104), W, 4, fc=RED, ec="none", zorder=7))
    state_pill(ax, "STALE", RED, x=W - 250)
    tx(ax, 34, 38, "Tracking lost", size=21, weight="bold", color="#FFD9D9")
    tx(ax, 34, 72, "Showing the last position for up to 0.3 s. Move so two bracket faces or the dome are in view.", size=13.5, color="#F2C4C4")
    rr(ax, W / 2 - 250, 640, 500, 76, r=22, alpha=0.9)
    tx(ax, W / 2, 666, "Do not use this overlay", size=20, weight="bold", ha="center", color="#FFB3B3")
    tx(ax, W / 2, 696, "Targets return when tracking recovers.", size=13.5, color=MUTED, ha="center")
    hint_bar(ax, [("r", "restart"), ("h", "details"), ("?", "help")])
    badge(ax, 14, 56, 1); badge(ax, ex + 24, ey - 60, 2); badge(ax, W / 2 + 270, 678, 3)
    caption(ax, ["1 Red banner with the reason and what to do.   2 Last pose frozen, grey, labelled (never shown as live).",
                 "3 Plain instruction. After 0.3 s with no pose the overlay is removed completely (back to SEARCHING)."])
    return fig


STATES = [("1_searching", w1_searching), ("2_verifying", w2_verifying), ("3_navigating_overview", w3_overview),
          ("4_navigating_focus", w4_focus), ("5_tracking_lost", w5_lost)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/ui_wireframes")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / "ui_wireframes_sheet.pdf") as pdf:
        for name, fn in STATES:
            fig = fn()
            fig.savefig(out / f"{name}.png", dpi=100)
            fig.savefig(out / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main()