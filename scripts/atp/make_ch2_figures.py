#!/usr/bin/env python3
"""
Chapter 2 figures in the Chapter 3/4 style (shared palette in fig_style.py).

  python scripts/atp/make_ch2_figures.py                      # all
  python scripts/atp/make_ch2_figures.py --only accuracy tree tre
  python scripts/atp/make_ch2_figures.py --out docs/figures

  accuracy   lit_accuracy_landscape.pdf       reported accuracy of AR navigation studies
  tree       lit_registration_taxonomy.pdf    families of registration methods
  tre        lit_fre_tre_layout.pdf           predicted RMS TRE, clustered vs spread fiducials

The accuracy values were read off the earlier version of the figure (check them against the papers).
The TRE map is the Fitzpatrick and West (2001) expression
    TRE^2(r) = FLE^2 / N * (1 + 1/3 * sum_k d_k^2 / f_k^2)
with FLE = 0.5 mm and six fiducials in the plane.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as FS  # noqa: E402
from make_ch3_figures import canvas, rbox, arr, poly, tag, save, P1_TINT, P2_TINT, P2_TEXT  # noqa: E402

W = FS.W
INK = FS.BLACK

# ---------------------------------------------------------------------------------------
# 1. Reported accuracy
# ---------------------------------------------------------------------------------------
# (label, centre mm, low, high or None, error type, setting)
STUDIES = [
    ("Zhang 2023", "2D/3D to X-ray, spine model", 0.50, 0.35, 0.63, "surface", "phantom"),
    ("Liu 2024", "point cloud, neurosurgical phantom", 0.75, None, None, "target", "phantom"),
    ("Besharati 2015", "projector, head phantom", 0.80, 0.54, 1.06, "target", "phantom"),
    ("Besharati 2015", "projector, 5 patients", 1.20, 0.65, 1.75, "target", "clinical"),
    ("Xu 2020", "projector AR, lumbar phantom", 1.64, 1.28, 2.00, "screw", "phantom"),
    ("Nicolai 2026", "review, head-mounted AR", 2.60, 0.30, 4.90, "screw", "clinical"),
    ("Müller 2020", "HoloLens, cadaver spines", 3.40, 1.80, 5.00, "screw", "cadaver"),
]
TYPE_COL = {"surface": FS.SLATE, "target": FS.BLUE, "screw": FS.GREEN}
TYPE_LAB = {"surface": "Surface registration error", "target": "Target or projection error",
            "screw": "Screw or entry point error"}
SET_MK = {"phantom": "o", "clinical": "s", "cadaver": "D"}
SET_LAB = {"phantom": "Phantom", "clinical": "Clinical", "cadaver": "Cadaver"}


def fig_accuracy(out):
    fig, ax = plt.subplots(figsize=(W, 3.8))
    fig.subplots_adjust(left=0.385, right=0.985, top=0.975, bottom=0.31)
    n = len(STUDIES)
    ax.axvline(2.0, c=FS.BLACK, ls="--", lw=1.1, zorder=1)
    for i, (au, desc, m, lo, hi, kind, setting) in enumerate(STUDIES):
        c = TYPE_COL[kind]
        if lo is not None:
            ax.plot([lo, hi], [i, i], c=c, lw=1.6, solid_capstyle="butt", zorder=2)
            for e in (lo, hi):
                ax.plot([e, e], [i - 0.13, i + 0.13], c=c, lw=1.4, zorder=2)
        ax.plot([m], [i], SET_MK[setting], c=c, mec=FS.BLACK, mew=0.8, ms=8 if setting != "cadaver" else 7.5, zorder=4)
        ax.text(-0.12, i - 0.17, au, ha="right", va="center", fontsize=9, fontweight="bold", color=INK,
                transform=ax.get_yaxis_transform())
        ax.text(-0.12, i + 0.17, desc, ha="right", va="center", fontsize=8, color=FS.SLATE,
                transform=ax.get_yaxis_transform())
    # axis
    ax.set_ylim(n - 0.45, -0.7)
    ax.set_xlim(0, 5.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.arange(0, 5.01, 0.5))
    ax.set_xticklabels([f"{v:g}" for v in np.arange(0, 5.01, 0.5)])
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("Reported error (mm); bars show SD or range where given")
    ax.text(2.05, -0.58, "2 mm criterion", ha="left", va="center", fontsize=9, fontweight="bold")
    # legends
    ch = [Line2D([], [], marker="o", ls="", color=TYPE_COL[k], mec=FS.BLACK, mew=0.8, ms=7, label=TYPE_LAB[k])
          for k in TYPE_COL]
    sh = [Line2D([], [], marker=SET_MK[k], ls="", color=FS.LIGHT, mec=FS.BLACK, mew=0.9, ms=7, label=SET_LAB[k])
          for k in SET_MK]
    l1 = fig.legend(handles=ch, title="Error measured", loc="lower left", bbox_to_anchor=(0.27, -0.015), ncol=1,
                    fontsize=8.8, title_fontsize=9, alignment="left", handletextpad=0.4, labelspacing=0.45)
    fig.add_artist(l1)
    fig.legend(handles=sh, title="Setting", loc="lower left", bbox_to_anchor=(0.66, -0.015), ncol=1, fontsize=8.8,
               title_fontsize=9, alignment="left", handletextpad=0.4, labelspacing=0.45)
    for l in fig.legends:
        l.get_title().set_fontweight("bold")
    save(fig, out, "lit_accuracy_landscape")


# ---------------------------------------------------------------------------------------
# 2. Families of registration methods
# ---------------------------------------------------------------------------------------
def fig_tree(out):
    Wu, Hu = 630, 296
    fig, ax = canvas(Wu, Hu)
    yr, ym, yl = 270, 172, 62
    root = (315, yr, 214, 42)
    mids = [(148, ym, 262, 60, "Extrinsic", "fiducial markers", "closed-form fit, error theory known", FS.BLUE, P1_TINT),
            (482, ym, 262, 60, "Intrinsic", "anatomy, no markers", "iterative or learned, less characterised", FS.SLATE, FS.PALE)]
    leaves = [(76, "Optical\ntrackers", "reflective spheres,\nclinical standard", FS.BLUE, 0),
              (222, "Camera + printed\nArUco fiducials", "", FS.GREEN, 1),
              (408, "Surface or\npoint cloud", "structured light,\ndeep learning", FS.SLATE, 0),
              (554, "Image-based\n2D/3D", "CT to X-ray or\nfluoroscopy", FS.SLATE, 0)]
    lw_, lh = 138, 84
    # connectors first (behind boxes)
    ymid1 = (yr - root[3] / 2 + ym + 30) / 2
    for (x, y, w, h, *_rest) in mids:
        poly(ax, [(315, yr - root[3] / 2), (315, ymid1), (x, ymid1), (x, y + h / 2)], color=FS.GREY, lw=1.6, z=1)
    ymid2 = (ym - 30 + yl + lh / 2) / 2
    for (x, y, w, h, *_r), pair in zip(mids, ((0, 1), (2, 3))):
        for k in pair:
            xl = leaves[k][0]
            poly(ax, [(x, y - h / 2), (x, ymid2), (xl, ymid2), (xl, yl + lh / 2)], color=FS.GREY, lw=1.6, z=1)
    # root
    rbox(ax, root[0], root[1], root[2], root[3], FS.NAVY, FS.NAVY, lw=1.4, r=8)
    tag(ax, root[0], root[1], "Registration of a model\nto the patient", color="white", fs=9.4, fontweight="bold")
    # middle level
    for (x, y, w, h, t, sub, desc, edge, tint) in mids:
        rbox(ax, x, y, w, h, tint, edge, lw=1.6, r=8)
        tag(ax, x, y + 18, t, color=INK, fs=10.4, fontweight="bold")
        tag(ax, x, y + 1, sub, color=INK, fs=8.8)
        tag(ax, x, y - 16, desc, color=FS.SLATE, fs=8.2)
    # leaves
    for (x, t, sub, edge, hot) in leaves:
        if hot:
            rbox(ax, x, yl, lw_ + 6, lh + 4, P2_TINT, FS.GREEN, lw=2.4, r=8, z=3)
            tag(ax, x, yl + 6, t, color=INK, fs=9.2, fontweight="bold")
            rbox(ax, x, yl - lh / 2 - 2, 92, 17, FS.GREEN, FS.GREEN, lw=1, r=8, z=5)
            tag(ax, x, yl - lh / 2 - 2, "THIS PROJECT", color="white", fs=7.6, fontweight="bold")
        else:
            rbox(ax, x, yl, lw_, lh, "white", edge, lw=1.5, r=8, z=3)
            tag(ax, x, yl + 17, t, color=INK, fs=9.2, fontweight="bold")
            tag(ax, x, yl - 18, sub, color=FS.SLATE, fs=8.2)
    save(fig, out, "lit_registration_taxonomy")


# ---------------------------------------------------------------------------------------
# 3. Predicted RMS TRE, clustered vs spread
# ---------------------------------------------------------------------------------------
CLUSTERED = [(12, 2), (3, 11), (-11, 6), (-5, -7), (10, -8), (2, -13)]
SPREAD = [(15, 45), (-45, 25), (50, 10), (-20, -30), (40, -35), (5, -50)]
FLE_MM = 0.5


def tre_map(fid, X, Y, fle=FLE_MM):
    F = np.c_[np.array(fid, float), np.zeros(len(fid))]
    c = F.mean(0); Q = F - c
    _, V = np.linalg.eigh(Q.T @ Q)
    f2 = [((Q - (Q @ V[:, k])[:, None] * V[:, k]) ** 2).sum(1).mean() for k in range(3)]
    P = np.stack([X - c[0], Y - c[1], np.zeros_like(X)], -1)
    s = 0
    for k in range(3):
        d = P - (P @ V[:, k])[..., None] * V[:, k]
        s = s + (d ** 2).sum(-1) / max(f2[k], 1e-9)
    return fle * np.sqrt((1 + s / 3) / len(fid))


def fig_tre(out):
    L = 150
    g = np.linspace(-L, L, 301)
    X, Y = np.meshgrid(g, g)
    fig = plt.figure(figsize=(W, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.13, left=0.085, right=0.9, bottom=0.17, top=0.9)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    vmax = 4.5
    for ax, fid, title, levels in ((axs[0], CLUSTERED, "(a) Clustered fiducials", (0.5, 1, 2)),
                                   (axs[1], SPREAD, "(b) Spread fiducials", (0.5, 1))):
        T = tre_map(fid, X, Y)
        im = ax.imshow(T, extent=(-L, L, -L, L), origin="lower", cmap="Blues", vmin=0, vmax=vmax, zorder=0)
        cs = ax.contour(X, Y, T, levels=list(levels), colors=[FS.NAVY], linewidths=1.2)
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8.5, inline=True, inline_spacing=4)
        fx, fy = zip(*fid)
        ax.plot(fx, fy, "o", c=FS.GREEN, mec="white", mew=0.9, ms=6.5, zorder=5)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100])
        ax.set_xlabel("x (mm)"); ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(True)
    axs[0].set_ylabel("y (mm)")
    axs[1].set_yticklabels([])
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Predicted RMS TRE (mm)"); cb.ax.tick_params(labelsize=8.5)
    save(fig, out, "lit_fre_tre_layout")


FIGS = {"accuracy": fig_accuracy, "tree": fig_tree, "tre": fig_tre}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", nargs="*", choices=list(FIGS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in (a.only or list(FIGS)):
        FIGS[k](out)


if __name__ == "__main__":
    main()
