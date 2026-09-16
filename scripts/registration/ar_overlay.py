"""
Overlay drawing for the live AR feed.

Flat dark panels with thin accent lines, compact single-line HUD rows,
buffered PIL text compositing. Matches the aesthetic of clinical
navigation systems (7D FLASH, Medivis).
"""

import cv2
import numpy as np

from ar_tre import compute_tre_covariance, project_covariance_ellipse
from ar_theme import (
    BG_PANEL, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT, ACCENT_DIM,
    COL_GOOD, COL_WARN, COL_BAD, STATUS_STALE,
    TYPE, LAYOUT,
    TRE_GOOD_MM, TRE_BAD_MM,
    tre_colour, is_in_envelope,
)

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False


# ====================================================================
# Buffered text renderer
# ====================================================================

class TextRenderer:
    """
    All draw() calls go to an RGBA buffer. flush() composites once.
    """

    _MONO = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    _SANS = [
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]

    def __init__(self, h, w):
        self.h = h
        self.w = w
        self._fonts = {}
        self._buf = None
        self._ctx = None
        if _PIL:
            for key in ("small", "medium", "large"):
                sz = max(int(h * getattr(TYPE, key)), 8)
                self._fonts[("mono", key)] = self._load(self._MONO, sz)
                self._fonts[("sans", key)] = self._load(self._SANS, sz)

    def _load(self, candidates, sz):
        for p in candidates:
            try:
                return ImageFont.truetype(p, sz)
            except (OSError, IOError):
                continue
        try:
            return ImageFont.load_default(size=sz)
        except TypeError:
            return ImageFont.load_default()

    def begin(self):
        if _PIL:
            self._buf = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
            self._ctx = ImageDraw.Draw(self._buf)

    def flush(self, frame):
        if not _PIL or self._buf is None:
            return
        ov = np.array(self._buf)
        a = ov[:, :, 3:4].astype(np.float32) / 255.0
        if a.max() > 0:
            rgb_bgr = ov[:, :, :3][:, :, ::-1]
            frame[:] = (frame * (1.0 - a) + rgb_bgr * a).astype(np.uint8)
        self._buf = self._ctx = None

    def measure(self, text, key="medium", fam="mono"):
        if _PIL and (fam, key) in self._fonts:
            bb = self._fonts[(fam, key)].getbbox(text)
            return bb[2] - bb[0], bb[3] - bb[1]
        return len(text) * 8, 14

    def text(self, frame, s, x, y, key="medium", col=TEXT_PRIMARY, fam="mono"):
        x, y = int(x), int(y)
        if _PIL and self._ctx and (fam, key) in self._fonts:
            font = self._fonts[(fam, key)]
            rgb = (col[2], col[1], col[0])
            # Single dark shadow (down-right 1px)
            self._ctx.text((x + 1, y + 1), s, font=font, fill=(0, 0, 0, 180))
            self._ctx.text((x, y), s, font=font, fill=(*rgb, 255))
        else:
            sz = max(int(self.h * getattr(TYPE, key, 0.02)), 8)
            sc = sz / 22.0
            ty = y + int(sz * 0.85)
            cv2.putText(frame, s, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        sc, (0, 0, 0), max(1, int(sc)) + 1, cv2.LINE_AA)
            cv2.putText(frame, s, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        sc, col, max(1, int(sc)), cv2.LINE_AA)


# ====================================================================
# Flat panel
# ====================================================================

def panel(frame, x, y, w, h, accent=ACCENT):
    """Flat dark rect with a thin accent line on top."""
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), BG_PANEL, -1)
    cv2.addWeighted(ov, LAYOUT.panel_alpha, frame,
                    1 - LAYOUT.panel_alpha, 0, frame)
    # Accent bar
    cv2.line(frame, (x, y), (x + w, y), accent, LAYOUT.accent_h, cv2.LINE_AA)
    # Subtle bottom/right border
    cv2.line(frame, (x, y + h), (x + w, y + h), BORDER, 1, cv2.LINE_AA)


# ====================================================================
# Overlay elements
# ====================================================================

def draw_crosshair(frame, cx, cy, sz=5, col=TEXT_PRIMARY):
    cx, cy = int(cx), int(cy)
    cv2.line(frame, (cx - sz, cy), (cx + sz, cy), col, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - sz), (cx, cy + sz), col, 1, cv2.LINE_AA)


def draw_targets(frame, targets_mm, labels, tre_vals, rvec, tvec,
                 K, D, vis_ids, corner_lut, fle, in_env, tr,
                 show_ell=True, doff=0):
    pts, _ = cv2.projectPoints(targets_mm.astype(np.float64), rvec, tvec, K, D)
    pts = pts.reshape(-1, 2)
    sz = max(4, int(tr.h * 0.007))

    for i, (pt, lbl, tre) in enumerate(zip(pts, labels, tre_vals)):
        col = tre_colour(tre, in_env)
        cx, cy = pt[0], pt[1] - doff
        draw_crosshair(frame, cx, cy, sz, col)

        if show_ell and np.isfinite(tre):
            cov = compute_tre_covariance(vis_ids, corner_lut, targets_mm[i], fle)
            ell = project_covariance_ellipse(cov, targets_mm[i], rvec, tvec, K)
            if ell:
                ec, (a1, a2), ang = ell
                ec = (ec[0], ec[1] - doff)
                a1, a2 = min(max(a1, 3), 100), min(max(a2, 3), 100)
                cv2.ellipse(frame, (int(ec[0]), int(ec[1])),
                            (int(a1), int(a2)), ang, 0, 360,
                            col, 1, cv2.LINE_AA)

        tr.text(frame, lbl, cx + sz + 3, cy - sz - 2, "small", col)


def draw_trajectories(frame, targets_mm, pairs, rvec, tvec, K, D, doff=0):
    if not pairs:
        return
    pts, _ = cv2.projectPoints(targets_mm.astype(np.float64), rvec, tvec, K, D)
    pts = pts.reshape(-1, 2)
    for ia, ib in pairs:
        p1 = (int(pts[ia][0]), int(pts[ia][1] - doff))
        p2 = (int(pts[ib][0]), int(pts[ib][1] - doff))
        cv2.line(frame, p1, p2, ACCENT, 1, cv2.LINE_AA)


def draw_wireframe(frame, verts, edges, rvec, tvec, K, D, cam_pos):
    if verts is None:
        return
    pts, _ = cv2.projectPoints(verts.astype(np.float64), rvec, tvec, K, D)
    pts = pts.reshape(-1, 2).astype(np.int32)
    for i0, i1, n in edges:
        mid = (verts[i0] + verts[i1]) / 2.0
        if np.dot(n, cam_pos - mid) < 0:
            continue
        cv2.line(frame, tuple(pts[i0]), tuple(pts[i1]),
                 ACCENT_DIM, 1, cv2.LINE_AA)


# ====================================================================
# HUD: compact single-line rows, flat panels
# ====================================================================

def _row(tr, frame, x, y, label, value, val_col=TEXT_PRIMARY):
    """Label on left (muted), value on right (bright). Returns row height."""
    tr.text(frame, label, x, y, "small", TEXT_SECONDARY)
    lw, _ = tr.measure(label, "small")
    tr.text(frame, value, x + lw + 8, y, "small", val_col)
    _, rh = tr.measure(value, "small")
    return int(rh * 1.8)


def _state_col(state):
    return {"SEARCHING": STATUS_STALE, "VERIFYING": ACCENT,
            "NAVIGATING": COL_GOOD, "STALE": COL_BAD}.get(state, ACCENT)


# ====================================================================
# Axis triad gizmo (phantom LPS orientation indicator)
# ====================================================================

# LPS axis colours matching medical imaging convention:
#   +X = patient Left  → red
#   +Y = Posterior     → green
#   +Z = Superior      → blue
_AXIS_COLS = {
    "X": (80, 80, 220),   # BGR red
    "Y": (100, 200, 80),  # BGR green
    "Z": (220, 140, 60),  # BGR blue
}


def draw_axis_triad(frame, tr, rvec, tvec, K, D, cx, cy, size_px):
    """
    Draw a small 3D coordinate gizmo in the corner showing phantom LPS
    axes rotated by the current camera pose. Uses only the rotation
    (not translation) so the triad sits at a fixed screen position.
    """
    # Unit axis endpoints in phantom frame (arbitrary units, we project
    # then scale to pixels)
    R, _ = cv2.Rodrigues(rvec)
    axes_3d = np.eye(3)  # X, Y, Z unit vectors

    # Rotate axes into camera frame (no translation, no perspective)
    axes_cam = R @ axes_3d  # (3, 3), columns are axis vectors in cam frame

    # Simple orthographic projection: x → screen x, y → screen y (flipped),
    # z is depth for occlusion ordering
    endpoints = []
    for i, name in enumerate(["X", "Y", "Z"]):
        v = axes_cam[:, i]
        # Screen offset from centre (y is flipped in image coords)
        ex = int(cx + v[0] * size_px)
        ey = int(cy - v[1] * size_px)
        depth = v[2]
        endpoints.append((name, ex, ey, depth))

    # Sort back-to-front so closer axes draw on top
    endpoints.sort(key=lambda e: e[3], reverse=True)

    # Draw each axis: line + tip circle + label
    tip_r = max(3, size_px // 12)
    for name, ex, ey, depth in endpoints:
        col = _AXIS_COLS[name]
        # Fade axes pointing away from viewer
        alpha = 0.4 if depth > 0.3 else 1.0
        line_col = tuple(int(c * alpha) for c in col)
        cv2.line(frame, (cx, cy), (ex, ey), line_col, 2, cv2.LINE_AA)
        cv2.circle(frame, (ex, ey), tip_r, col, -1, cv2.LINE_AA)
        # Label offset slightly past the tip
        label_x = ex + (5 if ex >= cx else -12)
        label_y = ey - 8
        tr.text(frame, name, label_x, label_y, "small", col, "sans")


# ====================================================================
# Target radar (grid of coloured dots showing TRE status per target)
# ====================================================================

def draw_target_radar(frame, tr, tre_vals, labels, in_env, x, y, w, h):
    """
    Draw a compact grid of dots representing each target, coloured by
    TRE status. Gives at-a-glance overview of which pedicles are
    green/amber/red across the whole spine.
    """
    if tre_vals is None or len(tre_vals) == 0:
        return

    # Panel background
    panel(frame, x, y, w, h, ACCENT_DIM)

    p = LAYOUT.panel_pad
    tr.text(frame, "TARGETS", x + p, y + p, "small", TEXT_SECONDARY)

    # Group targets by (level, side) to render one dot per pedicle,
    # merging isthmus + entry into a single worst-case status.
    # Labels: 'L3_R' isthmus, 'L3_E_R' entry.
    groups = {}
    for i, lbl in enumerate(labels):
        parts = lbl.split("_")
        if len(parts) < 2:
            continue
        level = parts[0]
        side = parts[-1]
        key = (level, side)
        # Keep the worst (max) TRE across isthmus + entry per pedicle
        cur = groups.get(key, -1)
        if tre_vals[i] > cur:
            groups[key] = tre_vals[i]

    # Sort by level (L1 → L5), left before right for reading order
    def _sort_key(item):
        (level, side), _tre = item
        try:
            n = int(level[1:])
        except (ValueError, IndexError):
            n = 99
        return (n, 0 if side == "R" else 1)

    ordered = sorted(groups.items(), key=_sort_key)
    if not ordered:
        return

    # Grid layout: 2 columns (L / R), rows = num levels
    label_h = int(tr.h * 0.028)
    inner_y = y + p + label_h + 4
    inner_x = x + p
    inner_w = w - p * 2

    dot_r = max(4, int(min(h, w) * 0.045))
    col_gap = int(inner_w * 0.35)
    row_h = int(dot_r * 2.8)

    # Determine levels present
    levels = sorted({lv for (lv, _) in groups.keys()},
                    key=lambda s: int(s[1:]) if s[1:].isdigit() else 99)

    for row_i, level in enumerate(levels):
        ry = inner_y + row_i * row_h + dot_r
        # Level label at left
        tr.text(frame, level, inner_x, ry - dot_r,
                "small", TEXT_MUTED, "mono")

        # L dot
        if (level, "L") in groups:
            tre = groups[(level, "L")]
            c = tre_colour(tre, in_env)
            cv2.circle(frame, (inner_x + int(inner_w * 0.35), ry),
                       dot_r, c, -1, cv2.LINE_AA)
        # R dot
        if (level, "R") in groups:
            tre = groups[(level, "R")]
            c = tre_colour(tre, in_env)
            cv2.circle(frame, (inner_x + int(inner_w * 0.65), ry),
                       dot_r, c, -1, cv2.LINE_AA)

    # Column headers
    header_y = inner_y - 2
    tr.text(frame, "L", inner_x + int(inner_w * 0.35) - 4, header_y,
            "small", TEXT_MUTED, "sans")
    tr.text(frame, "R", inner_x + int(inner_w * 0.65) - 4, header_y,
            "small", TEXT_MUTED, "sans")


def draw_hud(frame, tr, n_markers, n_total, pnp_ok, reproj_rms,
             dist_mm, elev, az, in_env, state, tre_vals, labels,
             countdown, show_help=False, show_wire=True):
    h, w = frame.shape[:2]
    pad = max(int(h * 0.015), 8)
    row_h = max(int(h * 0.045), 22)
    p = LAYOUT.panel_pad

    # ---- Top-left: status ----
    pw = int(w * 0.22)
    ph = p * 2 + row_h * 3
    panel(frame, pad, pad, pw, ph, _state_col(state))
    x0, y0 = pad + p, pad + p

    mcol = COL_GOOD if n_markers >= 4 else COL_BAD
    dy = _row(tr, frame, x0, y0, "MARKERS", f"{n_markers}/{n_total}", mcol)

    pnp_str = "OK" if pnp_ok else "FAIL"
    pcol = COL_GOOD if pnp_ok else COL_BAD
    dy += _row(tr, frame, x0, y0 + dy, "PNP", pnp_str, pcol)

    rms_s = f"{reproj_rms:.1f}px" if np.isfinite(reproj_rms) else "--"
    _row(tr, frame, x0, y0 + dy, "RMS", rms_s)

    # ---- Top-right: pose ----
    pw_tr = int(w * 0.20)
    panel(frame, w - pad - pw_tr, pad, pw_tr, ph,
          COL_GOOD if in_env else COL_BAD)
    x0 = w - pad - pw_tr + p
    y0 = pad + p

    d_s = f"{dist_mm / 10:.0f} cm" if dist_mm > 0 else "--"
    dy = _row(tr, frame, x0, y0, "DIST", d_s)

    e_s = f"{elev:.0f}\u00b0" if dist_mm > 0 else "--"
    dy += _row(tr, frame, x0, y0 + dy, "ELEV", e_s,
               COL_GOOD if in_env else COL_BAD)

    a_s = f"{az:.0f}\u00b0" if dist_mm > 0 else "--"
    _row(tr, frame, x0, y0 + dy, "AZ", a_s,
         COL_GOOD if in_env else COL_BAD)

    # ---- Bottom-left: TRE ----
    pw_bl = int(w * 0.30)
    ph_bl = p * 2 + row_h * 2 + int(h * 0.035)
    panel(frame, pad, h - pad - ph_bl, pw_bl, ph_bl, ACCENT)
    x0 = pad + p
    y0 = h - pad - ph_bl + p

    if tre_vals is not None and len(tre_vals) > 0 and np.isfinite(tre_vals).any():
        best = int(np.argmin(tre_vals))
        bv = tre_vals[best]
        bl = labels[best] if labels else "?"
        bc = tre_colour(bv, in_env)
        dy = _row(tr, frame, x0, y0, "TARGET", bl, bc)
        _row(tr, frame, x0, y0 + dy, "TRE", f"{bv:.2f} mm", bc)
    else:
        _row(tr, frame, x0, y0, "TARGET", "--", TEXT_MUTED)

    # Legend row: dot + label triplet at the bottom of the panel
    leg_y = h - pad - p - int(h * 0.015)
    dot_r = max(4, int(h * 0.005))
    seg = int((pw_bl - p * 2) / 3)
    for i, (c, t) in enumerate([(COL_GOOD, f"< {TRE_GOOD_MM}mm"),
                                 (COL_WARN, f"{TRE_GOOD_MM}-{TRE_BAD_MM}mm"),
                                 (COL_BAD, f"> {TRE_BAD_MM}mm")]):
        dx = x0 + i * seg
        cv2.circle(frame, (dx + dot_r, leg_y), dot_r, c, -1, cv2.LINE_AA)
        tr.text(frame, t, dx + dot_r * 3, leg_y - dot_r,
                "small", TEXT_SECONDARY)

    # ---- Bottom-right: shortcuts ----
    if show_help:
        keys = [("c", "confirm"), ("r", "reject"), ("t", "trajectories"),
                ("e", "ellipses"), ("w", "wireframe"), ("d", "detach"),
                ("g", "axis gizmo"), ("x", "target radar"),
                ("h", "toggle hud"), ("esc", "toggle help"),
                ("q", "quit")]
        sh_w = int(w * 0.18)
        sh_h = p * 2 + row_h * len(keys)
        sx = w - pad - sh_w
        sy = h - pad - sh_h

        # Opaque background (higher alpha) for readability
        ov = frame.copy()
        cv2.rectangle(ov, (sx, sy), (sx + sh_w, sy + sh_h), BG_PANEL, -1)
        cv2.addWeighted(ov, 0.95, frame, 0.05, 0, frame)
        cv2.line(frame, (sx, sy), (sx + sh_w, sy),
                 ACCENT_DIM, LAYOUT.accent_h, cv2.LINE_AA)
        cv2.line(frame, (sx, sy + sh_h), (sx + sh_w, sy + sh_h),
                 BORDER, 1, cv2.LINE_AA)

        for i, (k, n) in enumerate(keys):
            ry = sy + p + i * row_h
            tr.text(frame, k, sx + p, ry, "small", ACCENT)
            tr.text(frame, n, sx + p + int(w * 0.03), ry,
                    "small", TEXT_PRIMARY)
    else:
        # Layer toggle indicators + help hint
        hint_x = w - pad - int(w * 0.10)
        hint_y = h - pad - int(h * 0.03)
        tr.text(frame, "ESC  HELP", hint_x, hint_y, "small", TEXT_SECONDARY)

    # ---- State pill (top center) ----
    label_map = {"SEARCHING": "SEARCHING", "NAVIGATING": "NAVIGATING",
                 "STALE": "STALE POSE",
                 "VERIFYING": f"VERIFY {max(0, countdown):.1f}s"}
    txt = label_map.get(state, state)
    sc = _state_col(state)
    tw, th = tr.measure(txt, "large", "sans")
    pw_pill = tw + int(w * 0.04)
    ph_pill = th + int(h * 0.02)
    px = (w - pw_pill) // 2
    py = pad
    panel(frame, px, py, pw_pill, ph_pill, sc)

    # Status dot
    cv2.circle(frame, (px + int(w * 0.008), py + ph_pill // 2),
               max(4, int(h * 0.006)), sc, -1, cv2.LINE_AA)
    tr.text(frame, txt, px + int(w * 0.018), py + int(h * 0.008),
            "large", TEXT_PRIMARY, "sans")