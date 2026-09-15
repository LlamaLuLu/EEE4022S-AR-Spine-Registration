"""
All overlay drawing for the live AR feed.

Renders pedicle crosshairs, TRE confidence ellipses, trajectory lines,
STL wireframe, and the four-corner HUD. Text scales to frame size.
Uses Pillow for crisp monospace text when available, falls back to
OpenCV putText otherwise.
"""

import cv2
import numpy as np
from ar_tre import compute_tre_covariance, project_covariance_ellipse

# Try Pillow for better text; graceful fallback
try:
    from PIL import Image, ImageDraw, ImageFont
    _PILLOW_OK = True
except ImportError:
    _PILLOW_OK = False


# ====================================================================
# Colour palette (BGR)
# ====================================================================

COL_GREEN  = (100, 200, 80)
COL_RED    = (80, 80, 230)
COL_WHITE  = (230, 230, 230)
COL_CYAN   = (210, 190, 90)
COL_GREY   = (120, 120, 120)
COL_AMBER  = (50, 180, 240)
COL_BG     = (20, 20, 20)

TRE_GREEN_MM = 1.0
TRE_RED_MM   = 2.0

# Operational envelope bounds
ENVELOPE_ELEV = (30.0, 45.0)
ENVELOPE_AZ   = (30.0, 60.0)


def is_in_envelope(elev, az):
    return (ENVELOPE_ELEV[0] <= elev <= ENVELOPE_ELEV[1] and
            ENVELOPE_AZ[0] <= az <= ENVELOPE_AZ[1])


def tre_colour(tre_mm, in_envelope):
    if not in_envelope:
        return COL_RED
    if tre_mm < TRE_GREEN_MM:
        return COL_GREEN
    if tre_mm < TRE_RED_MM:
        return COL_AMBER
    return COL_RED


# ====================================================================
# Scaled text rendering
# ====================================================================

class TextRenderer:
    """
    Renders text scaled to frame dimensions. Uses Pillow when available
    for crisp monospace glyphs; falls back to OpenCV otherwise.
    """

    def __init__(self, frame_h, frame_w):
        self.h = frame_h
        self.w = frame_w
        # Base unit: 1/60th of frame height gives readable text at any res
        self.unit = max(frame_h / 60, 10)
        self._pil_font_sm = None
        self._pil_font_md = None
        self._pil_font_lg = None

        if _PILLOW_OK:
            try:
                sz_sm = int(self.unit * 0.85)
                sz_md = int(self.unit)
                sz_lg = int(self.unit * 1.4)
                # Try system monospace fonts in priority order
                for name in ["DejaVuSansMono.ttf", "Menlo.ttc",
                             "Consolas.ttf", "LiberationMono-Regular.ttf",
                             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
                    try:
                        self._pil_font_sm = ImageFont.truetype(name, sz_sm)
                        self._pil_font_md = ImageFont.truetype(name, sz_md)
                        self._pil_font_lg = ImageFont.truetype(name, sz_lg)
                        break
                    except (OSError, IOError):
                        continue
                if self._pil_font_sm is None:
                    self._pil_font_sm = ImageFont.load_default()
                    self._pil_font_md = self._pil_font_sm
                    self._pil_font_lg = self._pil_font_sm
            except Exception:
                pass

    def draw(self, frame, text, pos, size="md", colour=COL_WHITE):
        """
        Draw text at pos (x, y). size: 'sm', 'md', 'lg'.
        """
        x, y = int(pos[0]), int(pos[1])
        bgr = colour

        if _PILLOW_OK and self._pil_font_sm is not None:
            font = {"sm": self._pil_font_sm,
                    "md": self._pil_font_md,
                    "lg": self._pil_font_lg}.get(size, self._pil_font_md)
            # Convert BGR frame to RGB PIL image, draw, convert back
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            rgb = (bgr[2], bgr[1], bgr[0])
            # Outline for contrast
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
            draw.text((x, y), text, font=font, fill=rgb)
            frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            scale_map = {"sm": 0.35, "md": 0.45, "lg": 0.65}
            scale = scale_map.get(size, 0.45) * (self.unit / 14)
            thickness = max(1, int(scale))
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0,0,0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, bgr, thickness, cv2.LINE_AA)


# ====================================================================
# Drawing primitives
# ====================================================================

def draw_crosshair(frame, centre, size=6, colour=COL_WHITE, thickness=1):
    cx, cy = int(centre[0]), int(centre[1])
    cv2.line(frame, (cx - size, cy), (cx + size, cy), colour, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), colour, thickness, cv2.LINE_AA)


def draw_rounded_rect(frame, pt1, pt2, colour, alpha=0.55, radius=8):
    """Semi-transparent rounded rectangle."""
    overlay = frame.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    # OpenCV doesn't have rounded rects; approximate with filled rect
    # plus corner circles
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ====================================================================
# Target + trajectory drawing
# ====================================================================

def draw_targets(frame, targets_mm, labels, tre_values, rvec, tvec,
                 camera_matrix, dist_coeffs, visible_ids, corner_lut,
                 fle_mm, in_envelope, tr, show_ellipses=True, detach_offset=0):
    """Project targets and draw crosshairs + TRE ellipses."""
    pts_2d, _ = cv2.projectPoints(
        targets_mm.astype(np.float64), rvec, tvec, camera_matrix, dist_coeffs,
    )
    pts_2d = pts_2d.reshape(-1, 2)
    cross_size = max(4, int(tr.unit * 0.4))

    for i, (pt, label, tre_mm) in enumerate(zip(pts_2d, labels, tre_values)):
        col = tre_colour(tre_mm, in_envelope)
        cx, cy = pt[0], pt[1] - detach_offset

        draw_crosshair(frame, (cx, cy), size=cross_size, colour=col)

        if show_ellipses and np.isfinite(tre_mm):
            cov = compute_tre_covariance(visible_ids, corner_lut,
                                         targets_mm[i], fle_mm)
            ell = project_covariance_ellipse(cov, targets_mm[i],
                                            rvec, tvec, camera_matrix)
            if ell is not None:
                ecenter, (ea1, ea2), eangle = ell
                ecenter = (ecenter[0], ecenter[1] - detach_offset)
                ea1 = min(max(ea1, 2), 120)
                ea2 = min(max(ea2, 2), 120)
                cv2.ellipse(frame, (int(ecenter[0]), int(ecenter[1])),
                            (int(ea1), int(ea2)), eangle, 0, 360,
                            col, 1, cv2.LINE_AA)

        tr.draw(frame, label, (int(cx) + cross_size + 3, int(cy) - 2),
                size="sm", colour=col)


def draw_trajectories(frame, targets_mm, pairs, rvec, tvec,
                      camera_matrix, dist_coeffs, detach_offset=0):
    """Draw entry-to-isthmus trajectory lines."""
    if not pairs:
        return
    pts_2d, _ = cv2.projectPoints(
        targets_mm.astype(np.float64), rvec, tvec, camera_matrix, dist_coeffs,
    )
    pts_2d = pts_2d.reshape(-1, 2)

    for i_isth, i_entry in pairs:
        p1 = (int(pts_2d[i_isth][0]), int(pts_2d[i_isth][1] - detach_offset))
        p2 = (int(pts_2d[i_entry][0]), int(pts_2d[i_entry][1] - detach_offset))
        cv2.line(frame, p1, p2, COL_CYAN, 1, cv2.LINE_AA)


def draw_wireframe(frame, mesh_verts, mesh_edges, rvec, tvec,
                   camera_matrix, dist_coeffs, cam_pos):
    """Project and draw backface-culled STL wireframe."""
    if mesh_verts is None:
        return
    pts_2d, _ = cv2.projectPoints(
        mesh_verts.astype(np.float64), rvec, tvec, camera_matrix, dist_coeffs,
    )
    pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)

    for i0, i1, normal in mesh_edges:
        mid = (mesh_verts[i0] + mesh_verts[i1]) / 2.0
        if np.dot(normal, cam_pos - mid) < 0:
            continue
        cv2.line(frame, tuple(pts_2d[i0]), tuple(pts_2d[i1]),
                 COL_CYAN, 1, cv2.LINE_AA)


# ====================================================================
# HUD
# ====================================================================

def draw_hud(frame, tr, n_markers, n_total, pnp_ok, reproj_rms,
             distance_mm, elev_deg, az_deg, in_envelope,
             state, tre_values, labels, countdown_remaining,
             show_help=False):
    """Render the four-corner HUD with scaled text."""
    h, w = frame.shape[:2]
    pad = int(tr.unit * 0.7)
    line_h = int(tr.unit * 1.3)

    # Panel backgrounds
    draw_rounded_rect(frame, (0, 0), (int(w * 0.25), pad + line_h * 3), COL_BG)
    draw_rounded_rect(frame, (w - int(w * 0.25), 0), (w, pad + line_h * 3), COL_BG)
    draw_rounded_rect(frame, (0, h - pad - line_h * 3), (int(w * 0.36), h), COL_BG)

    # Top-left: pipeline status
    x0, y0 = pad, pad + line_h
    marker_col = COL_GREEN if n_markers >= 4 else COL_RED
    tr.draw(frame, f"markers {n_markers}/{n_total}", (x0, y0), "md", marker_col)
    pnp_col = COL_GREEN if pnp_ok else COL_RED
    tr.draw(frame, f"pnp: {'ok' if pnp_ok else 'FAIL'}", (x0, y0 + line_h), "md", pnp_col)
    tr.draw(frame, f"rms {reproj_rms:.1f}px", (x0, y0 + line_h * 2), "md")

    # Top-right: pose
    xr = w - int(w * 0.24)
    tr.draw(frame, f"dist {distance_mm / 10:.0f}cm", (xr, y0), "md")
    tr.draw(frame, f"elev {elev_deg:.0f}  az {az_deg:.0f}", (xr, y0 + line_h), "md")
    env_col = COL_GREEN if in_envelope else COL_RED
    tr.draw(frame, f"envelope: {'ok' if in_envelope else 'OUT'}", (xr, y0 + line_h * 2), "md", env_col)

    # Bottom-left: TRE legend + nearest target
    yb = h - pad - line_h * 2
    dot_r = max(4, int(tr.unit * 0.3))
    cv2.circle(frame, (pad + dot_r, yb), dot_r, COL_GREEN, -1)
    tr.draw(frame, f"<{TRE_GREEN_MM}mm", (pad + dot_r * 3, yb + dot_r // 2), "sm")
    off1 = int(w * 0.09)
    cv2.circle(frame, (pad + off1, yb), dot_r, COL_AMBER, -1)
    tr.draw(frame, f"{TRE_GREEN_MM}-{TRE_RED_MM}mm", (pad + off1 + dot_r * 2, yb + dot_r // 2), "sm")
    off2 = int(w * 0.19)
    cv2.circle(frame, (pad + off2, yb), dot_r, COL_RED, -1)
    tr.draw(frame, f">{TRE_RED_MM}mm", (pad + off2 + dot_r * 2, yb + dot_r // 2), "sm")

    if tre_values is not None and len(tre_values) > 0:
        best_idx = int(np.argmin(tre_values))
        best_lbl = labels[best_idx] if labels else "?"
        tr.draw(frame, f"nearest: {best_lbl}  TRE {tre_values[best_idx]:.2f}mm",
                (pad, yb + line_h), "md")

    # Bottom-right: shortcuts
    if show_help:
        keys = ["t traj", "e rings", "w wire", "d detach", "h hud", "q quit"]
        for i, s in enumerate(keys):
            tr.draw(frame, s, (w - int(w * 0.14), h - pad - (len(keys) - i) * int(line_h * 0.8)),
                    "sm", COL_GREY)
    else:
        tr.draw(frame, "? shortcuts", (w - int(w * 0.14), h - pad), "sm", COL_GREY)

    # State banner
    cx = w // 2
    if state == "SEARCHING":
        tr.draw(frame, "SEARCHING", (cx - int(tr.unit * 3.5), int(tr.unit * 1.8)), "lg", COL_GREY)
    elif state == "VERIFYING":
        rem = max(0, countdown_remaining)
        tr.draw(frame, f"VERIFY  {rem:.1f}s  [c]onfirm  [r]eject",
                (cx - int(tr.unit * 8), int(tr.unit * 1.8)), "md", COL_CYAN)
    elif state == "STALE":
        tr.draw(frame, "STALE POSE", (cx - int(tr.unit * 3.5), int(tr.unit * 1.8)), "lg", COL_RED)