"""
Slice views of the selected vertebra: axial (cross-section) and sagittal (side view along the pedicle).

Phantom frame: +X patient left, +Y posterior, +Z cranial.

axial     cut at constant Z through the pedicle. Drawn like a CT slice: anterior at the top,
          patient's left on the right of the picture.
sagittal  cut by the vertical plane that contains the pedicle axis (so the pedicle is cut along
          its full length and its cranio-caudal tilt is true). Anterior on the left, posterior
          on the right, cranial at the top.

Neither view needs the camera pose, so they cost nothing in tracking. The slice outline is built
once per pedicle and cached per pane size; only the trajectory (and error circle) is redrawn.
"""
import cv2
import numpy as np

TEAL = (195, 195, 60)           # BGR, same teal as the rest of the UI
BG = (22, 18, 14)               # BGR, matches the dark panel colour
TEXT = (190, 179, 159)
NEUTRAL = (150, 150, 150)


def slice_mesh(verts, tris, point, normal, e1, e2):
    """Cut a triangle mesh with the plane through `point` with unit `normal`.
    Returns segments (S, 2, 2) in plane coordinates (dot with e1, dot with e2), in mm."""
    point, normal = np.asarray(point, float), np.asarray(normal, float)
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    v = verts[tris].astype(np.float64)                 # (M, 3, 3)
    d = (v - point) @ normal                           # signed distance of each corner, (M, 3)
    d[d == 0] = 1e-9                                   # keep vertices off the plane
    pts, ok = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = d[:, a] * d[:, b] < 0
        t = np.where(cross, d[:, a] / (d[:, a] - d[:, b] + 1e-30), 0.0)
        p = v[:, a, :] + t[:, None] * (v[:, b, :] - v[:, a, :])
        pts.append(p)
        ok.append(cross)
    P = np.stack(pts, axis=1)                          # (M, 3, 3)
    mask = np.stack(ok, axis=1)                        # (M, 3)
    sel = mask.sum(axis=1) == 2
    if not sel.any():
        return np.zeros((0, 2, 2))
    q = P[sel][mask[sel]].reshape(-1, 2, 3) - point
    return np.stack([q @ e1, q @ e2], axis=-1)


class SliceView:
    """One slice of one vertebra plus the pedicle's trajectory, ready to draw at any pane size."""

    def __init__(self, mesh, entry_mm, isthmus_mm, kind="axial"):
        self.kind = kind
        entry, isth = np.asarray(entry_mm, float), np.asarray(isthmus_mm, float)
        mid = 0.5 * (entry + isth)
        if kind == "axial":
            self.n = np.array([0.0, 0.0, 1.0])
            self.e1 = np.array([1.0, 0.0, 0.0])        # image x: +X (patient left)
            self.e2 = np.array([0.0, 1.0, 0.0])        # image y down: +Y (posterior)
        else:
            h = np.array([entry[0] - isth[0], entry[1] - isth[1], 0.0])
            h /= max(np.linalg.norm(h), 1e-9)          # horizontal direction of the pedicle, towards the entry
            self.n = np.array([-h[1], h[0], 0.0])
            self.e1 = h                                # image x: towards the entry (posterior)
            self.e2 = np.array([0.0, 0.0, -1.0])       # image y down: -Z (cranial up)
        self.origin = mid
        self.segs = slice_mesh(mesh.verts, mesh.tris, mid, self.n, self.e1, self.e2)
        self.ok = len(self.segs) > 0
        self.entry = self._to2d(entry)
        self.isthmus = self._to2d(isth)
        pts = self.segs.reshape(-1, 2) if self.ok else np.stack([self.entry, self.isthmus])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.centre = 0.5 * (lo + hi)
        self.extent = np.maximum(hi - lo, 1.0)
        self._base = None
        self._base_key = None

    def _to2d(self, p):
        q = np.asarray(p, float) - self.origin
        return np.array([q @ self.e1, q @ self.e2])

    def _frame(self, w, h):
        pad = 1.18
        return min(w / (self.extent[0] * pad), h / (self.extent[1] * pad))

    def _px(self, xy, w, h):
        s = self._frame(w, h)
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.centre[0]) * s + w / 2.0,
                         (xy[..., 1] - self.centre[1]) * s + h / 2.0], axis=-1)

    def _build_base(self, w, h):
        img = np.empty((h, w, 3), np.uint8)
        img[:] = BG
        if self.ok:
            segs = self._px(self.segs, w, h)
            lines = np.zeros((h, w), np.uint8)
            cv2.polylines(lines, [np.round(s).astype(np.int32) for s in segs], False, 255, 1)
            lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))      # close pixel gaps
            pad = cv2.copyMakeBorder(lines, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            m = np.zeros((h + 4, w + 4), np.uint8)
            cv2.floodFill(pad, m, (0, 0), 255)                         # outside = reachable from the border
            inside = pad[1:-1, 1:-1] == 0
            tint = np.empty_like(img)
            tint[:] = TEAL
            img[inside] = (img[inside] * 0.82 + tint[inside] * 0.18).astype(np.uint8)
            cv2.polylines(img, [np.round(s * 16).astype(np.int32) for s in segs], False, TEAL, 2,
                          cv2.LINE_AA, shift=4)
        f = cv2.FONT_HERSHEY_SIMPLEX
        sc = max(0.5, min(w, h) / 700.0)

        def label(txt, cx, cy, align="c"):
            (tw, th), _ = cv2.getTextSize(txt, f, sc, 1)
            x = cx - tw / 2 if align == "c" else (cx if align == "l" else cx - tw)
            cv2.putText(img, txt, (int(x), int(cy + th / 2)), f, sc, TEXT, 1, cv2.LINE_AA)

        m_ = int(14 * sc * 1.6)
        e_ = int(10 * sc * 1.6)                                       # edge inset
        if self.kind == "axial":
            label("ANT", w / 2, m_); label("POST", w / 2, h - m_)
            label("R", e_, h / 2, "l"); label("L", w - e_, h / 2, "r")
        else:
            label("CRA", w / 2, m_); label("CAU", w / 2, h - m_)
            label("ANT", e_, m_, "l"); label("POST", w - e_, m_, "r")    # corners: the trajectory leaves mid-right
        L = int(round(10 * self._frame(w, h)))                        # 10 mm scale bar
        y = h - int(26 * sc * 1.4)
        x0 = 12
        cv2.line(img, (x0, y), (x0 + L, y), TEXT, 2, cv2.LINE_AA)
        cv2.putText(img, "10 mm", (x0, y - 8), f, sc * 0.9, TEXT, 1, cv2.LINE_AA)
        return img

    def render(self, w, h, colour, r95_mm=None, top=0):
        """Slice plus the trajectory (running on past the entry point) and, for the axial view, the
        true-scale 95 % error circle at the entry point. Returns a BGR image of size (h, w).
        `top` pixels at the top are left empty for the pane's title tag, so nothing collides with it."""
        if top > 0:
            body = self.render(w, max(h - top, 8), colour, r95_mm, 0)
            strip = np.empty((top, w, 3), np.uint8)
            strip[:] = BG
            return np.vstack([strip, body])
        if self._base is None or self._base_key != (w, h):
            self._base, self._base_key = self._build_base(w, h), (w, h)
        img = self._base.copy()
        s = self._frame(w, h)
        a = self._px(self.isthmus, w, h)
        b = self._px(self.entry, w, h)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        far = b + d * 45.0 * s
        sh = 4

        def q(p):
            return tuple(int(round(v * (1 << sh))) for v in p)
        cv2.line(img, q(a), q(far), (0, 0, 0), 5, cv2.LINE_AA, sh)
        cv2.line(img, q(a), q(far), colour, 3, cv2.LINE_AA, sh)
        if r95_mm is not None and self.kind == "axial":
            cv2.circle(img, q(b), int(round(r95_mm * s * (1 << sh))), colour, 2, cv2.LINE_AA, sh)
        cv2.circle(img, q(b), 7, (255, 255, 255), -1, cv2.LINE_AA, sh)      # entry
        cv2.circle(img, q(a), 6, (0, 0, 0), -1, cv2.LINE_AA, sh)            # isthmus
        return img