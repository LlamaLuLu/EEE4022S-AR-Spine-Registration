"""
Axial (cross-section) view of the selected vertebra.

The phantom frame: +X patient left, +Y posterior, +Z cranial. An axial slice is a cut at
constant Z. Drawn like a CT slice: viewed from the feet, anterior at the top, patient's
left on the right of the picture. The view needs no camera pose, so it is drawn straight
from the phantom model and never costs tracking time. The slice outline is cached per
pedicle; only the trajectory and the error circle are redrawn each frame.
"""
import cv2
import numpy as np

TEAL = (195, 195, 60)           # BGR, same teal as the rest of the UI
BG = (22, 18, 14)               # BGR, matches the dark panel colour
TEXT = (190, 179, 159)
NEUTRAL = (150, 150, 150)


def slice_mesh(verts, tris, z):
    """Cut a triangle mesh with the plane Z = z. Returns segments (S, 2, 2) in (X, Y) mm."""
    v = verts[tris].astype(np.float64)                 # (M, 3, 3)
    d = v[:, :, 2] - z
    d[d == 0] = 1e-9                                   # keep vertices off the plane
    pts, ok = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = d[:, a] * d[:, b] < 0
        t = np.where(cross, d[:, a] / (d[:, a] - d[:, b] + 1e-30), 0.0)
        p = v[:, a, :2] + t[:, None] * (v[:, b, :2] - v[:, a, :2])
        pts.append(p)
        ok.append(cross)
    P = np.stack(pts, axis=1)                          # (M, 3, 2)
    mask = np.stack(ok, axis=1)                        # (M, 3)
    sel = mask.sum(axis=1) == 2
    if not sel.any():
        return np.zeros((0, 2, 2))
    return P[sel][mask[sel]].reshape(-1, 2, 2)


class AxialSlice:
    """Slice of one vertebra at the pedicle's height, plus the pedicle's drawing geometry."""

    def __init__(self, mesh, entry_mm, isthmus_mm, name):
        self.name = name
        self.z = float(0.5 * (entry_mm[2] + isthmus_mm[2]))
        self.segs = slice_mesh(mesh.verts, mesh.tris, self.z)
        self.entry = np.asarray(entry_mm[:2], float)
        self.isthmus = np.asarray(isthmus_mm[:2], float)
        self.ok = len(self.segs) > 0
        pts = self.segs.reshape(-1, 2) if self.ok else np.stack([self.entry, self.isthmus])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.centre = 0.5 * (lo + hi)
        self.extent = hi - lo
        self._base = None
        self._base_key = None

    def _frame(self, w, h):
        """mm -> pixel mapping that fits the whole slice with a margin."""
        pad = 1.18
        s = min(w / (self.extent[0] * pad), h / (self.extent[1] * pad))
        return s, self.centre

    def _px(self, xy, w, h):
        s, c = self._frame(w, h)
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - c[0]) * s + w / 2.0, (xy[..., 1] - c[1]) * s + h / 2.0], axis=-1)

    def _build_base(self, w, h):
        img = np.empty((h, w, 3), np.uint8)
        img[:] = BG
        if self.ok:
            segs = self._px(self.segs, w, h)
            lines = np.zeros((h, w), np.uint8)
            cv2.polylines(lines, [np.round(s).astype(np.int32) for s in segs], False, 255, 1)
            lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))      # close pixel gaps
            # inside = not reachable from the border without crossing the outline
            ff = lines.copy()
            pad = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            m = np.zeros((h + 4, w + 4), np.uint8)
            cv2.floodFill(pad, m, (0, 0), 255)
            inside = pad[1:-1, 1:-1] == 0
            tint = np.empty_like(img)
            tint[:] = TEAL
            img[inside] = (img[inside] * 0.82 + tint[inside] * 0.18).astype(np.uint8)
            cv2.polylines(img, [np.round(s * 16).astype(np.int32) for s in segs], False, TEAL, 2,
                          cv2.LINE_AA, shift=4)
        # orientation labels and a 10 mm scale bar
        f = cv2.FONT_HERSHEY_SIMPLEX
        sc = max(0.5, min(w, h) / 700.0)
        for txt, pos in (("ANT", (w // 2, int(18 * sc * 2))), ("POST", (w // 2, h - 10)),
                         ("R", (10, h // 2)), ("L", (w - int(22 * sc * 1.6), h // 2))):
            (tw, th), _ = cv2.getTextSize(txt, f, sc, 1)
            org = (int(pos[0] - (tw / 2 if txt in ("ANT", "POST") else 0)), int(pos[1]))
            cv2.putText(img, txt, org, f, sc, TEXT, 1, cv2.LINE_AA)
        s, _ = self._frame(w, h)
        L = int(round(10 * s))
        y = h - int(26 * sc * 1.4)
        x0 = 12
        cv2.line(img, (x0, y), (x0 + L, y), TEXT, 2, cv2.LINE_AA)
        cv2.putText(img, "10 mm", (x0, y - 8), f, sc * 0.9, TEXT, 1, cv2.LINE_AA)
        return img

    def render(self, w, h, colour, r95_mm=None):
        """Draw the slice plus the trajectory (extended beyond the entry point) and, if given,
        the true-scale 95 % error circle at the entry point. Returns a BGR image."""
        if self._base is None or self._base_key != (w, h):
            self._base, self._base_key = self._build_base(w, h), (w, h)
        img = self._base.copy()
        s, _ = self._frame(w, h)
        a = self._px(self.isthmus, w, h)
        b = self._px(self.entry, w, h)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        far = b + d * 45.0 * s                          # guide runs out of the bone, as in the camera view
        sh = 4
        def q(p):
            return tuple(int(round(v * (1 << sh))) for v in p)
        cv2.line(img, q(a), q(far), (0, 0, 0), 5, cv2.LINE_AA, sh)
        cv2.line(img, q(a), q(far), colour, 3, cv2.LINE_AA, sh)
        if r95_mm is not None:
            cv2.circle(img, q(b), int(round(r95_mm * s * (1 << sh))), colour, 2, cv2.LINE_AA, sh)
        cv2.circle(img, q(b), 7, (255, 255, 255), -1, cv2.LINE_AA, sh)      # entry
        cv2.circle(img, q(a), 6, (0, 0, 0), -1, cv2.LINE_AA, sh)            # isthmus
        return img