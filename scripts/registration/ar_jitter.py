"""
Rolling pose jitter: how much the overlay wobbles from frame to frame, in mm.

No Qt and no drawing, so it can be unit tested and reused by the phone build.

Method. Keep the last N poses. For each target X, its position in the camera frame is
P_i = R_i X + t_i. A hand-held camera drifts slowly, so each axis is detrended with a
straight line over the window (least squares against time) and the leftover is the
jitter, as an RMS in mm. Slow, steady motion therefore scores about 0. Frame to frame
shake (marker corner noise, glare, blur, few or distant markers) scores high.
Depth is reported on its own, because the noise from a single camera is mostly along
the viewing axis.
"""
from collections import deque

import cv2
import numpy as np

MIN_SAMPLES = 6          # fewer poses than this and the number means little
MAX_GAP_S = 0.5          # a longer gap between poses starts a new window


class JitterMonitor:
    def __init__(self, n=15, max_gap=MAX_GAP_S):
        self.n = n
        self.max_gap = max_gap
        self._rot = deque(maxlen=n)       # 3x3 rotation matrices
        self._t = deque(maxlen=n)         # translations, shape (3,)
        self._time = deque(maxlen=n)

    def reset(self):
        self._rot.clear()
        self._t.clear()
        self._time.clear()

    def push(self, rvec, tvec, now):
        """Add one valid pose. Call reset() instead when there is no pose this frame."""
        if self._time and (now - self._time[-1] > self.max_gap or now <= self._time[-1]):
            self.reset()
        R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
        self._rot.append(R)
        self._t.append(np.asarray(tvec, float).ravel())
        self._time.append(float(now))

    def update(self, r):
        """Feed one FrameResult: only a live, solved pose counts."""
        if r.rvec is None or not r.pnp_ok or r.state not in ("VERIFYING", "NAVIGATING"):
            self.reset()
        else:
            self.push(r.rvec, r.tvec, r.now)

    @property
    def n_samples(self):
        return len(self._time)

    def measure(self, points_mm):
        """Per target (M, 3 array, phantom frame): (total, depth, lateral) jitter in mm,
        each an array of length M. Returns None until there are MIN_SAMPLES poses."""
        k = len(self._time)
        if k < MIN_SAMPLES:
            return None
        X = np.asarray(points_mm, float).reshape(-1, 3)
        R = np.stack(self._rot)                                   # (k, 3, 3)
        t = np.stack(self._t)                                     # (k, 3)
        P = np.einsum("kij,mj->kmi", R, X) + t[:, None, :]        # (k, M, 3) camera frame
        tt = np.asarray(self._time)
        tt = tt - tt.mean()
        A = np.stack([np.ones(k), tt], axis=1)                    # straight line against time
        coef, *_ = np.linalg.lstsq(A, P.reshape(k, -1), rcond=None)
        res = P.reshape(k, -1) - A @ coef
        var = (res ** 2).sum(axis=0) / (k - 2)                    # per axis, 2 fitted parameters
        var = var.reshape(-1, 3)                                  # (M, 3)
        total = np.sqrt(var.sum(axis=1))
        depth = np.sqrt(var[:, 2])
        lateral = np.sqrt(var[:, 0] + var[:, 1])
        return total, depth, lateral