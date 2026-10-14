"""
Click-to-check geometry (same maths as scripts/atp/measure_overlay_error.py), pure numpy and OpenCV.

The user clicks where the real entry dot is. The click is a ray; we take the point on that
ray at the same camera depth as the overlay target, express it in the phantom frame, and
compare it with the target. The error is in-plane: it has no component along the viewing
direction, because a single click cannot tell how deep the dot is.
"""
import cv2
import numpy as np


def entry_check(cam_px, rvec, tvec, K, D, target_mm):
    """cam_px: click in camera pixels. target_mm: the overlay target in the phantom frame.
    Returns a dict: err_mm, dv_mm (phantom frame, click minus target), Q_mm (click point,
    phantom frame), depth_mm (target depth in the camera), target_px (u, v), err_px."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    t = np.asarray(tvec, float).reshape(3, 1)
    tgt = np.asarray(target_mm, float).reshape(3)
    depth = float((R @ tgt.reshape(3, 1) + t)[2, 0])
    n = cv2.undistortPoints(np.array([[cam_px]], np.float64), K, D).reshape(2)
    Q = (R.T @ (np.array([[n[0] * depth], [n[1] * depth], [depth]]) - t)).ravel()
    dv = Q - tgt
    proj, _ = cv2.projectPoints(tgt.reshape(1, 3), rvec, tvec, K, D)
    tp = proj.reshape(2)
    return {
        "err_mm": float(np.linalg.norm(dv)), "dv_mm": dv, "Q_mm": Q, "depth_mm": depth,
        "target_px": (float(tp[0]), float(tp[1])),
        "err_px": float(np.hypot(cam_px[0] - tp[0], cam_px[1] - tp[1])),
    }