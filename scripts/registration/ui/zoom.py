"""
Zoom view geometry: a magnified crop of the camera frame around one 3D point,
plus the camera matrix that goes with it. Anything drawn with the returned
matrix (the same projection code the main view uses) lands exactly on the crop,
at full resolution, so lines stay sharp and the error circle stays true to scale.
Pure numpy and OpenCV, no Qt.
"""
import cv2
import numpy as np


def project_point(p_mm, rvec, tvec, K, D):
    """(pixel x, pixel y, depth mm) of a 3D point in the phantom frame, or None if behind the camera."""
    p = np.asarray(p_mm, np.float64).reshape(3)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))
    z = float((R @ p + np.asarray(tvec, np.float64).reshape(3))[2])
    if z <= 1e-6:
        return None
    c, _ = cv2.projectPoints(p.reshape(1, 1, 3), rvec, tvec, K, D)
    return float(c[0, 0, 0]), float(c[0, 0, 1]), z


def zoom_frame(clean, K, centre_px, px_per_mm, out_w, out_h, fov_mm):
    """
    Crop `clean` around centre_px so that fov_mm of the object spans the output width,
    and resize it to out_w x out_h.
    Returns (image, K_zoom, magnification). Areas outside the camera frame are black.
    """
    s = out_w / (fov_mm * px_per_mm)                 # output pixels per camera pixel
    x0 = centre_px[0] - out_w / (2.0 * s)            # crop origin in camera pixels
    y0 = centre_px[1] - out_h / (2.0 * s)
    M = np.array([[s, 0.0, -x0 * s],
                  [0.0, s, -y0 * s]], np.float64)
    interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
    img = cv2.warpAffine(clean, M, (out_w, out_h), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    return img, M3 @ np.asarray(K, np.float64), s