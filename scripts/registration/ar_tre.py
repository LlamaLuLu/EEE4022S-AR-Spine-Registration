"""
Real-time TRE computation for the live overlay.

Wraps fitzpatrick_tre.predict_tre for scalar values and adds the full
3x3 covariance computation + pinhole projection to get anisotropic
2D ellipses on the image plane.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRE_DIR = _REPO_ROOT / "scripts" / "tre_analysis"
if str(_TRE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRE_DIR))

from fitzpatrick_tre import predict_tre

DEFAULT_FLE_MM = 0.815


def _visible_centroids(visible_ids, corner_lut):
    """Extract fiducial centroids for the currently visible markers."""
    centroids = []
    for mid in visible_ids:
        if mid in corner_lut:
            centroids.append(corner_lut[mid].mean(axis=0))
    return np.array(centroids) if centroids else np.zeros((0, 3))


def compute_tre_scalar(visible_ids, corner_lut, targets_mm, fle_mm=DEFAULT_FLE_MM):
    """
    Fitzpatrick closed-form RMS TRE at each target using only visible markers.
    Returns (M,) array in mm.
    """
    fiducials = _visible_centroids(visible_ids, corner_lut)
    if len(fiducials) < 3:
        return np.full(len(targets_mm), float("inf"))
    return predict_tre(fiducials, targets_mm, fle_mm)


def compute_tre_covariance(visible_ids, corner_lut, target_mm, fle_mm=DEFAULT_FLE_MM):
    """
    Full 3x3 TRE covariance at a single target point (mm^2).

    Fitzpatrick 1998: in the principal-axis frame, the covariance is
    diagonal with sigma_k^2 = (FLE^2/N) * (1/3 + d_k^2 / (3*f_k^2)).
    Rotated back to phantom frame.
    """
    fiducials = _visible_centroids(visible_ids, corner_lut)
    N = len(fiducials)
    if N < 3:
        return np.eye(3) * 1e6

    centroid = fiducials.mean(axis=0)
    Fc = fiducials - centroid
    _, s, Vt = np.linalg.svd(Fc, full_matrices=False)
    f_sq = (s ** 2) / N

    t_pa = (target_mm - centroid) @ Vt.T
    d_sq = t_pa ** 2

    var_pa = np.zeros(3)
    for k in range(3):
        if f_sq[k] < 1e-12:
            var_pa[k] = 1e6 if d_sq[k] > 0 else (fle_mm ** 2 / N) / 3.0
        else:
            var_pa[k] = (fle_mm ** 2 / N) * (1.0 / 3.0 + d_sq[k] / (3.0 * f_sq[k]))

    V = Vt.T
    return V @ np.diag(var_pa) @ V.T


def project_covariance_ellipse(cov_3d, point_3d, rvec, tvec, camera_matrix):
    """
    Project a 3D TRE covariance to a 2D image-space ellipse via the
    Jacobian of the pinhole projection.

    Returns (center_px, (semi_major, semi_minor), angle_deg) or None.
    """
    pts_2d, _ = cv2.projectPoints(
        point_3d.reshape(1, 3).astype(np.float64),
        rvec, tvec, camera_matrix, None,
    )
    center = pts_2d.reshape(2)

    R, _ = cv2.Rodrigues(rvec)
    p_cam = R @ point_3d + tvec.flatten()
    X, Y, Z = p_cam
    if abs(Z) < 1e-6:
        return None

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]

    J_proj = np.array([
        [fx / Z, 0, -fx * X / (Z * Z)],
        [0, fy / Z, -fy * Y / (Z * Z)],
    ])
    J = J_proj @ R

    cov_2d = J @ cov_3d @ J.T
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.maximum(eigvals, 0)

    ax1 = float(np.sqrt(eigvals[1]))
    ax2 = float(np.sqrt(eigvals[0]))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))

    return (center, (ax1, ax2), angle)