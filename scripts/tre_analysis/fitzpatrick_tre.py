"""
Fitzpatrick TRE prediction for a rigid point-based registration.

Reference: Fitzpatrick, West & Maurer (1998), "Predicting error in
rigid-body point-based registration", IEEE TMI 17(5).

Assumes N discrete fiducial points with independent isotropic Gaussian
localisation error (variance FLE^2), rigid least-squares (Procrustes)
registration. For an ArUco marker board, we treat each marker's centroid
as one effective fiducial.
"""

import numpy as np


def predict_tre(fiducials_mm: np.ndarray,
                targets_mm: np.ndarray,
                fle_mm: float) -> np.ndarray:
    """
    Predicted RMS TRE at each target, in mm.

    fiducials_mm : (N, 3) fiducial positions in some common frame
    targets_mm   : (M, 3) target positions in the SAME frame
    fle_mm       : scalar fiducial localisation error (RMS, 3D)

    Returns (M,) array of predicted RMS TRE values in mm.
    """
    N = len(fiducials_mm)
    if N < 3:
        raise ValueError(f"Need at least 3 fiducials, got {N}")

    # Centre the fiducial cloud on its own centroid.
    centroid = fiducials_mm.mean(axis=0)
    F = fiducials_mm - centroid                    # (N, 3)

    # Principal axes of the fiducial cloud via SVD.
    # Columns of Vt.T are the principal axis directions in the fiducial frame.
    # Singular values give the RMS spread of fiducials along each axis:
    #   f_k^2 = (s_k^2) / N   (mean-squared distance from the axis-k plane)
    _, s, Vt = np.linalg.svd(F, full_matrices=False)
    f_sq = (s ** 2) / N                            # (3,) mean-squared spreads

    # For each target, express its displacement from the fiducial centroid
    # in the principal-axis frame, then compute distance-to-each-axis.
    T = targets_mm - centroid                      # (M, 3)
    T_pa = T @ Vt.T                                # (M, 3) in principal axes

    # Distance from target to principal axis k is the norm of the OTHER
    # two components. Equivalently, sum of squared coords excluding axis k.
    coord_sq = T_pa ** 2                           # (M, 3)
    total_sq = coord_sq.sum(axis=1, keepdims=True) # (M, 1)
    d_sq = total_sq - coord_sq                     # (M, 3): d_k^2 per axis

    # Fitzpatrick formula:
    #   TRE^2(target) = (FLE^2 / N) * (1 + (1/3) * sum_k d_k^2 / (f_a^2 + f_b^2))
    # where {a, b} are the OTHER two axes (not k). This is the standard form
    # from Fitzpatrick 1998 eq. (46). We compute f_a^2 + f_b^2 per axis.
    f_other_sum = f_sq.sum() - f_sq               # (3,) sum of the other two f_k^2

    # Guard against a degenerate (coplanar) fiducial set: if f_k is zero for
    # some k, the corresponding term blows up ONLY when d_k > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        term = d_sq / f_other_sum                  # (M, 3)
        term = np.where(f_other_sum > 1e-12, term, np.inf * (d_sq > 0))

    tre_sq = (fle_mm ** 2 / N) * (1.0 + term.sum(axis=1) / 3.0)
    return np.sqrt(tre_sq)


# --- Sanity check against a known symmetric case ---
if __name__ == "__main__":
    # A regular tetrahedron of fiducials around the origin, centroid at origin.
    # By symmetry all principal-axis spreads are equal, and TRE at the centroid
    # (target at origin) should equal FLE * sqrt(1/N) exactly.
    tetra = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float) * 50.0                        # 50mm scale

    fle = 0.5                                     # mm
    tre_at_centroid = predict_tre(tetra, np.zeros((1, 3)), fle)[0]
    expected = fle * np.sqrt(1.0 / len(tetra))
    print(f"Tetrahedron, target at centroid:")
    print(f"  predicted TRE = {tre_at_centroid:.4f} mm")
    print(f"  expected      = {expected:.4f} mm  (FLE * sqrt(1/N))")

    # Planar case: 4 markers on a plane, target off-plane should blow up in
    # the direction perpendicular to the plane.
    planar = np.array([
        [ 50,  50, 0],
        [ 50, -50, 0],
        [-50,  50, 0],
        [-50, -50, 0],
    ], dtype=float)
    off_plane = np.array([[0, 0, 30]])            # 30mm above the plane
    on_plane_off_centre = np.array([[30, 0, 0]])

    print(f"\nPlanar board:")
    print(f"  TRE at off-plane target   = {predict_tre(planar, off_plane, fle)[0]:.4f} mm")
    print(f"  TRE at in-plane off-centre = {predict_tre(planar, on_plane_off_centre, fle)[0]:.4f} mm")