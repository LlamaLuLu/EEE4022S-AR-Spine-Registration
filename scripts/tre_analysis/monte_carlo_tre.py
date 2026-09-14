"""
Monte Carlo simulation of TRE using actual solvePnP.

Coordinate frame: LPS phantom-frame (see design_variants.py).
Camera pose set by (distance, elevation, azimuth):
    elevation_deg : angle above horizontal XZ plane. 90 = directly above.
    azimuth_deg   : rotation around vertical +Y axis.
                    0=caudal (-Z), 90=patient left (+X), 180=cranial (+Z),
                    270=patient right (-X).

Visibility filters applied before PnP:
  1. Angle: marker rejected if its outward normal makes more than
     max-view-angle-deg with the marker-to-camera vector.
  2. Occlusion: marker rejected if its line-of-sight to the camera passes
     through the phantom bounding box.

Usage:
  python monte_carlo_tre.py \
      --calibration data/calibration/s22_ultra/camera_calibration.npz \
      --marker-size-mm 20 --pixel-noise-sigma 0.121 --n-trials 10000 \
      --azimuth-deg 45 --elevation-deg 60
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import cv2

from design_variants import (
    PLACEHOLDER_PEDICLE_TARGETS, PLACEHOLDER_PEDICLE_LABELS,
    load_pedicles_from_json,
)
from design_variants_oriented import VARIANTS_ORIENTED


# ---------- Phantom cylinder for occlusion ----------
# The phantom (vertebrae + processes) is approximated as a cylinder aligned
# with the spine axis (Z). Cross-section is a circle in the XY plane. This is
# tighter than an axis-aligned box: transverse processes stick out laterally
# but don't fill the box's corners.
PHANTOM_CYLINDER_CENTER_XY = np.array([+5.0, +50.0])   # (X, Y) of cylinder axis
PHANTOM_CYLINDER_RADIUS_MM = 50.0                       # captures body + processes
PHANTOM_CYLINDER_Z_MIN = -95.0                          # caudal end (near L5)
PHANTOM_CYLINDER_Z_MAX = +85.0                          # cranial end (near L1)


# ---------- Geometry helpers ----------

def make_camera_pose(distance_mm: float = 350.0,
                     elevation_deg: float = 30.0,
                     azimuth_deg: float = 45.0) -> tuple:
    """Camera position (LPS) from (distance, elevation, azimuth)."""
    el = np.deg2rad(elevation_deg)
    az = np.deg2rad(azimuth_deg)
    cam_pos = np.array([
        +distance_mm * np.cos(el) * np.sin(az),   # X
        +distance_mm * np.sin(el),                # Y (up)
        -distance_mm * np.cos(el) * np.cos(az),   # Z (az=0 -> -Z = caudal)
    ])

    # Build camera axes in world frame (OpenCV convention: +X right, +Y down,
    # +Z forward). Right-handed basis has right x down = forward.
    forward = -cam_pos / np.linalg.norm(cam_pos)  # camera looks at origin
    world_up = np.array([0.0, 0.0, 1.0])          # cranial (+Z) appears up
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    R_world_to_cam = np.stack([right, down, forward])
    rvec, _ = cv2.Rodrigues(R_world_to_cam)
    tvec = -R_world_to_cam @ cam_pos
    return rvec, tvec.reshape(3, 1)


def corners_from_oriented_marker(centroid_mm: np.ndarray,
                                 R_marker_to_phantom: np.ndarray,
                                 marker_size_mm: float) -> np.ndarray:
    s = marker_size_mm / 2.0
    local_offsets = np.array([[-s, +s, 0], [+s, +s, 0],
                              [+s, -s, 0], [-s, -s, 0]])
    return centroid_mm + local_offsets @ R_marker_to_phantom.T


# ---------- Visibility filters ----------

def ray_intersects_cylinder(origin: np.ndarray, direction: np.ndarray,
                             center_xy: np.ndarray, radius: float,
                             z_min: float, z_max: float) -> bool:
    """
    Ray-cylinder intersection for a cylinder aligned with the Z axis.
    Cross-section: circle centered at center_xy with given radius.
    Clipped to Z range [z_min, z_max].

    origin, direction: 3D vectors. Ray segment is origin + t * direction for
    t in [0, 1] (so |direction| = distance to endpoint).

    Returns True if the segment intersects the cylinder's INTERIOR.
    """
    ax = origin[0] - center_xy[0]
    ay = origin[1] - center_xy[1]
    dx = direction[0]
    dy = direction[1]
    dz = direction[2]

    A = dx * dx + dy * dy

    # Degenerate case: ray runs parallel to Z axis (no XY motion).
    if A < 1e-12:
        d_sq = ax * ax + ay * ay
        if d_sq > radius * radius:
            return False   # ray stays outside cylinder in XY
        # Ray is inside cylinder XY. Check Z overlap.
        z_low = min(origin[2], origin[2] + dz)
        z_high = max(origin[2], origin[2] + dz)
        return z_high >= z_min and z_low <= z_max

    B = 2.0 * (ax * dx + ay * dy)
    C = ax * ax + ay * ay - radius * radius

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return False   # ray never crosses cylinder wall in XY

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    # Clip [t1, t2] to segment [0, 1].
    t_enter = max(t1, 0.0)
    t_exit  = min(t2, 1.0)
    if t_enter > t_exit:
        return False   # cylinder interior segment doesn't overlap [0, 1]

    # Also require that during this interval, Z lies within cylinder's Z range.
    z_enter = origin[2] + t_enter * dz
    z_exit  = origin[2] + t_exit  * dz
    z_low = min(z_enter, z_exit)
    z_high = max(z_enter, z_exit)
    return z_high >= z_min and z_low <= z_max


def marker_is_visible(centroid: np.ndarray,
                      R_marker_to_phantom: np.ndarray,
                      cam_pos_phantom: np.ndarray,
                      max_view_angle_deg: float = 70.0,
                      check_occlusion: bool = True) -> bool:
    """Angle-based and occlusion-based visibility check."""
    to_cam = cam_pos_phantom - centroid
    dist = np.linalg.norm(to_cam)
    to_cam_unit = to_cam / dist

    # 1. Angle check: marker normal (local Z = column 2 of R) vs to_cam.
    normal = R_marker_to_phantom[:, 2]
    cos_angle = np.dot(normal, to_cam_unit)
    if cos_angle < np.cos(np.deg2rad(max_view_angle_deg)):
        return False

    # 2. Occlusion check: ray from marker (slightly offset along normal to
    #    avoid self-intersecting the bracket surface) toward camera.
    if check_occlusion:
        origin = centroid + normal * 0.5
        if ray_intersects_cylinder(origin, to_cam,
                                    PHANTOM_CYLINDER_CENTER_XY,
                                    PHANTOM_CYLINDER_RADIUS_MM,
                                    PHANTOM_CYLINDER_Z_MIN,
                                    PHANTOM_CYLINDER_Z_MAX):
            return False

    return True


def filter_visible_markers(oriented_markers: list,
                           rvec_true: np.ndarray, tvec_true: np.ndarray,
                           max_view_angle_deg: float,
                           check_occlusion: bool) -> list:
    R_true, _ = cv2.Rodrigues(rvec_true)
    cam_pos_phantom = -R_true.T @ tvec_true.ravel()
    return [
        (c, R) for c, R in oriented_markers
        if marker_is_visible(c, R, cam_pos_phantom,
                              max_view_angle_deg, check_occlusion)
    ]


# ---------- One MC trial ----------

def single_trial(marker_corners_3d, target_positions_3d,
                 rvec_true, tvec_true, K, dist,
                 pixel_noise_sigma, rng):
    corners_2d, _ = cv2.projectPoints(marker_corners_3d, rvec_true, tvec_true, K, dist)
    corners_2d = corners_2d.reshape(-1, 2)
    corners_2d_noisy = corners_2d + rng.normal(0.0, pixel_noise_sigma,
                                                corners_2d.shape).astype(np.float32)

    ok, rvec_est, tvec_est = cv2.solvePnP(
        marker_corners_3d.astype(np.float32),
        corners_2d_noisy.astype(np.float32),
        K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    R_true, _ = cv2.Rodrigues(rvec_true)
    R_est,  _ = cv2.Rodrigues(rvec_est)
    targets_cam_true = (R_true @ target_positions_3d.T).T + tvec_true.T
    targets_phantom_recovered = ((targets_cam_true - tvec_est.T) @ R_est)
    return np.sqrt(((target_positions_3d - targets_phantom_recovered) ** 2).sum(axis=1))


# ---------- MC across variants ----------

def run_monte_carlo(oriented_markers, target_positions_mm, K, dist,
                    marker_size_mm, pixel_noise_sigma, n_trials,
                    camera_pose, max_view_angle_deg=70.0,
                    check_occlusion=True, seed=0):
    rvec_true, tvec_true = camera_pose
    n_total = len(oriented_markers)
    visible = filter_visible_markers(oriented_markers, rvec_true, tvec_true,
                                      max_view_angle_deg, check_occlusion)
    n_visible = len(visible)

    if n_visible < 2:
        return None, 0, n_visible, n_total

    corners_3d = np.vstack([
        corners_from_oriented_marker(c, R, marker_size_mm)
        for c, R in visible
    ]).astype(np.float32)

    rng = np.random.default_rng(seed)
    M = len(target_positions_mm)
    sq_err_sum = np.zeros(M)
    n_ok = 0
    for _ in range(n_trials):
        err = single_trial(corners_3d, target_positions_mm,
                            rvec_true, tvec_true, K, dist,
                            pixel_noise_sigma, rng)
        if err is None:
            continue
        sq_err_sum += err ** 2
        n_ok += 1

    if n_ok == 0:
        return None, 0, n_visible, n_total
    return np.sqrt(sq_err_sum / n_ok), n_ok, n_visible, n_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--pixel-noise-sigma", type=float, default=0.25)
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--distance-mm", type=float, default=350.0)
    parser.add_argument("--elevation-deg", type=float, default=30.0)
    parser.add_argument("--azimuth-deg", type=float, default=45.0)
    parser.add_argument("--max-view-angle-deg", type=float, default=70.0)
    parser.add_argument("--no-occlusion-check", action="store_true")
    parser.add_argument("--pedicles", type=str, default=None)
    parser.add_argument("--outfile", type=str,
                        default="results/tre_analysis/monte_carlo_table.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.calibration)
    K, dist = data["camera_matrix"], data["dist_coeffs"]

    if args.pedicles:
        targets, labels = load_pedicles_from_json(args.pedicles)
    else:
        targets = PLACEHOLDER_PEDICLE_TARGETS
        labels = PLACEHOLDER_PEDICLE_LABELS

    pose = make_camera_pose(args.distance_mm, args.elevation_deg, args.azimuth_deg)
    print(f"Camera: distance={args.distance_mm}mm, "
          f"elevation={args.elevation_deg} deg, azimuth={args.azimuth_deg} deg")
    print(f"Noise: pixel sigma={args.pixel_noise_sigma}px, "
          f"{args.n_trials} trials/variant")
    print(f"Visibility: max angle={args.max_view_angle_deg} deg, "
          f"occlusion={'off' if args.no_occlusion_check else 'on'}\n")

    results = {}
    diagnostics = {}
    for name, oriented in VARIANTS_ORIENTED.items():
        tre, n_ok, n_vis, n_tot = run_monte_carlo(
            oriented, targets, K, dist,
            args.marker_size_mm, args.pixel_noise_sigma,
            args.n_trials, pose,
            max_view_angle_deg=args.max_view_angle_deg,
            check_occlusion=not args.no_occlusion_check,
            seed=args.seed,
        )
        diagnostics[name] = (n_vis, n_tot, n_ok)
        if tre is None:
            print(f"  {name:<15} FAILED  (visible {n_vis}/{n_tot} markers)")
            results[name] = None
        else:
            results[name] = tre
            print(f"  {name:<15} mean={tre.mean():.3f}mm  max={tre.max():.3f}mm  "
                  f"(visible {n_vis}/{n_tot}, PnP ok {n_ok}/{args.n_trials})")

    print("\nRanking by mean Monte Carlo TRE:")
    valid = {n: t for n, t in results.items() if t is not None}
    for name, tre in sorted(valid.items(), key=lambda kv: kv[1].mean()):
        print(f"  {name:<15} mean={tre.mean():.3f}mm  max={tre.max():.3f}mm")
    failed = [n for n, t in results.items() if t is None]
    if failed:
        print(f"\nFailed variants (insufficient visible markers): {failed}")

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + list(labels)
                        + ["mean_mm", "max_mm", "n_visible", "n_total"])
        for name, tre in results.items():
            n_vis, n_tot, _ = diagnostics[name]
            if tre is None:
                writer.writerow([name] + ["FAIL"] * len(labels)
                                + ["FAIL", "FAIL", n_vis, n_tot])
            else:
                writer.writerow([name] + [f"{v:.4f}" for v in tre]
                                + [f"{tre.mean():.4f}", f"{tre.max():.4f}",
                                    n_vis, n_tot])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()