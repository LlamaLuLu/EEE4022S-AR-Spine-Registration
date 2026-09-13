"""
Monte Carlo simulation of TRE using actual solvePnP.

Each marker is specified by BOTH its centroid AND its orientation
(a 3x3 rotation matrix mapping marker-local coords into the phantom
frame). Corner positions are computed correctly for markers on any
face of the bracket, not just the base plate.

Usage:
  python monte_carlo_tre.py \
      --calibration data/calibration/s22_ultra/camera_calibration.npz \
      --marker-size-mm 20 --pixel-noise-sigma 0.25 --n-trials 10000
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


# ---------- Geometry helpers ----------

def make_camera_pose(distance_mm: float = 350.0,
                     tilt_deg: float = 20.0) -> tuple:
    """(rvec, tvec) for a camera looking at phantom origin from above."""
    tilt = np.deg2rad(tilt_deg)
    cam_pos = np.array([0.0,
                        -distance_mm * np.sin(tilt),
                        +distance_mm * np.cos(tilt)])
    look = -cam_pos / np.linalg.norm(cam_pos)
    up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_world, -look);  right /= np.linalg.norm(right)
    up = np.cross(-look, right)
    R_world_to_cam = np.stack([right, up, look])
    rvec, _ = cv2.Rodrigues(R_world_to_cam)
    tvec = -R_world_to_cam @ cam_pos
    return rvec, tvec.reshape(3, 1)


def corners_from_oriented_marker(centroid_mm: np.ndarray,
                                 R_marker_to_phantom: np.ndarray,
                                 marker_size_mm: float) -> np.ndarray:
    """
    Corner positions in phantom frame, given a marker's centroid AND its
    orientation. R_marker_to_phantom: columns are the marker's local X, Y, Z
    axes expressed in phantom coords. Corners live in the marker's XY plane;
    local Z is the outward normal.
    """
    s = marker_size_mm / 2.0
    local_offsets = np.array([[-s, +s, 0], [+s, +s, 0],
                              [+s, -s, 0], [-s, -s, 0]])
    return centroid_mm + local_offsets @ R_marker_to_phantom.T


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
                    camera_pose, seed=0):
    """oriented_markers: list of (centroid_mm, R_marker_to_phantom) tuples."""
    corners_3d = np.vstack([
        corners_from_oriented_marker(c, R, marker_size_mm)
        for c, R in oriented_markers
    ]).astype(np.float32)
    rvec_true, tvec_true = camera_pose

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
        raise RuntimeError("All MC trials failed")
    return np.sqrt(sq_err_sum / n_ok), n_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--pixel-noise-sigma", type=float, default=0.25)
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--distance-mm", type=float, default=350.0)
    parser.add_argument("--tilt-deg", type=float, default=20.0)
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

    pose = make_camera_pose(args.distance_mm, args.tilt_deg)
    print(f"Camera at {args.distance_mm}mm, tilt {args.tilt_deg}°, "
          f"pixel noise σ={args.pixel_noise_sigma}px, {args.n_trials} trials/variant\n")

    results = {}
    for name, oriented in VARIANTS_ORIENTED.items():
        tre, n_ok = run_monte_carlo(oriented, targets, K, dist,
                                     args.marker_size_mm, args.pixel_noise_sigma,
                                     args.n_trials, pose, seed=args.seed)
        results[name] = tre
        print(f"  {name:<15} mean={tre.mean():.3f}mm  max={tre.max():.3f}mm  "
              f"({n_ok}/{args.n_trials} PnP ok)")

    print("\nRanking by mean Monte Carlo TRE:")
    for name, tre in sorted(results.items(), key=lambda kv: kv[1].mean()):
        print(f"  {name:<15} mean={tre.mean():.3f}mm  max={tre.max():.3f}mm")

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + list(labels) + ["mean_mm", "max_mm"])
        for name, tre in results.items():
            writer.writerow([name] + [f"{v:.4f}" for v in tre]
                            + [f"{tre.mean():.4f}", f"{tre.max():.4f}"])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()