"""
Measure the empirical fiducial localisation error (FLE) of the ArUco
marker centroid detection pipeline.

Method: point the camera at a static ArUco board and capture N frames.
For each detected marker, compute the 3D centroid from PnP (per frame).
The standard deviation of each marker's centroid across frames, converted
to 3D at the nominal working distance, is the FLE that feeds Fitzpatrick.

Usage:
  python measure_fle.py \
      --imagedir data/fle_measurement/s22_ultra/ \
      --calibration data/calibration/s22_ultra/camera_calibration.npz \
      --marker-size-mm 20 \
      --dictionary 6X6_50 \
      --outfile results/tre_analysis/fle_measurement.json

Assumes: static camera, static board, well-lit, focus locked at working
distance. Any motion during capture inflates the measured FLE.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2


DICTIONARIES = {
    "6X6_50":  cv2.aruco.DICT_6X6_50,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
}


def load_calibration(path: Path):
    """Load OpenCV camera calibration from .npz."""
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def detect_markers(img: np.ndarray, dictionary) -> tuple:
    """Return (corners_list, ids). corners_list is a list of (4,2) arrays."""
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    return corners, ids


def marker_centroid_3d(corners: np.ndarray,
                       marker_size_mm: float,
                       K: np.ndarray,
                       dist: np.ndarray) -> np.ndarray | None:
    """
    Recover 3D marker centre in camera frame using solvePnP on the 4 corners.
    corners: (4, 2) image points in the standard ArUco corner order.
    Returns (3,) position in mm, or None if PnP failed.
    """
    s = marker_size_mm / 2.0
    # OpenCV's canonical ArUco marker corners in marker frame (Z=0 plane).
    obj_pts = np.array([[-s, +s, 0], [+s, +s, 0],
                        [+s, -s, 0], [-s, -s, 0]], dtype=np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return tvec.ravel() if ok else None


def marker_corners_2d(corners: np.ndarray) -> np.ndarray:
    """Extract the 4 detected corner pixel positions as (4, 2)."""
    return corners.reshape(-1, 2)


def process_frames(image_paths: list,
                   K: np.ndarray,
                   dist: np.ndarray,
                   marker_size_mm: float,
                   dictionary) -> tuple:
    """
    Return (centroids_3d, corners_2d) where each is a dict:
      centroids_3d[marker_id]: (F, 3) 3D positions across frames
      corners_2d[marker_id]:   (F, 4, 2) 2D corner pixel positions across frames
    Only markers seen in ALL frames are included, so per-marker statistics
    are computed over an identical sample size.
    """
    per_marker_pos: dict[int, list] = {}
    per_marker_cor: dict[int, list] = {}

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  skipped (unreadable): {img_path.name}")
            continue

        corners, ids = detect_markers(img, dictionary)
        if ids is None:
            print(f"  no markers detected: {img_path.name}")
            continue

        for marker_corners, marker_id in zip(corners, ids.ravel()):
            pos = marker_centroid_3d(marker_corners[0], marker_size_mm, K, dist)
            if pos is None:
                continue
            mid = int(marker_id)
            per_marker_pos.setdefault(mid, []).append(pos)
            per_marker_cor.setdefault(mid, []).append(marker_corners_2d(marker_corners[0]))

    # Keep only markers detected in every frame.
    F = len(image_paths)
    centroids = {mid: np.array(pl) for mid, pl in per_marker_pos.items() if len(pl) == F}
    corners_by_marker = {mid: np.array(pl) for mid, pl in per_marker_cor.items() if len(pl) == F}
    return centroids, corners_by_marker


def compute_fle(centroids: dict) -> dict:
    """
    3D FLE per marker: RMS 3D deviation of the centroid from its mean position.
    """
    per_marker = {}
    for mid, positions in centroids.items():
        residuals = positions - positions.mean(axis=0)        # (F, 3)
        per_marker[mid] = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

    values = np.array(list(per_marker.values()))
    return {
        "per_marker_mm":  per_marker,
        "mean_mm":        float(values.mean()),
        "median_mm":      float(np.median(values)),
        "max_mm":         float(values.max()),
        "n_markers_used": len(per_marker),
    }


def compute_pixel_noise(corners_by_marker: dict) -> dict:
    """
    2D corner noise per marker: RMS pixel deviation from each corner's mean
    position, averaged across the 4 corners. Isotropic assumption (X and Y
    treated symmetrically), consistent with Monte Carlo's noise model.
    """
    per_marker = {}
    for mid, corners in corners_by_marker.items():           # (F, 4, 2)
        residuals = corners - corners.mean(axis=0)           # (F, 4, 2)
        # Per-corner RMS 2D pixel deviation, then mean across the 4 corners.
        per_corner_rms = np.sqrt((residuals ** 2).sum(axis=2).mean(axis=0))  # (4,)
        per_marker[mid] = float(per_corner_rms.mean())

    values = np.array(list(per_marker.values()))
    # Also return the average per-axis sigma (what MC actually samples from).
    # 2D RMS = sqrt(2) * sigma_per_axis for isotropic Gaussian noise.
    return {
        "per_marker_2d_rms_px":  per_marker,
        "mean_2d_rms_px":        float(values.mean()),
        "median_2d_rms_px":      float(np.median(values)),
        "sigma_per_axis_px":     float(values.mean() / np.sqrt(2.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagedir", type=str, required=True,
                        help="directory of static-board capture frames")
    parser.add_argument("--calibration", type=str, required=True,
                        help="camera calibration .npz")
    parser.add_argument("--marker-size-mm", type=float, required=True,
                        help="physical marker side length in mm")
    parser.add_argument("--dictionary", type=str, default="6X6_50",
                        choices=list(DICTIONARIES.keys()))
    parser.add_argument("--outfile", type=str, default="results/tre_analysis/fle_measurement.json")
    args = parser.parse_args()

    K, dist = load_calibration(Path(args.calibration))
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[args.dictionary])

    image_paths = sorted(Path(args.imagedir).glob("*.jpg")) + \
                  sorted(Path(args.imagedir).glob("*.png"))
    if len(image_paths) < 30:
        print(f"WARNING: only {len(image_paths)} frames found; "
              f"recommend >=100 for stable FLE estimate")

    print(f"Processing {len(image_paths)} frames from {args.imagedir}...")
    centroids, corners_by_marker = process_frames(
        image_paths, K, dist, args.marker_size_mm, dictionary)

    if not centroids:
        print("ERROR: no markers were detected consistently across all frames.")
        return

    fle = compute_fle(centroids)
    pxn = compute_pixel_noise(corners_by_marker)

    print(f"\n3D FLE per marker (RMS deviation of centroid from mean position):")
    for mid, v in sorted(fle["per_marker_mm"].items()):
        print(f"  marker {mid:3d}: {v:.4f} mm")

    print(f"\n2D corner noise per marker (RMS pixel deviation):")
    for mid, v in sorted(pxn["per_marker_2d_rms_px"].items()):
        print(f"  marker {mid:3d}: {v:.4f} px")

    print(f"\nAggregate over {fle['n_markers_used']} markers:")
    print(f"  3D FLE (for Fitzpatrick):")
    print(f"    mean   = {fle['mean_mm']:.4f} mm")
    print(f"    median = {fle['median_mm']:.4f} mm")
    print(f"    max    = {fle['max_mm']:.4f} mm")
    print(f"  2D corner noise (for Monte Carlo):")
    print(f"    mean RMS       = {pxn['mean_2d_rms_px']:.4f} px")
    print(f"    median RMS     = {pxn['median_2d_rms_px']:.4f} px")
    print(f"    sigma per axis = {pxn['sigma_per_axis_px']:.4f} px  ← use for --pixel-noise-sigma")

    # Combine both into one JSON output.
    result = {"fle": fle, "pixel_noise": pxn}
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nDrop-in commands:")
    print(f"  python evaluate_designs.py --fle {fle['median_mm']:.3f}")
    print(f"  python monte_carlo_tre.py ... --pixel-noise-sigma {pxn['sigma_per_axis_px']:.3f}")


if __name__ == "__main__":
    main()