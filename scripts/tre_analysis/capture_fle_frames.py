"""
Capture static frames for FLE measurement from IP Webcam.

Shows a live preview with the current camera-to-board distance overlaid,
computed from ArUco detection + solvePnP. When distance is close to the
target and everything looks static, press SPACE to start capture; ESC to
exit.

During capture: fetches N frames as fast as possible, saves as sequential
JPGs, and reports frame-to-frame centroid drift so you know if the setup
moved during capture.

Usage:
  python capture_fle_frames.py \
      --url http://192.168.1.42:8080/shot.jpg \
      --calibration data/calibration/s22_ultra/camera_calibration.npz \
      --marker-size-mm 20 \
      --outdir data/fle_measurement/s22_ultra/ \
      --target-distance-mm 350 \
      --n-frames 100
"""

import argparse
from pathlib import Path
import time
from urllib.request import urlopen
import numpy as np
import cv2


def fetch_frame(url: str) -> np.ndarray | None:
    """Grab a single JPEG frame from IP Webcam."""
    try:
        with urlopen(url, timeout=2.0) as resp:
            data = np.frombuffer(resp.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  fetch failed: {e}")
        return None


def measure_distance_mm(gray: np.ndarray,
                        detector,
                        marker_size_mm: float,
                        K: np.ndarray, dist: np.ndarray) -> tuple:
    """
    Detect markers, run PnP per marker, return (mean_distance_mm, n_markers).
    Returns (None, 0) if no markers detected.
    """
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, 0

    s = marker_size_mm / 2.0
    obj_pts = np.array([[-s, +s, 0], [+s, +s, 0],
                        [+s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    distances = []
    for marker_corners in corners:
        img_pts = marker_corners[0].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            distances.append(float(np.linalg.norm(tvec)))
    return (float(np.mean(distances)), len(distances)) if distances else (None, 0)


def draw_overlay(img: np.ndarray, dist_mm: float | None,
                 target_mm: float, n_markers: int, ready: bool) -> np.ndarray:
    """Overlay distance readout and status on the preview frame."""
    h, w = img.shape[:2]
    out = img.copy()
    if dist_mm is None:
        text = f"no markers detected (need at least 1)"
        colour = (0, 0, 255)
    else:
        delta = dist_mm - target_mm
        text = f"dist = {dist_mm:6.1f} mm  (target {target_mm:.0f}, delta {delta:+.1f}) | {n_markers} markers"
        colour = (0, 200, 0) if ready else (0, 165, 255)
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(out, "SPACE = start capture | ESC = quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if ready:
        cv2.putText(out, "READY", (w - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    return out


def preview_loop(url: str, detector, marker_size_mm: float,
                 K: np.ndarray, dist: np.ndarray,
                 target_mm: float, tolerance_mm: float) -> bool:
    """Show live preview until user presses SPACE (capture) or ESC (quit)."""
    print("Preview: SPACE to capture, ESC to quit")
    while True:
        img = fetch_frame(url)
        if img is None:
            time.sleep(0.2)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d, n = measure_distance_mm(gray, detector, marker_size_mm, K, dist)
        ready = d is not None and abs(d - target_mm) < tolerance_mm and n >= 4
        cv2.imshow("FLE preview", draw_overlay(img, d, target_mm, n, ready))
        key = cv2.waitKey(30) & 0xFF
        if key == 27:              # ESC
            return False
        if key == 32:              # SPACE
            return True


def capture_frames(url: str, n_frames: int, outdir: Path,
                   detector, marker_size_mm: float,
                   K: np.ndarray, dist: np.ndarray) -> list:
    """Fetch and save n_frames as sequential JPGs. Return per-frame distance list."""
    outdir.mkdir(parents=True, exist_ok=True)
    distances = []
    for i in range(n_frames):
        img = fetch_frame(url)
        if img is None:
            continue
        fname = outdir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d, _ = measure_distance_mm(gray, detector, marker_size_mm, K, dist)
        if d is not None:
            distances.append(d)
        print(f"  saved {fname.name}  dist={d:.1f}mm" if d else
              f"  saved {fname.name}  (no markers)")
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True,
                        help="IP Webcam frame URL, e.g. http://<phone>:8080/shot.jpg")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target-distance-mm", type=float, default=350.0)
    parser.add_argument("--tolerance-mm", type=float, default=10.0)
    parser.add_argument("--n-frames", type=int, default=100)
    args = parser.parse_args()

    data = np.load(args.calibration)
    K, dist = data["camera_matrix"], data["dist_coeffs"]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    proceed = preview_loop(args.url, detector, args.marker_size_mm,
                            K, dist, args.target_distance_mm, args.tolerance_mm)
    cv2.destroyAllWindows()

    if not proceed:
        print("Aborted.")
        return

    print(f"\nCapturing {args.n_frames} frames to {args.outdir}...")
    distances = capture_frames(args.url, args.n_frames, Path(args.outdir),
                                detector, args.marker_size_mm, K, dist)

    if distances:
        d = np.array(distances)
        print(f"\nCapture done. Distance stability across frames:")
        print(f"  mean   = {d.mean():.2f} mm")
        print(f"  std    = {d.std():.3f} mm  (should be << 1mm if nothing moved)")
        print(f"  range  = {d.max()-d.min():.3f} mm")
        if d.std() > 1.0:
            print(f"  WARNING: distance varied by more than 1mm — setup may have moved.")


if __name__ == "__main__":
    main()