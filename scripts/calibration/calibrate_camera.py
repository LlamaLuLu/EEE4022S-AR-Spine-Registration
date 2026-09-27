"""
Runs ChArUco camera calibration on the images saved by
capture_calibration_images.py and writes camera_calibration.npz
containing camera_matrix, dist_coeffs, and the RMS reprojection error.

Reports:
  - overall RMS reprojection error (training set)
  - per-image RMS, sorted worst first, with outliers flagged
  - held-out RMS on images excluded from the fit (--holdout N), which
    shows whether the calibration generalises beyond the images it saw

Usage:
    python calibrate_camera.py --imagedir calibration_images --outfile camera_calibration.npz
    python calibrate_camera.py --imagedir ... --outfile ... --holdout 10
    python calibrate_camera.py --imagedir ... --outfile ... --holdout 10 --exclude calib_007.png calib_023.png
"""
import argparse
import glob
import os
import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH_M = 0.029
MARKER_LENGTH_M = 0.0145
OUTLIER_FACTOR = 2.0   # flag images with RMS > 2x the median


def per_image_rms(obj_pts, img_pts, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(err ** 2))), err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", default="calibration_images")
    parser.add_argument("--outfile", default="camera_calibration.npz")
    parser.add_argument("--holdout", type=int, default=0,
                        help="Number of images held out for validation (e.g. 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the held-out split (reproducible)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Image filenames to drop (e.g. blurry outliers)")
    args = parser.parse_args()

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, DICTIONARY
    )
    detector = cv2.aruco.CharucoDetector(board)

    paths = sorted(glob.glob(f"{args.imagedir}/*.png"))
    paths = [p for p in paths if os.path.basename(p) not in set(args.exclude)]
    if args.exclude:
        print(f"Excluded {len(args.exclude)} image(s): {', '.join(args.exclude)}")
    if len(paths) < 5:
        print(f"Only found {len(paths)} images in {args.imagedir}, need more before calibrating.")
        return

    # --- Detect and match every image, keeping filenames aligned ---
    names, obj_list, img_list = [], [], []
    image_size = None
    for path in paths:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        corners, ids, _, _ = detector.detectBoard(gray)
        if (ids is None or corners is None
                or len(ids) != len(corners) or len(ids) <= 4):
            print(f"Skipping {os.path.basename(path)}, board not clearly detected.")
            continue
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 6:
            print(f"Skipping {os.path.basename(path)}, too few matched corners.")
            continue
        names.append(os.path.basename(path))
        obj_list.append(obj_pts)
        img_list.append(img_pts)

    n = len(names)
    print(f"Usable images: {n} of {len(paths)}")

    # --- Train / held-out split ---
    idx = np.arange(n)
    if args.holdout > 0:
        if n - args.holdout < 10:
            print(f"Holdout of {args.holdout} leaves fewer than 10 training images. "
                  f"Capture more or reduce --holdout.")
            return
        rng = np.random.default_rng(args.seed)
        test_idx = np.sort(rng.choice(idx, args.holdout, replace=False))
        train_idx = np.setdiff1d(idx, test_idx)
    else:
        test_idx = np.array([], dtype=int)
        train_idx = idx

    if len(train_idx) < 5:
        print("Not enough training images, capture more.")
        return

    # --- Calibrate on training set ---
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        [obj_list[i] for i in train_idx],
        [img_list[i] for i in train_idx],
        image_size, None, None,
    )

    # --- Per-image training error ---
    train_errs = []
    for k, i in enumerate(train_idx):
        e, _ = per_image_rms(obj_list[i], img_list[i], rvecs[k], tvecs[k], K, D)
        train_errs.append((names[i], e, len(obj_list[i])))

    med = np.median([e for _, e, _ in train_errs])
    print(f"\n=== Per-image RMS (training, {len(train_idx)} images), worst first ===")
    print(f"{'image':<18s} {'RMS px':>8s} {'corners':>8s}")
    for name, e, nc in sorted(train_errs, key=lambda x: -x[1]):
        flag = "  <-- outlier" if e > OUTLIER_FACTOR * med else ""
        print(f"{name:<18s} {e:8.3f} {nc:8d}{flag}")
    outliers = [nm for nm, e, _ in train_errs if e > OUTLIER_FACTOR * med]
    if outliers:
        print(f"\nOutliers (> {OUTLIER_FACTOR:.0f}x median {med:.3f}px). Inspect them; if blurry "
              f"or glared, rerun with:\n  --exclude {' '.join(outliers)}")

    # --- Held-out validation ---
    test_rms = None
    test_errs = []
    if len(test_idx) > 0:
        all_sq, total = 0.0, 0
        for i in test_idx:
            ok, rvec, tvec = cv2.solvePnP(obj_list[i], img_list[i], K, D)
            if not ok:
                print(f"Held-out {names[i]}: solvePnP failed")
                continue
            e, err = per_image_rms(obj_list[i], img_list[i], rvec, tvec, K, D)
            test_errs.append((names[i], e))
            all_sq += float(np.sum(err ** 2))
            total += len(err)
        test_rms = float(np.sqrt(all_sq / total))
        print(f"\n=== Held-out validation ({len(test_idx)} images, seed {args.seed}) ===")
        for name, e in test_errs:
            print(f"{name:<18s} {e:8.3f}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Training RMS:  {rms:.4f} px  ({len(train_idx)} images)")
    if test_rms is not None:
        ratio = test_rms / rms
        print(f"Held-out RMS:  {test_rms:.4f} px  ({len(test_idx)} images)")
        print(f"Ratio held-out / training: {ratio:.2f}"
              f"{'  (generalises well)' if ratio < 1.5 else '  (check for overfitting / poor pose variety)'}")
    print(f"Image size:    {image_size[0]}x{image_size[1]}")
    print("Camera matrix:\n", K)
    print("Distortion coefficients [k1 k2 p1 p2 k3]:\n", D.ravel())

    np.savez(args.outfile,
             camera_matrix=K, dist_coeffs=D, rms_error=rms,
             image_size=image_size,
             heldout_rms=np.nan if test_rms is None else test_rms,
             train_images=np.array([names[i] for i in train_idx]),
             heldout_images=np.array([names[i] for i in test_idx]),
             per_image_rms=np.array([e for _, e, _ in train_errs]))
    print(f"\nSaved calibration to {args.outfile}")


if __name__ == "__main__":
    main()