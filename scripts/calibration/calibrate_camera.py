"""
Runs ChArUco camera calibration on the images saved by
capture_calibration_images.py and writes camera_calibration.npz
containing camera_matrix, dist_coeffs, and the RMS reprojection error.

Usage:
    python calibrate_camera.py
    python calibrate_camera.py --imagedir calibration_images --outfile camera_calibration.npz
"""
import argparse
import glob
import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.015


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", default="calibration_images")
    parser.add_argument("--outfile", default="camera_calibration.npz")
    args = parser.parse_args()

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, DICTIONARY
    )
    detector = cv2.aruco.CharucoDetector(board)

    all_corners, all_ids = [], []
    image_size = None

    paths = sorted(glob.glob(f"{args.imagedir}/*.png"))
    if len(paths) < 5:
        print(f"Only found {len(paths)} images in {args.imagedir}, need more before calibrating.")
        return

    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if (charuco_ids is not None
                and charuco_corners is not None
                and len(charuco_ids) == len(charuco_corners)
                and len(charuco_ids) > 4):
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        else:
            print(f"Skipping {path}, board not clearly detected.")

    print(f"Using {len(all_corners)} of {len(paths)} images.")
    if len(all_corners) < 5:
        print("Not enough usable images, capture more or improve lighting/focus.")
        return

    all_obj_points, all_img_points = [], []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and len(obj_pts) >= 4:
            all_obj_points.append(obj_pts)
            all_img_points.append(img_pts)

    if len(all_obj_points) < 5:
        print("Not enough usable images after matching, capture more.")
        return

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, image_size, None, None
    )

    print(f"RMS reprojection error: {rms_error:.4f} px")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    np.savez(args.outfile, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
             rms_error=rms_error, image_size=image_size)
    print(f"Saved calibration to {args.outfile}")


if __name__ == "__main__":
    main()