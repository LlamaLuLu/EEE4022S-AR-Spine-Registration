"""
Live ArUco marker detection, pose estimation, and axis overlay.
Requires camera_calibration.npz from calibrate_camera.py.

Uses solvePnP directly rather than cv2.aruco.estimatePoseSingleMarkers,
which was deprecated in OpenCV 4.7 and removed in later versions.

Usage:
    python live_pose_overlay.py --source 0 --marker-length 0.05
    python live_pose_overlay.py --source http://196.42.111.25:8080/video --marker-length 0.05 --rotate 90

--rotate must match whatever value was used in capture_calibration_imgs.py
for this camera. The saved calibration's image_size reflects the rotated
frame dimensions, using a different rotation here would mismatch it.
"""
import argparse
import time
import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

ROTATE_MAP = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def open_source(source):
    try:
        return cv2.VideoCapture(int(source))
    except ValueError:
        return cv2.VideoCapture(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--calib", default="camera_calibration.npz")
    parser.add_argument("--marker-length", type=float, default=0.05,
                         help="Printed marker side length in metres (measure it, don't trust nominal)")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                         help="Must match the value used in capture_calibration_imgs.py for this camera")
    args = parser.parse_args()
    rotate_code = ROTATE_MAP[args.rotate]

    calib = np.load(args.calib)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]

    detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())

    half = args.marker_length / 2
    obj_points = np.array([
        [-half,  half, 0], [half,  half, 0],
        [ half, -half, 0], [-half, -half, 0],
    ], dtype=np.float32)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return

    print("Press 'q' to quit.")

    last_print = 0.0
    print_interval = 1.0  # seconds

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, check the connection.")
            break
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        now = time.time()
        should_print = now - last_print >= print_interval

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_id, corner in zip(ids.flatten(), corners):
                img_points = corner[0].astype(np.float32)
                ok_pnp, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                if not ok_pnp:
                    continue
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, half)

                # On-screen readout: distance from camera and full tvec
                tx, ty, tz = tvec.flatten()
                distance_cm = float(np.linalg.norm(tvec)) * 100
                text_pos = tuple(img_points[0].astype(int) + np.array([0, -10]))
                label = f"id{marker_id}: d={distance_cm:.1f}cm  z={tz*100:.1f}cm"
                cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)

                if should_print:
                    print(f"id {marker_id}: tvec (m) = [{tx:+.4f}, {ty:+.4f}, {tz:+.4f}]  "
                          f"distance = {distance_cm:.2f} cm")

            if should_print:
                last_print = now

        cv2.imshow("Marker detection + pose overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()