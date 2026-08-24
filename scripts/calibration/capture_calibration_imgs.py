"""
Live capture for camera calibration.

Usage:
    python capture_calibration_images.py --source 0
    python capture_calibration_images.py --source http://192.168.1.23:8080/video --rotate 90

Press 'c' to save the current frame (only saves if the ChArUco board is
detected, so you don't waste captures on blurry or out-of-frame shots).
Press 'q' to quit. Aim for at least 15-20 images to start, 40+ for the
final protocol you report, from varied distances and angles, with the
board close to the frame edges in some shots (Vally 2020's protocol).

--rotate corrects orientation if the phone stream comes in sideways or
upside down, a known IP Webcam quirk independent of how the phone is
physically held. Value is degrees clockwise needed to fix the preview:
if the image looks rotated 90 degrees left (counterclockwise), use
--rotate 90 to correct it.
"""
import argparse
import os
import cv2

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.015

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
    parser.add_argument("--source", default="0",
                         help="Camera index (e.g. 0) or stream URL (e.g. http://<phone-ip>:8080/video)")
    parser.add_argument("--outdir", default="calibration_images")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                         help="Degrees clockwise to rotate each frame, fixes a sideways/upside-down stream")
    args = parser.parse_args()
    rotate_code = ROTATE_MAP[args.rotate]

    os.makedirs(args.outdir, exist_ok=True)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, DICTIONARY
    )
    detector = cv2.aruco.CharucoDetector(board)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return

    count = len(os.listdir(args.outdir))
    print("Press 'c' to capture, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, check the connection.")
            break
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        display = frame.copy()
        detected = (
            charuco_ids is not None
            and charuco_corners is not None
            and len(charuco_ids) == len(charuco_corners)
            and len(charuco_ids) > 4
        )
        if marker_corners:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        if detected:
            # Draw a small dot at each detected checkerboard corner instead of
            # using drawDetectedCornersCharuco, which asserts strictly on array
            # shape in OpenCV 5 and trips on some frames.
            for corner in charuco_corners.reshape(-1, 2):
                cv2.circle(display, tuple(corner.astype(int)), 4, (0, 255, 0), -1)

        n_corners = 0 if charuco_ids is None else len(charuco_ids)
        status = f"Corners: {n_corners}  Saved: {count}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if detected else (0, 0, 255), 2)
        cv2.imshow("Calibration capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if detected:
                path = os.path.join(args.outdir, f"calib_{count:03d}.png")
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
                count += 1
            else:
                print("Board not clearly detected, not saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()