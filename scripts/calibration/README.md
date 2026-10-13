# Camera calibration and capture

Quick scaffolding to validate camera capture, ChArUco calibration, ArUco
detection, and pose overlay end to end, using the S22 Ultra as a stand-in
for the webcam that's on order.

## Setup

1. In the venv: `pip install opencv-contrib-python numpy reportlab` (need
   opencv >=4.7 for the ArUco/Charuco API used here, contrib build is
   required, plain `opencv-python` doesn't ship the aruco module).

2. Phone stream: install **IP Webcam** (Thyoni Tech / Pavel Khlebovich)
   from the Play Store on the S22 Ultra. Connect to the same WiFi network
   as the Mac. Before starting the server, check video preferences for
   resolution (1280x720 to start) and a focus setting, set it to a
   fixed/locked mode rather than continuous autofocus, since a shifting
   focal length during calibration invalidates the intrinsics. Scroll
   down, tap "Start server". Shows an address like `192.168.1.23:8080`.
   Test `http://192.168.1.23:8080/video` in a browser first to confirm
   the stream works before running any of the scripts.

## Order of operations

All commands run from repo root. Output goes under
`data/calibration/`, keyed by device (`printables/` is shared, not
device-specific, `s22_ultra/` holds this device's images and
calibration result).

```
python scripts/calibration/generate_printables.py --outdir data/calibration/printables
```
Generates `charuco_board.png`/`.pdf` and `test_markers.png`/`.pdf`.
Print the PDFs, not the PNGs, at 100% / actual size, not "Fit to page",
the PDF page is already set to A4 with the board placed at true size, so
scaling should be left alone. Each sheet has a printed 50mm reference
bar at the bottom, measure it with a ruler after printing, if it's not
50mm, that's the actual scale error to correct for, update
`SQUARE_LENGTH_M` / `MARKER_LENGTH_M` at the top of
`capture_calibration_imgs.py` and `calibrate_camera.py` to the measured
value rather than nominal. Tape the board's corners down flat on a
table, avoid glare.

```
python scripts/calibration/capture_calibration_imgs.py --source http://196.42.111.25:8080/video --outdir data/calibration/s22_ultra/calibration_imgs
```
Live preview with corner count shown on screen. Press `c` to save a
frame (only saves when the board is clearly detected), `q` to quit.
Aim for 15-20 images to start, from varied angles and distances.

```
python scripts/calibration/calibrate_camera.py --imagedir data/calibration/s22_ultra/calibration_imgs --outfile data/calibration/s22_ultra/camera_calibration.npz
```
Runs calibration on the saved images, prints the RMS reprojection
error, saves `camera_calibration.npz`. Vally (2020) got under 0.5px
with the full 40-image protocol at close range in good lighting, not
expected from a first-pass phone setup, this run is to validate the
code path.

```
python scripts/registration/live_pose_overlay.py --source http://196.42.111.25:8080/video --calib data/calibration/s22_ultra/camera_calibration.npz --marker-length 0.05
```
Live ArUco detection with a 3D axis drawn on each detected marker.
Sanity check that capture, detection, PnP, and overlay all work
together before any of it depends on the phantom or the real webcam.

## Known limitations of this v0

- Phone autofocus/computational photography can still drift mid-session
  even with focus locked in-app, this calibration is a placeholder, not
  numbers for the report.
- Paper board on a table isn't rigid enough for final numbers, fine for
  validating code.
- Board is undersized (5x7, 30mm) to fit A4. Reprint at 6x8 / 50mm+ once
  larger-format printing is available, per Vally's optimal-conditions
  findings.