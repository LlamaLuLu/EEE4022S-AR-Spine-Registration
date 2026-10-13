# Registration pipeline and live AR overlay

Live AR overlay for fiducial-based spine surgical navigation. Detects
ArUco markers on the marker board (V3 L-bracket or dome), recovers camera
pose via multi-marker solvePnP, and projects one selected pedicle (or all
levels) onto the video feed with a calibrated expected error.

## Layout

```
scripts/registration/
  live_ar_overlay_qt.py    Primary entry point (Qt window, frame loop, keys)
  live_ar_overlay.py       cv2 fallback entry point
  live_pose_overlay.py     V0 single-marker debug viewer
  generate_bracket_markers.py   marker stickers, print sheet, marker_map.json

  Core (no Qt, reusable by other front ends)
  ar_board.py       board geometry, ArUco ID to 3D corners, multi-marker solvePnP
  ar_tracker.py     detection, PnP, state machine, TRE; returns a FrameResult
  ar_tre.py         Fitzpatrick TRE, 3x3 covariance, image-plane ellipse
  ar_targets.py     pedicle targets from Slicer mrk.json, label parsing
  ar_selection.py   which pedicle is selected (keys, rail and clicks all use it)
  ar_error.py       calibrated expected error, 95 % radius, grade, error circle
  ar_guidance.py    trust badge text and "move the camera" hints
  ar_theme.py       colours and envelope limits (also used by scripts/atp)
  replay_source.py  replay a folder of saved frames as a camera source

  ui/               Qt widgets and on-frame drawing
    hud.py          HUD widgets: state pill, status and pose panels, axis triad
    rail.py         docked left rail: trust badge, pedicle selector, target card, help
    styles.py       Qt style sheet (QSS) for the HUD
    overlay.py      on-frame drawing: crosshairs, ellipses, trajectories
    stl.py          STL loading and per-vertebra mesh drawing
```

The split: the tracker produces a `FrameResult`, the window reads it, and
the drawing code paints it. Pose estimation and CSV logging do not depend
on the UI.

## Dependencies

```
pip install opencv-contrib-python numpy PySide6 Pillow
```

PySide6 is required for the Qt entry point. Pillow is used by the cv2
fallback and the marker generator.

## Camera setup

### USB tether (recommended for low latency)

USB tethering removes WiFi jitter and gives consistent 3-5 ms frame grabs.

1. Enable Developer Options on the phone (Settings, About Phone, tap
   Build Number 7 times)
2. Enable USB Debugging in Developer Options
3. Install ADB: `brew install android-platform-tools`
4. Connect the phone via USB and approve the debugging prompt
5. Start IP Webcam on the phone (tap "Start server")
6. Forward the port:
   ```
   adb forward tcp:8080 tcp:8080
   ```
7. Use `--source http://localhost:8080/video`

### WiFi streaming (fallback)

Use the IP shown by IP Webcam, e.g. `--source http://192.168.1.42:8080/video`.
Expect occasional latency spikes of 50-100 ms.

### Replay (no camera)

Pass a folder of saved `*_clean.png` frames, or one image, as the source:

```
python scripts/registration/live_ar_overlay_qt.py --source data/screenshots/atp2_crisp
```

Use `,` and `.` to step frames. The dome board needs
`--marker-map data/dome_markers/marker_map.json`; without it the built-in
V3 bracket is used.

## Workflow

### 1. Generate bracket markers

```
python scripts/registration/generate_bracket_markers.py \
    --outdir data/bracket_markers
```

Produces 8 marker PNGs (20 mm at 600 DPI), `bracket_markers.pdf` (A4 print
sheet with cut guides and a 50 mm ruler) and `marker_map.json`. Print at
100 % scale, check the ruler, cut and stick on the bracket.

### 2. Launch the overlay

```
python scripts/registration/live_ar_overlay_qt.py \
    --source http://localhost:8080/video \
    --calib data/calibration/s22_ultra/camera_calibration.npz \
    --isthmus annotations/pedicle_isthmus_targets.json \
    --entry annotations/pedicle_entry_targets.json \
    --marker-map data/bracket_markers/marker_map.json
```

Per-vertebra meshes in `phantom/meshes/v3_overlay` load automatically. If
they are missing, `--stl` (single mesh) is used. `--log file.csv` writes the
per-frame ATP log.

### 3. Keyboard shortcuts

| Key | Action |
|---|---|
| `W` / `S` (or Up / Down) | Previous / next vertebra |
| `A` / `D` (or Left / Right) | Left / right pedicle |
| `1` to `5` | Jump to that vertebra |
| `[` / `]` | Walk the ten pedicles in order |
| `T` or `0` | All levels |
| `Y` | Bone outline on / off |
| `E` | Error circle (one pedicle) or noise ellipses (all levels) |
| `F` | Fullscreen |
| `P` | Save a photo (frame, clean frame, pose JSON, window) |
| `C` / `R` | Confirm / reject registration |
| `H` | Engineer mode: marker count, PnP, RMS, pose, ArUco boxes |
| `Shift+P` | Per-marker residuals (engineer mode only) |
| `N` | ATP step marker in the log |
| `,` / `.` | Replay: previous / next frame |
| `?` | Help overlay (`Esc` closes it) |
| `Q` | Quit |

The rail buttons (vertebra, pedicle, all levels) do the same as the keys.

## State machine

```
SEARCHING ──(PnP lock)──▶ VERIFYING ──(2 s countdown or 'c')──▶ NAVIGATING
    ▲                         │                                      │
    │                    ('r' reject)                           ('r' reject)
    │                         │                                      │
    └─────────────────────────┘──────────────────────────────────────┘

    PnP dropout < 300 ms in NAVIGATING ──▶ STALE (holds last pose)
    PnP dropout > 300 ms anywhere ──▶ SEARCHING
```

SEARCHING: no targets. VERIFYING: every vertebra outline is shown so the
fit can be checked. NAVIGATING: the selected pedicle with its trajectory,
depth scale and error circle. STALE: last pose held briefly, targets hidden.

## Window layout

```
┌──────────────┬──────────────────────────────────────┐
│ ● NAVIGATING │                                  ◇   │
│ Tracking good│                                 axis │
│ 8 of 8 markers                                       │
│ VERTEBRA     │          camera view                 │
│ L1 L2 L3 L4 L5                                      │
│ PEDICLE      │                                      │
│ Left  Right  │                                      │
│ All levels   │                                      │
│ TARGET       │                                      │
│ L4 left      │                                      │
│ ±2.9 mm      │      L4 LEFT  W/S level  A/D side    │
│ Help ? Engineer H                                   │
└──────────────┴──────────────────────────────────────┘
```

The window resizes to the camera's aspect ratio on start, so the video
fills it. Fullscreen on a taller screen leaves bars above and below,
because the whole frame is always shown.

## Performance

Measured in replay mode on a MacBook Pro (no camera), UI v2:

| Stage | One pedicle | All levels |
|---|---|---|
| Detection + PnP | 11-12 ms | 11-12 ms |
| On-frame overlay | 10-12 ms | 37 ms |
| Qt widget update | 2 ms | 2 ms |

All-levels is slower because every entry point and trajectory is drawn.
Live USB-tether timings from the earlier build (3-5 ms grab, about 30 FPS)
have not been re-measured for UI v2. Timing prints once per second in the
terminal.

## Design rationale

The overlay shows the expected 95 % error at the selected entry point
("±x mm") as a true-scale circle, not the raw predicted TRE. It is
calibrated to measured phantom data (see `ar_error.py`): the predicted
TRE sets the shape, and the measured error sets the floor. Colours follow
alarm conventions: green up to 3.0 mm, amber up to 4.0 mm, red above that
or outside the characterised pose envelope (V3 bracket only). The number
describes the population of targets and views on a phantom, not a
guarantee at a single point.