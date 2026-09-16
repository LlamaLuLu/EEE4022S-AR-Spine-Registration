# Registration pipeline

Live AR overlay for fiducial-based spine surgical navigation. Detects
ArUco markers on the V3 L-bracket, recovers camera pose via multi-marker
solvePnP, and projects pedicle targets with real-time TRE confidence
onto the video feed.

## Modules

| File | Purpose |
|---|---|
| `live_ar_overlay_qt.py` | **Primary entry point (Qt).** QMainWindow with QTimer frame loop, HUD as Qt widgets, keyboard shortcuts. |
| `ar_qt_hud.py` | Qt HUD widgets: StatusPanel, PosePanel, TrePanel, StatePill, ShortcutsPanel, AxisTriadWidget, TargetRadarWidget. Only repaint when values change. |
| `ar_qt_styles.py` | Qt Style Sheets (QSS) defining the visual theme. |
| `ar_board.py` | Board geometry. Corner lookup table mapping ArUco IDs to 3D positions, multi-marker solvePnP, pose decomposition. |
| `ar_tre.py` | TRE computation. Fitzpatrick closed-form scalar TRE, full 3x3 covariance, image-plane ellipse projection. |
| `ar_overlay.py` | On-frame drawing (crosshairs, TRE ellipses, trajectories, wireframe). Drawn on the numpy frame before Qt display. Also used by the cv2 fallback. |
| `ar_theme.py` | Colour palette, typography scale, layout constants. Single source of truth for both Qt and cv2 paths. |
| `ar_targets.py` | Pedicle target loading from Slicer mrk.json files. Label parsing, trajectory pairing. |
| `ar_stl.py` | Lightweight STL loader for wireframe registration verification. |
| `generate_bracket_markers.py` | ArUco sticker PNGs, A4 print-ready PDF, and marker_map.json for the L-bracket. |
| `live_ar_overlay.py` | cv2 fallback entry point. Functional but visually limited compared to Qt version. |
| `live_pose_overlay.py` | V0 single-marker debug viewer. |

## Dependencies

```
pip install opencv-contrib-python numpy PySide6 Pillow
```

PySide6 is required for the Qt entry point. Pillow is used by the cv2
fallback and the marker generator.

## Camera setup

### USB tether (recommended for low latency)

USB tethering eliminates WiFi jitter and gives consistent 3-5ms frame
grab times at 30 FPS.

1. Enable Developer Options on your phone (Settings → About Phone →
   tap Build Number 7 times)
2. Enable USB Debugging in Developer Options
3. Install ADB: `brew install android-platform-tools`
4. Connect phone via USB, approve the debugging prompt on the phone
5. Start IP Webcam on the phone (tap "Start server")
6. Forward the port through USB:
   ```
   adb forward tcp:8080 tcp:8080
   ```
7. Use `--source http://localhost:8080/video` when launching

### WiFi streaming (fallback)

Use the IP shown by IP Webcam directly (e.g.
`--source http://192.168.1.42:8080/video`). Expect occasional latency
spikes of 50-100ms due to WiFi jitter.

## Workflow

### 1. Generate bracket markers

```
python scripts/registration/generate_bracket_markers.py \
    --outdir data/bracket_markers
```

Produces `data/bracket_markers/`:
- 8 individual marker PNGs (20mm at 600 DPI)
- `bracket_markers.pdf` (A4 print sheet with cut guides and 50mm ruler)
- `marker_map.json` (ID-to-slot mapping)

Print the PDF at 100% scale (not "fit to page"), verify the ruler,
cut along dashed lines, stick on the bracket at the labelled positions.

### 2. Launch the overlay

```
python scripts/registration/live_ar_overlay_qt.py \
    --source http://localhost:8080/video \
    --calib data/calibration/s22_ultra/camera_calibration.npz \
    --isthmus annotations/pedicle_isthmus_targets.json \
    --entry annotations/pedicle_entry_targets.json \
    --marker-map data/bracket_markers/marker_map.json \
    --stl phantom/meshes/v1_print/BASE_v1.4.stl
```

### 3. Keyboard shortcuts

| Key | Action |
|---|---|
| `c` | Confirm registration (during VERIFY countdown) |
| `r` | Reject and re-register |
| `t` | Toggle trajectory lines |
| `e` | Toggle TRE confidence ellipses |
| `w` | Toggle wireframe overlay |
| `d` | Toggle detach mode (overlay floats above anatomy) |
| `g` | Toggle axis triad gizmo |
| `x` | Toggle target radar |
| `h` | Toggle HUD panels |
| `f` | Toggle fullscreen |
| `esc` | Toggle shortcut panel |
| `q` | Quit |

## State machine

```
SEARCHING ──(PnP lock)──▶ VERIFYING ──(2s countdown or 'c')──▶ NAVIGATING
    ▲                         │                                      │
    │                    ('r' reject)                           ('r' reject)
    │                         │                                      │
    └─────────────────────────┘──────────────────────────────────────┘

    PnP dropout < 300ms in NAVIGATING ──▶ STALE (holds last pose)
    PnP dropout > 300ms anywhere ──▶ SEARCHING
```

During SEARCHING: no targets drawn, grey banner.
During VERIFYING: wireframe projected for visual alignment check.
During NAVIGATING: full overlay with targets, trajectories, TRE rings.
During STALE: last known pose held briefly, red banner, targets suppressed.

## HUD layout

```
┌─────────────┐          ● NAVIGATING           ┌─────────────┐
│  DISTANCE   │                                 │  ◇ Axis     │
│  ELEVATION  │                                 │    Triad    │
│  AZIMUTH    │                                 └─────────────┘
├─────────────┤                                 ┌─────────────┐
│  MARKERS    │                                 │  TARGETS    │
│  PNP        │                                 │  L1 ● ●     │
│  RMS        │                                 │  L2 ● ●     │
└─────────────┘                                 │  ...        │
                                                └─────────────┘
┌─────────────┐                                 ┌─────────────┐
│  TARGET L5  │                                 │  c confirm  │
│  TRE 0.48mm │                                 │  r reject   │
│  ● ● ●      │                                 │  ...        │
└─────────────┘                                 └─────────────┘
```

## Performance

Per-frame timing (USB tether, S22 Ultra at 1080p):

| Stage | Typical | Notes |
|---|---|---|
| Frame grab | 3-5 ms | USB tether; 50-100ms over WiFi |
| Detection + PnP | 6-10 ms | ArUco detect + multi-marker solvePnP |
| On-frame overlay | 0-5 ms | Crosshairs, ellipses, trajectories |
| Qt widget update | 2-3 ms | Only repaints on value change |
| **Total** | **~15 ms** | **30 FPS (capped by QTimer at 33ms)** |

Timing is logged to the terminal once per second during operation.

## Design rationale

The overlay follows clinical alarm colour conventions: green (TRE < 1mm),
amber (1 to 2mm), red (> 2mm or outside the characterised pose envelope).
Crosshair marks the optimal insertion point; the anisotropic ellipse around
it communicates the TRE uncertainty envelope, with axis lengths and
orientation derived from the full Fitzpatrick covariance projected through
the camera model. This is novel contribution #2 of the project.