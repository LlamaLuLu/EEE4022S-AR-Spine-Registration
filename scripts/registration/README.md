# Registration pipeline

Live AR overlay for fiducial-based spine surgical navigation. Detects
ArUco markers on the V3 L-bracket, recovers camera pose via multi-marker
solvePnP, and projects pedicle targets with real-time TRE confidence
onto the video feed.

## Modules

| File | Purpose |
|---|---|
| `live_ar_overlay.py` | Main entry point. State machine (SEARCHING → VERIFYING → NAVIGATING), CLI, frame loop. |
| `ar_board.py` | Board geometry. Builds the corner lookup table mapping ArUco IDs to 3D positions, runs multi-marker solvePnP, decomposes pose to distance/elevation/azimuth. |
| `ar_tre.py` | TRE computation. Fitzpatrick closed-form scalar TRE, full 3x3 covariance, and pinhole projection to get anisotropic 2D ellipses on the image plane. |
| `ar_overlay.py` | All drawing. Crosshairs, TRE ellipses, trajectory lines, wireframe, HUD. Text scales to frame size via Pillow (falls back to OpenCV if Pillow is missing). |
| `ar_targets.py` | Pedicle target loading from Slicer mrk.json files. Parses isthmus/entry labels, merges into one array, builds trajectory index pairs. |
| `ar_stl.py` | Lightweight STL loader for the registration verification wireframe. Binary and ASCII format, with vertex deduplication and backface-cull normals. |
| `generate_bracket_markers.py` | Generates the 8 ArUco sticker images for the L-bracket + a `marker_map.json` that tells the overlay which ID sits at which board position. Run once before printing stickers. |
| `live_pose_overlay.py` | V0 single-marker debug viewer (pre-existing). Kept for quick sanity checks on camera + calibration. |

## Dependencies

```
pip install opencv-contrib-python numpy Pillow
```

Pillow is optional but recommended for readable HUD text. Without it
the overlay falls back to OpenCV's bitmap font.

## Workflow

### 1. Generate bracket markers

```
python scripts/registration/generate_bracket_markers.py \
    --outdir data/bracket_markers
```

Produces `data/bracket_markers/`:
- 8 individual marker PNGs (20mm at 300 DPI)
- `bracket_sheet.png` (composite for printing)
- `marker_map.json` (ID-to-slot mapping)

Print the sheet at 100% scale, cut, and stick each marker on the
L-bracket at the labelled position (Face A slots 0 through 3, Face B slots 4
through 7).

### 2. Launch the overlay

```
python scripts/registration/live_ar_overlay.py \
    --source http://192.168.1.42:8080/video \
    --calib data/calibration/s22_ultra/camera_calibration.npz \
    --isthmus annotations/pedical_isthmus_targets.json \
    --entry annotations/pedicle_entry_targets.json \
    --marker-map data/bracket_markers/marker_map.json
```

Optional: pass `--stl phantom/spine_phantom.stl` for wireframe
registration verification during the VERIFY phase.

### 3. Keyboard shortcuts

| Key | Action |
|---|---|
| `c` | Confirm registration (during VERIFY countdown) |
| `r` | Reject and re-register |
| `t` | Toggle trajectory lines |
| `e` | Toggle TRE confidence ellipses |
| `w` | Toggle wireframe overlay |
| `d` | Toggle detach mode (overlay floats above anatomy) |
| `h` | Toggle HUD |
| `?` | Toggle shortcut reminder |
| `q` | Quit |

## State machine

```
SEARCHING ──(PnP lock)──▶ VERIFYING ──(3s countdown or 'c')──▶ NAVIGATING
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

## Design rationale

The overlay follows clinical alarm colour conventions: green (TRE < 1mm),
amber (1 to 2mm), red (> 2mm or outside the characterised pose envelope).
Crosshair marks the optimal insertion point; the anisotropic ellipse around
it communicates the TRE uncertainty envelope, with axis lengths and
orientation derived from the full Fitzpatrick covariance projected through
the camera model. This is novel contribution #2 of the project.