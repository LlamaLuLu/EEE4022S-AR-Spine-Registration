"""
Live AR overlay for spine navigation (Qt version).

Qt-based UI: video frame shown in a QLabel, HUD widgets stacked on top.
Only widgets whose values change get repainted.

Scaffold: video stream + StatusPanel + keyboard shortcuts (q to quit).
Other HUD widgets (pose, TRE, radar, triad, state pill, shortcuts)
will follow after this scaffold verifies.

Usage (from repo root):
    python scripts/registration/live_ar_overlay_qt.py \
        --source http://192.168.1.42:8080/video \
        --calib data/calibration/s22_ultra/camera_calibration.npz \
        --isthmus annotations/pedical_isthmus_targets.json \
        --entry annotations/pedicle_entry_targets.json \
        --marker-map data/bracket_markers/marker_map.json \
        --stl phantom/meshes/v1_print/BASE_v1.4.stl

UI v2 (pedicle selection) is the default: the per-vertebra meshes in
phantom/meshes/v3_overlay are loaded automatically. Select one pedicle at a
time. Keys: W/S or Up/Down step the vertebra, A/D or Left/Right pick the side,
1-5 jump to a level, [ and ] walk the ten pedicles in order, T (or 0) shows all
levels, Y wireframe, E error circle, Z zoom view (+ / - or the wheel change the zoom), X axial view, L sagittal (lateral) view, V move those two beside the main view or back to the rail, F fullscreen, P photo, M debug mode (ArUco
boxes, marker count, pose, Shift+P residuals), H or ? help.
If those meshes are missing or fail to load, the original single-mesh
wireframe (--stl) is used instead. Pass --vertebrae-dir none to force it.
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
from replay_source import ReplayCapture, is_replay_source
import numpy as np

from PySide6.QtCore import Qt, QTimer, QSize, QEvent
from PySide6.QtGui import QImage, QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QStackedLayout,
    QVBoxLayout, QHBoxLayout, QSizePolicy,
)

# ArUco + TRE pipeline (unchanged from cv2 version)
from ar_board import (
    ARUCO_DICT, build_corner_lut, load_marker_map, 
    load_corner_lut, multi_marker_pnp, decompose_pose,
)
from ar_tre import compute_tre_scalar, DEFAULT_FLE_MM
from ar_targets import load_targets
from ui.overlay import (
    draw_targets, draw_trajectories, draw_wireframe,
    is_in_envelope,
)
from ar_tracker import Tracker, VERIFY_COUNTDOWN_SEC
from ar_selection import Selection, parse_label, SIDE_NAMES
from ar_guidance import trust_badge
from ar_error import (ANCHORS, error_radius95, grade, grade_colour,
                      draw_error_circle)
from ui.stl import (
    load_stl_for_wireframe, load_vertebra_meshes,
    draw_vertebra_layers, _blend_lines, STYLE as VERT_STYLE,
)
from ar_theme import envelope_tier

# Qt HUD
from ui.styles import QSS
from ui.hud import AxisTriadWidget
from ui.rail import Rail, HelpOverlay, RAIL_QSS
from ui.views import ViewPane, ZoomControls, ClickLabel, image_pos
from ui.axial import SliceView, NEUTRAL
from ui.zoom import project_point, zoom_frame


EXTENSION_MM = 70.0     # guide band length beyond the entry point
TICK_MM = 5.0           # a tick mark every 5 mm
LABEL_MM = 10.0         # longer tick every 10 mm; numbers every 10 mm when they fit
BAND_ALPHA = 0.70       # opacity of the outer (approach) guide band
MAIN_STYLE = "background-color: black;"
INSET_STYLE = "background-color: black; border: 2px solid #3CC3C3;"
INSET_FRAC = 0.30        # zoom layout 'inset': camera picture-in-picture width, as a share of the video area
SLICE_COL_FRAC = 0.27    # width of the axial / sagittal column beside the main view, as a share of the stage
SLICE_DOCK_DEFAULT = "right"  # where axial and sagittal start: "right" (beside the main view) or "rail" (V swaps)
FOV_STEPS = (150.0, 120.0, 90.0, 60.0, 40.0)   # zoom steps: width of the field of view in mm (wide to tight)
WINDOW_MIN_ASPECT = 1.6                          # window width / height: lower = taller default window
AUTO_ZOOM = True                                  # open zoom on the first pedicle pick (Z or Reset close it)
ZOOM_DEFAULT = 2                                 # index into FOV_STEPS when zoom is first opened (90 mm)
ZOOM_LINE_PX = 2                                 # vertebra outline thickness in the zoom view
ZOOM_ALL_PX = 1                                  # outline thickness of the all-levels zoom (1 or 2)
ZOOM_ALL_ALPHA = 0.9                             # outline strength of the all-levels zoom (lower = lighter lines)
ZOOM_TINT = 0.16                                 # faint fill of the selected vertebra in the zoom view
BOX_FILL = 0.22                                  # faint teal fill inside the crop box on the camera view
BOX_BGR = (195, 195, 60)                         # teal, the colour of the inset border
ZOOM_STYLE = {k: {**v, "thickness": ZOOM_LINE_PX} for k, v in VERT_STYLE.items()}
ZOOM_STYLE["selected"]["fill_alpha"] = ZOOM_TINT
ZOOM_ALL_STYLE = {**VERT_STYLE, "verify": {**VERT_STYLE["verify"], "alpha": ZOOM_ALL_ALPHA, "thickness": ZOOM_ALL_PX}}   # all levels: 1 px, lighter
OVERVIEW_EXT_MM = 50.0  # thin extension beyond the entry point in the all-levels view
OUTER_COLOUR = (200, 150, 55)    # BGR blue, outer trajectory (beyond the entry point)
INNER_COLOUR = (60, 215, 250)    # BGR yellow, inner trajectory (entry to isthmus)
GUIDE_BY_STATUS = True          # True: the whole trajectory (outer guide band AND inner line) takes the TRE
                                 # status colour (green/amber/red) of the worse of the entry and isthmus points.
                                 # The two are still told apart by width. False: blue guide + yellow inner line.
MARKERS_BLACK = True            # True: in single-vertebra focus, crosshairs / dots are black (white rim).
                                 # The all-levels view always uses coloured crosshairs.
TICK_ALPHA = 0.85       # opacity of the tick marks
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_VERTEBRAE_DIR = _REPO / "phantom" / "meshes" / "v3_overlay"
DEFAULT_FALLBACK_STL = _REPO / "phantom" / "meshes" / "v2_print" / "spine_phantom_v2.stl"
DEFAULT_ISTHMUS = _REPO / "annotations" / "pedicle_isthmus_targets.json"
DEFAULT_ENTRY = _REPO / "annotations" / "pedicle_entry_targets.json"
ROTATE_MAP = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
              180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def open_source(source):
    if is_replay_source(source):
        return ReplayCapture(source)
    try:
        return cv2.VideoCapture(int(source))
    except ValueError:
        return cv2.VideoCapture(source)


def _blend_text(frame, txt, centre, angle_deg, colour, alpha, scale):
    """Small anti-aliased text rotated by angle_deg (clockwise on screen), with a soft shadow."""
    S = 2                                           # draw at 2x then downsample for smooth edges
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(txt, font, scale * S, 2)
    d = int(np.hypot(tw, th + bl)) + 8
    d += d % 2
    canvas = np.zeros((d, d), np.uint8)
    cv2.putText(canvas, txt, ((d - tw) // 2, (d + th) // 2), font, scale * S, 255, 2, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((d / 2, d / 2), -angle_deg, 1.0)
    canvas = cv2.warpAffine(canvas, M, (d, d), flags=cv2.INTER_LINEAR)
    mask = cv2.resize(canvas, (d // S, d // S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    h, w = frame.shape[:2]
    m = mask.shape[0]
    x0, y0 = int(centre[0] - m / 2), int(centre[1] - m / 2)
    for dx, dy, col, a in ((1, 1, (0, 0, 0), 0.55 * alpha), (0, 0, colour, alpha)):
        xa, ya = x0 + dx, y0 + dy
        xs, ys = max(xa, 0), max(ya, 0)
        xe, ye = min(xa + m, w), min(ya + m, h)
        if xe <= xs or ye <= ys:
            continue
        mk = mask[ys - ya:ye - ya, xs - xa:xe - xa, None] * a
        roi = frame[ys:ye, xs:xe].astype(np.float32)
        frame[ys:ye, xs:xe] = (roi * (1 - mk) + np.array(col, np.float32) * mk).astype(np.uint8)


def draw_focus_trajectory(frame, entry_mm, isthmus_mm, rvec, tvec, K, D,
                          ext_mm=EXTENSION_MM, status_colour=None):
    """
    Focus-mode trajectory for one pedicle, in the style of the 7D FLASH tool view:
      * beyond the entry point: a thick translucent blue guide band (the approach path)
        with a scale along one edge: a tick every TICK_MM, a longer tick and a
        rotated "10 mm", "20 mm" ... label every LABEL_MM, counted from the entry;
      * inside the bone: a thin yellow line to the isthmus with a fine tick every
        TICK_MM for depth.
    entry_mm and isthmus_mm are 3D points in the phantom frame (mm).
    """
    entry = np.asarray(entry_mm, float)
    isth = np.asarray(isthmus_mm, float)
    axis = entry - isth
    depth = np.linalg.norm(axis)
    if depth < 1e-6:
        return
    axis /= depth                   # unit vector pointing out of the bone

    fh = frame.shape[0]
    band = max(8, int(round(fh / 85)))          # ~13 px at 1080p
    thin = max(1, int(round(fh / 540)))         # ~2 px at 1080p
    tick_s = max(5, int(round(fh / 120)))       # minor tick length (~9 px)
    txt_scale = max(0.4, fh / 1900.0)           # ~0.57 at 1080p
    white = (240, 240, 240)

    def proj(pts):
        p, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3),
                                 rvec, tvec, K, D)
        return p.reshape(-1, 2)

    # ---- guide band beyond the entry, thin line inside the bone ----
    p_e = proj([entry, entry + axis * ext_mm, isth])
    _blend_lines(frame, np.round(np.stack([p_e[0], p_e[1]])).astype(np.int32).reshape(1, 2, 2),
                 status_colour if status_colour is not None else OUTER_COLOUR,
                 BAND_ALPHA, band)
    _blend_lines(frame, np.round(np.stack([p_e[2], p_e[0]])).astype(np.int32).reshape(1, 2, 2),
                 status_colour if status_colour is not None else INNER_COLOUR,
                 0.95, thin)

    # ---- scale along one edge ----
    ks = np.arange(np.ceil(-depth / TICK_MM) * TICK_MM, ext_mm + 1e-6, TICK_MM)
    ks = ks[np.abs(ks) > 1e-6]                  # the entry point itself is marked by the arrowhead
    pts = proj([entry + axis * k for k in ks])
    pa = proj([entry + axis * (k - 0.5) for k in ks])
    pb = proj([entry + axis * (k + 0.5) for k in ks])
    tang = pb - pa
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    perp = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    # use the same side for the whole scale: the one pointing right / down on screen
    mean_t = tang.mean(axis=0)
    side = np.array([-mean_t[1], mean_t[0]])
    if side[0] < -1e-6 or (abs(side[0]) <= 1e-6 and side[1] < 0):
        side = -side
    perp = np.where((perp @ side)[:, None] < 0, -perp, perp)

    major = np.isclose(np.mod(ks, LABEL_MM), 0.0, atol=1e-6) | np.isclose(np.mod(ks, LABEL_MM), LABEL_MM, atol=1e-6)
    in_band = ks > 0
    start_off = np.where(in_band, band / 2.0, thin / 2.0)[:, None]
    length = np.where(major, tick_s * 1.7, tick_s * 0.9)[:, None]
    a_pt = pts + perp * start_off
    b_pt = pts + perp * (start_off + length)
    _blend_lines(frame, np.round(np.stack([a_pt, b_pt], axis=1)).astype(np.int32),
                 white, TICK_ALPHA, 1)

    # ---- rotated labels beyond the entry ----
    # Label every LABEL_MM, or every 20/30 mm if the view is too small for the numbers to fit.
    (tw2, th), _ = cv2.getTextSize("00", cv2.FONT_HERSHEY_SIMPLEX, txt_scale, 1)
    p10 = proj([entry, entry + axis * LABEL_MM])
    spacing = max(float(np.linalg.norm(p10[1] - p10[0])), 1.0)
    step = LABEL_MM * max(1, int(np.ceil(tw2 * 1.5 / spacing)))
    lab = [i for i in np.flatnonzero(in_band)
           if np.isclose(np.mod(ks[i], step), 0.0, atol=1e-6)
           or np.isclose(np.mod(ks[i], step), step, atol=1e-6)]
    for n_, i in enumerate(lab):
        ang = np.degrees(np.arctan2(tang[i, 1], tang[i, 0]))
        if ang > 90:
            ang -= 180
        elif ang <= -90:
            ang += 180
        txt = f"{int(round(ks[i]))}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, txt_scale, 1)
        c = pts[i] + perp[i] * (band / 2.0 + tick_s * 1.7 + th * 0.9 + 3)
        _blend_text(frame, txt, c, ang, white, 0.9, txt_scale)
    if lab:                                     # unit, once, just past the last tick
        i = int(np.argmax(ks))
        pm = proj([entry + axis * (ext_mm + 0.9 * step)])[0]
        ang = np.degrees(np.arctan2(tang[i, 1], tang[i, 0]))
        if ang > 90:
            ang -= 180
        elif ang <= -90:
            ang += 180
        _blend_text(frame, "mm", pm + perp[i] * (band / 2.0 + tick_s * 1.7 + th * 0.9 + 3),
                    ang, white, 0.9, txt_scale)


def draw_overview_trajectory(frame, entry_mm, isthmus_mm, rvec, tvec, K, D,
                             ext_mm=OVERVIEW_EXT_MM, status_colour=None):
    """
    All-levels view: thin line from the isthmus through the entry point and on
    for ext_mm beyond it, so the approach direction is visible without the
    ruler. Coloured by status if status_colour is given, otherwise blue
    outside the bone and yellow inside it.
    """
    entry = np.asarray(entry_mm, float)
    isth = np.asarray(isthmus_mm, float)
    axis = entry - isth
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return
    axis /= n
    fh = frame.shape[0]
    w = max(2, int(round(fh / 540)))
    p, _ = cv2.projectPoints(np.array([isth, entry, entry + axis * ext_mm], np.float64)
                             .reshape(-1, 1, 3), rvec, tvec, K, D)
    p = np.round(p.reshape(-1, 2)).astype(np.int32)
    inner = p[[0, 1]].reshape(1, 2, 2)
    outer = p[[1, 2]].reshape(1, 2, 2)
    _blend_lines(frame, inner, (0, 0, 0), 0.45, w + 2)           # dark underlay for contrast
    _blend_lines(frame, outer, (0, 0, 0), 0.45, w + 2)
    _blend_lines(frame, inner, status_colour if status_colour is not None else INNER_COLOUR, 0.95, w)
    _blend_lines(frame, outer, status_colour if status_colour is not None else OUTER_COLOUR, 0.95, w)


class FrameGrabber:
    """Background thread that continuously grabs frames, keeping only the
    latest. The main loop calls read() and always gets the freshest frame
    instead of a stale one from FFmpeg's internal buffer. seq counts
    frames received, so the caller can skip frames it has already seen."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                self.frame = frame
                self.seq += 1

    def read(self):
        with self.lock:
            return self.ok, self.frame, self.seq

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ====================================================================
# Main window
# ====================================================================

class ARWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # --- Calibration ---
        calib = np.load(args.calib)
        self.K = calib["camera_matrix"]
        self.D = calib["dist_coeffs"]

        # --- Board geometry ---
        self.corner_lut = load_corner_lut(args.marker_map)
        self.n_total = len(self.corner_lut)
        print(f"Board: {self.corner_lut.source or 'default V3'} | "
              f"IDs {sorted(self.corner_lut)}, gate {self.corner_lut.min_markers}, "
              f"{self.corner_lut.marker_size_mm:g} mm")

        self.is_v3 = set(self.corner_lut) <= set(range(8))
        
        # --- Targets ---
        self.target_pts, self.target_labels, self.traj_pairs = load_targets(
            args.isthmus, args.entry
        )
        print(f"Targets: {len(self.target_pts)}, "
              f"trajectory pairs: {len(self.traj_pairs)}")

        # Which pedicle is selected (single source of truth for keys, rail, clicks)
        self.selection = Selection(self.target_labels)
        self.entry_idx = np.array(self.selection.all_entry_idx, dtype=int)
        # Validated error anchor for this board (see ar_error.py)
        self.anchor = ANCHORS["bracket" if self.is_v3 else "dome"]

        # --- Per-vertebra meshes (UI v2) ---
        self.vertebrae = None
        vdir = args.vertebrae_dir
        if str(vdir).lower() not in ("none", "off", ""):
            try:
                meshes = load_vertebra_meshes(
                    vdir, prefix=args.vertebrae_prefix,
                    scale=args.stl_scale, offset=tuple(args.stl_offset),
                    cell_mm=0)           # already decimated in Blender
            except Exception as exc:     # corrupt file, bad path, etc.
                print(f"Vertebra meshes failed to load ({exc})")
                meshes = {}
            if meshes:
                self.vertebrae = meshes
                missing = [lv for lv in range(1, 6) if lv not in meshes]
                print(f"Vertebra meshes: levels {sorted(meshes)} | "
                      f"{sum(len(m.tris) for m in meshes.values())} triangles"
                      + (f" | MISSING L{missing}" if missing else ""))
            else:
                print(f"No usable vertebra meshes in {vdir}: "
                      + (f"falling back to {args.stl}."
                         if args.stl and Path(args.stl).exists() else
                         "fallback --stl is missing too, so no mesh will be drawn."))

        # --- Wireframe ---
        self.mesh_verts = None
        self.mesh_edge_i0 = self.mesh_edge_i1 = self.mesh_edge_normals = None
        if args.stl and self.vertebrae is None and Path(args.stl).exists():
            verts, edges = load_stl_for_wireframe(args.stl)
            if verts is not None:
                # Apply scale (Blender metres → mm) and offset
                verts_mm = (verts * args.stl_scale
                            + np.array(args.stl_offset, dtype=np.float32))

                # Runtime decimation: merge vertices within grid_mm of
                # each other, which collapses small triangles and reduces
                # edge count. grid=1mm keeps fine detail; grid=3-4mm gives
                # a recognisable shape at much higher FPS.
                grid = args.stl_decimate
                if grid > 0:
                    rounded = np.round(verts_mm / grid) * grid
                    unique, inverse = np.unique(
                        rounded.astype(np.float32), axis=0, return_inverse=True)

                    # Rebuild edges with merged indices, drop degenerate edges
                    i0_raw = np.array([e[0] for e in edges], dtype=np.int32)
                    i1_raw = np.array([e[1] for e in edges], dtype=np.int32)
                    normals_raw = np.array([e[2] for e in edges], dtype=np.float32)

                    new_i0 = inverse[i0_raw]
                    new_i1 = inverse[i1_raw]
                    valid = new_i0 != new_i1  # drop collapsed edges

                    # Deduplicate edges
                    edge_keys = np.stack([np.minimum(new_i0[valid], new_i1[valid]),
                                         np.maximum(new_i0[valid], new_i1[valid])], axis=1)
                    _, unique_idx = np.unique(edge_keys, axis=0, return_index=True)

                    self.mesh_verts = unique
                    self.mesh_edge_i0 = new_i0[valid][unique_idx]
                    self.mesh_edge_i1 = new_i1[valid][unique_idx]
                    self.mesh_edge_normals = normals_raw[valid][unique_idx]

                    print(f"STL decimated: {len(unique)} verts, "
                          f"{len(unique_idx)} edges (grid={grid}mm)")
                else:
                    self.mesh_verts = verts_mm
                    self.mesh_edge_i0 = np.array([e[0] for e in edges], dtype=np.int32)
                    self.mesh_edge_i1 = np.array([e[1] for e in edges], dtype=np.int32)
                    self.mesh_edge_normals = np.array([e[2] for e in edges], dtype=np.float32)

                print(f"  STL bounds (after transform): "
                      f"min={self.mesh_verts.min(axis=0).round(1)} "
                      f"max={self.mesh_verts.max(axis=0).round(1)}")

        # Debug: print marker centroid bounds for comparison with STL
        centroids = np.array([v.mean(axis=0) for v in self.corner_lut.values()])
        print(f"Marker centroid bounds: min={centroids.min(axis=0).round(1)} "
              f"max={centroids.max(axis=0).round(1)}")
        print(f"Target bounds: min={self.target_pts.min(axis=0).round(1)} "
              f"max={self.target_pts.max(axis=0).round(1)}")

        # --- Video ---
        self.rotate_code = ROTATE_MAP[args.rotate]
        cap = open_source(args.source)
        if not cap.isOpened():
            print(f"Could not open source: {args.source}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Threaded grabber: continuously reads frames in background,
        # so the main loop always gets the freshest frame instead of
        # a stale one from FFmpeg's internal buffer.
        self.grabber = FrameGrabber(cap)

        self.detector = cv2.aruco.ArucoDetector(
            ARUCO_DICT, cv2.aruco.DetectorParameters()
        )

        # --- Tracking (detection, PnP, state machine, TRE) ---
        self.tracker = Tracker(self.detector, self.corner_lut, self.K, self.D,
                               self.target_pts, args.fle, self.is_v3)
        self.show_ellipses = self.show_wire = True
        self.zoom_idx = ZOOM_DEFAULT
        self._wheel_acc = 0
        self.views = {"zoom": False, "axial": True, "sagittal": True}   # axial and sagittal sit in the rail, on by default
        self._slice_cache = {}
        self._last_r = None
        self.check = None                      # last click-to-check result (see _do_check)
        self._toast = None                     # (text, until) shown in the hint bar
        self._zoom_xf = None                   # last zoom view's mapping back to camera pixels
        self.engineer = False                         # M: diagnostics for the engineer, hidden by default
        self._fitted = False                          # window shaped to the camera frame once

        # --- Timing instrumentation ---
        self._frame_count = 0
        self._last_log = time.perf_counter()

        # --- CSV data logger (for ATP tests) ---
        self._csv_file = None
        self._csv_writer = None
        self._step = 0
        if args.log:
            import csv
            self._csv_file = open(args.log, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            # Header
            tre_headers = [f"TRE_{l}" for l in self.target_labels]
            self._csv_writer.writerow([
                "timestamp", "step", "state", "n_markers", "pnp_ok", "rms_px",
                "distance_mm", "elevation_deg", "azimuth_deg",
                "cam_x", "cam_y", "cam_z", "in_envelope", "env_tier",
                "amb_rejected", "grab_ms", "detect_pnp_ms", "overlay_ms", "qt_ms",
                "visible_ids",
            ] + tre_headers)
            print(f"Logging to: {args.log}")

        # --- Screenshot ---
        self._screenshot_dir = Path(args.screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_count = 0
        self._last_frame_bgr = None
        self._last_clean_bgr = None
        self._last_pose = None

        # --- Build UI ---
        self._build_ui()

        # --- Timer for frame updates (~30 FPS) ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(33)

        print("Qt overlay running. Press 'q' or close window to quit.")

    @property
    def state(self):
        return self.tracker.state

    def _build_ui(self):
        """Docked left rail + video. Overlays (triad, hint bar, help) sit on the video."""
        self.setWindowTitle("AR Spine Navigation")
        self.setFocusPolicy(Qt.StrongFocus)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.rail = Rail(self.n_total, self.corner_lut.min_markers, self.selection.levels)
        self.rail.selector.levelClicked.connect(lambda lv: self._select(self.selection.select, lv))
        self.rail.selector.sideClicked.connect(lambda s_: self._select(self.selection.set_side, s_))
        self.rail.selector.overviewClicked.connect(lambda: self._select(self.selection.overview))
        self.rail.helpClicked.connect(self._toggle_help)
        self.rail.viewToggled.connect(self._toggle_view)
        self.rail.optionToggled.connect(self._toggle_option)
        self.rail.engineerClicked.connect(self._toggle_engineer)
        lay.addWidget(self.rail)

        # The stage holds the camera view and the zoom view. They are placed by hand
        # (see _layout_views) so the zoom can fill it with the camera as an inset.
        self.stage = QWidget()
        self.stage.setObjectName("viewGrid")
        self.stage.setAttribute(Qt.WA_StyledBackground, True)
        self.stage.setMinimumSize(640, 400)
        self.stage.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label = ClickLabel(self.stage)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(MAIN_STYLE)
        self._main_style = MAIN_STYLE
        self.zoom_pane = ViewPane("ZOOM", self.stage)
        self.video_label.clicked.connect(self._on_main_click)
        self.zoom_pane.clicked.connect(self._on_zoom_click)
        self.zoom_pane.hide()
        self.slice_dock = "rail"                      # where the panes are right now (the default move happens after the build)
        self._main_w = 0
        self._frame_aspect = 9 / 16                   # camera height / width, updated from the frames
        lay.addWidget(self.stage, 1)
        self.setCentralWidget(central)

        self.axis_triad = AxisTriadWidget(parent=self.stage)
        self.axial_pane = self.rail.axial_pane            # the axial and sagittal views live in the rail
        self.sagittal_pane = self.rail.sagittal_pane
        # debug panels float over the video (top left) so the rail keeps its space for the axial view
        self.debug_box = self.rail.engineer_box
        self.debug_box.setParent(self.stage)
        self.debug_box.setObjectName("debugCard")
        self.debug_box.setAttribute(Qt.WA_StyledBackground, True)
        self.debug_box.setMinimumWidth(230)
        self.debug_box.setStyleSheet(
            "#debugCard { background-color: rgba(14,18,22,228); border-top: 2px solid #3CC3C3; border-radius: 10px; }"
            "#debugCard QFrame#hudPanel, #debugCard QFrame#hudPanel[accent=\"default\"],"
            "#debugCard QFrame#hudPanel[accent=\"good\"], #debugCard QFrame#hudPanel[accent=\"warn\"],"
            "#debugCard QFrame#hudPanel[accent=\"bad\"], #debugCard QFrame#hudPanel[accent=\"stale\"]"
            " { background: transparent; border: none; }"
            "#debugCard QLabel#railTitle { color: #3CC3C3; font-size: 10px; letter-spacing: 2px; }"
            "#debugCard QLabel#hudLabel { font-size: 10px; letter-spacing: 1px; }"
            "#debugCard QLabel#hudValue { font-size: 13px; }")
        lay_ = self.debug_box.layout()
        lay_.setContentsMargins(14, 10, 14, 10)
        lay_.setSpacing(2)
        for panel in (self.rail.status_panel, self.rail.pose_panel):
            panel.setMinimumSize(0, 0)                    # the panels force a tall minimum height: drop it
            panel.layout().setContentsMargins(0, 2, 0, 2)
            panel.layout().setSpacing(0)
        self.debug_box.hide()
        self.zoom_ctl = ZoomControls(self.stage)
        self.zoom_ctl.hide()
        self.zoom_ctl.zoomIn.connect(lambda: self._step_zoom(+1))
        self.zoom_ctl.zoomOut.connect(lambda: self._step_zoom(-1))
        self.zoom_ctl.reset.connect(self._reset_zoom)
        self.hint_bar = QLabel(parent=self.stage)
        self.hint_bar.setObjectName("hintBar")
        self.hint_bar.hide()
        self.help = HelpOverlay(parent=self.stage)
        self.stage.installEventFilter(self)              # re-layout when the stage resizes

        # Pre-polish the widgets that appear later, and warm the video label's two styles, so their
        # first appearance does not stall a frame.
        for wd in (self.zoom_pane, self.zoom_ctl, self.axial_pane, self.sagittal_pane):
            wd.ensurePolished()
            for ch in wd.findChildren(QWidget):
                ch.ensurePolished()
        self.video_label.setStyleSheet(INSET_STYLE)
        self.video_label.ensurePolished()
        self.video_label.setStyleSheet(MAIN_STYLE)
        self.video_label.ensurePolished()
        if SLICE_DOCK_DEFAULT == "right":
            self._toggle_layout()
        self.rail.set_option("outline", self.show_wire)
        self.rail.set_option("circle", self.show_ellipses)
        self.rail.set_view("axial", self.views["axial"])
        self.rail.set_view("sagittal", self.views["sagittal"])
        self.resize(1360, 800)

    # ---- overlay placement ----

    def eventFilter(self, obj, event):
        if obj is self.stage and event.type() == QEvent.Resize:
            self._layout_views()
        return super().eventFilter(obj, event)

    def _layout_views(self):
        """Place the camera view, the zoom view and (if docked on the right) the slice column."""
        W, H = self.stage.width(), self.stage.height()
        zoom_on = self.views["zoom"]
        slices = [(self.axial_pane, self.views["axial"]), (self.sagittal_pane, self.views["sagittal"])]
        on_panes = [p_ for p_, on in slices if on]
        gap = 3
        right = bool(on_panes) and self.slice_dock == "right"
        cw = int(min(max(W * SLICE_COL_FRAC, 260), 480)) if right else 0
        Wm = W - cw - (gap if right else 0)               # width left for the main view
        self._main_w = Wm
        # slice panes: a column on the right of the stage, or in the rail
        if self.slice_dock == "right":
            self.rail.slice_host.hide()
            n = len(on_panes)
            y = 0
            for p_, on in slices:
                if not on:
                    p_.hide()
                    continue
                eh = (H - gap * (n - 1)) // n
                p_.setGeometry(Wm + gap, y, cw, eh)
                p_.show()
                p_.raise_()
                y += eh + gap
        else:
            for p_, on in slices:
                p_.setVisible(on)
            self.rail.slice_host.setVisible(bool(on_panes))
        # main view: the camera fills the area, or (zoom) the zoom fills it with the camera as an inset
        style = MAIN_STYLE
        if not zoom_on:
            self.zoom_pane.hide()
            self.video_label.setGeometry(0, 0, Wm, H)
        else:
            iw = max(int(Wm * INSET_FRAC), 320)
            ih = int(iw * self._frame_aspect)
            m = 14
            self.zoom_pane.setGeometry(0, 0, Wm, H)
            self.zoom_pane.show()
            self.video_label.setGeometry(Wm - iw - m, H - ih - m, iw, ih)
            style = INSET_STYLE
        if style != self._main_style:
            self._main_style = style
            self.video_label.setStyleSheet(style)
        self.video_label.raise_()
        self.zoom_ctl.setVisible(zoom_on)
        if zoom_on:
            self.zoom_ctl.adjustSize()
            cwid, chei = self.zoom_ctl.width(), self.zoom_ctl.height()
            ytop = self.axis_triad.y() + self.axis_triad.height() + 8      # never over the axes gizmo
            self.zoom_ctl.move(Wm - cwid - 14, max(self.video_label.y() - 10 - chei, ytop))
            self.zoom_ctl.raise_()
        self._place_overlays()

    def _place_overlays(self):
        w, h = self.stage.width(), self.stage.height()
        mw = self._main_w or w                               # width of the main view (the slice column is beyond it)
        pad = max(int(h * 0.02), 12)
        self.axis_triad.move(mw - self.axis_triad.width() - pad, pad)
        # the hint bar is centred in the space the camera inset leaves free
        free_w = mw
        if self.views["zoom"]:
            free_w = self.video_label.x() - 14
        self.hint_bar.adjustSize()
        self.hint_bar.move(max((free_w - self.hint_bar.width()) // 2, 14), h - self.hint_bar.height() - pad)
        self.help.adjustSize()
        self.help.move((w - self.help.width()) // 2, (h - self.help.height()) // 2)
        if self.engineer:
            self.debug_box.adjustSize()
            self.debug_box.move(14, 52 if self.views["zoom"] else 14)    # below the zoom tag
            self.debug_box.show()
            self.debug_box.raise_()
        else:
            self.debug_box.hide()
        for wd in (self.axis_triad, self.hint_bar, self.help):
            wd.raise_()

    @property
    def focus_level(self):
        return self.selection.level

    def _toggle_help(self):
        on = not self.help.isVisible()
        self.help.setVisible(on)
        self.rail.set_help(on)
        self._place_overlays()

    def _toggle_fullscreen(self):
        """Pause frame processing during the macOS fullscreen animation,
        otherwise the busy event loop can freeze it."""
        self.timer.stop()
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        QTimer.singleShot(900, lambda: self.timer.start(33))

    # ---- click to check: where is the real entry dot compared with the overlay? ----

    def _on_main_click(self, pos):
        if self.views["zoom"]:                    # the camera inset is too small to click on
            return
        clean = self._last_clean_bgr
        if clean is None:
            return
        hit = image_pos(self.video_label, pos, clean.shape[1], clean.shape[0])
        if hit is not None:
            self._do_check(hit, "camera")

    def _on_zoom_click(self, pos):
        if self._zoom_xf is None:
            return
        w, h, x0, y0, mag = self._zoom_xf
        hit = image_pos(self.zoom_pane, pos, w, h)
        if hit is not None:
            self._do_check((x0 + hit[0] / mag, y0 + hit[1] / mag), "zoom")

    def _say(self, text, secs=8.0):
        self._toast = (text, time.time() + secs)

    def _do_check(self, cam_px, view):
        """The click marks the real entry dot. Back-project it onto the plane through the selected entry
        target (square to the camera) and measure the in-plane error, as the ATP-2 tool does."""
        r, sel = self._last_r, self.selection
        if sel.is_overview:
            self._say("Pick a pedicle first (W / S, A / D), then click its real entry dot")
            return
        if r is None or r.state != "NAVIGATING" or r.rvec is None:
            self._say("Tracking is needed to check the overlay")
            return
        R, _ = cv2.Rodrigues(np.asarray(r.rvec, float).reshape(3, 1))
        t = np.asarray(r.tvec, float).reshape(3, 1)
        e = sel.entry_idx
        tgt = np.asarray(self.target_pts[e], float)
        depth = float((R @ tgt.reshape(3, 1) + t)[2, 0])
        n = cv2.undistortPoints(np.array([[cam_px]], np.float64), self.K, self.D).reshape(2)
        Q = (R.T @ (np.array([[n[0] * depth], [n[1] * depth], [depth]]) - t)).ravel()
        dv = Q - tgt
        err = float(np.linalg.norm(dv))
        proj, _ = cv2.projectPoints(tgt.reshape(1, 3), r.rvec, r.tvec, self.K, self.D)
        tp = proj.reshape(2)
        err_px = float(np.hypot(cam_px[0] - tp[0], cam_px[1] - tp[1]))
        r95 = float(error_radius95(np.asarray(r.tre_values), self.anchor)[e]) if r.tre_values is not None else float("nan")
        pred = float(r.tre_values[e]) if r.tre_values is not None else float("nan")
        self.check = {"sel": sel.current, "Q": Q, "err": err, "within": bool(err <= r95)}
        self._say(f"{sel.name.upper()}   {err:.1f} mm off   expected within {r95:.1f}   Esc clears")
        print(f"Check {sel.name}: {err:.2f} mm (dx {dv[0]:+.2f}, dy {dv[1]:+.2f}, dz {dv[2]:+.2f}), "
              f"expected within {r95:.2f} mm")
        self._redraw_cached()                     # show the mark now, and in the saved window picture
        shot = self._save_screenshot(quiet=True)  # clean frame + pose, so the offline tool can repeat it
        self._log_check({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "shot": shot or "",
            "label": self.target_labels[e], "level": f"L{sel.level}", "side": sel.side, "view": view,
            "distance_mm": round(float(r.distance_mm), 1), "elev_deg": round(float(r.elev_deg), 1),
            "az_deg": round(float(r.az_deg), 1), "n_markers": int(r.n_markers),
            "tier": r.env_tier, "rms_px": None if np.isnan(r.reproj_rms) else round(float(r.reproj_rms), 2),
            "target_u": round(float(tp[0]), 2), "target_v": round(float(tp[1]), 2),
            "dot_u": round(float(cam_px[0]), 2), "dot_v": round(float(cam_px[1]), 2),
            "err_px": round(err_px, 2), "depth_mm": round(depth, 1), "err_mm": round(err, 3),
            "dx_mm": round(float(dv[0]), 3), "dy_mm": round(float(dv[1]), 3), "dz_mm": round(float(dv[2]), 3),
            "pred_tre_mm": round(pred, 3), "expected_95_mm": round(r95, 3),
        })

    def _log_check(self, row):
        import csv
        path = self._screenshot_dir / "click_checks.csv"
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                wr.writeheader()
            wr.writerow(row)

    def _draw_check(self, frame, rvec, tvec, K, D, target_mm):
        """The clicked point (pinned to the phantom) with a line to the overlay's entry point and the error."""
        pts = np.vstack([self.check["Q"], np.asarray(target_mm, float)]).astype(np.float64)
        proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
        (qx, qy), (tx, ty) = proj.reshape(2, 2)
        fw = frame.shape[1]
        colour = (140, 210, 100) if self.check["within"] else (90, 90, 215)       # BGR green / red
        arm = max(7, fw // 150)
        q, tg = (int(round(qx)), int(round(qy))), (int(round(tx)), int(round(ty)))
        cv2.line(frame, tg, q, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(frame, tg, q, (255, 255, 255), 1, cv2.LINE_AA)
        for col, th in (((0, 0, 0), 4), (colour, 2)):
            cv2.line(frame, (q[0] - arm, q[1] - arm), (q[0] + arm, q[1] + arm), col, th, cv2.LINE_AA)
            cv2.line(frame, (q[0] - arm, q[1] + arm), (q[0] + arm, q[1] - arm), col, th, cv2.LINE_AA)
        txt = f"{self.check['err']:.1f} mm"
        fs = min(max(fw / 2200.0, 0.45), 1.0)
        org = (q[0] + arm + 6, q[1] - arm - 4)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, colour, 2, cv2.LINE_AA)

    def _redraw_cached(self):
        """After a layout change, redraw the views from the last camera frame right away, instead of
        showing stale pixmaps at the new size until the next frame arrives."""
        r, clean = self._last_r, getattr(self, "_last_clean_bgr", None)
        if r is None or clean is None:
            return
        frame = clean.copy()
        self._draw_overlays(frame, r)
        if self.views["zoom"]:
            self._render_zoom(clean, r, frame)
        for kind in ("axial", "sagittal"):
            if self.views[kind]:
                self._render_slice(kind, r)
        self._show_frame(frame)

    def _toggle_view(self, name):
        """Switch an extra view on or off; the side column shows only if some view is on."""
        self.views[name] = not self.views[name]
        self.rail.set_view(name, self.views[name])
        self._layout_views()
        self._redraw_cached()

    def _step_zoom(self, d):
        """+ / - : move along FOV_STEPS (clamped). Opens the zoom view first if it is off."""
        if not self.views["zoom"]:
            if d > 0:
                self._toggle_view("zoom")
            return
        self.zoom_idx = min(max(self.zoom_idx + d, 0), len(FOV_STEPS) - 1)

    def _reset_zoom(self):
        """Back to the plain camera view (the zoom level is remembered for next time)."""
        if self.views["zoom"]:
            self._toggle_view("zoom")

    def wheelEvent(self, event):
        if self.views["zoom"] and self.zoom_pane.geometry().contains(self.stage.mapFrom(self, event.position().toPoint())):
            self._wheel_acc += event.angleDelta().y()
            while abs(self._wheel_acc) >= 120:             # one step per wheel notch (trackpad: per 120 units)
                self._step_zoom(1 if self._wheel_acc > 0 else -1)
                self._wheel_acc -= 120 if self._wheel_acc > 0 else -120
            event.accept()
        else:
            super().wheelEvent(event)

    def _toggle_option(self, name):
        """Y outline, E error circle, V side views: the keys and the rail buttons both come here."""
        if name == "outline":
            self.show_wire = not self.show_wire
            self.rail.set_option("outline", self.show_wire)
        elif name == "circle":
            self.show_ellipses = not self.show_ellipses
            self.rail.set_option("circle", self.show_ellipses)
        elif name == "layout":
            self._toggle_layout()
            return
        self._redraw_cached()

    def _toggle_layout(self):
        """V: move the axial and sagittal views between the rail and a column beside the main view."""
        panes = [self.axial_pane, self.sagittal_pane]
        if self.slice_dock == "rail":
            self.slice_dock = "right"
            for p_ in panes:
                p_.setParent(self.stage)
                p_.hide()
        else:
            self.slice_dock = "rail"
            self.rail.dock_slices(panes)
        self.rail.set_option("layout", self.slice_dock == "right")
        self._layout_views()
        self._redraw_cached()

    def _toggle_engineer(self):
        self.engineer = not self.engineer
        self.rail.set_engineer(self.engineer)
        self._place_overlays()

    def _friendly(self, j):
        """'L3_E_R' -> 'L3 right'."""
        p = parse_label(self.target_labels[j])
        return f"L{p[0]} {SIDE_NAMES[p[1]]}" if p else str(self.target_labels[j])

    def _hint_free_width(self):
        """Width the hint bar may use: the main view, minus the camera inset in zoom mode, minus margins."""
        mw = self._main_w or self.stage.width()
        free = (self.video_label.x() - 14) if self.views["zoom"] else mw
        return max(free - 28, 120)

    def _update_hint(self, r):
        """One line at the bottom of the video: what the keys do right now. Longer wording is
        swapped for shorter wording when the bar would not fit beside the camera inset."""
        name = self.selection.name.upper()
        if self._toast is not None and time.time() < self._toast[1]:
            options = [self._toast[0], self._toast[0].replace("   Esc clears", "")]
        elif r.state != "NAVIGATING":
            options = [{"VERIFYING": "C  confirm      R  reject",
                        "STALE": "Bring the markers back into view"}.get(r.state, "")]
        elif self.selection.is_overview:
            options = ["ALL LEVELS      W/S or A/D  pick a pedicle", "ALL LEVELS   W/S  A/D  pick", "ALL LEVELS"]
        else:
            options = [f"{name}      W/S  level      A/D  side      T  all levels",
                       f"{name}    W/S  level    A/D  side",
                       f"{name}    W/S  A/D", name]
        if not options[0]:
            self.hint_bar.hide()
            return
        avail = self._hint_free_width()
        key = (tuple(options), avail)
        if key != getattr(self, "_hint_key", None):
            self._hint_key = key
            for txt in options:                       # first wording that fits
                self.hint_bar.setText(txt)
                self.hint_bar.adjustSize()
                if self.hint_bar.width() <= avail:
                    break
            self._place_overlays()
        self.hint_bar.show()
        self.hint_bar.raise_()


    # ================================================================
    # Frame loop
    # ================================================================

    def next_frame(self):
        t0 = time.perf_counter()

        ok, frame, seq = self.grabber.read()
        if not ok or frame is None or seq == getattr(self, "_last_seq", None):
            return          # no new camera frame since the last tick
        self._last_seq = seq
        if self.rotate_code is not None:
            frame = cv2.rotate(frame, self.rotate_code)
        # Clean copy (no overlays) for measurement screenshots
        clean = frame.copy()

        t1 = time.perf_counter()

        now = time.time()
        r = self.tracker.process(frame, now)          # tracking: no drawing

        t2 = time.perf_counter()

        self._draw_overlays(frame, r)                 # drawing onto the numpy frame
        if self.views["zoom"]:
            self._render_zoom(clean, r, frame)        # separate view, redrawn at full resolution
        if self.views["axial"]:
            self._render_slice("axial", r)
        if self.views["sagittal"]:
            self._render_slice("sagittal", r)

        t3 = time.perf_counter()

        self._update_widgets(r)                       # Qt HUD

        # --- Push frame to video label ---
        self._last_r = r
        self._last_frame_bgr = frame.copy()
        self._last_clean_bgr = clean
        self._last_pose = self._pose_dict(r)
        self._show_frame(frame)

        t4 = time.perf_counter()

        # --- Log timing once per second ---
        self._frame_count += 1
        elapsed = t4 - self._last_log
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            print(f"grab={((t1-t0)*1000):.0f}ms  "
                  f"detect+pnp={((t2-t1)*1000):.0f}ms  "
                  f"overlay={((t3-t2)*1000):.0f}ms  "
                  f"qt={((t4-t3)*1000):.0f}ms  "
                  f"FPS={fps:.1f}")
            self._frame_count = 0
            self._last_log = t4

        self._log_row(r, (t1 - t0, t2 - t1, t3 - t2, t4 - t3))

    # ---- rendering: draws the FrameResult onto the numpy frame ----

    def _draw_overlays(self, frame, r):
        """Main view: engineer boxes, then the scene with the real camera matrix."""
        if self.engineer and r.ids is not None and len(r.ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, r.corners, r.ids,
                                          borderColor=(60, 160, 60))
        self._draw_scene(frame, r, self.K, self.D)

    def _draw_scene(self, frame, r, K, D, zoom=False):
        """Draw the model and the selected pedicle onto `frame`, projected with camera matrix K.
        The main view passes the camera's K; the zoom view passes a K that matches its crop."""
        state, rvec, tvec = r.state, r.rvec, r.tvec
        if state not in ("VERIFYING", "NAVIGATING", "STALE"):
            return

        if self.show_wire and self.vertebrae is not None and state != "STALE":
            # UI v2: VERIFYING shows every level; NAVIGATING shows the
            # focused level bold with neighbours ghosted (or all levels
            # if no focus is set).
            sel = self.focus_level if state == "NAVIGATING" else None
            if zoom:                                     # thicker line, exact faint fill
                draw_vertebra_layers(frame, self.vertebrae, rvec, tvec, K, D, selected=sel,
                                     style=ZOOM_STYLE if sel is not None else ZOOM_ALL_STYLE,
                                     exact_fill=sel is not None)
            else:
                draw_vertebra_layers(frame, self.vertebrae,
                                     rvec, tvec, K, D, selected=sel)
        elif (self.show_wire and self.mesh_verts is not None
                and state != "STALE"):
            draw_wireframe(frame, self.mesh_verts,
                           self.mesh_edge_i0, self.mesh_edge_i1,
                           self.mesh_edge_normals,
                           rvec, tvec, K, D, r.cam_pos)

        if state == "NAVIGATING" and r.tre_values is not None:
            # One pedicle selected: its entry, isthmus, trajectory and guide, nothing else.
            # Overview: every entry point with a thin trajectory line.
            # Colours grade the expected 95 % error (measured, see ar_error.py),
            # not the raw predicted TRE.
            sel_ = self.selection
            focus = not sel_.is_overview
            r95 = error_radius95(np.asarray(r.tre_values), self.anchor)
            status = [grade(v, r.env_tier) for v in r95]
            colour = [grade_colour(st) for st in status]
            if focus:
                ix = np.array([sel_.isthmus_idx, sel_.entry_idx], dtype=int)
            else:
                ix = self.entry_idx                      # isthmus dots hidden
            f_pts = self.target_pts[ix]
            f_labels = [self.target_labels[i] for i in ix]
            f_tre = np.asarray(r.tre_values)[ix]
            f_col = [colour[i] for i in ix]

            def worse(ia, ib):                           # guide takes the worse end
                return colour[ia if r95[ia] >= r95[ib] else ib] if GUIDE_BY_STATUS else None

            if focus:
                ia, ib = sel_.isthmus_idx, sel_.entry_idx
                draw_focus_trajectory(
                    frame, self.target_pts[ib], self.target_pts[ia],
                    rvec, tvec, K, D, status_colour=worse(ia, ib))
            else:
                for ia, ib in self.traj_pairs:           # (isthmus, entry)
                    draw_overview_trajectory(
                        frame, self.target_pts[ib], self.target_pts[ia],
                        rvec, tvec, K, D, status_colour=worse(ia, ib))

            _NoOpTR = type("_NR", (), {
                "h": frame.shape[0], "w": frame.shape[1],
                "text": lambda self, *a, **k: None,
                "measure": lambda self, *a, **k: (0, 0),
            })()
            draw_targets(frame, f_pts, f_labels,
                         f_tre, rvec, tvec, K, D,
                         r.visible_ids, self.corner_lut, self.args.fle,
                         r.env_tier, _NoOpTR,
                         show_ell=self.show_ellipses and not focus,   # random-noise ellipses only in overview
                         marker_col=(0, 0, 0) if (MARKERS_BLACK and focus) else None,
                         colours=f_col)
            if focus and self.show_ellipses:
                # true-scale 95 % error circle around the entry point
                e = sel_.entry_idx
                draw_error_circle(frame, self.target_pts[e], r95[e],
                                  rvec, tvec, K, D, colour[e])
            if focus and self.check is not None and self.check["sel"] == sel_.current:
                self._draw_check(frame, rvec, tvec, K, D, self.target_pts[sel_.entry_idx])

    def _render_zoom(self, clean, r, frame):
        """Zoom view: crop the clean camera frame around the selected entry point, magnify it,
        and draw the same scene on it with the matching camera matrix."""
        pane = self.zoom_pane
        sel = self.selection
        can_in, can_out = self.zoom_idx < len(FOV_STEPS) - 1, self.zoom_idx > 0
        self._zoom_xf = None
        self.zoom_ctl.set_state(None, can_in, can_out)
        if r.state != "NAVIGATING" or r.rvec is None:
            pane.set_tag("ZOOM")
            pane.show_message("Waiting for tracking")
            return
        w, h = pane.render_size()
        if sel.is_overview:
            # all levels: centre on the middle of the ten entry points and fit them with a margin;
            # the +/- steps then scale that fit (the default step is the fit itself)
            pts = [project_point(self.target_pts[i], r.rvec, r.tvec, self.K, self.D) for i in sel.all_entry_idx]
            pts = [p_ for p_ in pts if p_ is not None]
            if not pts:
                pane.show_message("Targets are behind the camera")
                return
            xy = np.array([[p_[0], p_[1]] for p_ in pts])
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            e = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, float(np.mean([p_[2] for p_ in pts])))
            ppm = 0.5 * (self.K[0, 0] + self.K[1, 1]) / e[2]
            need_px = max((hi[0] - lo[0]) * 1.5, (hi[1] - lo[1]) * 1.5 * w / h, 160.0)
            fov = need_px / ppm * FOV_STEPS[self.zoom_idx] / FOV_STEPS[ZOOM_DEFAULT]
            name = "ALL LEVELS"
        else:
            e = project_point(self.target_pts[sel.entry_idx], r.rvec, r.tvec, self.K, self.D)
            if e is None:
                pane.show_message("Entry point is behind the camera")
                return
            ppm = 0.5 * (self.K[0, 0] + self.K[1, 1]) / e[2]      # camera pixels per mm at the entry point
            fov = FOV_STEPS[self.zoom_idx]
            name = sel.name.upper()
        img, Kz, mag = zoom_frame(clean, self.K, (e[0], e[1]), ppm, w, h, fov)
        self._draw_scene(img, r, Kz, self.D, zoom=True)
        pane.set_tag(f"ZOOM  {name}  {fov:.0f} mm")
        pane.show_frame(img)
        self._zoom_xf = (w, h, e[0] - w / (2.0 * mag), e[1] - h / (2.0 * mag), mag)
        self.zoom_ctl.set_state(mag, can_in, can_out)
        # box on the camera view around the area the zoom is showing
        fh = frame.shape[0]
        x0, y0 = e[0] - w / (2.0 * mag), e[1] - h / (2.0 * mag)
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x0 + w / mag)), int(round(y0 + h / mag)))
        t = max(3, fh // 220)
        fw = frame.shape[1]
        cx0, cy0, cx1, cy1 = max(p0[0], 0), max(p0[1], 0), min(p1[0], fw), min(p1[1], fh)
        if cx1 > cx0 and cy1 > cy0:                    # faint teal fill so the box stands out
            roi = frame[cy0:cy1, cx0:cx1]
            tint = np.empty_like(roi)
            tint[:] = BOX_BGR
            cv2.addWeighted(roi, 1.0 - BOX_FILL, tint, BOX_FILL, 0, roi)
        cv2.rectangle(frame, p0, p1, (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.rectangle(frame, p0, p1, BOX_BGR, t, cv2.LINE_AA)

    def _slice_view(self, kind):
        """Cached slice for the selected pedicle (depends on the selection only, not the camera)."""
        sel = self.selection
        key = (kind, sel.current)
        if key not in self._slice_cache:
            mesh = None if self.vertebrae is None else self.vertebrae.get(sel.level)
            self._slice_cache[key] = None if mesh is None else SliceView(
                mesh, self.target_pts[sel.entry_idx], self.target_pts[sel.isthmus_idx], kind)
        return self._slice_cache[key]

    def _render_slice(self, kind, r):
        """Rail inset: axial cross-section or sagittal side view of the selected vertebra,
        with the planned trajectory."""
        pane = self.axial_pane if kind == "axial" else self.sagittal_pane
        title = kind.upper()
        sel = self.selection
        pane.set_tag(title if sel.is_overview else f"{title}  {sel.name.upper()}")
        if sel.is_overview:
            pane.show_message("Pick a pedicle\nW / S  A / D")
            return
        sl = self._slice_view(kind)
        if sl is None or not sl.ok:
            pane.show_message("No slice for this level")
            return
        colour, r95_mm = NEUTRAL, None
        if r.state == "NAVIGATING" and r.tre_values is not None:
            r95_all = error_radius95(np.asarray(r.tre_values), self.anchor)
            r95_mm = float(r95_all[sel.entry_idx])
            colour = grade_colour(grade(r95_mm, r.env_tier))
        w, h = pane.render_size()                       # fill the whole pane: no empty borders
        top = int((pane.tag.height() + 14) * pane.devicePixelRatioF())      # keep clear of the title tag
        pane.show_frame(sl.render(w, h, colour, r95_mm, top))

    # ---- HUD: pushes the FrameResult into the Qt widgets ----

    def _update_widgets(self, r):
        rail = self.rail
        rail.state_pill.set_state(r.state, r.verify_remaining)
        rail.trust.update_state(*trust_badge(r, self.is_v3, self.n_total))
        rail.selector.set_state(self.selection.level, self.selection.side)

        j, r95_mm, status = self._error_summary(r)
        if j is None:
            rail.card.update_state(self.selection.name, None, "muted",
                                   "Waiting for tracking")
        elif self.selection.is_overview:
            rail.card.update_state("All levels", r95_mm, status,
                                   f"Worst entry point: {self._friendly(j)}. 95% of the time within this.")
        else:
            if self.check is not None and self.check["sel"] == self.selection.current:
                extra = f"Last check: {self.check['err']:.1f} mm off the real dot. Click it again to redo."
            else:
                extra = "Click the real entry dot to check the overlay."
            rail.card.update_state(self.selection.name, r95_mm, status,
                                   f"Expected error at the entry point. 95% of the time within this. {extra}")

        if self.engineer:
            rail.status_panel.update_values(r.n_markers, r.pnp_ok, r.reproj_rms)
            rail.pose_panel.update_values(r.distance_mm, r.elev_deg, r.az_deg, r.env_tier)

        if r.pnp_ok and r.state != "SEARCHING":
            self.axis_triad.set_rvec(r.rvec)
            if not self.axis_triad.isVisible():
                self.axis_triad.show()
        else:
            self.axis_triad.set_rvec(None)

        self._update_hint(r)

    def _error_summary(self, r):
        """(target index, 95 % error mm, status). One pedicle selected: its entry point.
        Overview: the worst entry point. Index is None when there is nothing to show."""
        if r.tre_values is None or r.state == "SEARCHING":
            return None, None, "muted"
        r95 = error_radius95(np.asarray(r.tre_values), self.anchor)
        if self.selection.is_overview:
            j = int(self.entry_idx[np.argmax(r95[self.entry_idx])])
        else:
            j = self.selection.entry_idx
        if not np.isfinite(r95[j]):
            return None, None, "muted"
        return j, float(r95[j]), grade(r95[j], r.env_tier)

    def _pose_dict(self, r):
        """Pose and status saved next to screenshots (None if there is no pose)."""
        if r.rvec is None:
            return None
        return {
            "rvec": np.asarray(r.rvec).ravel().tolist(),
            "tvec": np.asarray(r.tvec).ravel().tolist(),
            "state": r.state,
            "pnp_ok": bool(r.pnp_ok),
            "n_markers": int(r.n_markers),
            "rms_px": None if np.isnan(r.reproj_rms) else float(r.reproj_rms),
            "visible_ids": [int(i) for i in r.visible_ids],
            "distance_mm": float(r.distance_mm),
            "elevation_deg": float(r.elev_deg),
            "azimuth_deg": float(r.az_deg),
            "env_tier": r.env_tier,
            "focus_level": self.selection.level,
            "focus_side": self.selection.side,
            "expected_error_95_mm": self._error_summary(r)[1],
        }

    def _log_row(self, r, dts):
        """One CSV row per frame (ATP logging). dts = grab, detect+pnp, overlay, qt seconds."""
        if self._csv_writer is None:
            return
        if r.tre_values is not None:
            tre_vals = [f"{v:.3f}" for v in r.tre_values]
        else:
            tre_vals = [""] * len(self.target_labels)
        cam = r.cam_pos
        self._csv_writer.writerow([
            f"{r.now:.3f}", self._step, r.state, r.n_markers, r.pnp_ok,
            f"{r.reproj_rms:.2f}" if not np.isnan(r.reproj_rms) else "",
            f"{r.distance_mm:.1f}", f"{r.elev_deg:.1f}", f"{r.az_deg:.1f}",
            f"{cam[0]:.1f}", f"{cam[1]:.1f}", f"{cam[2]:.1f}",
            r.in_env, r.env_tier, int(r.amb_rejected),
            f"{dts[0]*1000:.1f}", f"{dts[1]*1000:.1f}",
            f"{dts[2]*1000:.1f}", f"{dts[3]*1000:.1f}",
            "|".join(str(i) for i in r.visible_ids),
        ] + tre_vals)
        self._csv_file.flush()

    def _fit_window_to_frame(self, fw, fh):
        """Once, on the first frame: make the video area the same shape as the
        camera frame, so it fills the window with no black bars."""
        self._fitted = True
        if self.isFullScreen() or self.isMaximized():
            return
        avail = self.screen().availableGeometry()
        vh = self.stage.height()
        vw = round(vh * fw / fh)
        total_w = self.rail.width() + vw
        if total_w > avail.width():                   # small screen: fit the width instead
            total_w = avail.width()
            vw = total_w - self.rail.width()
            vh = round(vw * fh / fw)
        # keep the window tall enough for the rail's axial view (the camera view may get thin
        # bars in plain mode; the zoom view always fills the area)
        vh = max(vh, int(total_w / WINDOW_MIN_ASPECT))
        vh = min(vh, avail.height() - 90)               # leave room for the title bar and dock
        self.resize(total_w, vh)

    def _show_frame(self, frame_bgr):
        """Convert BGR numpy frame to QPixmap and set on the video label."""
        h, w, _ = frame_bgr.shape
        if not self._fitted:
            self._fit_window_to_frame(w, h)
        if abs(h / w - self._frame_aspect) > 1e-6:
            self._frame_aspect = h / w
            self._layout_views()
        # BGR -> RGB for Qt
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # Fit the whole frame inside the label (nothing cropped)
        pix = pix.scaled(self.video_label.contentsRect().size(),
                         Qt.KeepAspectRatio,
                         Qt.FastTransformation)
        self.video_label.setPixmap(pix)

    # ================================================================
    # Input
    # ================================================================

    def keyPressEvent(self, event: QKeyEvent):
        k = event.key()
        text = event.text().lower()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        sel = self.selection
        if text == "q":
            self.close()
        elif k == Qt.Key_Escape:
            if self.help.isVisible():
                self._toggle_help()
            elif self.check is not None:
                self.check = None
                self._toast = None
                self._redraw_cached()
        elif text in ("h", "?"):
            self._toggle_help()
        elif k == Qt.Key_P and shift:
            if self.engineer:
                self.tracker.print_residuals = True      # debug, one-shot
            else:
                print("Shift+P: turn on debug mode (M) first")
        # ---- selection: level, side, overview ----
        elif k in (Qt.Key_W, Qt.Key_Up):
            self._select(sel.step_level, -1)
        elif k in (Qt.Key_S, Qt.Key_Down):
            self._select(sel.step_level, +1)
        elif k in (Qt.Key_A, Qt.Key_Left):
            self._select(sel.set_side, "L")
        elif k in (Qt.Key_D, Qt.Key_Right):
            self._select(sel.set_side, "R")
        elif text in ("1", "2", "3", "4", "5"):
            self._select(sel.select, int(text))
        elif text == "]":
            self._select(sel.step_pedicle, +1)
        elif text == "[":
            self._select(sel.step_pedicle, -1)
        elif text in ("t", "0"):
            self._select(sel.overview)
        # ---- view and tools ----
        elif text == "y":
            self._toggle_option("outline")
        elif text == "e":
            self._toggle_option("circle")
        elif text == "z":
            self._toggle_view("zoom")
        elif text == "x":
            self._toggle_view("axial")
        elif text == "l":
            self._toggle_view("sagittal")
        elif text == "v":
            self._toggle_layout()
        elif text in ("+", "="):
            self._step_zoom(+1)
        elif text in ("-", "_"):
            self._step_zoom(-1)
        elif text == "f":
            self._toggle_fullscreen()
        elif text == "p":
            self._save_screenshot()
        elif text == "m":
            self._toggle_engineer()
        # ---- registration and logging ----
        elif text == "c":
            self.tracker.confirm()
        elif text == "r":
            self.tracker.reset()
        elif text == "n":
            self._step += 1
            print(f"--- step {self._step} ---")
        elif text in (",", ".") and isinstance(self.grabber.cap, ReplayCapture):
            print("Replay:", self.grabber.cap.step(-1 if text == "," else +1))

    def _select(self, change, *args):
        """Run a Selection change (step_level, set_side, ...) and report it."""
        was_overview = self.selection.is_overview
        if change(*args):
            print("Selected:", self.selection.name)
            self.check = None
            if AUTO_ZOOM and was_overview and not self.selection.is_overview and not self.views["zoom"]:
                self._toggle_view("zoom")                 # first pedicle pick opens the zoom view (only Z / Reset close it)

    def _save_screenshot(self, quiet=False):
        """Save raw frame (with on-frame overlays) and full window (with HUD). Returns the file stem."""
        if self._last_frame_bgr is None:
            return None
        self._screenshot_count += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = self._screenshot_dir / f"shot_{stamp}_{self._screenshot_count:03d}"
        cv2.imwrite(f"{base}_frame.png", self._last_frame_bgr)
        if self._last_clean_bgr is not None:
            cv2.imwrite(f"{base}_clean.png", self._last_clean_bgr)
        if self._last_pose is not None:
            import json
            pose = dict(self._last_pose)
            pose.update({
                "timestamp": time.time(),
                "camera_matrix": np.asarray(self.K).tolist(),
                "dist_coeffs": np.asarray(self.D).ravel().tolist(),
                "calib": str(self.args.calib),
                "image_size": [int(self._last_clean_bgr.shape[1]),
                               int(self._last_clean_bgr.shape[0])],
            })
            with open(f"{base}_pose.json", "w") as f:
                json.dump(pose, f, indent=2)
        else:
            print("  (no valid pose, measurement data not saved)")
        self.grab().save(f"{base}_window.png")
        if not quiet:
            print(f"Screenshot saved: {base}_*.png")
        if self._csv_writer is not None:
            self._csv_writer.writerow([f"{time.time():.3f}", self._step,
                                       "SCREENSHOT", base.name])
            self._csv_file.flush()
        return base.name

    def closeEvent(self, event):
        self.timer.stop()
        self.grabber.release()
        if self._csv_file is not None:
            self._csv_file.close()
            print(f"Log closed: {self.args.log}")
        super().closeEvent(event)


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--calib",
                    default="data/calibration/s22_ultra_v1/camera_calibration.npz")
    ap.add_argument("--isthmus", default=str(DEFAULT_ISTHMUS))
    ap.add_argument("--entry", default=str(DEFAULT_ENTRY))
    ap.add_argument("--stl", default=str(DEFAULT_FALLBACK_STL),
                    help="Combined spine mesh, used only if the per-vertebra "
                         "meshes fail to load (default: v2_print/spine_phantom_v2.stl)")
    ap.add_argument("--stl-scale", type=float, default=1000.0,
                    help="STL scale factor (default 1000: Blender metres → mm)")
    ap.add_argument("--stl-offset", type=float, nargs=3, default=[0, 0, 0],
                    metavar=("X", "Y", "Z"),
                    help="Translation after scaling (mm, phantom LPS frame)")
    ap.add_argument("--stl-decimate", type=float, default=1.0,
                    help="Vertex merge grid in mm (default 1.0; try 2-4 for"
                         " faster rendering on dense meshes)")
    ap.add_argument("--vertebrae-dir", default=str(DEFAULT_VERTEBRAE_DIR),
                    help="Folder with per-vertebra STLs (phantom_l1..5.stl) for "
                         "level focus (default: phantom/meshes/v3_overlay). "
                         "Use 'none' to skip and use the combined --stl.")
    ap.add_argument("--vertebrae-prefix", default="phantom_l")
    ap.add_argument("--marker-map", default=None)
    ap.add_argument("--fle", type=float, default=DEFAULT_FLE_MM)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--log", default=None,
                    help="CSV file path for per-frame data logging (ATP tests)")
    ap.add_argument("--screenshot-dir", default="data/screenshots",
                    help="Directory for screenshots (press 's' to capture)")
    args = ap.parse_args()

    # HiDPI is on by default in Qt6, no explicit attributes needed.
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS + RAIL_QSS)

    window = ARWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()