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
levels, Y wireframe, F fullscreen, P photo, Shift+P prints per-marker residuals.
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

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QStackedLayout,
    QVBoxLayout, QHBoxLayout,
)

# ArUco + TRE pipeline (unchanged from cv2 version)
from ar_board import (
    ARUCO_DICT, build_corner_lut, load_marker_map, 
    load_corner_lut, multi_marker_pnp, decompose_pose,
)
from ar_tre import compute_tre_scalar, DEFAULT_FLE_MM
from ar_targets import load_targets
from ar_overlay import (
    draw_targets, draw_trajectories, draw_wireframe,
    is_in_envelope,
)
from ar_tracker import Tracker, VERIFY_COUNTDOWN_SEC
from ar_selection import Selection
from ar_error import (ANCHORS, error_radius95, grade, grade_colour,
                      draw_error_circle)
from ar_stl import (
    load_stl_for_wireframe, load_vertebra_meshes,
    draw_vertebra_layers, _blend_lines,
)
from ar_theme import envelope_tier

# Qt HUD
from ar_qt_styles import QSS
from ar_qt_hud import (
    StatusPanel, PosePanel, TrePanel, StatePill,
    ShortcutsPanel, AxisTriadWidget, TargetRadarWidget,
)


EXTENSION_MM = 70.0     # guide band length beyond the entry point
TICK_MM = 5.0           # a tick mark every 5 mm
LABEL_MM = 10.0         # longer tick every 10 mm; numbers every 10 mm when they fit
BAND_ALPHA = 0.70       # opacity of the outer (approach) guide band
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
        """Video label as central widget, HUD widgets as its children."""
        self.setWindowTitle("AR Spine Navigation")

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 480)
        self.setCentralWidget(self.video_label)

        # All HUD widgets are direct children of the video label
        self.status_panel = StatusPanel(n_total=self.n_total,
                                        gate=self.corner_lut.min_markers,
                                        parent=self.video_label)
        self.pose_panel = PosePanel(parent=self.video_label)
        self.tre_panel = TrePanel(parent=self.video_label)
        self.state_pill = StatePill(parent=self.video_label)
        self.shortcuts_panel = ShortcutsPanel(parent=self.video_label)
        self.axis_triad = AxisTriadWidget(parent=self.video_label)
        self.target_radar = TargetRadarWidget(parent=self.video_label)

        # Focus indicator (bottom centre), styled to match the HUD panels
        self.focus_label = QLabel(parent=self.video_label)
        self.focus_label.setStyleSheet(
            "background-color: rgba(16, 20, 26, 215); color: #E1E8EB;"
            "border-top: 2px solid #3CC3C3; padding: 6px 14px;"
            "font-family: Menlo, Consolas, monospace; font-size: 15px;")
        self.focus_label.hide()

        # Adjust sizes now so positioning works
        for w in [self.status_panel, self.pose_panel, self.tre_panel,
                  self.state_pill, self.shortcuts_panel]:
            w.adjustSize()
            w.show()
            w.raise_()
        self.axis_triad.show()
        self.axis_triad.raise_()
        self.target_radar.show()
        self.target_radar.raise_()

        # Toggle flags for optional widgets
        self._show_triad = True
        self._show_radar = True
        self._show_shortcuts = True

        self.resize(1280, 800)

    def resizeEvent(self, event):
        """Reposition HUD widgets when window resizes."""
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        pad = max(int(h * 0.02), 12)
        gap = 12

        # ---- Top-left column: pose on top, status below ----
        self.pose_panel.adjustSize()
        self.pose_panel.move(pad, pad)

        self.status_panel.adjustSize()
        self.status_panel.move(pad, pad + self.pose_panel.height() + gap)

        # ---- Top-right column: axis triad on top, target radar below ----
        triad_x = w - self.axis_triad.width() - pad
        self.axis_triad.move(triad_x, pad)

        radar_x = w - self.target_radar.width() - pad
        radar_y = pad + self.axis_triad.height() + gap
        self.target_radar.move(radar_x, radar_y)

        # ---- Bottom-left: TRE panel ----
        self.tre_panel.adjustSize()
        tre_y = h - self.tre_panel.height() - pad
        self.tre_panel.move(pad, tre_y)

        # ---- Top-center: state pill ----
        self.state_pill.adjustSize()
        self.state_pill.move((w - self.state_pill.width()) // 2, pad)

        # ---- Bottom-centre: focus indicator ----
        self._place_focus_label()

        # ---- Bottom-right: shortcuts panel ----
        self.shortcuts_panel.adjustSize()
        self.shortcuts_panel.move(
            w - self.shortcuts_panel.width() - pad,
            h - self.shortcuts_panel.height() - pad
        )

        # Ensure all widgets render on top of the pixmap
        for widget in [self.status_panel, self.pose_panel, self.tre_panel,
                       self.state_pill, self.shortcuts_panel,
                       self.axis_triad, self.target_radar]:
            widget.raise_()

    def _place_focus_label(self):
        self.focus_label.adjustSize()
        self.focus_label.move((self.width() - self.focus_label.width()) // 2,
                              self.height() - self.focus_label.height()
                              - max(int(self.height() * 0.02), 12))

    @property
    def focus_level(self):
        return self.selection.level

    def _update_focus_label(self):
        if self.state != "NAVIGATING":
            if self.focus_label.isVisible():
                self.focus_label.hide()
            return
        sel = self.selection
        if sel.is_overview:
            txt = "ALL LEVELS      W/S or A/D: select a pedicle"
        else:
            prev_l, next_l = sel.neighbour_levels()
            txt = (f"{sel.name.upper()}      "
                   f"W/S: {'L%d' % prev_l if prev_l else '-'} / {'L%d' % next_l if next_l else '-'}    "
                   f"A/D: left / right    T: all levels")
        if self.focus_label.text() != txt:
            self.focus_label.setText(txt)
            self._place_focus_label()
        self.focus_label.show()
        self.focus_label.raise_()

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

        t3 = time.perf_counter()

        self._update_widgets(r)                       # Qt HUD

        # --- Push frame to video label ---
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
        state, rvec, tvec = r.state, r.rvec, r.tvec
        if r.ids is not None and len(r.ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, r.corners, r.ids,
                                          borderColor=(60, 160, 60))
        if state not in ("VERIFYING", "NAVIGATING", "STALE"):
            return

        if self.show_wire and self.vertebrae is not None and state != "STALE":
            # UI v2: VERIFYING shows every level; NAVIGATING shows the
            # focused level bold with neighbours ghosted (or all levels
            # if no focus is set).
            sel = self.focus_level if state == "NAVIGATING" else None
            draw_vertebra_layers(frame, self.vertebrae,
                                 rvec, tvec, self.K, self.D, selected=sel)
        elif (self.show_wire and self.mesh_verts is not None
                and state != "STALE"):
            draw_wireframe(frame, self.mesh_verts,
                           self.mesh_edge_i0, self.mesh_edge_i1,
                           self.mesh_edge_normals,
                           rvec, tvec, self.K, self.D, r.cam_pos)

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
                    rvec, tvec, self.K, self.D, status_colour=worse(ia, ib))
            else:
                for ia, ib in self.traj_pairs:           # (isthmus, entry)
                    draw_overview_trajectory(
                        frame, self.target_pts[ib], self.target_pts[ia],
                        rvec, tvec, self.K, self.D, status_colour=worse(ia, ib))

            _NoOpTR = type("_NR", (), {
                "h": frame.shape[0], "w": frame.shape[1],
                "text": lambda self, *a, **k: None,
                "measure": lambda self, *a, **k: (0, 0),
            })()
            draw_targets(frame, f_pts, f_labels,
                         f_tre, rvec, tvec, self.K, self.D,
                         r.visible_ids, self.corner_lut, self.args.fle,
                         r.env_tier, _NoOpTR,
                         show_ell=self.show_ellipses and not focus,   # random-noise ellipses only in overview
                         marker_col=(0, 0, 0) if (MARKERS_BLACK and focus) else None,
                         colours=f_col)
            if focus and self.show_ellipses:
                # true-scale 95 % error circle around the entry point
                e = sel_.entry_idx
                draw_error_circle(frame, self.target_pts[e], r95[e],
                                  rvec, tvec, self.K, self.D, colour[e])

    # ---- HUD: pushes the FrameResult into the Qt widgets ----

    def _update_widgets(self, r):
        self.status_panel.update_values(r.n_markers, r.pnp_ok, r.reproj_rms)
        self.status_panel.set_accent({
            "SEARCHING": "stale",
            "VERIFYING": "default",
            "NAVIGATING": "good",
            "STALE": "bad",
        }.get(r.state, "default"))

        self.pose_panel.update_values(r.distance_mm, r.elev_deg, r.az_deg, r.env_tier)
        self.pose_panel.set_accent(
            {"full": "good", "reduced": "warn"}.get(r.env_tier, "bad"))

        label, r95_mm, status = self._error_summary(r)
        self.tre_panel.update_error(label, r95_mm, status)

        self.state_pill.set_state(r.state, r.verify_remaining)

        # Axis triad
        if self._show_triad and r.pnp_ok and r.state != "SEARCHING":
            self.axis_triad.set_rvec(r.rvec)
            if not self.axis_triad.isVisible():
                self.axis_triad.show()
        else:
            self.axis_triad.set_rvec(None)

        # Target radar
        if self._show_radar and r.state == "NAVIGATING":
            self.target_radar.set_tre_values(r.tre_values, self.target_labels, r.env_tier)
            if not self.target_radar.isVisible():
                self.target_radar.show()
        else:
            self.target_radar.set_tre_values(None, [], r.env_tier)

        self._update_focus_label()

        # Reposition state pill (its width changes with countdown text)
        w = self.width()
        self.state_pill.move((w - self.state_pill.width()) // 2,
                             max(int(self.height() * 0.02), 12))

    def _error_summary(self, r):
        """(target label, 95 % error mm, status) for the HUD.
        One pedicle selected: its entry point. Overview: the worst entry point."""
        if r.tre_values is None or r.state == "SEARCHING":
            return "--", None, "muted"
        r95 = error_radius95(np.asarray(r.tre_values), self.anchor)
        if self.selection.is_overview:
            j = int(self.entry_idx[np.argmax(r95[self.entry_idx])])
        else:
            j = self.selection.entry_idx
        if not np.isfinite(r95[j]):
            return "--", None, "muted"
        return self.target_labels[j], float(r95[j]), grade(r95[j], r.env_tier)

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

    def _show_frame(self, frame_bgr):
        """Convert BGR numpy frame to QPixmap and set on the video label."""
        h, w, _ = frame_bgr.shape
        # BGR -> RGB for Qt
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # Scale to fill the label preserving aspect
        pix = pix.scaled(self.video_label.size(),
                         Qt.KeepAspectRatioByExpanding,
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
            # Toggle shortcuts panel
            self._show_shortcuts = not self._show_shortcuts
            self.shortcuts_panel.setVisible(self._show_shortcuts)
        elif k == Qt.Key_P and shift:
            self.tracker.print_residuals = True          # debug, one-shot
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
            self.show_wire = not self.show_wire
        elif text == "e":
            self.show_ellipses = not self.show_ellipses
        elif text == "f":
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif text == "p":
            self._save_screenshot()
        elif text == "h":
            # Toggle all HUD panels
            for widget in [self.status_panel, self.pose_panel, self.tre_panel,
                           self.state_pill]:
                widget.setVisible(not widget.isVisible())
        elif text == "g":
            self._show_triad = not self._show_triad
            self.axis_triad.setVisible(self._show_triad)
        elif text == "x":
            self._show_radar = not self._show_radar
            self.target_radar.setVisible(self._show_radar)
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
        if change(*args):
            print("Selected:", self.selection.name)

    def _save_screenshot(self):
        """Save raw frame (with on-frame overlays) and full window (with HUD)."""
        if self._last_frame_bgr is None:
            return
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
        print(f"Screenshot saved: {base}_*.png")
        if self._csv_writer is not None:
            self._csv_writer.writerow([f"{time.time():.3f}", self._step,
                                       "SCREENSHOT", base.name])
            self._csv_file.flush()

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
    app.setStyleSheet(QSS)

    window = ARWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()