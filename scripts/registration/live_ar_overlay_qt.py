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

UI v2 (per-vertebra focus) is the default: the per-vertebra meshes in
phantom/meshes/v3_overlay are loaded automatically. Keys: 1-5 focus one level,
A (or 0) shows all levels, Left/Right arrows (or [ and ]) step through levels.
If those meshes are missing or fail to load, the original single-mesh
wireframe (--stl) is used instead. Pass --vertebrae-dir none to force it.
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
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
from ar_stl import (
    load_stl_for_wireframe, load_vertebra_meshes, level_from_label,
    draw_vertebra_layers, _blend_lines,
)
from ar_theme import envelope_tier, tre_colour

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
STALE_HOLD_SEC = 0.3
VERIFY_COUNTDOWN_SEC = 2.0
DETACH_OFFSET_PX = 60
ROTATE_MAP = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
              180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def open_source(source):
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
                          doff=0, ext_mm=EXTENSION_MM, status_colour=None):
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
        p = p.reshape(-1, 2)
        p[:, 1] -= doff
        return p

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
                             doff=0, ext_mm=OVERVIEW_EXT_MM, status_colour=None):
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
    p[:, 1] -= int(doff)
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

        # Per-level index (for focus mode): level -> target indices, and
        # (isthmus, entry) pairs remapped into each level's sub-array.
        self.level_idx = {}
        for i, lbl in enumerate(self.target_labels):
            lv, _side = level_from_label(lbl)
            if lv is not None:
                self.level_idx.setdefault(lv, []).append(i)
        self.level_idx = {lv: np.array(ix, dtype=int)
                          for lv, ix in self.level_idx.items()}
        self.entry_idx = np.array([i for i, l in enumerate(self.target_labels)
                                   if "_E_" in l], dtype=int)
        self.level_pairs = {}
        for lv, ix in self.level_idx.items():
            pos = {g: k for k, g in enumerate(ix)}
            self.level_pairs[lv] = [(pos[a], pos[b]) for a, b in self.traj_pairs
                                    if a in pos and b in pos]

        # --- Per-vertebra meshes (UI v2) ---
        self.vertebrae = None
        self.focus_level = None          # None = show all levels
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

        # --- State ---
        self.state = "SEARCHING"
        self.prev_rvec = self.prev_tvec = None
        self.last_good_time = self.verify_start_time = 0.0
        self._pre_stale_state = None   # state to resume after a brief dropout
        self.show_traj = self.show_ellipses = self.show_wire = True
        self.detach = False

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

    def _update_focus_label(self):
        show = (self.vertebrae is not None and self.state == "NAVIGATING"
                and self.focus_level is not None)
        if show:
            levels = sorted(self.level_idx)
            k = levels.index(self.focus_level)
            prev_t = f"L{levels[k - 1]}" if k > 0 else "all levels"
            next_t = f"L{levels[k + 1]}" if k < len(levels) - 1 else "all levels"
            txt = (f"FOCUS  L{self.focus_level}      "
                   f"Left/Up: {prev_t}    Right/Down: {next_t}    A: all levels")
            if self.focus_label.text() != txt:
                self.focus_label.setText(txt)
                self._place_focus_label()
            self.focus_label.show()
            self.focus_label.raise_()
        elif self.focus_label.isVisible():
            self.focus_label.hide()

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        # --- Multi-marker PnP ---
        pnp_ok = False
        n_markers = 0
        reproj_rms = float("nan")
        distance_mm = elev_deg = az_deg = 0.0
        cam_pos = np.zeros(3)
        in_env = False
        env_tier = "outside"
        amb_rejected = False   # single-face solution discarded by the side check
        tre_values = None
        visible_ids = []
        rvec = tvec = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids,
                                          borderColor=(60, 160, 60))
            pnp_ok, rvec, tvec, n_markers, reproj_rms = multi_marker_pnp(
                ids, corners, self.corner_lut, self.K, self.D,
                self.prev_rvec, self.prev_tvec,
            )
            if pnp_ok:
                cam_pos, distance_mm, elev_deg, az_deg = decompose_pose(rvec, tvec)
                visible_ids = [m for m in ids.flatten() if m in self.corner_lut]

                # Reject mirrored PnP solution (planar ambiguity).
                # The mirror places the camera hundreds of mm on the wrong
                # side. Use generous margins (100mm) to allow oblique views.
                if self.is_v3:
                    face_a_only = all(m < 4 for m in visible_ids)
                    face_b_only = all(m >= 4 for m in visible_ids)

                    if face_a_only and cam_pos[2] > -103.7 + 100:
                        pnp_ok = False  # camera way too cranial for Face A
                        amb_rejected = True
                    elif face_b_only and cam_pos[0] < 70.1 - 100:
                        pnp_ok = False  # camera way too medial for Face B
                        amb_rejected = True

                # Per-marker reprojection error (debug: find geometry mismatches)
                if pnp_ok and (reproj_rms > 5.0 or getattr(self, "_print_residuals", False)):
                    self._print_residuals = False
                    for mid, c2d in zip(ids.flatten(), corners):
                        if mid in self.corner_lut:
                            obj = self.corner_lut[mid]
                            rep, _ = cv2.projectPoints(
                                obj, rvec, tvec, self.K, self.D)
                            err = np.linalg.norm(
                                rep.reshape(4, 2) - c2d.reshape(4, 2), axis=1)
                            print(f"  ID {mid}: reproj err = "
                                  f"{err.mean():.1f}px (corners: "
                                  f"{err.round(1).tolist()})")

                if pnp_ok:
                    self.prev_rvec, self.prev_tvec = rvec.copy(), tvec.copy()
                    self.last_good_time = now
                    env_tier = envelope_tier(elev_deg, az_deg, distance_mm) if self.is_v3 else "full"
                    in_env = env_tier == "full"
                    tre_values = compute_tre_scalar(visible_ids, self.corner_lut,
                                                    self.target_pts, self.args.fle)

                    if self.state == "STALE":
                        # Brief dropout recovered: resume where we were.
                        # VERIFYING keeps its countdown, NAVIGATING stays put.
                        self.state = self._pre_stale_state or "VERIFYING"
                        self._pre_stale_state = None
                    elif self.state == "SEARCHING":
                        self.state = "VERIFYING"
                        self.verify_start_time = now

        # --- Stale / lost ---
        if not pnp_ok:
            if (self.prev_rvec is not None and
                    (now - self.last_good_time) < STALE_HOLD_SEC):
                if self.state not in ("SEARCHING", "STALE"):
                    self._pre_stale_state = self.state
                    self.state = "STALE"
                rvec, tvec = self.prev_rvec, self.prev_tvec
                cam_pos, distance_mm, elev_deg, az_deg = decompose_pose(rvec, tvec)
                env_tier = envelope_tier(elev_deg, az_deg, distance_mm) if self.is_v3 else "reduced"
                in_env = env_tier == "full"
                tre_values = compute_tre_scalar(visible_ids, self.corner_lut,
                                                self.target_pts, self.args.fle)
            else:
                self.state = "SEARCHING"
                self._pre_stale_state = None
                self.prev_rvec = self.prev_tvec = None
                visible_ids = []

        # --- Verify countdown ---
        if self.state == "VERIFYING":
            if (now - self.verify_start_time) >= VERIFY_COUNTDOWN_SEC:
                self.state = "NAVIGATING"
                print("Registration confirmed (auto).")

        t2 = time.perf_counter()

        # --- On-frame overlays (drawn onto numpy frame) ---
        doff = DETACH_OFFSET_PX if self.detach else 0

        if self.state in ("VERIFYING", "NAVIGATING", "STALE"):
            if (self.show_wire and self.vertebrae is not None
                    and self.state != "STALE"):
                # UI v2: VERIFYING shows every level; NAVIGATING shows the
                # focused level bold with neighbours ghosted (or all levels
                # if no focus is set).
                sel = self.focus_level if self.state == "NAVIGATING" else None
                draw_vertebra_layers(frame, self.vertebrae,
                                     rvec, tvec, self.K, self.D, selected=sel)
            elif (self.show_wire and self.mesh_verts is not None
                    and self.state != "STALE"):
                draw_wireframe(frame, self.mesh_verts,
                               self.mesh_edge_i0, self.mesh_edge_i1,
                               self.mesh_edge_normals,
                               rvec, tvec, self.K, self.D, cam_pos)

            if self.state == "NAVIGATING" and tre_values is not None:
                # Focus mode: one level shows entry, isthmus, trajectory and guide.
                # All-levels mode: entry points and trajectory lines only.
                fl = self.focus_level
                focus = fl is not None and fl in self.level_idx
                tre_arr = np.asarray(tre_values)
                if focus:
                    ix = self.level_idx[fl]
                else:
                    ix = self.entry_idx                      # isthmus dots hidden
                f_pts = self.target_pts[ix]
                f_labels = [self.target_labels[i] for i in ix]
                f_tre = tre_arr[ix]

                if self.show_traj:
                    if focus:
                        pos = {g: k for k, g in enumerate(ix)}
                        for ia, ib in self.traj_pairs:       # (isthmus, entry)
                            if ia in pos and ib in pos:
                                # guide colour follows the worse of the two ends
                                oc = (tre_colour(max(tre_arr[ia], tre_arr[ib]), env_tier)
                                      if GUIDE_BY_STATUS else None)
                                draw_focus_trajectory(
                                    frame, self.target_pts[ib], self.target_pts[ia],
                                    rvec, tvec, self.K, self.D, doff, status_colour=oc)
                    else:
                        for ia, ib in self.traj_pairs:       # (isthmus, entry)
                            oc = (tre_colour(max(tre_arr[ia], tre_arr[ib]), env_tier)
                                  if GUIDE_BY_STATUS else None)
                            draw_overview_trajectory(
                                frame, self.target_pts[ib], self.target_pts[ia],
                                rvec, tvec, self.K, self.D, doff, status_colour=oc)

                _NoOpTR = type("_NR", (), {
                    "h": frame.shape[0], "w": frame.shape[1],
                    "text": lambda self, *a, **k: None,
                    "measure": lambda self, *a, **k: (0, 0),
                })()
                draw_targets(frame, f_pts, f_labels,
                             f_tre, rvec, tvec, self.K, self.D,
                             visible_ids, self.corner_lut, self.args.fle,
                             env_tier, _NoOpTR, self.show_ellipses, doff,
                             (0, 0, 0) if (MARKERS_BLACK and focus) else None)

        t3 = time.perf_counter()

        # --- Update Qt widgets (only if values changed) ---
        self.status_panel.update_values(n_markers, pnp_ok, reproj_rms)
        self.status_panel.set_accent({
            "SEARCHING": "stale",
            "VERIFYING": "default",
            "NAVIGATING": "good",
            "STALE": "bad",
        }.get(self.state, "default"))

        self.pose_panel.update_values(distance_mm, elev_deg, az_deg, env_tier)
        self.pose_panel.set_accent(
            {"full": "good", "reduced": "warn"}.get(env_tier, "bad"))

        tp_vals, tp_labels = tre_values, self.target_labels
        if (self.focus_level is not None and self.state == "NAVIGATING"
                and tre_values is not None and self.focus_level in self.level_idx):
            # Focus mode: show the selected vertebra's entry point nearest the camera.
            ent = [i for i in self.level_idx[self.focus_level]
                   if "_E_" in self.target_labels[i]]
            if ent:
                j = min(ent, key=lambda i: np.linalg.norm(self.target_pts[i] - cam_pos))
                tp_vals = np.asarray(tre_values)[[j]]
                tp_labels = [self.target_labels[j]]
        self.tre_panel.update_values(tp_vals, tp_labels, env_tier)

        # State pill
        countdown = 0.0
        if self.state == "VERIFYING":
            countdown = VERIFY_COUNTDOWN_SEC - (now - self.verify_start_time)
        self.state_pill.set_state(self.state, countdown)

        # Axis triad
        if self._show_triad and pnp_ok and self.state != "SEARCHING":
            self.axis_triad.set_rvec(rvec)
            if not self.axis_triad.isVisible():
                self.axis_triad.show()
        else:
            self.axis_triad.set_rvec(None)

        # Target radar
        if self._show_radar and self.state == "NAVIGATING":
            self.target_radar.set_tre_values(tre_values, self.target_labels, env_tier)
            if not self.target_radar.isVisible():
                self.target_radar.show()
        else:
            self.target_radar.set_tre_values(None, [], env_tier)

        self._update_focus_label()

        # Reposition state pill (its width changes with countdown text)
        w = self.width()
        self.state_pill.move((w - self.state_pill.width()) // 2,
                             max(int(self.height() * 0.02), 12))

        # --- Push frame to video label ---
        self._last_frame_bgr = frame.copy()
        self._last_clean_bgr = clean
        if rvec is not None:
            self._last_pose = {
                "rvec": np.asarray(rvec).ravel().tolist(),
                "tvec": np.asarray(tvec).ravel().tolist(),
                "state": self.state,
                "pnp_ok": bool(pnp_ok),
                "n_markers": int(n_markers),
                "rms_px": None if np.isnan(reproj_rms) else float(reproj_rms),
                "visible_ids": [int(i) for i in visible_ids],
                "distance_mm": float(distance_mm),
                "elevation_deg": float(elev_deg),
                "azimuth_deg": float(az_deg),
                "env_tier": env_tier,
                "focus_level": self.focus_level,
            }
        else:
            self._last_pose = None
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

        # --- CSV data log (every frame) ---
        if self._csv_writer is not None:
            tre_vals = []
            if tre_values is not None:
                tre_vals = [f"{v:.3f}" for v in tre_values]
            else:
                tre_vals = [""] * len(self.target_labels)
            self._csv_writer.writerow([
                f"{now:.3f}", self._step, self.state, n_markers, pnp_ok,
                f"{reproj_rms:.2f}" if not np.isnan(reproj_rms) else "",
                f"{distance_mm:.1f}", f"{elev_deg:.1f}", f"{az_deg:.1f}",
                f"{cam_pos[0]:.1f}", f"{cam_pos[1]:.1f}", f"{cam_pos[2]:.1f}",
                in_env, env_tier, int(amb_rejected),
                f"{(t1-t0)*1000:.1f}", f"{(t2-t1)*1000:.1f}",
                f"{(t3-t2)*1000:.1f}", f"{(t4-t3)*1000:.1f}",
                "|".join(str(i) for i in visible_ids),
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
        if text == "q":
            self.close()
        elif k == Qt.Key_Escape:
            # Toggle shortcuts panel
            self._show_shortcuts = not self._show_shortcuts
            self.shortcuts_panel.setVisible(self._show_shortcuts)
        elif text == "t":
            self.show_traj = not self.show_traj
        elif text == "e":
            self.show_ellipses = not self.show_ellipses
        elif text == "w":
            self.show_wire = not self.show_wire
        elif text in ("1", "2", "3", "4", "5"):
            self._set_focus(int(text))
        elif text in ("a", "0"):
            self._set_focus(None)
        elif k in (Qt.Key_Right, Qt.Key_Down) or text == "]":
            self._step_focus(+1)
        elif k in (Qt.Key_Left, Qt.Key_Up) or text == "[":
            self._step_focus(-1)
        elif text == "d":
            self.detach = not self.detach
        elif text == "g":
            self._show_triad = not self._show_triad
            self.axis_triad.setVisible(self._show_triad)
        elif text == "x":
            self._show_radar = not self._show_radar
            self.target_radar.setVisible(self._show_radar)
        elif text == "h":
            # Toggle all HUD panels
            for widget in [self.status_panel, self.pose_panel, self.tre_panel,
                           self.state_pill]:
                widget.setVisible(not widget.isVisible())
        elif text == "f":
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif text == "c" and self.state == "VERIFYING":
            self.state = "NAVIGATING"
            print("Registration confirmed (manual).")
        elif text == "r":
            self.state = "SEARCHING"
            self.prev_rvec = self.prev_tvec = None
        elif text == "s":
            self._save_screenshot()
        elif text == "n":
            self._step += 1
            print(f"--- step {self._step} ---")
        elif text == "p":
            self._print_residuals = True

    def _set_focus(self, level):
        """Focus one vertebra level (1-5), or None for all levels."""
        if level is not None and level not in self.level_idx:
            print(f"No targets for L{level}")
            return
        self.focus_level = level
        print("Focus:", "all levels" if level is None else f"L{level}")

    def _step_focus(self, delta):
        levels = sorted(self.level_idx)
        if not levels:
            return
        if self.focus_level is None:
            self._set_focus(levels[0] if delta > 0 else levels[-1])
        else:
            k = levels.index(self.focus_level) + delta
            self._set_focus(levels[k] if 0 <= k < len(levels) else None)

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