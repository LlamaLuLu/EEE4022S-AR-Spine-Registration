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
"""

import argparse
import sys
import time
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
    multi_marker_pnp, decompose_pose,
)
from ar_tre import compute_tre_scalar, DEFAULT_FLE_MM
from ar_targets import load_targets
from ar_overlay import (
    draw_targets, draw_trajectories, draw_wireframe,
    is_in_envelope,
)
from ar_stl import load_stl_for_wireframe

# Qt HUD
from ar_qt_styles import QSS
from ar_qt_hud import (
    StatusPanel, PosePanel, TrePanel, StatePill,
    ShortcutsPanel, AxisTriadWidget, TargetRadarWidget,
)


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
        if args.marker_map:
            slot_to_id = load_marker_map(args.marker_map)
            print(f"Loaded marker map: {slot_to_id}")
        else:
            slot_to_id = list(range(8))
        self.corner_lut = build_corner_lut(slot_to_id=slot_to_id)
        self.n_total = len(slot_to_id)

        # --- Targets ---
        self.target_pts, self.target_labels, self.traj_pairs = load_targets(
            args.isthmus, args.entry
        )
        print(f"Targets: {len(self.target_pts)}, "
              f"trajectory pairs: {len(self.traj_pairs)}")

        # --- Wireframe ---
        self.mesh_verts, self.mesh_edges = None, None
        if args.stl:
            self.mesh_verts, self.mesh_edges = load_stl_for_wireframe(args.stl)
            if self.mesh_verts is not None:
                print(f"STL wireframe: {len(self.mesh_verts)} verts, "
                      f"{len(self.mesh_edges)} edges")

        # --- Video ---
        self.rotate_code = ROTATE_MAP[args.rotate]
        self.cap = open_source(args.source)
        if not self.cap.isOpened():
            print(f"Could not open source: {args.source}")
            sys.exit(1)

        # Minimize internal buffer: always grab the newest frame, not a
        # stale one that was captured seconds ago over WiFi.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.detector = cv2.aruco.ArucoDetector(
            ARUCO_DICT, cv2.aruco.DetectorParameters()
        )

        # --- State ---
        self.state = "SEARCHING"
        self.prev_rvec = self.prev_tvec = None
        self.last_good_time = self.verify_start_time = 0.0
        self.show_traj = self.show_ellipses = self.show_wire = True
        self.detach = False

        # --- Timing instrumentation ---
        self._frame_count = 0
        self._last_log = time.perf_counter()

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
                                        parent=self.video_label)
        self.pose_panel = PosePanel(parent=self.video_label)
        self.tre_panel = TrePanel(parent=self.video_label)
        self.state_pill = StatePill(parent=self.video_label)
        self.shortcuts_panel = ShortcutsPanel(parent=self.video_label)
        self.axis_triad = AxisTriadWidget(parent=self.video_label)
        self.target_radar = TargetRadarWidget(parent=self.video_label)

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

    # ================================================================
    # Frame loop
    # ================================================================

    def next_frame(self):
        t0 = time.perf_counter()

        ok, frame = self.cap.read()
        if not ok:
            return
        if self.rotate_code is not None:
            frame = cv2.rotate(frame, self.rotate_code)

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
                self.prev_rvec, self.prev_tvec = rvec.copy(), tvec.copy()
                self.last_good_time = now
                cam_pos, distance_mm, elev_deg, az_deg = decompose_pose(rvec, tvec)
                in_env = is_in_envelope(elev_deg, az_deg)
                visible_ids = [m for m in ids.flatten() if m in self.corner_lut]
                tre_values = compute_tre_scalar(visible_ids, self.corner_lut,
                                                self.target_pts, self.args.fle)

                if self.state in ("SEARCHING", "STALE"):
                    self.state = "VERIFYING"
                    self.verify_start_time = now

        # --- Stale / lost ---
        if not pnp_ok:
            if (self.prev_rvec is not None and
                    (now - self.last_good_time) < STALE_HOLD_SEC):
                if self.state != "SEARCHING":
                    self.state = "STALE"
                rvec, tvec = self.prev_rvec, self.prev_tvec
                cam_pos, distance_mm, elev_deg, az_deg = decompose_pose(rvec, tvec)
                in_env = is_in_envelope(elev_deg, az_deg)
                tre_values = compute_tre_scalar(visible_ids, self.corner_lut,
                                                self.target_pts, self.args.fle)
            else:
                self.state = "SEARCHING"
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
            if (self.show_wire and self.mesh_verts is not None
                    and self.state != "STALE"):
                draw_wireframe(frame, self.mesh_verts, self.mesh_edges,
                               rvec, tvec, self.K, self.D, cam_pos)

            if self.state == "NAVIGATING" and tre_values is not None:
                if self.show_traj:
                    draw_trajectories(frame, self.target_pts, self.traj_pairs,
                                      rvec, tvec, self.K, self.D, doff)

                _NoOpTR = type("_NR", (), {
                    "h": frame.shape[0], "w": frame.shape[1],
                    "text": lambda self, *a, **k: None,
                    "measure": lambda self, *a, **k: (0, 0),
                })()
                draw_targets(frame, self.target_pts, self.target_labels,
                             tre_values, rvec, tvec, self.K, self.D,
                             visible_ids, self.corner_lut, self.args.fle,
                             in_env, _NoOpTR, self.show_ellipses, doff)

        t3 = time.perf_counter()

        # --- Update Qt widgets (only if values changed) ---
        self.status_panel.update_values(n_markers, pnp_ok, reproj_rms)
        self.status_panel.set_accent({
            "SEARCHING": "stale",
            "VERIFYING": "default",
            "NAVIGATING": "good",
            "STALE": "bad",
        }.get(self.state, "default"))

        self.pose_panel.update_values(distance_mm, elev_deg, az_deg, in_env)
        self.pose_panel.set_accent("good" if in_env else "bad")

        self.tre_panel.update_values(tre_values, self.target_labels, in_env)

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
            self.target_radar.set_tre_values(tre_values, self.target_labels, in_env)
            if not self.target_radar.isVisible():
                self.target_radar.show()
        else:
            self.target_radar.set_tre_values(None, [], in_env)

        # Reposition state pill (its width changes with countdown text)
        w = self.width()
        self.state_pill.move((w - self.state_pill.width()) // 2,
                             max(int(self.height() * 0.02), 12))

        # --- Push frame to video label ---
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
                         Qt.SmoothTransformation)
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

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--calib",
                    default="data/calibration/s22_ultra/camera_calibration.npz")
    ap.add_argument("--isthmus", default=None)
    ap.add_argument("--entry", default=None)
    ap.add_argument("--stl", default=None)
    ap.add_argument("--marker-map", default=None)
    ap.add_argument("--fle", type=float, default=DEFAULT_FLE_MM)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    args = ap.parse_args()

    # HiDPI is on by default in Qt6, no explicit attributes needed.
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    window = ARWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()