"""
Tracking half of the live AR overlay: marker detection, multi-marker PnP,
the SEARCHING / VERIFYING / NAVIGATING / STALE state machine, envelope tier
and TRE. No drawing and no Qt in here, so it can be replayed, tested and
reused by other front ends (phone, glasses). The logic is moved line for line
from ARWindow.next_frame; only its home has changed.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from ar_board import multi_marker_pnp, decompose_pose
from ar_tre import compute_tre_scalar
from ar_theme import envelope_tier

STALE_HOLD_SEC = 0.3
VERIFY_COUNTDOWN_SEC = 1.0


@dataclass
class FrameResult:
    """Everything the renderer and logger need about one frame."""
    now: float
    state: str
    corners: object = None          # raw ArUco detections (for drawing the boxes)
    ids: object = None
    pnp_ok: bool = False
    n_markers: int = 0
    reproj_rms: float = float("nan")
    rvec: object = None
    tvec: object = None
    cam_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance_mm: float = 0.0
    elev_deg: float = 0.0
    az_deg: float = 0.0
    env_tier: str = "outside"
    in_env: bool = False
    amb_rejected: bool = False      # single-face solution discarded by the side check
    tre_values: object = None
    visible_ids: list = field(default_factory=list)
    verify_remaining: float = 0.0   # seconds left in the VERIFYING countdown
    board: str = ""                 # which board this frame was solved against
    board_switched: bool = False    # auto mode changed board on this frame


class Tracker:
    def __init__(self, detector, selector, K, D, target_pts, fle):
        self.detector = detector
        self.selector = selector
        self.K, self.D = K, D
        self.target_pts = target_pts
        self.fle = fle

        self.state = "SEARCHING"
        self.prev_rvec = self.prev_tvec = None
        self.last_good_time = self.verify_start_time = 0.0
        self._pre_stale_state = None    # state to resume after a brief dropout
        self.print_residuals = False    # one-shot debug print (engineer mode)

    @property
    def board(self):
        return self.selector.board

    @property
    def corner_lut(self):
        return self.selector.board.lut

    @property
    def is_v3(self):
        return self.selector.board.is_v3
    
    # ---- commands from the UI ----
    def confirm(self):
        if self.state == "VERIFYING":
            self.state = "NAVIGATING"
            print("Registration confirmed (manual).")

    def reset(self):
        self.state = "SEARCHING"
        self.prev_rvec = self.prev_tvec = None

    # ---- per-frame ----
    def process(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        switched = self.selector.update(ids)
        if switched:
            self.reset()            # the old pose belongs to the other board

        pnp_ok = False
        n_markers = 0
        reproj_rms = float("nan")
        distance_mm = elev_deg = az_deg = 0.0
        cam_pos = np.zeros(3)
        in_env = False
        env_tier = "outside"
        amb_rejected = False
        tre_values = None
        visible_ids = []
        rvec = tvec = None

        if ids is not None and len(ids) > 0:
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
                if pnp_ok and (reproj_rms > 5.0 or self.print_residuals):
                    self.print_residuals = False
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
                                                    self.target_pts, self.fle)

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
                                                self.target_pts, self.fle)
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

        remaining = 0.0
        if self.state == "VERIFYING":
            remaining = VERIFY_COUNTDOWN_SEC - (now - self.verify_start_time)

        return FrameResult(
            now=now, state=self.state, corners=corners, ids=ids,
            pnp_ok=pnp_ok, n_markers=n_markers, reproj_rms=reproj_rms,
            rvec=rvec, tvec=tvec, cam_pos=cam_pos,
            distance_mm=distance_mm, elev_deg=elev_deg, az_deg=az_deg,
            env_tier=env_tier, in_env=in_env, amb_rejected=amb_rejected,
            tre_values=tre_values, visible_ids=visible_ids,
            verify_remaining=remaining, board=self.board.name, board_switched=switched,
        )