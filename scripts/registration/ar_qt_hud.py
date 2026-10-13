"""
Qt HUD widgets for the AR overlay.

Each widget owns its own paint state and only updates when its input
values change. Layered on top of the video QLabel via QStackedLayout.

Scaffold: StatusPanel is fully implemented. PosePanel, TrePanel,
StatePill, ShortcutsPanel, AxisTriad, TargetRadar to follow after
we verify this renders correctly on your Mac.
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (
    QFrame, QLabel, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
    QGraphicsDropShadowEffect,
)


# ====================================================================
# Helpers
# ====================================================================

def _apply_shadow(widget, blur=24, offset_y=4, alpha=140):
    """Add a soft drop shadow to a widget."""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, offset_y)
    shadow.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)


def _row(label_text, value_text=""):
    """
    Build a label/value row. Returns (container_widget, value_label).
    Caller keeps the value_label to update its text.
    """
    row = QWidget()
    row.setAttribute(Qt.WA_TranslucentBackground, True)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    label = QLabel(label_text.upper())
    label.setObjectName("hudLabel")

    value = QLabel(value_text)
    value.setObjectName("hudValue")
    value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    layout.addWidget(label)
    layout.addStretch(1)
    layout.addWidget(value)

    return row, value


def _tier(in_envelope):
    """Normalise a bool (2-tier) or tier string (3-tier) to a tier string."""
    if in_envelope is True:
        return "full"
    if in_envelope is False or in_envelope is None:
        return "outside"
    return in_envelope


TIER_STATUS = {"full": "good", "reduced": "warn", "outside": "bad"}
TIER_TEXT = {"full": "FULL", "reduced": "REDUCED", "outside": "OUTSIDE"}


# ====================================================================
# Base panel
# ====================================================================

class HudPanel(QFrame):
    """
    Rounded translucent panel with an accent bar on top.
    Accent colour is set via a Qt property so the QSS picks it up.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hudPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("accent", "default")

    def set_accent(self, accent):
        """
        accent: 'good' | 'warn' | 'bad' | 'stale' | 'default'
        Triggers a style refresh so the border colour updates.
        """
        if self.property("accent") != accent:
            self.setProperty("accent", accent)
            self.style().unpolish(self)
            self.style().polish(self)


# ====================================================================
# Status panel (top-left)
# ====================================================================

class StatusPanel(HudPanel):
    """
    Shows MARKERS n/8, PNP OK|FAIL, RMS x.xpx.
    Accent bar colour reflects overall state (searching/verifying/etc).
    """

    def __init__(self, n_total=8, gate=4, parent=None):
        super().__init__(parent)
        self.n_total = n_total
        self.gate = gate

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        markers_row, self._markers_val = _row("Markers", f"0/{n_total}")
        pnp_row, self._pnp_val = _row("PnP", "--")
        rms_row, self._rms_val = _row("RMS", "--")

        layout.addWidget(markers_row)
        layout.addWidget(pnp_row)
        layout.addWidget(rms_row)

        # Drop shadow disabled: conflicts with translucent parent on some
        # Qt versions and makes the whole panel invisible. Re-enable once
        # positioning is verified.
        # _apply_shadow(self)

        # Explicit size so the panel is visible even before content updates
        self.setMinimumSize(240, 120)

        # Cache last values to skip unnecessary setText calls
        self._last = {"markers": None, "pnp": None, "rms": None}

    def _set_status(self, label, status):
        """Set the QSS status property on a value label."""
        if label.property("status") != status:
            label.setProperty("status", status)
            label.style().unpolish(label)
            label.style().polish(label)

    def update_values(self, n_markers, pnp_ok, rms):
        # Markers
        m_str = f"{n_markers}/{self.n_total}"
        if self._last["markers"] != m_str:
            self._markers_val.setText(m_str)
            self._set_status(self._markers_val,
                             "good" if n_markers >= self.gate else "bad")
            self._last["markers"] = m_str

        # PnP
        p_str = "OK" if pnp_ok else "FAIL"
        if self._last["pnp"] != p_str:
            self._pnp_val.setText(p_str)
            self._set_status(self._pnp_val, "good" if pnp_ok else "bad")
            self._last["pnp"] = p_str

        # RMS
        if rms is None or rms != rms:  # NaN check
            r_str = "--"
            r_status = "muted"
        else:
            r_str = f"{rms:.1f}px"
            r_status = "good" if rms < 2.0 else "warn"
        if self._last["rms"] != r_str:
            self._rms_val.setText(r_str)
            self._set_status(self._rms_val, r_status)
            self._last["rms"] = r_str


# ====================================================================
# Pose panel (top-right)
# ====================================================================

class PosePanel(HudPanel):
    """
    Shows DIST cm, ELEV deg, AZ deg.
    Accent bar colour reflects operational envelope status.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        dist_row, self._dist_val = _row("Distance", "--")
        elev_row, self._elev_val = _row("Elevation", "--")
        az_row, self._az_val = _row("Azimuth", "--")
        zone_row, self._zone_val = _row("Zone", "--")

        layout.addWidget(dist_row)
        layout.addWidget(elev_row)
        layout.addWidget(az_row)
        layout.addWidget(zone_row)

        self.setMinimumSize(240, 150)
        self._last = {"dist": None, "elev": None, "az": None, "zone": None}

    def _set_status(self, label, status):
        if label.property("status") != status:
            label.setProperty("status", status)
            label.style().unpolish(label)
            label.style().polish(label)

    def update_values(self, distance_mm, elev_deg, az_deg, in_envelope):
        if distance_mm > 0:
            d_str = f"{distance_mm / 10:.0f} cm"
            e_str = f"{elev_deg:+.0f}\u00b0"
            a_str = f"{az_deg:.0f}\u00b0"
            tier = _tier(in_envelope)
            e_status = TIER_STATUS[tier]
            z_str = TIER_TEXT[tier]
        else:
            d_str = e_str = a_str = z_str = "--"
            e_status = "muted"

        if self._last["dist"] != d_str:
            self._dist_val.setText(d_str)
            self._last["dist"] = d_str
        if self._last["elev"] != (e_str, e_status):
            self._elev_val.setText(e_str)
            self._set_status(self._elev_val, e_status)
            self._last["elev"] = (e_str, e_status)
        if self._last["az"] != (a_str, e_status):
            self._az_val.setText(a_str)
            self._set_status(self._az_val, e_status)
            self._last["az"] = (a_str, e_status)
        if self._last["zone"] != (z_str, e_status):
            self._zone_val.setText(z_str)
            self._set_status(self._zone_val, e_status)
            self._last["zone"] = (z_str, e_status)


# ====================================================================
# TRE panel (bottom-left)
# ====================================================================

class TrePanel(HudPanel):
    """
    Shows best target label and its TRE, plus a legend row.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        tgt_row, self._tgt_val = _row("Target", "--")
        tre_row, self._tre_val = _row("TRE", "--")

        # Legend row
        legend = QWidget()
        legend.setAttribute(Qt.WA_TranslucentBackground, True)
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 4, 0, 0)
        legend_layout.setSpacing(16)

        for colour_hex, text in [("#64D28C", "< 1.0mm"),
                                 ("#EBB450", "1.0-2.0mm"),
                                 ("#D75A5A", "> 2.0mm")]:
            item = QWidget()
            item.setAttribute(Qt.WA_TranslucentBackground, True)
            il = QHBoxLayout(item)
            il.setContentsMargins(0, 0, 0, 0)
            il.setSpacing(6)
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                f"background-color: {colour_hex}; border-radius: 5px;"
            )
            label = QLabel(text)
            label.setObjectName("legendLabel")
            il.addWidget(dot)
            il.addWidget(label)
            legend_layout.addWidget(item)
        legend_layout.addStretch(1)

        layout.addWidget(tgt_row)
        layout.addWidget(tre_row)
        layout.addWidget(legend)

        self.setMinimumSize(260, 130)
        self.setMaximumWidth(280)
        self._last = {"tgt": None, "tre": None}

    def _set_status(self, label, status):
        if label.property("status") != status:
            label.setProperty("status", status)
            label.style().unpolish(label)
            label.style().polish(label)

    def update_values(self, tre_values, labels, in_envelope):
        import numpy as np
        if tre_values is not None and len(tre_values) > 0 and np.isfinite(tre_values).any():
            best = int(np.argmin(tre_values))
            bv = tre_values[best]
            bl = labels[best] if labels else "?"
            tre_str = f"{bv:.2f} mm"
            tier = _tier(in_envelope)
            if tier == "outside" or bv >= 2.0:
                status = "bad"
            elif tier == "reduced":
                status = "warn"
            elif bv < 1.0:
                status = "good"
            else:
                status = "warn"
        else:
            bl = "--"
            tre_str = "--"
            status = "muted"

        # Cache text AND status, so a tier change recolours even when
        # the text is unchanged.
        if self._last["tgt"] != (bl, status):
            self._tgt_val.setText(bl)
            self._set_status(self._tgt_val, status)
            self._last["tgt"] = (bl, status)
        if self._last["tre"] != (tre_str, status):
            self._tre_val.setText(tre_str)
            self._set_status(self._tre_val, status)
            self._last["tre"] = (tre_str, status)


# ====================================================================
# State pill (top-center)
# ====================================================================

class StatePill(QFrame):
    """
    Small rounded pill with a coloured dot + state label.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statePill")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("state", "searching")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 20, 8)
        layout.setSpacing(10)

        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        self._dot.setStyleSheet("background-color: #555F64; border-radius: 5px;")

        self._text = QLabel("SEARCHING")
        self._text.setObjectName("statePillText")

        layout.addWidget(self._dot)
        layout.addWidget(self._text)

        self.setMinimumHeight(40)
        self._current_state = "SEARCHING"
        self._current_countdown = -1.0

    def set_state(self, state, countdown=0.0):
        text_map = {
            "SEARCHING": "SEARCHING",
            "VERIFYING": f"VERIFY  {max(0, countdown):.1f}s",
            "NAVIGATING": "NAVIGATING",
            "STALE": "STALE POSE",
        }
        dot_col = {
            "SEARCHING": "#555F64",
            "VERIFYING": "#3CC3C3",
            "NAVIGATING": "#64D28C",
            "STALE": "#D75A5A",
        }.get(state, "#3CC3C3")

        new_text = text_map.get(state, state)
        if state != self._current_state or (
                state == "VERIFYING" and countdown != self._current_countdown):
            self._text.setText(new_text)
            self._dot.setStyleSheet(
                f"background-color: {dot_col}; border-radius: 5px;"
            )
            self._current_state = state
            self._current_countdown = countdown

        # Update the QSS state attribute for accent bar
        prop_state = state.lower()
        if self.property("state") != prop_state:
            self.setProperty("state", prop_state)
            self.style().unpolish(self)
            self.style().polish(self)

        self.adjustSize()


# ====================================================================
# Shortcuts panel (bottom-right)
# ====================================================================

class ShortcutsPanel(QFrame):
    """
    List of keyboard shortcuts.
    """

    SHORTCUTS = [
        ("c", "confirm"),
        ("r", "reject"),
        ("e", "ellipses"),
        ("w", "wireframe"),
        ("g", "axis gizmo"),
        ("x", "target radar"),
        ("h", "toggle hud"),
        ("f", "fullscreen"),
        ("s", "screenshot"),
        ("n", "next step"),
        ("p", "print marker residuals"),
        ("esc", "toggle help"),
        ("q", "quit"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("shortcutsPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QGridLayout(self)
        layout.setContentsMargins(16, 12, 20, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(5)

        for i, (k, n) in enumerate(self.SHORTCUTS):
            key_label = QLabel(k)
            key_label.setObjectName("shortcutKey")
            key_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            name_label = QLabel(n)
            name_label.setObjectName("shortcutName")
            layout.addWidget(key_label, i, 0)
            layout.addWidget(name_label, i, 1)


# ====================================================================
# Axis triad widget (LPS orientation gizmo)
# ====================================================================

class AxisTriadWidget(QWidget):
    """
    Small QPainter widget showing phantom LPS axes rotated by current
    camera pose. Only redraws on set_rvec().
    """

    _AXIS_COLS = {
        "X": QColor(215, 90, 90),   # patient Left → red
        "Y": QColor(100, 210, 140), # Posterior → green
        "Z": QColor(60, 140, 220),  # Superior → blue
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(120, 120)
        self._axes_cam = None

    def set_rvec(self, rvec):
        """Update the rotation and trigger repaint."""
        import cv2
        import numpy as np
        if rvec is None:
            if self._axes_cam is not None:
                self._axes_cam = None
                self.update()
            return
        R, _ = cv2.Rodrigues(rvec)
        new = R @ np.eye(3)
        if self._axes_cam is None or not np.allclose(new, self._axes_cam,
                                                    atol=0.01):
            self._axes_cam = new
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx, cy = self.width() // 2, self.height() // 2
        size = min(cx, cy) - 12

        # Low-opacity dark circle backdrop (draws even when axes aren't set)
        backdrop = QColor(16, 20, 22, 130)
        painter.setBrush(QBrush(backdrop))
        painter.setPen(QPen(QColor(60, 195, 195, 90), 1))
        radius = size + 8
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        if self._axes_cam is None:
            return

        # Sort axes back-to-front by depth (z in camera frame)
        endpoints = []
        for i, name in enumerate(["X", "Y", "Z"]):
            v = self._axes_cam[:, i]
            ex = int(cx + v[0] * size)
            ey = int(cy - v[1] * size)
            depth = v[2]
            endpoints.append((name, ex, ey, depth))
        endpoints.sort(key=lambda e: e[3], reverse=True)

        for name, ex, ey, depth in endpoints:
            col = QColor(self._AXIS_COLS[name])
            if depth > 0.3:
                col.setAlpha(120)
            # Line
            pen = QPen(col, 2.5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(cx, cy, ex, ey)
            # Tip
            painter.setBrush(QBrush(col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(ex - 5, ey - 5, 10, 10)
            # Label
            painter.setPen(col)
            painter.setFont(QFont("Helvetica Neue", 11, QFont.Bold))
            label_x = ex + (6 if ex >= cx else -18)
            label_y = ey - 8
            painter.drawText(label_x, label_y, name)


# ====================================================================
# Target radar widget
# ====================================================================

class TargetRadarWidget(QWidget):
    """
    Grid of dots showing per-pedicle TRE status (L1-L5, L/R sides).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedSize(140, 180)
        self.setStyleSheet(
            "background-color: rgba(16, 20, 22, 220);"
            "border-top: 2px solid #267878;"
            "border-radius: 6px;"
        )
        self._groups = {}  # {(level, side): worst_tre}
        self._in_env = False

    def set_tre_values(self, tre_values, labels, in_envelope):
        import numpy as np
        if tre_values is None or len(tre_values) == 0:
            new_groups = {}
        else:
            groups = {}
            for i, lbl in enumerate(labels):
                parts = lbl.split("_")
                if len(parts) < 2:
                    continue
                level = parts[0]
                side = parts[-1]
                key = (level, side)
                cur = groups.get(key, -1)
                if tre_values[i] > cur:
                    groups[key] = float(tre_values[i])
            new_groups = groups

        if new_groups != self._groups or in_envelope != self._in_env:
            self._groups = new_groups
            self._in_env = in_envelope
            self.update()

    def _tre_colour(self, tre):
        tier = _tier(self._in_env)
        red, amber, green = QColor(215, 90, 90), QColor(235, 180, 80), QColor(100, 210, 140)
        if tier == "outside" or tre >= 2.0:
            return red
        if tier == "reduced":
            return amber
        return green if tre < 1.0 else amber

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Title
        painter.setPen(QColor(135, 145, 150))
        painter.setFont(QFont("Menlo", 11, QFont.Bold))
        painter.drawText(16, 26, "TARGETS")

        # Column headers
        painter.setPen(QColor(78, 85, 96))
        painter.setFont(QFont("Menlo", 10))
        painter.drawText(72, 50, "L")
        painter.drawText(112, 50, "R")

        if not self._groups:
            return

        # Rows: L1 to L5
        levels = sorted({lv for (lv, _) in self._groups.keys()},
                        key=lambda s: int(s[1:]) if s[1:].isdigit() else 99)

        row_h = 22
        y0 = 58
        painter.setFont(QFont("Menlo", 10))
        for i, level in enumerate(levels):
            ry = y0 + i * row_h
            painter.setPen(QColor(135, 145, 150))
            painter.drawText(16, ry + 5, level)

            for j, side in enumerate(["L", "R"]):
                x = 68 + j * 40
                if (level, side) in self._groups:
                    col = self._tre_colour(self._groups[(level, side)])
                    painter.setBrush(QBrush(col))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(x, ry - 5, 11, 11)