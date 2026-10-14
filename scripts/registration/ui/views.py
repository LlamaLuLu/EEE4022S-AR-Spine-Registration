"""
Panes for the multi-view layout. The main camera view stays in the window; every
extra view (zoom now, axial later) is a ViewPane in the column on its right.
A pane only shows what it is given: a BGR image or a short message.
"""
import cv2
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy

MAX_RENDER_PX = 1100        # cap on the longest side a view is rendered at


def image_pos(label, pos, iw, ih):
    """Where a click at `pos` (label coordinates) lands in an image of iw x ih pixels that the label
    shows scaled to fit and centred. Returns (u, v) in image pixels, or None outside the picture."""
    cr = label.contentsRect()
    k = min(cr.width() / iw, cr.height() / ih)
    dw, dh = iw * k, ih * k
    ox = cr.x() + (cr.width() - dw) / 2.0
    oy = cr.y() + (cr.height() - dh) / 2.0
    u, v = (pos.x() - ox) / k, (pos.y() - oy) / k
    return (u, v) if (0 <= u < iw and 0 <= v < ih) else None


class ClickLabel(QLabel):
    """A QLabel that reports left clicks."""
    clicked = Signal(QPoint)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(e.position().toPoint())
        super().mousePressEvent(e)


class ViewPane(QLabel):
    clicked = Signal(QPoint)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(e.position().toPoint())
        super().mousePressEvent(e)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("viewPane")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # the layout decides the size
        self.tag = QLabel(title, self)
        self.tag.setObjectName("paneTag")
        self.tag.move(10, 10)
        self._title = title

    def render_size(self):
        """Pixel size to render this view at: the pane's size on screen (sharp on HiDPI), capped."""
        dpr = self.devicePixelRatioF()
        cr = self.contentsRect()
        w, h = max(int(cr.width() * dpr), 2), max(int(cr.height() * dpr), 2)
        k = min(1.0, MAX_RENDER_PX / max(w, h))
        return max(int(w * k), 2), max(int(h * k), 2)

    def set_tag(self, text):
        if self.tag.text() != text:
            self.tag.setText(text)
            self.tag.adjustSize()

    def show_frame(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))
        dpr = self.devicePixelRatioF()
        pix = pix.scaled(self.contentsRect().size() * dpr, Qt.KeepAspectRatio, Qt.FastTransformation)
        pix.setDevicePixelRatio(dpr)
        self.setPixmap(pix)

    def show_message(self, text):
        self.clear()
        self.setText(text)


class ZoomControls(QFrame):
    """Small floating control for the zoom view: [ - ] x5 [ + ] [ Reset ]."""
    zoomIn = Signal()
    zoomOut = Signal()
    reset = Signal()

    QSS = """
    #zoomControls { background: rgba(14,18,22,215); border: 1px solid rgba(60,195,195,120); border-radius: 12px; }
    #zoomControls QPushButton { color: #E8EEF2; background: rgba(255,255,255,18); border: none;
                                border-radius: 8px; padding: 0px; margin: 0px; font-size: 16px; font-weight: 600; }
    #zoomControls QPushButton:hover { background: rgba(60,195,195,90); }
    #zoomControls QPushButton:disabled { color: rgba(232,238,242,70); background: rgba(255,255,255,8); }
    #zoomControls QPushButton#zoomReset { font-size: 12px; }
    #zoomControls QLabel { color: #9FB3BD; font-size: 12px; padding: 0px; margin: 0px; background: transparent; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("zoomControls")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(self.QSS)
        self.minus = QPushButton("\u2212", self)
        self.plus = QPushButton("+", self)
        self.label = QLabel("x1.0", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.reset_btn = QPushButton("Reset", self)
        self.reset_btn.setObjectName("zoomReset")
        self.minus.setToolTip("Zoom out  (-)")
        self.plus.setToolTip("Zoom in  (+)")
        self.reset_btn.setToolTip("Back to the camera view  (Z)")
        for b in (self.minus, self.plus, self.reset_btn):
            b.setFocusPolicy(Qt.NoFocus)                 # keys keep going to the main window
            b.setCursor(Qt.PointingHandCursor)
        self.minus.clicked.connect(self.zoomOut)
        self.plus.clicked.connect(self.zoomIn)
        self.reset_btn.clicked.connect(self.reset)
        # Placed by hand (no layout): equal padding on all four sides, every item the same height.
        PAD, GAP, BH = 9, 6, 32
        x = PAD
        for w, wd in ((self.minus, 32), (self.label, 46), (self.plus, 32), (self.reset_btn, 62)):
            w.setGeometry(x, PAD, wd, BH)
            x += wd + GAP
        self.setFixedSize(x - GAP + PAD, BH + 2 * PAD)

    def set_state(self, magnification, can_in, can_out):
        txt = f"x{magnification:.1f}" if magnification else "zoom"
        if self.label.text() != txt:
            self.label.setText(txt)
        self.plus.setEnabled(can_in)
        self.minus.setEnabled(can_out)