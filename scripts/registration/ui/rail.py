"""
Docked left rail for the AR window: state, trust badge, pedicle selector,
target card, engineer diagnostics (hidden by default) and the help / engineer
buttons. The rail only displays state and emits signals; the window owns the
logic (see ar_selection.py, ar_guidance.py).
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QGridLayout,
)

from .hud import HudPanel, StatePill, StatusPanel, PosePanel
from .views import ViewPane

RAIL_WIDTH = 300

RAIL_QSS = """
QFrame#rail {
    background-color: #0E1216;
    border-right: 1px solid #2D373C;
}
QLabel#railTitle {
    color: #8791A0;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 11px;
    letter-spacing: 2px;
    background: transparent;
}
QPushButton#railBtn {
    background-color: #161B20;
    color: #E1E8EB;
    border: 1px solid #2D373C;
    border-radius: 6px;
    padding: 9px 0px;
    font-size: 15px;
    font-weight: 600;
}
QPushButton#railBtn:hover { border-color: #3CC3C3; }
QPushButton#railBtn[active="true"] {
    background-color: #3CC3C3;
    color: #0B1014;
    border-color: #3CC3C3;
}
QPushButton#railBtn:disabled { color: #4E5560; border-color: #1E252B; }
QPushButton#railLink {
    background: transparent;
    color: #8791A0;
    border: none;
    text-align: left;
    padding: 6px 2px;
    font-size: 13px;
}
QPushButton#railLink:hover { color: #E1E8EB; }
QPushButton#railLink[active="true"] { color: #3CC3C3; }
QLabel#trustTitle {
    color: #E1E8EB;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 18px;
    font-weight: 700;
    background: transparent;
}
QLabel#trustTitle[status="good"] { color: #64D28C; }
QLabel#trustTitle[status="warn"] { color: #EBB450; }
QLabel#trustTitle[status="bad"]  { color: #D75A5A; }
QLabel#trustTitle[status="muted"] { color: #8791A0; }
QLabel#trustReason {
    color: #8791A0;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
    background: transparent;
}
QLabel#cardName {
    color: #E1E8EB;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 20px;
    font-weight: 700;
    background: transparent;
}
QLabel#cardValue {
    color: #8791A0;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 30px;
    font-weight: 700;
    background: transparent;
}
QLabel#cardValue[status="good"] { color: #64D28C; }
QLabel#cardValue[status="warn"] { color: #EBB450; }
QLabel#cardValue[status="bad"]  { color: #D75A5A; }
QLabel#cardCaption {
    color: #8791A0;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 12px;
    background: transparent;
}
QFrame#helpOverlay {
    background-color: rgba(14, 18, 22, 235);
    border: 1px solid #2D373C;
    border-top: 2px solid #3CC3C3;
    border-radius: 8px;
}
QLabel#helpKey {
    color: #3CC3C3;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 14px;
    background: transparent;
}
QLabel#helpText {
    color: #E1E8EB;
    font-size: 14px;
    background: transparent;
}
QLabel#helpHead {
    color: #8791A0;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 11px;
    letter-spacing: 2px;
    background: transparent;
}
QWidget#viewGrid { background-color: #2D373C; }
QLabel#viewPane {
    background-color: #05080A;
    color: #8791A0;
    font-size: 14px;
}
QLabel#paneTag {
    background-color: rgba(16, 20, 22, 215);
    color: #3CC3C3;
    border-radius: 4px;
    padding: 3px 8px;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
    letter-spacing: 1px;
}
QLabel#hintBar {
    background-color: rgba(16, 20, 22, 215);
    color: #E1E8EB;
    border-top: 2px solid #3CC3C3;
    padding: 6px 14px;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 15px;
}
"""


def _set_prop(widget, name, value):
    """Set a dynamic Qt property and re-apply the stylesheet if it changed."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _button(text, object_name="railBtn"):
    b = QPushButton(text)
    b.setObjectName(object_name)
    b.setFocusPolicy(Qt.NoFocus)          # keep key presses on the main window
    b.setCursor(Qt.PointingHandCursor)
    b.setProperty("active", "false")
    return b


class TrustBadge(HudPanel):
    """One answer to 'can I trust the overlay right now?', with the reason."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        self.title = QLabel("--")
        self.title.setObjectName("trustTitle")
        self.reason = QLabel("")
        self.reason.setObjectName("trustReason")
        self.reason.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.reason)
        self.setMinimumHeight(74)
        self._last = None

    def update_state(self, status, title, reason):
        key = (status, title, reason)
        if key == self._last:
            return
        self._last = key
        self.title.setText(title)
        self.reason.setText(reason)
        self.reason.setVisible(bool(reason))
        self.set_accent(status)
        _set_prop(self.title, "status", "muted" if status in ("stale", "default") else status)


class PedicleSelector(QWidget):
    """Level buttons, left / right, and all levels. Emits clicks; shows the current selection."""
    levelClicked = Signal(int)
    sideClicked = Signal(str)
    overviewClicked = Signal()

    def __init__(self, levels, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        t1 = QLabel("VERTEBRA")
        t1.setObjectName("railTitle")
        v.addWidget(t1)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.level_btns = {}
        for lv in levels:
            b = _button(f"L{lv}")
            b.clicked.connect(lambda _=False, lv=lv: self.levelClicked.emit(lv))
            row.addWidget(b)
            self.level_btns[lv] = b
        v.addLayout(row)

        t2 = QLabel("PEDICLE")
        t2.setObjectName("railTitle")
        v.addWidget(t2)
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.side_btns = {}
        for side, text in (("L", "Left"), ("R", "Right")):
            b = _button(text)
            b.clicked.connect(lambda _=False, s=side: self.sideClicked.emit(s))
            row2.addWidget(b)
            self.side_btns[side] = b
        v.addLayout(row2)

        self.all_btn = _button("All levels", "railLink")
        self.all_btn.clicked.connect(self.overviewClicked.emit)
        v.addWidget(self.all_btn)
        self._last = object()

    def set_state(self, level, side):
        if self._last == (level, side):
            return
        self._last = (level, side)
        for lv, b in self.level_btns.items():
            _set_prop(b, "active", "true" if lv == level else "false")
        for s, b in self.side_btns.items():
            _set_prop(b, "active", "true" if (s == side and level is not None) else "false")
        _set_prop(self.all_btn, "active", "true" if level is None else "false")


class ViewsMenu(QWidget):
    """One toggle per extra view. Emits the view's name when clicked."""
    toggled = Signal(str)

    def __init__(self, views, parent=None, title="VIEWS"):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("railTitle")
        v.addWidget(t)
        self.btns = {}
        for name, text in views:
            b = _button(text, "railLink")
            b.clicked.connect(lambda _=False, n=name: self.toggled.emit(n))
            v.addWidget(b)
            self.btns[name] = b

    def set_active(self, name, on):
        _set_prop(self.btns[name], "active", "true" if on else "false")


class TargetCard(HudPanel):
    """The selected pedicle and its expected error, in one glance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        t = QLabel("TARGET")
        t.setObjectName("railTitle")
        self.name = QLabel("--")
        self.name.setObjectName("cardName")
        self.value = QLabel("--")
        self.value.setObjectName("cardValue")
        self.caption = QLabel("")
        self.caption.setObjectName("cardCaption")
        self.caption.setWordWrap(True)
        for w in (t, self.name, self.value, self.caption):
            lay.addWidget(w)
        self._last = None

    def update_state(self, name, r95_mm, status, caption):
        key = (name, None if r95_mm is None else round(r95_mm, 1), status, caption)
        if key == self._last:
            return
        self._last = key
        self.name.setText(name)
        self.value.setText("--" if r95_mm is None else f"\u00b1{r95_mm:.1f} mm")
        _set_prop(self.value, "status", status)
        self.caption.setText(caption)
        self.set_accent({"good": "good", "warn": "warn", "bad": "bad"}.get(status, "stale"))


HELP_ROWS = [
    ("Choose a pedicle", [("W / S", "previous / next vertebra"),
                          ("A / D", "left / right pedicle"),
                          ("1 - 5", "jump to a vertebra"),
                          ("[ / ]", "walk the ten pedicles"),
                          ("T", "all levels")]),
    ("View", [("Z", "zoom view on / off"),
              ("+ / -", "zoom in / out (or mouse wheel)"),
              ("X", "axial view on / off"),
              ("L", "sagittal view on / off"),
              ("V", "axial / sagittal: beside the view / in the rail"),
              ("Y", "bone outline on / off"),
              ("E", "error circle on / off"),
              ("F", "fullscreen"),
              ("P", "save a photo")]),
    ("Check accuracy", [("Click", "on the real entry dot: error in mm"),
                        ("Esc", "clear the mark")]),
    ("Registration", [("C", "confirm"),
                      ("R", "reject and re-check")]),
    ("Other", [("M", "debug mode"),
               ("H or ?", "this help"),
               ("Q", "quit")]),
]


class HelpOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("helpOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        g = QGridLayout(self)
        g.setContentsMargins(26, 20, 30, 22)
        g.setHorizontalSpacing(18)
        g.setVerticalSpacing(7)
        row = 0
        for head, items in HELP_ROWS:
            h = QLabel(head.upper())
            h.setObjectName("helpHead")
            g.addWidget(h, row, 0, 1, 2)
            row += 1
            for k, text in items:
                a = QLabel(k)
                a.setObjectName("helpKey")
                a.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                b = QLabel(text)
                b.setObjectName("helpText")
                g.addWidget(a, row, 0)
                g.addWidget(b, row, 1)
                row += 1
        self.hide()


class Rail(QFrame):
    helpClicked = Signal()
    viewToggled = Signal(str)
    optionToggled = Signal(str)
    engineerClicked = Signal()

    def __init__(self, n_total, gate, levels, parent=None):
        super().__init__(parent)
        self.setObjectName("rail")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedWidth(RAIL_WIDTH)

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 12)
        v.setSpacing(12)

        self.state_pill = StatePill()
        v.addWidget(self.state_pill, 0, Qt.AlignLeft)
        self.trust = TrustBadge()
        v.addWidget(self.trust)
        self.selector = PedicleSelector(levels)
        v.addWidget(self.selector)
        self.card = TargetCard()                     # the selected pedicle and its expected error
        v.addWidget(self.card)
        # two short columns: the extra views, and the display options (their keys stay the same)
        self.views_menu = ViewsMenu([("zoom", "Zoom   Z"), ("axial", "Axial   X"), ("sagittal", "Sagittal   L")])
        self.views_menu.toggled.connect(self.viewToggled.emit)
        self.options_menu = ViewsMenu([("outline", "Outline   Y"), ("circle", "Error circle   E"),
                                       ("layout", "Side views   V")], title="OPTIONS")
        self.options_menu.toggled.connect(self.optionToggled.emit)
        menus = QHBoxLayout()
        menus.setSpacing(10)
        menus.addWidget(self.views_menu, 1)
        menus.addWidget(self.options_menu, 1)
        v.addLayout(menus)

        # engineer diagnostics, hidden until engineer mode is on
        self.engineer_box = QWidget()
        ev = QVBoxLayout(self.engineer_box)
        ev.setContentsMargins(0, 0, 0, 0)
        ev.setSpacing(10)
        t = QLabel("DEBUG")
        t.setObjectName("railTitle")
        ev.addWidget(t)
        self.status_panel = StatusPanel(n_total=n_total, gate=gate)
        self.pose_panel = PosePanel()
        ev.addWidget(self.status_panel)
        ev.addWidget(self.pose_panel)
        self.engineer_box.hide()
        v.addWidget(self.engineer_box)

        # axial and sagittal views: share the free space between the views list and the bottom links.
        # The window can also move them out to a column beside the main view (V).
        self.slice_host = QWidget()
        self.slice_layout = QVBoxLayout(self.slice_host)
        self.slice_layout.setContentsMargins(0, 0, 0, 0)
        self.slice_layout.setSpacing(6)
        self.axial_pane = ViewPane("AXIAL", self.slice_host)
        self.sagittal_pane = ViewPane("SAGITTAL", self.slice_host)
        for p in (self.axial_pane, self.sagittal_pane):
            p.setMinimumSize(110, 110)
            p.hide()
            self.slice_layout.addWidget(p, 1)
        self.slice_host.hide()
        v.addWidget(self.slice_host, 100)
        v.addStretch(1)

        bottom = QHBoxLayout()
        self.help_btn = _button("Help  H", "railLink")
        self.help_btn.clicked.connect(self.helpClicked.emit)
        self.eng_btn = _button("Debug mode  M", "railLink")
        self.eng_btn.clicked.connect(self.engineerClicked.emit)
        bottom.addWidget(self.eng_btn)
        bottom.addStretch(1)
        bottom.addWidget(self.help_btn)
        v.addLayout(bottom)

    def dock_slices(self, panes):
        """Put the slice panes (back) into the rail, in order."""
        for p in panes:
            self.slice_layout.addWidget(p, 1)

    def set_engineer(self, on):
        self.engineer_box.setVisible(on)
        _set_prop(self.eng_btn, "active", "true" if on else "false")

    def set_view(self, name, on):
        self.views_menu.set_active(name, on)

    def set_option(self, name, on):
        self.options_menu.set_active(name, on)

    def set_help(self, on):
        _set_prop(self.help_btn, "active", "true" if on else "false")