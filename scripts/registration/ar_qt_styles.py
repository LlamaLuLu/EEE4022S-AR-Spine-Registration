"""
Qt Style Sheets for the AR overlay HUD.

Colours mirror ar_theme.py. Change the palette there and regenerate
the RGB strings below to keep everything in sync.

Fonts use a system-appropriate cascade: SF Pro / SF Mono on macOS,
Segoe UI / Consolas on Windows, Ubuntu / DejaVu on Linux.
"""

# RGB strings mirror ar_theme BGR values reversed:
#   BG_PANEL     = BGR (22, 20, 16)   -> RGB (16, 20, 22)
#   ACCENT       = BGR (195, 195, 60) -> RGB (60, 195, 195)   #3CC3C3
#   TEXT_PRIMARY = BGR (235, 232, 225)-> RGB (225, 232, 235)  #E1E8EB
#   TEXT_SECONDARY = BGR (150, 145, 135) -> RGB (135, 145, 150)  #8791A0
#   COL_GOOD  = BGR (140, 210, 100) -> RGB (100, 210, 140)   #64D28C
#   COL_WARN  = BGR (80, 180, 235)  -> RGB (235, 180, 80)    #EBB450
#   COL_BAD   = BGR (90, 90, 215)   -> RGB (215, 90, 90)     #D75A5A

QSS = """
/* ================= Base panel ================= */

QFrame#hudPanel {
    background-color: rgba(16, 20, 22, 220);
    border-top: 2px solid #3CC3C3;
    border-bottom: 1px solid rgba(60, 55, 45, 255);
    border-radius: 6px;
}
QFrame#hudPanel[accent="good"] { border-top: 2px solid #64D28C; }
QFrame#hudPanel[accent="warn"] { border-top: 2px solid #EBB450; }
QFrame#hudPanel[accent="bad"]  { border-top: 2px solid #D75A5A; }
QFrame#hudPanel[accent="stale"] { border-top: 2px solid #555F64; }

/* ================= Labels and values ================= */

QLabel#hudLabel {
    color: #8791A0;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
    letter-spacing: 1px;
    background: transparent;
}
QLabel#hudValue {
    color: #E1E8EB;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 20px;
    font-weight: 600;
    background: transparent;
}
QLabel#hudValue[status="good"] { color: #64D28C; }
QLabel#hudValue[status="warn"] { color: #EBB450; }
QLabel#hudValue[status="bad"]  { color: #D75A5A; }
QLabel#hudValue[status="muted"] { color: #4E5560; }

/* ================= State pill ================= */

QFrame#statePill {
    background-color: rgba(16, 20, 22, 235);
    border-radius: 20px;
    border-top: 2px solid #3CC3C3;
}
QFrame#statePill[state="searching"] { border-top: 2px solid #555F64; }
QFrame#statePill[state="verifying"] { border-top: 2px solid #3CC3C3; }
QFrame#statePill[state="navigating"] { border-top: 2px solid #64D28C; }
QFrame#statePill[state="stale"] { border-top: 2px solid #D75A5A; }

QLabel#statePillText {
    color: #E1E8EB;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2px;
    background: transparent;
}

/* ================= Shortcuts panel ================= */

QFrame#shortcutsPanel {
    background-color: rgba(16, 20, 22, 240);
    border-top: 2px solid #267878;
    border-radius: 6px;
}
QLabel#shortcutKey {
    color: #3CC3C3;
    font-family: "Menlo", "Consolas", monospace;
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}
QLabel#shortcutName {
    color: #E1E8EB;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
    background: transparent;
}

/* ================= Legend row ================= */

QLabel#legendLabel {
    color: #8791A0;
    font-family: "Menlo", "Consolas", monospace;
    font-size: 11px;
    background: transparent;
}
"""