"""
Shared look for every report figure (biomedical palette: blues, teals, greens, greys, black, white).

Import with:   import fig_style as FS      then use FS.NAVY, FS.TEAL, ... and FS.apply().

Rules the palette follows
  * Dark vs light of the SAME hue encodes a state (e.g. calibrated vs drifted focus).
  * Hue encodes the prototype (blue = P1 bracket, teal = P2 dome).
  * Tracking classes always use the same three tones (FULL, PART, NONE).
  * Black dashed line = the 2 mm criterion, everywhere.
"""
import matplotlib as mpl

# --- core hues --------------------------------------------------------------------------
NAVY = "#0B3C5D"      # darkest blue: titles, emphasis, P1 dark
BLUE = "#1B6CA8"      # P1 (bracket), calibrated / main series
SKY = "#8DBBDD"       # P1 light (drifted focus), random-noise share
TEAL = "#0E8F7E"      # P2 (dome), main series
MINT = "#92D1C4"      # P2 light (f35)
GREEN = "#2E9E6B"     # good / full accuracy
SLATE = "#4A5560"     # measured / neutral dark
GREY = "#8A949E"      # reference lines, text on figures
LIGHT = "#D9DEE3"     # panels, "none" class fill
PALE = "#EEF1F4"      # very light background wash
BLACK = "#000000"
WHITE = "#FFFFFF"

# --- tracking classes (same everywhere) --------------------------------------------------
FULL = "#0E8F7E"      # full accuracy / reliable tracking
PART = "#9ED8CB"      # tracking holds / degraded
NONE = "#D9DEE3"      # no tracking (also drawn with hatching where it is a region)

# --- prototype and state colours ---------------------------------------------------------
P1_CAL, P1_DRIFT = BLUE, SKY
P2_F50, P2_F35 = TEAL, MINT

W = 6.3               # text width in inches


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": SLATE, "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": SLATE, "ytick.color": SLATE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "grid.color": LIGHT, "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
