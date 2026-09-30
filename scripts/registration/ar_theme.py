"""
Visual theme for the live AR overlay.

Inspired by 7D FLASH and Medivis: deep black foundation, single teal
accent, flat panels with thin top-edge accent lines, compact text.
All colours BGR (OpenCV convention).
"""

from dataclasses import dataclass

# Foundation
BG_PANEL     = (22, 20, 16)      # RGB #10141A  near-black panel fill
BORDER       = (60, 55, 45)      # RGB #2D373C  subtle panel edge

# Text
TEXT_PRIMARY   = (235, 232, 225)   # RGB #E1E8EB
TEXT_SECONDARY = (150, 145, 135)   # RGB #8791A0
TEXT_MUTED     = (90, 85, 78)      # RGB #4E5560

# Accent
ACCENT         = (195, 195, 60)    # RGB #3CC3C3  teal (Medivis-style)
ACCENT_DIM     = (120, 120, 38)    # RGB #267878

# Status
COL_GOOD       = (140, 210, 100)   # RGB #64D28C  green
COL_WARN       = (80, 180, 235)    # RGB #EBB450  amber
COL_BAD        = (90, 90, 215)     # RGB #D75A5A  red
STATUS_STALE   = (100, 95, 85)     # RGB #555F64  grey

# TRE thresholds
TRE_GOOD_MM = 1.0
TRE_BAD_MM  = 2.0   # Gertzbein-Robbins safe zone

# ====================================================================
# Operational envelope (measured, ATP-3, V3 L-bracket)
# ====================================================================
# Distance is camera to phantom origin (base plate top centre), in mm.
# Azimuth: 0 = caudal (Face A), 90 = patient left (Face B).
#
# FULL:    both bracket faces visible, worst-target TRE p95 < 2 mm,
#          no visible tracking loss.
# REDUCED: tracking holds (dropouts < 300 ms) but only one face is
#          visible, so worst-target TRE is ~2.1 to 2.5 mm.
# Outside both: tracking unreliable or lost.
#
# Each range was measured with the other two parameters held near
# the centre of the full envelope (~45 cm, ~33 deg elev, ~45 deg az),
# so combined extremes (e.g. far AND steep) are not characterised.

ENV_FULL_DIST_MM   = (550.0, 1000.0)
ENV_FULL_ELEV      = (10.0, 35.0)
ENV_FULL_AZ        = (30.0, 55.0)

ENV_REDUCED_DIST_MM = (330.0, 1240.0)
ENV_REDUCED_ELEV    = (0.0, 52.0)
ENV_REDUCED_AZ      = (325.0, 120.0)   # wraps through 0

# Legacy names (kept so older imports still work)
ENVELOPE_ELEV = ENV_FULL_ELEV
ENVELOPE_AZ   = ENV_FULL_AZ


@dataclass(frozen=True)
class TypeScale:
    small:   float = 0.022    # legends, shortcuts
    medium:  float = 0.028    # HUD metrics
    large:   float = 0.036    # state banner

TYPE = TypeScale()

@dataclass(frozen=True)
class Layout:
    panel_alpha:  float = 0.82
    panel_pad:    int   = 12
    accent_h:     int   = 2

LAYOUT = Layout()


def _in_range(v, lo, hi):
    return lo <= v <= hi


def _in_az_range(az, lo, hi):
    """Azimuth range that may wrap through 0/360."""
    az = az % 360.0
    if lo <= hi:
        return lo <= az <= hi
    return az >= lo or az <= hi


def envelope_tier(elev, az, dist_mm=None):
    """
    Returns "full", "reduced" or "outside".
    dist_mm is optional; if None, distance is not checked.
    """
    def dist_ok(rng):
        return dist_mm is None or _in_range(dist_mm, *rng)

    if (dist_ok(ENV_FULL_DIST_MM)
            and _in_range(elev, *ENV_FULL_ELEV)
            and _in_az_range(az, *ENV_FULL_AZ)):
        return "full"
    if (dist_ok(ENV_REDUCED_DIST_MM)
            and _in_range(elev, *ENV_REDUCED_ELEV)
            and _in_az_range(az, *ENV_REDUCED_AZ)):
        return "reduced"
    return "outside"


def is_in_envelope(elev, az, dist_mm=None):
    """2-tier view: True only inside the full-accuracy envelope."""
    return envelope_tier(elev, az, dist_mm) == "full"


def tre_colour(tre_mm, in_envelope):
    """
    in_envelope may be a bool (2-tier) or a tier string (3-tier).
    In the reduced tier, targets are capped at amber.
    """
    if in_envelope is False or in_envelope == "outside":
        return COL_BAD
    if tre_mm >= TRE_BAD_MM:
        return COL_BAD
    if in_envelope == "reduced":
        return COL_WARN
    if tre_mm < TRE_GOOD_MM:
        return COL_GOOD
    return COL_WARN