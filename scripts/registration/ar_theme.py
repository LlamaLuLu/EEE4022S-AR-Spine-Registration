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
TRE_BAD_MM  = 2.0

# Operational envelope
ENVELOPE_ELEV = (30.0, 45.0)
ENVELOPE_AZ   = (30.0, 60.0)

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

def tre_colour(tre_mm, in_envelope):
    if not in_envelope:
        return COL_BAD
    if tre_mm < TRE_GOOD_MM:
        return COL_GOOD
    if tre_mm < TRE_BAD_MM:
        return COL_WARN
    return COL_BAD

def is_in_envelope(elev, az):
    return (ENVELOPE_ELEV[0] <= elev <= ENVELOPE_ELEV[1] and
            ENVELOPE_AZ[0] <= az <= ENVELOPE_AZ[1])