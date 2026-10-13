"""
Expected overlay error: the number the UI shows to the surgeon.

Why not the raw Fitzpatrick TRE? It models random fiducial noise only. ATP-2
measured 1.4 to 1.6 mm RMS overlay error against 0.4 to 1.0 mm predicted, and
the gap is a fixed systematic offset that the prediction cannot see. Showing the
prediction alone would be overconfident, so we anchor to what was measured.

Model (per target, per frame)
    expected RMS  = sqrt( RMS_ref^2 + max(0, TRE^2 - TRE_ref^2) )
    95% radius    = RAYLEIGH_95 * expected RMS
RMS_ref is the measured overlay RMS at the ATP-2 conditions, TRE_ref the mean
predicted TRE at those same conditions. At validated conditions the display
equals what was measured; when conditions get worse (TRE grows) the error grows;
it never claims to be better than the measurement.

The 2D radial error is treated as Rayleigh distributed, i.e. isotropic Gaussian in
x and y. The Monte Carlo simulation does not assume this: it adds Gaussian noise to
the corners (its input) and reports the resulting RMS. The assumption is that the
overlay error across targets and views is roughly isotropic and zero-mean, which the
signed per-target offsets support (dx, dy, dz change sign between targets). It is a
statement about the population of targets and views, not a guarantee at one point,
since each target has its own fixed bias.
Checked against the ATP-2 data (err_mm column), Rayleigh(RMS) against observed:
    dome f50 n=64  median 1.19 vs 1.35   p90 2.17 vs 1.94   p95 2.48 vs 2.31
    bracket n=22   median 1.39 vs 1.42   p90 2.54 vs 2.47   p95 2.90 vs 2.49
The body of the distribution matches and the 95 % radius is slightly conservative.

Validated on a phantom only. The systematic part includes phantom seating and
ground-truth dot placement, so it is not a clinical accuracy claim.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from ar_theme import COL_GOOD, COL_WARN, COL_BAD

RAYLEIGH_95 = math.sqrt(-math.log(0.05))     # 1.731 : 95 % radius per unit RMS radial error

GOOD_MM = 3.0      # 95 % radius up to here: green (corresponds to predicted TRE of about 1.2 mm)
BAD_MM = 4.0       # beyond here: red     (predicted TRE of about 2 mm, the old red line)


@dataclass(frozen=True)
class Anchor:
    rms_mm: float      # measured overlay RMS at the ATP-2 conditions
    tre_mm: float      # mean predicted TRE at those conditions
    source: str


ANCHORS = {
    # data/atp_v2/atp2_dome_f50.csv: err_mm RMS, mean of pred_tre_mm. Dome camera focus is
    # f50 (the f35 run, RMS 1.56 mm, predates the focus fix). Bracket camera focus is f35.
    "dome": Anchor(1.43, 1.03, "ATP-2 dome, f50 focus: n=64, RMS 1.43 mm, mean predicted TRE 1.03 mm"),
    # data/atp/atp2_crisp.csv: err_mm RMS, mean of pred_tre_mm
    "bracket": Anchor(1.67, 1.02, "ATP-2 bracket, f35 crisp focus: n=22, RMS 1.67 mm, mean predicted TRE 1.02 mm"),
}


def expected_rms(tre, anchor):
    tre = np.asarray(tre, float)
    return np.sqrt(anchor.rms_mm ** 2 + np.maximum(0.0, tre ** 2 - anchor.tre_mm ** 2))


def error_radius95(tre, anchor):
    """95 % overlay error radius in mm (scalar or array). inf where TRE is not finite."""
    return RAYLEIGH_95 * expected_rms(tre, anchor)


def grade(r95, tier):
    """'good' | 'warn' | 'bad'. Outside the envelope is bad, the reduced tier is capped at warn."""
    if tier == "outside" or not np.isfinite(r95) or r95 > BAD_MM:
        return "bad"
    if tier == "reduced" or r95 > GOOD_MM:
        return "warn"
    return "good"


_COLOURS = {"good": COL_GOOD, "warn": COL_WARN, "bad": COL_BAD}


def grade_colour(status):
    return _COLOURS[status]


def draw_error_circle(frame, point_mm, r_mm, rvec, tvec, K, D, colour):
    """
    Circle of radius r_mm (true scale, in the plane facing the camera) around a 3D point.
    Dashed when large enough to read, solid when tiny. Dark underlay for contrast.
    """
    p = np.asarray(point_mm, np.float64).reshape(3)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))
    z = float((R @ p + np.asarray(tvec, np.float64).reshape(3))[2])
    if z <= 1e-6 or not np.isfinite(r_mm):
        return
    px_per_mm = 0.5 * (K[0, 0] + K[1, 1]) / z
    r = max(2, int(round(r_mm * px_per_mm)))
    c, _ = cv2.projectPoints(p.reshape(1, 1, 3), rvec, tvec, K, D)
    cx, cy = int(round(c[0, 0, 0])), int(round(c[0, 0, 1]))
    w = max(1, int(round(frame.shape[0] / 540)))

    def ring(col, thick):
        if r < 14:
            cv2.circle(frame, (cx, cy), r, col, thick, cv2.LINE_AA)
        else:
            for a in range(0, 360, 30):                      # 12 dashes, 15 degrees each
                cv2.ellipse(frame, (cx, cy), (r, r), 0, a, a + 15, col, thick, cv2.LINE_AA)

    ring((0, 0, 0), w + 2)
    ring(colour, w)
    return r