"""
Plain-language guidance for the HUD: the trust badge and the "move the camera"
hint. Pure functions of a FrameResult, no Qt, so they can be reused by other
front ends.
"""
from ar_theme import ENV_FULL_DIST_MM, ENV_FULL_ELEV, ENV_FULL_AZ


def zone_hint(dist_mm, elev_deg, az_deg):
    """One instruction that moves the camera back toward the full-accuracy zone, or None.
    Azimuth: 0 = caudal (patient's feet), 90 = patient's left. Elevation is above the base plane."""
    lo_d, hi_d = ENV_FULL_DIST_MM
    lo_e, hi_e = ENV_FULL_ELEV
    lo_a, hi_a = ENV_FULL_AZ
    cands = []                                   # (how far out, text); the worst violation wins
    if dist_mm < lo_d:
        cands.append(((lo_d - dist_mm) / lo_d, "Move the camera back"))
    elif dist_mm > hi_d:
        cands.append(((dist_mm - hi_d) / hi_d, "Move the camera closer"))
    if elev_deg < lo_e:
        cands.append(((lo_e - elev_deg) / 30.0, "Raise the camera"))
    elif elev_deg > hi_e:
        cands.append(((elev_deg - hi_e) / 30.0, "Lower the camera"))
    centre, half = (lo_a + hi_a) / 2.0, (hi_a - lo_a) / 2.0
    delta = ((az_deg - centre + 180.0) % 360.0) - 180.0
    excess = abs(delta) - half
    if excess > 0:
        text = ("Move the camera toward the patient's left" if delta < 0
                else "Move the camera toward the patient's feet")
        cands.append((excess / 90.0, text))
    return max(cands)[1] if cands else None


def trust_badge(r, is_v3, n_total):
    """(status, title, reason) for the single trust badge.
    status is one of 'good', 'warn', 'bad', 'stale' (neutral grey), 'default' (teal)."""
    if r.state == "SEARCHING":
        return "stale", "Searching", "Show the markers to the camera"
    if r.state == "STALE":
        return "bad", "Tracking lost", "Bring the markers back into view"
    if r.state == "VERIFYING":
        return "default", "Checking registration", "Does the outline sit on the bone?"
    # NAVIGATING
    markers = f"{len(r.visible_ids)} of {n_total} markers"
    if is_v3 and r.env_tier != "full":
        hint = zone_hint(r.distance_mm, r.elev_deg, r.az_deg)
        status = "bad" if r.env_tier == "outside" else "warn"
        return status, "Tracking limited", hint or markers
    return "good", "Tracking good", markers