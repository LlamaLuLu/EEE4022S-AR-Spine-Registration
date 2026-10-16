"""
Scene description: one plain, JSON-safe snapshot of everything a front end needs to
show a frame. No Qt, no drawing. The desktop window reads it, and the phone and glasses
builds can read the same thing, so they only have to reimplement the renderer.

    scene = build_scene(result, selection, target_pts, target_labels, anchor, is_v3, n_total)
    text  = scene_to_json(scene)            # save it, stream it, replay it, compare it
    scene = scene_from_json(text)

Units: mm and degrees. Target positions are in the phantom frame (+X patient left,
+Y posterior, +Z cranial). The pose maps phantom to camera: X_cam = R(rvec) X + tvec.
Anything that is not finite becomes null, so the output is strict JSON.
"""
import json
import math

import numpy as np

from ar_error import error_radius95, grade
from ar_guidance import trust_badge
from ar_selection import parse_label

SCENE_VERSION = 1


def _num(x, nd=3):
    """Finite float rounded, else None (JSON has no NaN or inf)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _vec(v, nd=5):
    return None if v is None else [_num(a, nd) for a in np.asarray(v, float).ravel()]


def error_summary(r, selection, anchor):
    """(target index, 95 % error mm, grade). One pedicle selected: its entry point.
    Overview: the worst entry point. (None, None, "muted") when there is nothing to show."""
    if r.tre_values is None or r.state == "SEARCHING":
        return None, None, "muted"
    r95 = error_radius95(np.asarray(r.tre_values), anchor)
    if selection.is_overview:
        entries = np.asarray(selection.all_entry_idx, dtype=int)
        j = int(entries[np.argmax(r95[entries])])
    else:
        j = selection.entry_idx
    if not np.isfinite(r95[j]):
        return None, None, "muted"
    return j, float(r95[j]), grade(r95[j], r.env_tier)


def build_scene(r, selection, target_pts, target_labels, anchor, is_v3, n_total, jitter=None, board=None):
    """r: a FrameResult. jitter: (total_mm, depth_mm, n_poses) or None."""
    status, title, reason = trust_badge(r, is_v3, n_total)
    tre = None if r.tre_values is None else np.asarray(r.tre_values, float)
    r95 = None if tre is None else error_radius95(tre, anchor)

    targets = []
    for i, (p, lbl) in enumerate(zip(target_pts, target_labels)):
        parsed = parse_label(lbl)
        level, side, is_entry = parsed if parsed else (None, None, None)
        targets.append({
            "i": i, "label": str(lbl), "level": level, "side": side,
            "kind": None if is_entry is None else ("entry" if is_entry else "isthmus"),
            "xyz": _vec(p, 3),
            "tre_mm": None if tre is None else _num(tre[i]),
            "r95_mm": None if r95 is None else _num(r95[i]),
        })

    j, e95, grd = error_summary(r, selection, anchor)
    error = None if j is None else {
        "target": j, "label": str(target_labels[j]), "r95_mm": _num(e95), "grade": grd}

    return {
        "version": SCENE_VERSION,
        "t": _num(r.now, 3),
        "state": r.state,
        "verify_remaining_s": _num(r.verify_remaining, 1),
        "trust": {"status": status, "title": title, "reason": reason},
        "pose": None if r.rvec is None else {"rvec": _vec(r.rvec), "tvec": _vec(r.tvec, 3)},
        "quality": {
            "board": board,
            "pnp_ok": bool(r.pnp_ok), "n_markers": int(r.n_markers),
            "n_total": int(n_total),
            "visible_ids": [int(i) for i in r.visible_ids],
            "rms_px": _num(r.reproj_rms, 2),
            "distance_mm": _num(r.distance_mm, 1), "elev_deg": _num(r.elev_deg, 1),
            "az_deg": _num(r.az_deg, 1), "env_tier": r.env_tier,
        },
        "selection": {
            "overview": bool(selection.is_overview), "name": selection.name,
            "level": selection.level, "side": selection.side,
            "entry": selection.entry_idx, "isthmus": selection.isthmus_idx,
        },
        "pedicles": [{"level": lv, "side": sd, "entry": v["entry"], "isthmus": v["isthmus"]}
                     for (lv, sd), v in sorted(selection.pedicles.items())],
        "targets": targets,
        "error": error,
        "jitter": None if jitter is None else {
            "total_mm": _num(jitter[0]), "depth_mm": _num(jitter[1]), "n": int(jitter[2])},
    }


def scene_to_json(scene, indent=None):
    return json.dumps(scene, indent=indent, allow_nan=False)


def scene_from_json(text):
    scene = json.loads(text)
    if scene.get("version") != SCENE_VERSION:
        raise ValueError(f"scene version {scene.get('version')!r}, expected {SCENE_VERSION}")
    return scene