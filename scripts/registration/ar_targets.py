"""
Pedicle target loading and trajectory pairing.

Loads isthmus and entry targets from separate Slicer mrk.json files,
merges them into a single array, and builds index pairs for drawing
entry-to-isthmus trajectory lines.

Label convention:
    Isthmus: L3_R, L3_L    (level_side)
    Entry:   L3_E_R, L3_E_L (level_E_side)
"""

import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRE_DIR = _REPO_ROOT / "scripts" / "tre_analysis"
if str(_TRE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRE_DIR))

from design_variants import (
    load_pedicles_from_json,
    PLACEHOLDER_PEDICLE_TARGETS,
    PLACEHOLDER_PEDICLE_LABELS,
)


def parse_pedicle_label(label):
    """
    Extract (level, side, is_entry) from a pedicle label.

    'L3_R'   -> ('L3', 'R', False)
    'L3_E_R' -> ('L3', 'R', True)
    """
    parts = label.split("_")
    level = parts[0]
    side = parts[-1]
    is_entry = "E" in parts[1:-1] or (len(parts) == 3 and parts[1] == "E")
    return level, side, is_entry


def build_trajectory_pairs(labels):
    """
    Returns list of (isthmus_index, entry_index) into the merged
    target array, matched by (level, side).
    """
    isthmus_idx, entry_idx = {}, {}
    for i, lbl in enumerate(labels):
        level, side, is_entry = parse_pedicle_label(lbl)
        key = (level, side)
        if is_entry:
            entry_idx[key] = i
        else:
            isthmus_idx[key] = i

    return [(isthmus_idx[k], entry_idx[k])
            for k in isthmus_idx if k in entry_idx]


def load_targets(isthmus_path=None, entry_path=None):
    """
    Load and merge pedicle targets from one or two Slicer JSON files.

    Returns (target_pts: Nx3, target_labels: list[str],
             trajectory_pairs: list[(int, int)]).
    """
    isthmus_pts, isthmus_labels = None, None
    entry_pts, entry_labels = None, None

    if isthmus_path and Path(isthmus_path).exists():
        isthmus_pts, isthmus_labels = load_pedicles_from_json(isthmus_path)

    if entry_path and Path(entry_path).exists():
        entry_pts, entry_labels = load_pedicles_from_json(entry_path)

    if isthmus_pts is not None and entry_pts is not None:
        target_pts = np.vstack([isthmus_pts, entry_pts])
        target_labels = list(isthmus_labels) + list(entry_labels)
    elif isthmus_pts is not None:
        target_pts = isthmus_pts
        target_labels = list(isthmus_labels)
    elif entry_pts is not None:
        target_pts = entry_pts
        target_labels = list(entry_labels)
    else:
        target_pts = PLACEHOLDER_PEDICLE_TARGETS
        target_labels = list(PLACEHOLDER_PEDICLE_LABELS)

    pairs = build_trajectory_pairs(target_labels)
    return target_pts, target_labels, pairs