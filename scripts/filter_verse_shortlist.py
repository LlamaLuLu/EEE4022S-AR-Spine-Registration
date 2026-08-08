"""
Stage 2 filter: rank VerSe cases by centroids and voxel spacing.

For each shortlisted case:
  - Read centroids JSON, check L1-L5 (labels 20-24) presence
  - Flag L6 (25) and T13 (28)
  - Read segmentation NIfTI header for voxel spacing and image dimensions
  - Print ranked table
"""

import json
from pathlib import Path
import nibabel as nib

# --- Configuration ---
DATA_ROOT = Path("data/verse20/derivatives")
SHORTLIST = [
    "sub-verse504",
    "sub-verse534",
    "sub-verse535",
    "sub-verse646",
    "sub-verse808",
    "sub-verse835",
]

# VerSe label conventions (from OSF wiki)
L1_TO_L5 = {20, 21, 22, 23, 24}
L6_LABEL = 25
T13_LABEL = 28


def find_file(case_dir: Path, suffix: str) -> Path | None:
    """Find a file in the case directory ending with the given suffix."""
    matches = list(case_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def analyse_case(case_id: str) -> dict:
    """Extract centroids and voxel info for one case."""
    case_dir = DATA_ROOT / case_id
    result = {"case_id": case_id}

    # --- Centroids ---
    ctd_file = find_file(case_dir, "_ctd.json")
    if ctd_file is None:
        result["error"] = "no centroids JSON found"
        return result

    with open(ctd_file) as f:
        centroids = json.load(f)

    # Centroids JSON is a list of dicts. First entry is orientation metadata,
    # rest have a "label" field per vertebra.
    labels_present = {entry["label"] for entry in centroids if "label" in entry}

    result["labels"] = sorted(labels_present)
    result["l1_l5_present"] = L1_TO_L5.issubset(labels_present)
    result["l6_flag"] = L6_LABEL in labels_present
    result["t13_flag"] = T13_LABEL in labels_present

    # --- Segmentation NIfTI header ---
    seg_file = find_file(case_dir, "_msk.nii.gz")
    if seg_file is None:
        result["error"] = "no segmentation NIfTI found"
        return result

    img = nib.load(seg_file)
    # zooms returns voxel spacing in mm for each axis
    spacing = img.header.get_zooms()
    shape = img.shape

    result["voxel_spacing_mm"] = tuple(round(float(s), 3) for s in spacing)
    result["dims_voxels"] = shape
    result["physical_extent_mm"] = tuple(
        round(float(s) * d, 1) for s, d in zip(spacing, shape)
    )

    return result


def rank_key(r: dict) -> tuple:
    """Ranking: L1-L5 present first, then smallest max voxel spacing."""
    if "error" in r or not r.get("l1_l5_present"):
        return (1, 0)  # sink to bottom
    max_spacing = max(r["voxel_spacing_mm"])
    return (0, max_spacing)


def main():
    results = [analyse_case(cid) for cid in SHORTLIST]
    results.sort(key=rank_key)

    print(f"\n{'Case':<16} {'L1-L5':<7} {'L6':<4} {'T13':<5} "
          f"{'Voxel (mm)':<22} {'Dims (voxels)':<22} {'Extent (mm)':<25}")
    print("-" * 105)

    for r in results:
        if "error" in r:
            print(f"{r['case_id']:<16} ERROR: {r['error']}")
            continue

        vox = f"{r['voxel_spacing_mm']}"
        dims = f"{r['dims_voxels']}"
        extent = f"{r['physical_extent_mm']}"

        print(f"{r['case_id']:<16} "
              f"{'YES' if r['l1_l5_present'] else 'NO':<7} "
              f"{'YES' if r['l6_flag'] else 'no':<4} "
              f"{'YES' if r['t13_flag'] else 'no':<5} "
              f"{vox:<22} {dims:<22} {extent:<25}")

    print()


if __name__ == "__main__":
    main()