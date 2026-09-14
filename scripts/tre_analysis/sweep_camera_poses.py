"""
Sweep Monte Carlo TRE across a grid of camera poses.

Runs monte_carlo_tre for each (elevation, azimuth) combination, tabulates
visibility and mean TRE per variant per pose. Reveals which variants work
at which viewing angles.

Usage:
  python scripts/tre_analysis/sweep_camera_poses.py \
      --calibration data/calibration/s22_ultra/camera_calibration.npz \
      --marker-size-mm 20 --pixel-noise-sigma 0.121
"""

import argparse
import csv
from pathlib import Path
import numpy as np

from design_variants import (
    PLACEHOLDER_PEDICLE_TARGETS, PLACEHOLDER_PEDICLE_LABELS,
    load_pedicles_from_json,
)
from design_variants_oriented import VARIANTS_ORIENTED
from monte_carlo_tre import make_camera_pose, run_monte_carlo


DEFAULT_ELEVATIONS = [15, 30, 45, 60, 90]              # angle above horizontal
DEFAULT_AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]  # rotation around vertical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--pixel-noise-sigma", type=float, default=0.25)
    parser.add_argument("--n-trials", type=int, default=2000,
                        help="fewer trials per pose since sweeping many")
    parser.add_argument("--distance-mm", type=float, default=350.0)
    parser.add_argument("--elevations", type=float, nargs="+",
                        default=DEFAULT_ELEVATIONS)
    parser.add_argument("--azimuths", type=float, nargs="+",
                        default=DEFAULT_AZIMUTHS)
    parser.add_argument("--pedicles", type=str, default=None)
    parser.add_argument("--outfile", type=str,
                        default="results/tre_analysis/pose_sweep.csv")
    args = parser.parse_args()

    data = np.load(args.calibration)
    K, dist = data["camera_matrix"], data["dist_coeffs"]

    if args.pedicles:
        targets, _ = load_pedicles_from_json(args.pedicles)
    else:
        targets = PLACEHOLDER_PEDICLE_TARGETS

    print(f"Sweeping {len(args.elevations)} elevations x {len(args.azimuths)} azimuths "
          f"= {len(args.elevations)*len(args.azimuths)} poses per variant")
    print(f"Trials per pose: {args.n_trials}\n")

    rows = []
    print(f"{'variant':<15}{'el':>4}{'az':>5}{'vis':>6}{'mean_mm':>10}{'max_mm':>10}")
    print("-" * 50)

    for el in args.elevations:
        for az in args.azimuths:
            pose = make_camera_pose(args.distance_mm, el, az)
            for name, oriented in VARIANTS_ORIENTED.items():
                tre, n_ok, n_vis, n_tot = run_monte_carlo(
                    oriented, targets, K, dist,
                    args.marker_size_mm, args.pixel_noise_sigma,
                    args.n_trials, pose,
                )
                if tre is None:
                    print(f"{name:<15}{el:>4}{az:>5}{n_vis:>3}/{n_tot}    FAIL       -")
                    rows.append({"variant": name, "elevation": el, "azimuth": az,
                                 "n_visible": n_vis, "n_total": n_tot,
                                 "mean_mm": "FAIL", "max_mm": "FAIL"})
                else:
                    print(f"{name:<15}{el:>4}{az:>5}{n_vis:>3}/{n_tot} "
                          f"{tre.mean():>9.4f}{tre.max():>10.4f}")
                    rows.append({"variant": name, "elevation": el, "azimuth": az,
                                 "n_visible": n_vis, "n_total": n_tot,
                                 "mean_mm": f"{tre.mean():.4f}",
                                 "max_mm": f"{tre.max():.4f}"})

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "elevation", "azimuth",
            "n_visible", "n_total", "mean_mm", "max_mm",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Quick summary
    print("\nSummary: how often each variant succeeds across all poses")
    for name in VARIANTS_ORIENTED.keys():
        var_rows = [r for r in rows if r["variant"] == name]
        n_success = sum(1 for r in var_rows if r["mean_mm"] != "FAIL")
        n_total = len(var_rows)
        print(f"  {name:<15} {n_success}/{n_total} poses succeeded")


if __name__ == "__main__":
    main()