"""
Run the Fitzpatrick TRE predictor across all design variants and produce
a comparison table.

Outputs:
  - stdout: per-variant per-target TRE, plus summary statistics
  - results/tre_analysis/predictions_table.csv: same data for the report

Usage:
  python evaluate_designs.py                       # uses placeholder FLE
  python evaluate_designs.py --fle 0.42            # uses measured FLE
  python evaluate_designs.py --pedicles path.json  # uses re-annotated targets
"""

import argparse
import csv
from pathlib import Path
import numpy as np

from fitzpatrick_tre import predict_tre
from design_variants import (
    VARIANTS,
    PLACEHOLDER_PEDICLE_TARGETS,
    PLACEHOLDER_PEDICLE_LABELS,
    load_pedicles_from_json,
)


def evaluate_all(variants: dict,
                 targets: np.ndarray,
                 fle_mm: float) -> dict:
    """Return {variant_name: (M,) array of TRE per target in mm}."""
    return {
        name: predict_tre(markers, targets, fle_mm)
        for name, markers in variants.items()
    }


def summary_stats(tre_per_target: np.ndarray) -> dict:
    """Mean, max, and per-target vector for one variant's TRE."""
    return {
        "mean_mm": float(tre_per_target.mean()),
        "max_mm":  float(tre_per_target.max()),
        "min_mm":  float(tre_per_target.min()),
        "std_mm":  float(tre_per_target.std()),
    }


def print_table(results: dict,
                target_labels: list,
                summaries: dict) -> None:
    """Pretty-print per-variant per-target TRE plus summary statistics."""
    variants = list(results.keys())

    # Header: target labels across the top
    hdr = f"{'Variant':<15}" + "".join(f"{lbl:>8}" for lbl in target_labels) \
                             + f"{'mean':>8}{'max':>8}"
    print(hdr)
    print("-" * len(hdr))

    for v in variants:
        row = f"{v:<15}" + "".join(f"{tre:>8.3f}" for tre in results[v])
        row += f"{summaries[v]['mean_mm']:>8.3f}{summaries[v]['max_mm']:>8.3f}"
        print(row)


def save_csv(results: dict,
             target_labels: list,
             summaries: dict,
             out_path: Path) -> None:
    """Write a CSV: one row per variant, columns are targets + summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + list(target_labels)
                        + ["mean_mm", "max_mm", "min_mm", "std_mm"])
        for name, tre in results.items():
            s = summaries[name]
            writer.writerow([name] + [f"{v:.4f}" for v in tre]
                            + [f"{s[k]:.4f}" for k in
                               ("mean_mm", "max_mm", "min_mm", "std_mm")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fle", type=float, default=0.5,
                        help="FLE in mm (default: 0.5 placeholder)")
    parser.add_argument("--pedicles", type=str, default=None,
                        help="path to re-annotated pedicle targets JSON")
    parser.add_argument("--outdir", type=str, default="results/tre_analysis",
                        help="output directory for CSV")
    args = parser.parse_args()

    # Load targets: real ones if a JSON path was provided, otherwise placeholders.
    if args.pedicles:
        targets, labels = load_pedicles_from_json(args.pedicles)
        print(f"Loaded {len(targets)} pedicle targets from {args.pedicles}")
    else:
        targets = PLACEHOLDER_PEDICLE_TARGETS
        labels = PLACEHOLDER_PEDICLE_LABELS
        print(f"Using PLACEHOLDER pedicle targets ({len(targets)} points)")

    print(f"FLE = {args.fle} mm\n")

    # Run predictor on each variant.
    results = evaluate_all(VARIANTS, targets, args.fle)
    summaries = {name: summary_stats(tre) for name, tre in results.items()}

    print_table(results, labels, summaries)

    # Overall ranking by mean TRE.
    ranked = sorted(summaries.items(), key=lambda kv: kv[1]["mean_mm"])
    print("\nRanking by mean TRE (best first):")
    for name, s in ranked:
        print(f"  {name:<15} mean={s['mean_mm']:.3f} mm   max={s['max_mm']:.3f} mm")

    # Save CSV for the report.
    out_path = Path(args.outdir) / "predictions_table.csv"
    save_csv(results, labels, summaries, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()