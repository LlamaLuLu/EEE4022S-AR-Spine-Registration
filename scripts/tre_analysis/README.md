# TRE analysis pipeline

Simulation and prediction pipeline for evaluating fiducial marker board
layouts before physical construction. Predicts Target Registration Error
(TRE) at each pedicle target for a set of candidate marker layouts, so
only the winning layout gets 3D printed.

## Motivation

Sachin Daya's 2025 cranial pipeline used arbitrarily-placed markers on
a flat base plate without formally engaging with the FRE/TRE distinction.
This project designs the marker layout using Fitzpatrick's prospective
TRE prediction (Fitzpatrick 1998), then validates the prediction with
Monte Carlo simulation of the actual PnP pipeline. The winning layout
minimises predicted TRE at the 10 lumbar pedicle targets.

## Contents

```
scripts/tre_analysis/
├── fitzpatrick_tre.py     Closed-form TRE predictor (Fitzpatrick 1998 eq 46)
├── design_variants.py     Marker layouts for V0-V3, pedicle target loader
├── measure_fle.py         Empirical fiducial localisation error measurement
├── evaluate_designs.py    Runs Fitzpatrick across variants, produces table
└── monte_carlo_tre.py     Simulates real PnP with noisy corners, empirical TRE
```

Fitzpatrick and design_variants are library modules imported by the
harnesses. measure_fle, evaluate_designs, and monte_carlo_tre are
runnable scripts.

## Design variants

Four hand-designed layouts, each isolating one variable:

- V0 baseline: 4 coplanar markers on the base plate around the phantom.
  Replicates the Sachin approach constrained to the real 60x170mm base
  plate footprint. Never physically built.
- V1 primary: L-bracket with 90° between faces, 6 markers (3 per face)
  spread to face corners. The proposed design under Fitzpatrick
  optimisation.
- V2 clustered: same L-bracket geometry as V1 but 6 markers clustered
  near face centres. Isolates the spread variable to demonstrate that
  bracket shape alone is not sufficient.
- V3 8-marker: same L-bracket as V1 but 8 markers (4 per face) at face
  corners. Isolates the count variable to quantify diminishing returns.

Only the winning variant is 3D printed and physically evaluated.
The others exist as numpy arrays for simulation only.

## Coordinate frame

All positions are in the phantom frame:
- origin: centre of base plate top surface
- +X: patient left (lateral)
- +Y: cranial (along spine axis)
- +Z: posterior-up (vertical, away from base plate)
- units: mm

Pedicle targets from Slicer's mrk.json export are in LPS by default. If
the JSON was annotated on the transformed STL (post-Blender), coordinates
should already be in the phantom frame. If it contains original CT
coordinates, transform before loading.

## Pipeline flow

```
camera_calibration.npz  →  measure_fle.py       →  FLE (mm)  ─┐
pedicle_targets.json    →  design_variants.py   →  targets   ─┤
                                                              ├→ evaluate_designs.py → predictions_table.csv → design decision
                                      fitzpatrick_tre.py     ─┘

                                                              ├→ monte_carlo_tre.py → monte_carlo_table.csv → validation
camera_calibration.npz (K matrix) ────────────────────────────┘
```

evaluate_designs.py is the primary path to a decision. monte_carlo_tre.py
validates that the Fitzpatrick ranking survives real PnP behaviour.

## Usage

Measure FLE first (once per camera setup):
```bash
python scripts/tre_analysis/measure_fle.py \
    --imagedir data/fle_measurement/s22_ultra/ \
    --calibration data/calibration/s22_ultra/camera_calibration.npz \
    --marker-size-mm 20 \
    --dictionary 6X6_50
```

Predict TRE across all variants:
```bash
python scripts/tre_analysis/evaluate_designs.py \
    --fle 0.35 \
    --pedicles annotations/pedicle_targets_phantom_frame.json
```

Validate with Monte Carlo:
```bash
python scripts/tre_analysis/monte_carlo_tre.py \
    --calibration data/calibration/s22_ultra/camera_calibration.npz \
    --marker-size-mm 20 \
    --pixel-noise-sigma 0.3 \
    --n-trials 10000 \
    --pedicles annotations/pedicle_targets_phantom_frame.json
```

All three write CSVs to results/tre_analysis/ by default.

## Assumptions and caveats

Fitzpatrick 1998 assumes:
- Independent isotropic 3D Gaussian noise per fiducial
- Rigid least-squares (Procrustes) registration
- Discrete point fiducials

This pipeline treats each ArUco marker's centroid as one effective
Fitzpatrick fiducial, with FLE measured empirically as the RMS 3D
deviation of the centroid across static-board frames. This sidesteps
the corner-correlation problem within a single marker.

The Fitzpatrick prediction is a first-order approximation and does not
capture PnP-specific failure modes (notably the planar-flip ambiguity
on near-coplanar boards). Monte Carlo simulation uses the actual
cv2.solvePnP call and captures these effects. Where the two predictions
disagree, Monte Carlo takes precedence for the design decision.

Monte Carlo currently assumes all markers have their face parallel to
the phantom XY plane. This is correct for V0 (base plate markers) but a
simplification for V1-V3 (L-bracket markers with two different face
orientations). Fine for validation of Fitzpatrick ranking, would need
per-marker orientation matrices for absolute magnitude claims.

FLE is measured at one working distance under one lighting condition.
Real operating conditions will vary. Chapter 3 methodology should note
this and optionally include FLE at 25cm and 50cm for range reporting.

## Outputs

```
results/tre_analysis/
├── fle_measurement.json       Per-marker and aggregate FLE
├── predictions_table.csv      Fitzpatrick TRE per variant per target
└── monte_carlo_table.csv      Monte Carlo TRE per variant per target
```

## Dependencies

Python 3.10+, numpy, opencv-python (>= 5.0.0). No other external
libraries required for the analysis itself.

## References

- Fitzpatrick, West, Maurer (1998). "Predicting error in rigid-body
  point-based registration." IEEE TMI 17(5).
- Bao et al. (2014). Fiducial marker configuration optimisation.
- Kam et al. (2020). "Improved Pose Estimation of ArUco Tags Using a
  Novel 3D Placement Strategy." Sensors 20:4825.
- Ganeshan et al. (2025). Monocular NIR optical tracking with
  retroreflective fiducial markers. PMC12845559.