# TRE simulation and marker layout analysis

Simulation and prediction pipeline for evaluating fiducial marker array
layouts before they are printed. Predicts Target Registration Error (TRE)
at each pedicle target for a set of candidate layouts (the four L-bracket
variants V0 to V3 and the dome designs D1 to D4), so only the winning
layout of each prototype is built.

All commands below are run from the **repo root**, not from
`scripts/tre_analysis/`.

## Motivation

Sachin Daya's 2025 cranial pipeline used arbitrarily-placed markers on
a flat base plate without formally engaging with the FRE/TRE distinction.
This project designs the marker layout using Fitzpatrick's prospective
TRE prediction (Fitzpatrick 1998), then validates the prediction with
Monte Carlo simulation of the actual PnP pipeline. The winning layout
minimises predicted TRE at the 10 lumbar pedicle targets. The simulator
(`mc_validation.py`) was then fitted to the P1 acceptance tests and used to
design the dome.

## Contents

```
scripts/tre_analysis/
├── fitzpatrick_tre.py           Closed-form TRE predictor (Fitzpatrick 1998 eq 46)
├── design_variants.py           Marker centroid layouts for V0 to V3, pedicle target loader
├── design_variants_oriented.py  Layouts with per-marker orientation, including the domes
├── evaluate_designs.py          Fitzpatrick TRE across variants, writes predictions_table.csv
├── monte_carlo_tre.py           Monte Carlo of the PnP pipeline, writes monte_carlo_table.csv
├── mc_validation.py             Simulator fitted to the ATP data: visibility model, overlay
│                                error, design, tilt, ATP-2 and ATP-4 comparisons
├── ring_sweep.py                Dome ring radius and top marker height sweeps
├── sweep_camera_poses.py        Monte Carlo over a grid of camera poses
├── measure_fle.py               FLE and pixel noise from static-board frames
├── generate_fle_board.py        Prints the 20 mm ArUco board used for the FLE capture
└── capture_fle_frames.py        Captures static frames from IP Webcam for the FLE measurement
```

`fitzpatrick_tre.py`, `design_variants.py` and `design_variants_oriented.py`
are library modules imported by the others. Everything else is runnable.

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

V3 was built as the first prototype (P1). The dome designs D1 to D4 are
defined in `design_variants_oriented.py` (`DOME_CONFIGS`); D1_final was
built as the second prototype (P2). The other variants exist for simulation
only.

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
camera_calibration.npz  →  measure_fle.py       →  FLE (mm) + pixel sigma  ─┐
pedicle_entry_targets.json →  design_variants.py   →  targets                 ─┤
                                                                            ├→ evaluate_designs.py → predictions_table.csv → design decision
                                       fitzpatrick_tre.py                  ─┘

                                                                            ├→ monte_carlo_tre.py → monte_carlo_table.csv → validation
camera_calibration.npz (K matrix)  ─────────────────────────────────────────┘
```

evaluate_designs.py is the primary path to a decision. monte_carlo_tre.py
validates that the Fitzpatrick ranking survives real PnP behaviour.

## Usage

All commands run from the repo root.

Measure FLE (once per camera setup):
```bash
python scripts/tre_analysis/measure_fle.py \
    --imagedir data/fle_measurement/s22_ultra/ \
    --calibration data/calibration/s22_ultra/camera_calibration.npz \
    --marker-size-mm 20 \
    --dictionary 6X6_50 \
    --outfile results/tre_analysis/fle_measurement.json
```

Predict TRE across all variants:
```bash
python scripts/tre_analysis/evaluate_designs.py \
    --fle 0.35 \
    --pedicles annotations/pedicle_entry_targets.json \
    --outdir results/tre_analysis
```

Validate with Monte Carlo:
```bash
python scripts/tre_analysis/monte_carlo_tre.py \
    --calibration data/calibration/s22_ultra/camera_calibration.npz \
    --marker-size-mm 20 \
    --pixel-noise-sigma 0.3 \
    --n-trials 10000 \
    --pedicles annotations/pedicle_entry_targets.json \
    --outfile results/tre_analysis/monte_carlo_table.csv
```

The defaults of 0.35 mm and 0.3 px are placeholders. Use the values from
`measure_fle.py` for absolute numbers; with the placeholders only the ranking
between variants is meaningful.

The fitted simulator is run through its subcommands (`fit`, `design`, `tilt`,
`atp2`, `atp4`, `fle`, `predict`); see the header of `mc_validation.py` for
the options.

## Quick usage via Makefile

From the repo root:

```bash
make help              # list all available targets
make measure-fle       # empirical FLE from captures
make evaluate          # Fitzpatrick predictions
make monte-carlo       # PnP validation
make analyse           # both analyses back to back
make test              # unit tests
```

Parameters such as `FLE`, `PIXEL_NOISE`, `N_TRIALS`, `DISTANCE`, `ELEVATION`
and `AZIMUTH` can be overridden on the command line, for example
`make evaluate FLE=0.42`.

## Path handling

Scripts don't hardcode any paths. Every input and output is set through
CLI flags with sensible defaults that assume repo-root execution:

| Default | Used by |
|---|---|
| `data/calibration/s22_ultra/camera_calibration.npz` | measure_fle, monte_carlo_tre |
| `data/fle_measurement/s22_ultra/` | measure_fle |
| `annotations/pedicle_entry_targets.json` | evaluate_designs, monte_carlo_tre |
| `results/tre_analysis/` | all output files |

If run from `scripts/tre_analysis/` directly, pass explicit paths with
`../../` prefixes.

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

Monte Carlo uses per-marker orientation matrices (design_variants_oriented.py)
so V1-V3 markers on the L-bracket faces are correctly placed in 3D. Earlier
versions of this script assumed all markers were parallel to the phantom XY
plane, which underestimated the L-bracket's PnP advantage.

FLE is measured at one working distance under one lighting condition, so
real operating conditions will vary.

## Outputs

```
results/tre_analysis/
├── fle_measurement.json       Per-marker and aggregate FLE + pixel noise
├── predictions_table.csv      Fitzpatrick TRE per variant per target
└── monte_carlo_table.csv      Monte Carlo TRE per variant per target
```

## Dependencies

numpy, opencv-contrib-python, scipy and matplotlib for the simulator and its
figures, and Pillow for the FLE board. See `requirements.txt` in the repo
root.

## References

- Fitzpatrick, West, Maurer (1998). "Predicting error in rigid-body
  point-based registration." IEEE TMI 17(5).
- Bao et al. (2014). Fiducial marker configuration optimisation.
- Oščádal et al. (2020). "Improved Pose Estimation of ArUco Tags Using a
  Novel 3D Placement Strategy." Sensors 20:4825.
- Ganeshan et al. (2025). Monocular NIR optical tracking with
  retroreflective fiducial markers. PMC12845559.