# Makefile for EEE4022S AR Spine Registration
#
# Standard invocations for the TRE analysis pipeline. Run from repo root:
#   make help              # list all targets
#   make measure-fle       # empirical FLE measurement
#   make evaluate          # Fitzpatrick predictions across variants
#   make monte-carlo       # PnP-based validation
#   make analyse           # run evaluate + monte-carlo together
#
# Override defaults per invocation, e.g.:
#   make evaluate FLE=0.42
#   make monte-carlo PIXEL_NOISE=0.4 N_TRIALS=5000

# ---- Paths (relative to repo root) ----
CALIB       := data/calibration/s22_ultra/camera_calibration.npz
FLE_IMGDIR  := data/fle_measurement/s22_ultra
PEDICLES    := annotations/pedicle_targets_phantom_frame.json
RESULTS_DIR := results/tre_analysis
FLE_JSON    := $(RESULTS_DIR)/fle_measurement.json

# ---- Parameters (override on command line) ----
MARKER_SIZE ?= 20
DICTIONARY  ?= 6X6_50
FLE         ?= 0.35
PIXEL_NOISE ?= 0.30
N_TRIALS    ?= 10000
DISTANCE    ?= 350
TILT        ?= 20

# ---- Python entry points ----
PYTHON      := python
TRE_DIR     := scripts/tre_analysis

.PHONY: help measure-fle evaluate monte-carlo analyse clean-results

help:
	@echo "TRE analysis pipeline targets:"
	@echo ""
	@echo "  make measure-fle        Empirical FLE from static-board frames"
	@echo "  make evaluate           Fitzpatrick TRE predictions per variant"
	@echo "  make monte-carlo        PnP-based Monte Carlo validation"
	@echo "  make analyse            Run evaluate then monte-carlo"
	@echo "  make clean-results      Remove generated CSVs and JSON"
	@echo ""
	@echo "Override any parameter with FLAG=value:"
	@echo "  make evaluate FLE=0.42"
	@echo "  make monte-carlo N_TRIALS=5000 PIXEL_NOISE=0.4"

# ---- FLE measurement ----
measure-fle:
	$(PYTHON) $(TRE_DIR)/measure_fle.py \
	    --imagedir $(FLE_IMGDIR) \
	    --calibration $(CALIB) \
	    --marker-size-mm $(MARKER_SIZE) \
	    --dictionary $(DICTIONARY) \
	    --outfile $(FLE_JSON)

# ---- Fitzpatrick TRE predictions ----
evaluate:
	$(PYTHON) $(TRE_DIR)/evaluate_designs.py \
	    --fle $(FLE) \
	    --pedicles $(PEDICLES) \
	    --outdir $(RESULTS_DIR)

# ---- Monte Carlo validation ----
monte-carlo:
	$(PYTHON) $(TRE_DIR)/monte_carlo_tre.py \
	    --calibration $(CALIB) \
	    --marker-size-mm $(MARKER_SIZE) \
	    --pixel-noise-sigma $(PIXEL_NOISE) \
	    --n-trials $(N_TRIALS) \
	    --distance-mm $(DISTANCE) \
	    --tilt-deg $(TILT) \
	    --pedicles $(PEDICLES) \
	    --outfile $(RESULTS_DIR)/monte_carlo_table.csv

# ---- Both analyses back to back ----
analyse: evaluate monte-carlo

# ---- Cleanup ----
clean-results:
	rm -rf $(RESULTS_DIR)