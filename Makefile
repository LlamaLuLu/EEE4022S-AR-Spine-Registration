# TRE analysis pipeline. Run from the repo root; `make help` lists the targets.
#
# Override any parameter on the command line, e.g.
#   make evaluate FLE=0.42
#   make monte-carlo N_TRIALS=5000 PIXEL_NOISE=0.4

CALIB       := data/calibration/s22_ultra/camera_calibration.npz
FLE_IMGDIR  := data/fle_measurement/s22_ultra
PEDICLES    := annotations/pedicle_entry_targets.json
RESULTS_DIR := results/tre_analysis
FLE_JSON    := $(RESULTS_DIR)/fle_measurement.json

MARKER_SIZE ?= 20
DICTIONARY  ?= 6X6_50
FLE         ?= 0.35
PIXEL_NOISE ?= 0.30
N_TRIALS    ?= 10000
DISTANCE    ?= 350
ELEVATION   ?= 30
AZIMUTH     ?= 45

PYTHON      := python
TRE_DIR     := scripts/tre_analysis

.PHONY: help measure-fle evaluate monte-carlo analyse test clean-results

help:
	@echo "  make measure-fle    FLE and pixel noise from the static-board frames"
	@echo "  make evaluate       Fitzpatrick TRE prediction for each marker layout"
	@echo "  make monte-carlo    Monte Carlo PnP validation"
	@echo "  make analyse        evaluate, then monte-carlo"
	@echo "  make test           run the unit tests"
	@echo "  make clean-results  delete predictions_table.csv and monte_carlo_table.csv"

measure-fle:
	$(PYTHON) $(TRE_DIR)/measure_fle.py \
	    --imagedir $(FLE_IMGDIR) \
	    --calibration $(CALIB) \
	    --marker-size-mm $(MARKER_SIZE) \
	    --dictionary $(DICTIONARY) \
	    --outfile $(FLE_JSON)

evaluate:
	$(PYTHON) $(TRE_DIR)/evaluate_designs.py \
	    --fle $(FLE) \
	    --pedicles $(PEDICLES) \
	    --outdir $(RESULTS_DIR)

monte-carlo:
	$(PYTHON) $(TRE_DIR)/monte_carlo_tre.py \
	    --calibration $(CALIB) \
	    --marker-size-mm $(MARKER_SIZE) \
	    --pixel-noise-sigma $(PIXEL_NOISE) \
	    --n-trials $(N_TRIALS) \
	    --distance-mm $(DISTANCE) \
	    --elevation-deg $(ELEVATION) \
	    --azimuth-deg $(AZIMUTH) \
	    --pedicles $(PEDICLES) \
	    --outfile $(RESULTS_DIR)/monte_carlo_table.csv

analyse: evaluate monte-carlo

test:
	$(PYTHON) -m unittest discover -s tests

# Only the two files the targets above generate; the other results are kept.
clean-results:
	rm -f $(RESULTS_DIR)/predictions_table.csv $(RESULTS_DIR)/monte_carlo_table.csv
