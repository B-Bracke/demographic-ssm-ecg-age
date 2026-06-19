#!/bin/bash
# ==============================================================================
# ECG Age Validation Pipeline
#
# Runs the full evaluation: optional calibration → data preparation →
# performance plots → survival analysis.
#
# Usage:
#   ./run_validation.sh <prediction_csv> <output_folder> [calibration_samples] [bias_only] [age_threshold]
#
# Example (without calibration — as used in the official submission):
#   ./run_validation.sh ./predictions.csv ./results/
#
# Example (with calibration):
#   ./run_validation.sh ./predictions.csv ./results/ 512
# ==============================================================================

set -e

PREDICTION_FILE=$1
OUTPUT_FOLDER=$2
CALIB_SAMPLES=${3:-0}
BIAS_ONLY=${4:-false}
AGE_THRESHOLD=${5:-8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$PREDICTION_FILE" ] || [ -z "$OUTPUT_FOLDER" ]; then
    echo "Usage: ./run_validation.sh <prediction_csv> <output_folder> [calibration_samples] [bias_only] [age_threshold]"
    exit 1
fi

mkdir -p "$OUTPUT_FOLDER/plots"
mkdir -p "$OUTPUT_FOLDER/survival"

# Step 0: Optional calibration
DATA_TO_USE="$PREDICTION_FILE"

if [ "$CALIB_SAMPLES" -gt 0 ]; then
    echo "[Step 0/3] Calibrating with $CALIB_SAMPLES samples..."
    CALIBRATED_FILE="$OUTPUT_FOLDER/calibrated_predictions.csv"

    python "$SCRIPT_DIR/calibrate_predictions.py" \
        --input "$PREDICTION_FILE" \
        --output "$CALIBRATED_FILE" \
        --samples "$CALIB_SAMPLES" \
        --seed 42 \
        $( [ "$BIAS_ONLY" = true ] && echo "--bias_only" )

    DATA_TO_USE="$CALIBRATED_FILE"
    echo "Using calibrated predictions."
else
    echo "Skipping calibration (samples=0). Using raw predictions."
fi

# Step 1: Data preparation
echo ""
echo "[Step 1/3] Preparing data..."
python "$SCRIPT_DIR/prepare_data.py" \
    --input "$DATA_TO_USE" \
    --output_dir "$OUTPUT_FOLDER" \
    --age_threshold "$AGE_THRESHOLD"

# Step 2: Performance plots
echo ""
echo "[Step 2/3] Generating performance plots..."
python "$SCRIPT_DIR/plot_performance.py" \
    --input_dir "$OUTPUT_FOLDER" \
    --output_dir "$OUTPUT_FOLDER/plots"

# Step 3: Survival analysis
echo ""
echo "[Step 3/3] Running survival analysis..."
python "$SCRIPT_DIR/survival_analysis.py" \
    --input_dir "$OUTPUT_FOLDER" \
    --output_dir "$OUTPUT_FOLDER/survival"

echo ""
echo "Pipeline completed. Results saved in: $OUTPUT_FOLDER"