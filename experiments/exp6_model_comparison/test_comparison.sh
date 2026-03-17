#!/bin/bash
# Quick local test for WP1.2 comparison
# Tests with small sample size before HPC submission

set -e

echo "========================================================================"
echo "WP1.2: Local Test (100 samples)"
echo "========================================================================"

# Configuration
SVGP_MODEL="results/hyperparameterization/model_optimized_ind2500_random_rep1.pth"
UNET_MODEL="results/unet_experiment/sail_mse_seed42"
OUTPUT_DIR="results/model_comparison_test"
NUM_SAMPLES=100
PARCEL_SIZE=51

# Check if models exist
if [ ! -f "$SVGP_MODEL" ]; then
    echo "ERROR: SVGP model not found: $SVGP_MODEL"
    exit 1
fi

if [ ! -d "$UNET_MODEL" ]; then
    echo "ERROR: U-Net model directory not found: $UNET_MODEL"
    exit 1
fi

echo "Configuration:"
echo "  SVGP model: $SVGP_MODEL"
echo "  U-Net model: $UNET_MODEL"
echo "  Num samples: $NUM_SAMPLES (test)"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run test
python experiments/exp6_model_comparison/compare_svgp_unet.py \
    --svgp-model "$SVGP_MODEL" \
    --unet-model "$UNET_MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --parcel-size "$PARCEL_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --svgp-batch-size 256 \
    --unet-batch-size 128 \
    --seed 42

echo ""
echo "========================================================================"
echo "✓ Test completed successfully"
echo ""
echo "Review results in: $OUTPUT_DIR"
echo ""
echo "If results look good, submit full job:"
echo "  sbatch hpc/exp6_model_comparison/submit_comparison.sh"
echo "========================================================================"
