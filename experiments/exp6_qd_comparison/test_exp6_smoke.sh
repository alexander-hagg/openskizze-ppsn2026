#!/bin/bash

# Quick smoke test for Experiment 6
# Tests core functionality without full optimization run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "================================================================"
echo "EXPERIMENT 6: SMOKE TEST"
echo "================================================================"
echo ""

# Check dependencies
echo "Checking Python environment..."
python -c "import torch; import gpytorch; import ribs; print('✓ All dependencies available')"

# Check model files exist
echo ""
echo "Checking model files..."

UNET_MODEL="results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth"
SVGP_MODEL="results/exp3_hpo/hyperparameterization/model_optimized_ind2500_kmeans_rep1.pth"

if [ -f "$UNET_MODEL" ]; then
    echo "✓ U-Net model found: $UNET_MODEL"
else
    echo "✗ U-Net model not found: $UNET_MODEL"
    echo "  Run Experiment 5 first or adjust path in scripts"
fi

if [ -f "$SVGP_MODEL" ]; then
    echo "✓ SVGP model found: $SVGP_MODEL"
else
    echo "✗ SVGP model not found: $SVGP_MODEL"
    echo "  Run Experiment 3 first or adjust path in scripts"
fi

# Quick optimization test (minimal generations)
echo ""
echo "Running quick optimization test (10 generations)..."

python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model unet \
    --parcel-size 60 \
    --generations 10 \
    --num-emitters 4 \
    --batch-size 2 \
    --seed 42 \
    --output-dir results/exp6_qd_comparison/smoke_test \
    2>&1 | tail -20

echo ""
echo "✓ Optimization test completed"

# Check if archive was created
SMOKE_ARCHIVE="results/exp6_qd_comparison/smoke_test/archive_unet_size27_seed42.pkl"
if [ -f "$SMOKE_ARCHIVE" ]; then
    echo "✓ Archive created: $SMOKE_ARCHIVE"
    
    # Quick validation test (max 10 solutions)
    echo ""
    echo "Running quick validation test (max 10 solutions)..."
    
    python experiments/exp6_qd_comparison/validate_archives.py \
        --archive "$SMOKE_ARCHIVE" \
        --output-dir results/exp6_qd_comparison/smoke_test/validation \
        --max-solutions 10 \
        2>&1 | tail -15
    
    echo ""
    echo "✓ Validation test completed"
    
    # Check validation results
    VALIDATION_FILE="results/exp6_qd_comparison/smoke_test/validation/archive_unet_size27_seed42_validated.npz"
    if [ -f "$VALIDATION_FILE" ]; then
        echo "✓ Validation results created: $VALIDATION_FILE"
        
        # Show metrics
        METRICS_FILE="results/exp6_qd_comparison/smoke_test/validation/archive_unet_size27_seed42_validated_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            echo ""
            echo "Validation metrics:"
            cat "$METRICS_FILE" | python -m json.tool | grep -A 5 '"metrics"'
        fi
    fi
else
    echo "✗ Archive not created"
fi

echo ""
echo "================================================================"
echo "SMOKE TEST COMPLETE"
echo "================================================================"
echo ""
echo "To run full experiment:"
echo "  bash hpc/exp6_qd_comparison/run_exp6.sh all"
