#!/bin/bash
#SBATCH --job-name=svgp_unet_cmp
#SBATCH --output=logs/model_comparison_%j.out
#SBATCH --error=logs/model_comparison_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# WP1.2: SVGP vs U-Net Comparison
# ============================================================================
#
# Compares SVGP and U-Net models on 1000 diverse layouts to determine
# which model to use for GUI optimization.
#
# Usage:
#   sbatch hpc/exp6_model_comparison/submit_comparison.sh
#
# Expected runtime: ~30-60 minutes
# ============================================================================

echo "========================================================================"
echo "WP1.2: SVGP vs U-Net Model Comparison"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load environment
module purge
module load anaconda3/2023.03
source activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

# Configuration
SVGP_MODEL="results/exp3_hpo/hyperparameterization/model_combined_ind5000_kmeans_rep2.pth"
UNET_MODEL="results/exp5_unet/unet_experiment/sail_mse_seed42"
OUTPUT_DIR="results/exp6_comparison/model_comparison"
NUM_SAMPLES=1000
PARCEL_SIZE=51

echo "Configuration:"
echo "  SVGP model: $SVGP_MODEL"
echo "  U-Net model: $UNET_MODEL"
echo "  Num samples: $NUM_SAMPLES"
echo "  Parcel size: $PARCEL_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run comparison
echo "Running comparison..."
python experiments/exp6_model_comparison/compare_svgp_unet.py \
    --svgp-model "$SVGP_MODEL" \
    --unet-model "$UNET_MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --parcel-size "$PARCEL_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --svgp-batch-size 256 \
    --unet-batch-size 128 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ WP1.2 COMPLETED SUCCESSFULLY"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Review:"
    echo "  1. Comparison metrics: $OUTPUT_DIR/svgp_vs_unet_comparison.json"
    echo "  2. Visualization: $OUTPUT_DIR/comparison_plots.png"
    echo ""
    echo "Next step: Review recommendation and proceed with WP3 (evaluation_unet.py)"
else
    echo "✗ WP1.2 FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"
echo "End time: $(date)"
