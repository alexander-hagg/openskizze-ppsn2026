#!/bin/bash
#SBATCH --job-name=unet_multiscale
#SBATCH --output=logs/unet_multiscale_%A_%a.out
#SBATCH --error=logs/unet_multiscale_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2  # 3 parcel sizes

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# U-NET Training for Multiple Parcel Sizes
# ============================================================================
# Trains 3 separate U-NET models for parcel sizes: 60, 120, 240m
# Each model optimized for its grid dimensions
#
# Prerequisites:
#   - Exp1 spatial data generated for all 3 sizes
#   - Files: sail_{size}x{size}_rep{1,2,3}_spatial.npz
#
# Usage:
#   sbatch hpc/exp5_unet/submit_unet_multiscale.sh
#
# Expected Runtime:
#   - 60m:  ~6 hours (smallest grid)
#   - 120m: ~8 hours
#   - 240m: ~10 hours
# ============================================================================

echo "======================================================================"
echo "U-NET Multi-Scale Training Job"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Create directories
mkdir -p logs
mkdir -p results/unet_experiment

# Configuration arrays
PARCEL_SIZES=(60 120 240)
BATCH_SIZES=(32 16 8 4)  # Decrease with size due to memory constraints

# Get configuration for this task
SIZE=${PARCEL_SIZES[$SLURM_ARRAY_TASK_ID]}
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Configuration:"
echo "  Parcel size: ${SIZE}m"
echo "  Batch size:  ${BATCH}"
echo "  Max epochs:  200"
echo "  Early stop:  20 epochs patience"
echo "  Output dir:  results/unet_experiment/unet_${SIZE}m"
echo ""

# Check for spatial data
DATA_DIR="results/exp1_gp_training_data/sail_data"
SPATIAL_FILES=(
    "${DATA_DIR}/sail_${SIZE}x${SIZE}_rep1_spatial.npz"
    "${DATA_DIR}/sail_${SIZE}x${SIZE}_rep2_spatial.npz"
    "${DATA_DIR}/sail_${SIZE}x${SIZE}_rep3_spatial.npz"
)

echo "Checking for spatial data files..."
MISSING=0
for FILE in "${SPATIAL_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "  ✗ MISSING: $FILE"
        MISSING=1
    else
        echo "  ✓ Found: $FILE"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Missing spatial data files!"
    echo "Generate with: bash hpc/exp1_gp_training_data/run_gp_experiment.sh 1"
    echo "Make sure to use --collect-spatial-data flag"
    exit 1
fi

echo ""
echo "All spatial data files found. Starting training..."
echo ""

# Run training
python experiments/exp5_unet/train_unet_klam.py \
    --parcel-size ${SIZE} \
    --data-dir ${DATA_DIR} \
    --output-dir results/unet_experiment/unet_${SIZE}m \
    --max-epochs 200 \
    --batch-size ${BATCH} \
    --early-stopping-patience 20 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Training Complete"
echo "======================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: U-Net model trained for ${SIZE}m parcels"
    echo "  Model saved: results/unet_experiment/unet_${SIZE}m/best_model.pth"
    
    # Check model file
    MODEL_FILE="results/unet_experiment/unet_${SIZE}m/best_model.pth"
    if [ -f "$MODEL_FILE" ]; then
        SIZE_MB=$(du -m "$MODEL_FILE" | cut -f1)
        echo "  Model size: ${SIZE_MB} MB"
    fi
else
    echo "✗ FAILED: Training failed with exit code $EXIT_CODE"
    echo "  Check logs: logs/unet_multiscale_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
fi

echo "======================================================================"
exit $EXIT_CODE
