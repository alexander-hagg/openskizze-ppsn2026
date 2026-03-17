#!/bin/bash

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# Submit All U-NET Training Configurations
# ============================================================================
# Submits separate SLURM jobs for all U-NET training configurations:
#   - Training data: SAIL vs Random
#   - Loss function: MSE vs MSE+Gradient
#   - 3 replicates each
#
# Total: 2 × 2 × 3 = 12 jobs
#
# Usage:
#   bash hpc/exp5_unet/submit_all_unet_configs.sh
# ============================================================================

echo "======================================================================"
echo "Submitting All U-NET Training Configurations"
echo "======================================================================"
echo ""

# Define configurations
DATA_TYPES=("sail" "random")
LOSS_TYPES=("mse" "mse_grad")
SEEDS=(42 43 44)

# Counter
TOTAL=0
SUBMITTED=0

# Submit all combinations
for DATA_TYPE in "${DATA_TYPES[@]}"; do
    for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            TOTAL=$((TOTAL + 1))
            
            echo "[$TOTAL] Submitting: DATA_TYPE=${DATA_TYPE}, LOSS_TYPE=${LOSS_TYPE}, SEED=${SEED}"
            
            JOB_ID=$(sbatch --parsable \
                --export=ALL,DATA_TYPE=${DATA_TYPE},LOSS_TYPE=${LOSS_TYPE},SEED=${SEED} \
                hpc/exp5_unet/submit_unet_training.sh)
            
            if [ $? -eq 0 ]; then
                echo "    ✓ Job ID: ${JOB_ID}"
                SUBMITTED=$((SUBMITTED + 1))
            else
                echo "    ✗ Failed to submit"
            fi
            echo ""
        done
    done
done

echo "======================================================================"
echo "Summary"
echo "======================================================================"
echo "Total configurations: ${TOTAL}"
echo "Successfully submitted: ${SUBMITTED}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "======================================================================"
