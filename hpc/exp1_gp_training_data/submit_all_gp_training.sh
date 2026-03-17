#!/bin/bash

# ============================================================================
# GP EXPERIMENT: Submit All GP Training Jobs
# ============================================================================
#
# Submits 9 individual SLURM jobs (3 datasets × 3 replicates).
# Each job is independent and can be resubmitted individually if needed.
#
# Prerequisites:
#   1. Run submit_gp_exp_sail_data.sh (Phase 1A)
#   2. Run submit_gp_exp_random_data.sh (Phase 1B)
#   3. Run prepare_training_datasets.py locally (Phase 2)
#
# Usage:
#   bash submit_all_gp_training.sh
#
# Expected outputs:
#   9 SLURM job IDs
#   9 model files: results/gp_experiment/gp_<dataset>_rep<N>.pth
# ============================================================================

echo "=========================================="
echo "Submitting GP Training Jobs"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Counter for tracking submissions
SUBMITTED=0

# Submit jobs for each dataset and replicate
for DATASET in optimized random combined; do
    for REPLICATE in 1 2 3; do
        echo "Submitting: ${DATASET} replicate ${REPLICATE}"
        
        JOB_ID=$(sbatch --parsable \
            --job-name=gp_${DATASET}_r${REPLICATE} \
            --output=logs/gp_train_${DATASET}_rep${REPLICATE}_%j.out \
            --error=logs/gp_train_${DATASET}_rep${REPLICATE}_%j.err \
            --export=ALL,DATASET=${DATASET},REPLICATE=${REPLICATE} \
            hpc/exp1_gp_training_data/submit_gp_exp_training.sh)
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Job ID: ${JOB_ID}"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "  ✗ Failed to submit"
        fi
    done
done

echo ""
echo "=========================================="
echo "Submitted ${SUBMITTED}/9 jobs successfully"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all: scancel -u \$USER -n gp_optimized_r1,gp_optimized_r2,gp_optimized_r3,gp_random_r1,gp_random_r2,gp_random_r3,gp_combined_r1,gp_combined_r2,gp_combined_r3"
echo ""
