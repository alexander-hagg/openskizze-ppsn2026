#!/bin/bash

# ============================================================================
# GP HYPERPARAMETER OPTIMIZATION: Submit All Jobs
# ============================================================================
#
# Submits 90 individual SLURM jobs for hyperparameter optimization.
# Each job is independent and can be resubmitted individually if needed.
#
# Configuration space:
#   - Datasets: optimized, random, combined
#   - Inducing points: 100, 500, 1000, 2500, 5000
#   - K-means init: 0 (False), 1 (True)
#   - Replicates: 1, 2, 3
#
# Total: 3 datasets × 5 inducing × 2 kmeans × 3 replicates = 90 jobs
#
# Prerequisites:
#   python experiments/prepare_training_datasets.py
#
# Usage:
#   bash submit_all_gp_hpo.sh
# ============================================================================

echo "=========================================="
echo "Submitting GP HPO Jobs"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Configuration space
DATASETS=(optimized random combined)
NUM_INDUCING=(100 500 1000 2500 5000)
KMEANS_OPTIONS=(0 1)  # 0=False, 1=True
REPLICATES=(1 2 3)

# Counter for tracking submissions
SUBMITTED=0
FAILED=0

# Submit jobs for each configuration
for DATASET in "${DATASETS[@]}"; do
    for INDUCING in "${NUM_INDUCING[@]}"; do
        for KMEANS in "${KMEANS_OPTIONS[@]}"; do
            for REPLICATE in "${REPLICATES[@]}"; do
                # Build config name
                if [ "$KMEANS" -eq 1 ]; then
                    KMEANS_STR="kmeans"
                else
                    KMEANS_STR="random"
                fi
                
                CONFIG_NAME="ind${INDUCING}_${KMEANS_STR}"
                RUN_NAME="${DATASET}_${CONFIG_NAME}_rep${REPLICATE}"
                
                echo "Submitting: ${RUN_NAME}"
                
                JOB_ID=$(sbatch --parsable \
                    --job-name=hpo_${DATASET}_i${INDUCING}_k${KMEANS}_r${REPLICATE} \
                    --output=logs/gp_hpo_${RUN_NAME}_%j.out \
                    --error=logs/gp_hpo_${RUN_NAME}_%j.err \
                    --export=ALL,DATASET=${DATASET},INDUCING=${INDUCING},KMEANS=${KMEANS},REPLICATE=${REPLICATE} \
                    hpc/exp3_hpo/submit_gp_hpo.sh)
                
                if [ $? -eq 0 ]; then
                    echo "  ✓ Job ID: ${JOB_ID}"
                    SUBMITTED=$((SUBMITTED + 1))
                else
                    echo "  ✗ Failed to submit"
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Submitted ${SUBMITTED}/90 jobs successfully"
if [ $FAILED -gt 0 ]; then
    echo "Failed: ${FAILED} jobs"
fi
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After completion, analyze results:"
echo "  python experiments/exp3_hpo/analyze_hpo_results.py --results-dir results/exp3_hpo/hyperparameterization"
echo ""
