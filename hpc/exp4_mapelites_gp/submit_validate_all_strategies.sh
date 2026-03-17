#!/bin/bash

# ============================================================================
# Validate MAP-Elites Archives: Submit Both Top and Random Strategies
# ============================================================================
#
# Submits validation jobs for BOTH strategies to test selection bias hypothesis:
# 1. TOP 50 - highest GP predictions (already done)
# 2. RANDOM 50 - random samples from archive
#
# This tests whether GP failure is due to:
# - Selection bias (only top samples are wrong) OR
# - False optimum (entire archive is in wrong region)
#
# Usage:
#   bash submit_validate_both_strategies.sh
# ============================================================================

echo "=========================================="
echo "Submitting Validation Jobs (Top + Random)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Configuration space (use subset for quick test)
EMITTERS=(4 16 64)
BATCH_SIZES=(4 16 64)
REPLICATES=(1 2 3)

# Validation settings
TOP_N="${TOP_N:-250}"
STRATEGIES=("top" "random" "diverse")

# Counter for tracking submissions
SUBMITTED=0
SKIPPED=0

# Submit jobs for each configuration and strategy
for NUM_EMITTERS in "${EMITTERS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for REPLICATE in "${REPLICATES[@]}"; do
            for STRATEGY in "${STRATEGIES[@]}"; do
                RUN_NAME="emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"
                ARCHIVE_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/archive_final.pkl"
                VALIDATION_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/validation_${STRATEGY}/validation_metrics.yaml"
                
                # Check if archive exists
                if [ ! -f "$ARCHIVE_PATH" ]; then
                    echo "Skipping ${RUN_NAME} (${STRATEGY}): archive not found"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
                
                # Check if already validated
                if [ -f "$VALIDATION_PATH" ]; then
                    echo "Skipping ${RUN_NAME} (${STRATEGY}): already validated"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
                
                echo "Submitting: ${RUN_NAME} (${STRATEGY})"
                
                JOB_ID=$(sbatch --parsable \
                    --job-name=val_${STRATEGY}_e${NUM_EMITTERS}_b${BATCH_SIZE}_r${REPLICATE} \
                    --output=logs/validate_${STRATEGY}_${RUN_NAME}_%j.out \
                    --error=logs/validate_${STRATEGY}_${RUN_NAME}_%j.err \
                    --export=ALL,NUM_EMITTERS=${NUM_EMITTERS},BATCH_SIZE=${BATCH_SIZE},REPLICATE=${REPLICATE},TOP_N=${TOP_N},STRATEGY=${STRATEGY} \
                    hpc/exp4_mapelites_gp/submit_validate_archive.sh)
                
                if [ $? -eq 0 ]; then
                    echo "  ✓ Job ID: ${JOB_ID}"
                    SUBMITTED=$((SUBMITTED + 1))
                else
                    echo "  ✗ Failed to submit"
                fi
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Submitted ${SUBMITTED} validation jobs"
echo "Skipped ${SKIPPED} (missing archive or already validated)"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After completion, compare strategies:"
echo "  python experiments/exp4_mapelites_gp/compare_validation_strategies.py"
echo ""
