#!/bin/bash

# ============================================================================
# MAP-Elites Archive Validation: Submit All Jobs
# ============================================================================
#
# Submits 75 individual SLURM jobs to validate MAP-Elites archives with
# real KLAM_21 simulations.
#
# Configuration space:
#   - Emitters: 4, 16, 64, 256, 512
#   - Batch sizes: 4, 16, 64, 256, 512
#   - Replicates: 1, 2, 3
#
# Total: 5 emitters × 5 batch sizes × 3 replicates = 75 jobs
#
# Usage:
#   bash submit_all_validate_archives.sh
# ============================================================================

echo "=========================================="
echo "Submitting Archive Validation Jobs"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Configuration space (must match sweep)
EMITTERS=(4 16 64 256 512)
BATCH_SIZES=(4 16 64 256 512)
REPLICATES=(1 2 3)

# Validation settings
TOP_N="${TOP_N:-50}"
STRATEGY="${STRATEGY:-top}"

# Counter for tracking submissions
SUBMITTED=0
SKIPPED=0
FAILED=0

# Submit jobs for each configuration
for NUM_EMITTERS in "${EMITTERS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for REPLICATE in "${REPLICATES[@]}"; do
            RUN_NAME="emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"
            ARCHIVE_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/archive_final.pkl"
            VALIDATION_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/validation/validation_metrics.yaml"
            
            # Check if archive exists
            if [ ! -f "$ARCHIVE_PATH" ]; then
                echo "Skipping ${RUN_NAME}: archive not found"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            
            # Check if already validated
            if [ -f "$VALIDATION_PATH" ]; then
                echo "Skipping ${RUN_NAME}: already validated"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            
            echo "Submitting: ${RUN_NAME}"
            
            JOB_ID=$(sbatch --parsable \
                --job-name=val_e${NUM_EMITTERS}_b${BATCH_SIZE}_r${REPLICATE} \
                --output=logs/validate_${RUN_NAME}_%j.out \
                --error=logs/validate_${RUN_NAME}_%j.err \
                --export=ALL,NUM_EMITTERS=${NUM_EMITTERS},BATCH_SIZE=${BATCH_SIZE},REPLICATE=${REPLICATE},TOP_N=${TOP_N},STRATEGY=${STRATEGY} \
                hpc/exp4_mapelites_gp/submit_validate_archive.sh)
            
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

echo ""
echo "=========================================="
echo "Submitted ${SUBMITTED} jobs"
echo "Skipped ${SKIPPED} jobs (missing archive or already validated)"
if [ $FAILED -gt 0 ]; then
    echo "Failed: ${FAILED} jobs"
fi
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
