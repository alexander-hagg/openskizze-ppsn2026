#!/bin/bash

# ============================================================================
# Diagnose All MAP-Elites Archives: Submit Batch Jobs
# ============================================================================
#
# Submits diagnostic jobs for all MAP-Elites archives to analyze:
# 1. Genome distribution shift from training data
# 2. GP predictive uncertainty on archive vs test data
#
# This helps understand why archives validate poorly with KLAM_21.
#
# Usage:
#   bash submit_diagnose_all_archives.sh           # Submit missing only
#   bash submit_diagnose_all_archives.sh --all     # Submit all (overwrite)
#   bash submit_diagnose_all_archives.sh --dry-run # Show what would run
#   bash submit_diagnose_all_archives.sh --status  # Show completion status
# ============================================================================

# Parse arguments
DRY_RUN=false
FORCE_ALL=false
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|--dry)
            DRY_RUN=true
            shift
            ;;
        --all|--force)
            FORCE_ALL=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run|--all|--status]"
            exit 1
            ;;
    esac
done

echo "=========================================="
if $STATUS_ONLY; then
    echo "Diagnostics Status Check"
elif $DRY_RUN; then
    echo "Diagnostics: DRY RUN"
else
    echo "Submitting Diagnostic Jobs"
fi
echo "=========================================="

# Create logs directory
mkdir -p logs

# Configuration space (should match your sweep)
EMITTERS=(4 16 64)
BATCH_SIZES=(4 16 64)
REPLICATES=(1 2 3)

# Model and data paths
GP_MODEL="${GP_MODEL:-results/exp3_hpo/hyperparameterization/model_combined_ind1000_random_rep1.pth}"
TRAINING_DATA="${TRAINING_DATA:-results/exp1_gp_training_data/training_datasets/dataset_combined.npz}"

echo "Using:"
echo "  GP Model:      $GP_MODEL"
echo "  Training Data: $TRAINING_DATA"
echo ""

# Counters
SUBMITTED=0
SKIPPED=0
COMPLETED=0
INCOMPLETE=0
MISSING=0

# Submit jobs for each configuration
for NUM_EMITTERS in "${EMITTERS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for REPLICATE in "${REPLICATES[@]}"; do
            RUN_NAME="emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"
            ARCHIVE_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/archive_final.pkl"
            DIAGNOSTICS_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/diagnostics/diagnostic_metrics.yaml"
            
            # Check if archive exists
            if [ ! -f "$ARCHIVE_PATH" ]; then
                STATUS="✗ NO ARCHIVE"
                MISSING=$((MISSING + 1))
                if $STATUS_ONLY; then
                    echo "  $STATUS: ${RUN_NAME}"
                fi
                continue
            fi
            
            # Check completion status
            if [ -f "$DIAGNOSTICS_PATH" ]; then
                STATUS="✓ COMPLETE"
                COMPLETED=$((COMPLETED + 1))
                
                if $STATUS_ONLY; then
                    echo "  $STATUS: ${RUN_NAME}"
                    continue
                fi
                
                if ! $FORCE_ALL; then
                    SKIPPED=$((SKIPPED + 1))
                    if $DRY_RUN; then
                        echo "  SKIP (complete): ${RUN_NAME}"
                    fi
                    continue
                fi
            else
                STATUS="⚠ PENDING"
                INCOMPLETE=$((INCOMPLETE + 1))
                if $STATUS_ONLY; then
                    echo "  $STATUS: ${RUN_NAME}"
                    continue
                fi
            fi
            
            if $DRY_RUN; then
                echo "  WOULD SUBMIT: ${RUN_NAME}"
                SUBMITTED=$((SUBMITTED + 1))
                continue
            fi
            
            echo "Submitting: ${RUN_NAME}"
            
            JOB_ID=$(sbatch --parsable \
                --job-name=diag_e${NUM_EMITTERS}_b${BATCH_SIZE}_r${REPLICATE} \
                --output=logs/diagnose_${RUN_NAME}_%j.out \
                --error=logs/diagnose_${RUN_NAME}_%j.err \
                --export=ALL,NUM_EMITTERS=${NUM_EMITTERS},BATCH_SIZE=${BATCH_SIZE},REPLICATE=${REPLICATE},GP_MODEL=${GP_MODEL},TRAINING_DATA=${TRAINING_DATA} \
                hpc/exp4_mapelites_gp/submit_diagnose_archive.sh)
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Job ID: ${JOB_ID}"
                SUBMITTED=$((SUBMITTED + 1))
            else
                echo "  ✗ Failed to submit"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if $STATUS_ONLY; then
    echo "  Completed:  ${COMPLETED}"
    echo "  Pending:    ${INCOMPLETE}"
    echo "  No archive: ${MISSING}"
elif $DRY_RUN; then
    echo "  Would submit: ${SUBMITTED}"
    echo "  Would skip:   ${SKIPPED} (already complete)"
    echo ""
    echo "Run without --dry-run to actually submit jobs"
else
    echo "  Submitted: ${SUBMITTED}"
    echo "  Skipped:   ${SKIPPED} (already complete)"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo ""
    echo "After completion, review diagnostics:"
    echo "  ls -la results/exp4_mapelites_gp/mapelites_gp/*/diagnostics/"
fi
echo ""
