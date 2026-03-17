#!/bin/bash

# ============================================================================
# MAP-Elites GP Sweep: Submit All Jobs
# ============================================================================
#
# Submits individual SLURM jobs for MAP-Elites GP parameter sweep.
# Each job is independent and can be resubmitted individually if needed.
#
# Configuration space:
#   - Emitters: 4, 16, 64, 256, 512
#   - Batch sizes: 4, 16, 64, 256, 512
#   - Replicates: 1, 2, 3
#
# Total: 5 emitters × 5 batch sizes × 3 replicates = 75 jobs
#
# Usage:
#   bash submit_all_mapelites_gp_sweep.sh           # Submit missing jobs only
#   bash submit_all_mapelites_gp_sweep.sh --all     # Submit all jobs (overwrite)
#   bash submit_all_mapelites_gp_sweep.sh --dry-run # Show what would be submitted
#   bash submit_all_mapelites_gp_sweep.sh --status  # Show completion status
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
    echo "MAP-Elites GP Sweep: Status Check"
elif $DRY_RUN; then
    echo "MAP-Elites GP Sweep: DRY RUN"
else
    echo "Submitting MAP-Elites GP Sweep Jobs"
fi
echo "=========================================="

# Create logs directory
mkdir -p logs
mkdir -p results/exp4_mapelites_gp/mapelites_gp

# Configuration space
EMITTERS=(4 16 64)
BATCH_SIZES=(4 16 64)
REPLICATES=(1 2 3)

# Configuration
GP_MODEL="${GP_MODEL:-results/exp3_hpo/hyperparameterization/model_combined_ind1000_random_rep1.pth}"
echo "Using GP model: ${GP_MODEL}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5000}"  # Reduced from 20K to limit GP exploitation
echo "Number of generations: ${NUM_GENERATIONS}"
PARCEL_SIZE="${PARCEL_SIZE:-60}"  # Must match GP training data: {60, 120, 240}m
echo "Parcel size: ${PARCEL_SIZE}m"
LAMBDA_UCB="${LAMBDA_UCB:-1.0}"  # UCB penalty (score = mean - lambda*std), SAIL uses 1.0
echo "Lambda UCB: ${LAMBDA_UCB}"
GENOME_BOUNDS="${GENOME_BOUNDS:-15.0}"  # Clip to [-15, 15], covers ~99% of training data
echo "Genome bounds: [-${GENOME_BOUNDS}, ${GENOME_BOUNDS}]"

# Counters
SUBMITTED=0
SKIPPED=0
FAILED=0
COMPLETED=0
INCOMPLETE=0
MISSING=0

# Submit jobs for each configuration
for NUM_EMITTERS in "${EMITTERS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for REPLICATE in "${REPLICATES[@]}"; do
            RUN_NAME="emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"
            OUTPUT_DIR="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}"
            ARCHIVE_FILE="${OUTPUT_DIR}/archive_final.pkl"
            HISTORY_FILE="${OUTPUT_DIR}/history.pkl"
            
            # Check completion status
            if [ -f "$ARCHIVE_FILE" ] && [ -f "$HISTORY_FILE" ]; then
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
            elif [ -d "$OUTPUT_DIR" ]; then
                STATUS="⚠ INCOMPLETE"
                INCOMPLETE=$((INCOMPLETE + 1))
                if $STATUS_ONLY; then
                    echo "  $STATUS: ${RUN_NAME}"
                    # Show what files exist
                    ls -la "$OUTPUT_DIR" 2>/dev/null | head -5 | sed 's/^/      /'
                    continue
                fi
            else
                STATUS="✗ MISSING"
                MISSING=$((MISSING + 1))
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
                --job-name=mapgp_e${NUM_EMITTERS}_b${BATCH_SIZE}_r${REPLICATE} \
                --output=logs/mapelites_gp_${RUN_NAME}_%j.out \
                --error=logs/mapelites_gp_${RUN_NAME}_%j.err \
                --export=ALL,NUM_EMITTERS=${NUM_EMITTERS},BATCH_SIZE=${BATCH_SIZE},REPLICATE=${REPLICATE},GP_MODEL=${GP_MODEL},NUM_GENERATIONS=${NUM_GENERATIONS},PARCEL_SIZE=${PARCEL_SIZE},LAMBDA_UCB=${LAMBDA_UCB},GENOME_BOUNDS=${GENOME_BOUNDS} \
                hpc/exp4_mapelites_gp/submit_mapelites_gp_sweep.sh)
            
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
echo "Summary (75 total configurations)"
echo "=========================================="

if $STATUS_ONLY; then
    echo "  Completed:  ${COMPLETED}"
    echo "  Incomplete: ${INCOMPLETE}"
    echo "  Missing:    ${MISSING}"
elif $DRY_RUN; then
    echo "  Would submit: ${SUBMITTED}"
    echo "  Would skip:   ${SKIPPED} (already complete)"
    echo ""
    echo "Run without --dry-run to actually submit jobs"
else
    echo "  Submitted: ${SUBMITTED}"
    echo "  Skipped:   ${SKIPPED} (already complete)"
    if [ $FAILED -gt 0 ]; then
        echo "  Failed:    ${FAILED}"
    fi
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo ""
    echo "After completion, validate archives:"
    echo "  bash hpc/exp4_mapelites_gp/submit_all_validate_archives.sh"
fi
echo ""
