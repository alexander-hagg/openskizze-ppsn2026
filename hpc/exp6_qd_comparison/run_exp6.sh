#!/bin/bash

# Experiment 6: QD Comparison with Offline Surrogates
# Master script for running the full experiment pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs

# Print usage
usage() {
    cat << EOF
Usage: $0 <phase> [options]

Phases:
    opt         Run MAP-Elites optimization (Phase 1)
    validate    Validate archives with KLAM_21 (Phase 2)
    analyze     Analyze results (Phase 3)
    all         Run all phases sequentially
    status      Check job status
    clean       Clean results (use with caution!)

Options:
    --dry-run   Print commands without executing

Examples:
    # Run full experiment
    bash hpc/exp6_qd_comparison/run_exp6.sh all
    
    # Run individual phases
    bash hpc/exp6_qd_comparison/run_exp6.sh opt
    bash hpc/exp6_qd_comparison/run_exp6.sh validate
    bash hpc/exp6_qd_comparison/run_exp6.sh analyze
    
    # Check status
    bash hpc/exp6_qd_comparison/run_exp6.sh status
EOF
}

# Parse arguments
PHASE=${1:-help}
DRY_RUN=false

if [ "$2" == "--dry-run" ]; then
    DRY_RUN=true
fi

# Execute command
run_cmd() {
    echo "Executing: $@"
    if [ "$DRY_RUN" = false ]; then
        "$@"
    fi
}

# Phase 1: Optimization
run_optimization() {
    echo "================================================================"
    echo "PHASE 1: MAP-ELITES OPTIMIZATION WITH OFFLINE SURROGATES"
    echo "================================================================"
    echo ""
    
    # Baselines: unet and svgp with UCB=0.0, 3 replicates each (6 jobs)
    # Exploration: svgp and hybrid with UCB={0.1,1.0,10.0}, 3 replicates each (18 jobs)
    # Total: 24 jobs
    
    JOB_COUNT=0
    SUBMIT_SCRIPT="$SCRIPT_DIR/submit_qd_optimization.sh"
    
    for SEED in 42 43 44; do
        # Baselines: unet and svgp, UCB=0.0 only
        for MODEL in unet svgp; do
            echo "Submitting: MODEL=$MODEL UCB_LAMBDA=0.0 SEED=$SEED"
            run_cmd sbatch --export=ALL,MODEL=$MODEL,UCB_LAMBDA=0.0,SEED=$SEED "$SUBMIT_SCRIPT"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
        
        # Exploration: svgp and hybrid, UCB sweep
        for MODEL in svgp hybrid; do
            for UCB_LAMBDA in 0.1 1.0 10.0; do
                echo "Submitting: MODEL=$MODEL UCB_LAMBDA=$UCB_LAMBDA SEED=$SEED"
                run_cmd sbatch --export=ALL,MODEL=$MODEL,UCB_LAMBDA=$UCB_LAMBDA,SEED=$SEED "$SUBMIT_SCRIPT"
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
    
    echo ""
    echo "Submitted $JOB_COUNT optimization jobs. Monitor with: squeue -u $USER"
    echo "Expected completion time: ~1 hour per job"
}

# Phase 2: Validation
run_validation() {
    echo "================================================================"
    echo "PHASE 2: ARCHIVE VALIDATION WITH KLAM_21"
    echo "================================================================"
    echo ""
    
    # Check if archives exist
    ARCHIVE_DIR="results/exp6_qd_comparison"
    
    mapfile -t ARCHIVES < <(find "$ARCHIVE_DIR" -name "archive_*.pkl" -type f | sort)
    N_ARCHIVES=${#ARCHIVES[@]}
    
    if [ $N_ARCHIVES -eq 0 ]; then
        echo "ERROR: No archives found in $ARCHIVE_DIR"
        echo "Please run optimization phase first!"
        exit 1
    fi
    
    echo "Found $N_ARCHIVES archives to validate"
    echo ""
    
    SUBMIT_SCRIPT="$SCRIPT_DIR/submit_validation.sh"
    JOB_COUNT=0
    
    for ARCHIVE_PATH in "${ARCHIVES[@]}"; do
        echo "Submitting validation for: $ARCHIVE_PATH"
        run_cmd sbatch --export=ALL,ARCHIVE_PATH="$ARCHIVE_PATH" "$SUBMIT_SCRIPT"
        JOB_COUNT=$((JOB_COUNT + 1))
    done
    
    echo ""
    echo "Submitted $JOB_COUNT validation jobs. Monitor with: squeue -u $USER"
    echo "Expected completion time: ~5-10 hours per archive"
}

# Phase 3: Analysis
run_analysis() {
    echo "================================================================"
    echo "PHASE 3: RESULT ANALYSIS"
    echo "================================================================"
    echo "Submitting analysis job..."
    echo ""
    
    # Check if validation results exist
    VALIDATION_DIR="results/exp6_qd_comparison/validation"
    N_VALIDATED=$(find "$VALIDATION_DIR" -name "*_validated.npz" -type f 2>/dev/null | wc -l)
    
    if [ $N_VALIDATED -eq 0 ]; then
        echo "ERROR: No validation results found in $VALIDATION_DIR"
        echo "Please run validation phase first!"
        exit 1
    fi
    
    echo "Found $N_VALIDATED validated archives"
    
    run_cmd sbatch "$SCRIPT_DIR/submit_analysis.sh"
    
    echo ""
    echo "Job submitted! Monitor with: squeue -u $USER"
    echo "Results will be saved to: results/exp6_qd_comparison/analysis"
}

# Check status
check_status() {
    echo "================================================================"
    echo "EXPERIMENT 6: JOB STATUS"
    echo "================================================================"
    echo ""
    
    echo "Current running/pending jobs:"
    squeue -u $USER --name=exp6_qd_opt,exp6_validate,exp6_analyze || echo "No jobs found"
    echo ""
    
    echo "Recent job history:"
    sacct --name=exp6_qd_opt,exp6_validate,exp6_analyze -S $(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,MaxRSS | tail -20
    echo ""
    
    # Check for completed archives
    ARCHIVE_DIR="results/exp6_qd_comparison"
    VALIDATION_DIR="$ARCHIVE_DIR/validation"
    ANALYSIS_DIR="$ARCHIVE_DIR/analysis"
    
    echo "Progress:"
    echo "  Archives: $(find "$ARCHIVE_DIR" -name "archive_*.pkl" -type f 2>/dev/null | wc -l)"
    echo "  Validated: $(find "$VALIDATION_DIR" -name "*_validated.npz" -type f 2>/dev/null | wc -l)"
    echo "  Analysis plots: $(find "$ANALYSIS_DIR" -name "*.png" -type f 2>/dev/null | wc -l)"
}

# Clean results
clean_results() {
    echo "================================================================"
    echo "WARNING: CLEAN RESULTS"
    echo "================================================================"
    echo "This will DELETE all Experiment 6 results!"
    echo ""
    read -p "Are you sure? Type 'yes' to confirm: " confirm
    
    if [ "$confirm" == "yes" ]; then
        echo "Removing results/exp6_qd_comparison..."
        rm -rf results/exp6_qd_comparison
        echo "Done!"
    else
        echo "Cancelled"
    fi
}

# Main logic
case $PHASE in
    opt)
        run_optimization
        ;;
    validate)
        run_validation
        ;;
    analyze)
        run_analysis
        ;;
    all)
        echo "================================================================"
        echo "RUNNING FULL EXPERIMENT 6 PIPELINE"
        echo "================================================================"
        echo ""
        
        run_optimization
        echo ""
        echo "Waiting for optimization jobs to complete..."
        echo "Check status with: bash hpc/exp6_qd_comparison/run_exp6.sh status"
        echo ""
        echo "When optimization is complete, run:"
        echo "  bash hpc/exp6_qd_comparison/run_exp6.sh validate"
        echo "  bash hpc/exp6_qd_comparison/run_exp6.sh analyze"
        ;;
    status)
        check_status
        ;;
    clean)
        clean_results
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo ""
        usage
        exit 1
        ;;
esac
