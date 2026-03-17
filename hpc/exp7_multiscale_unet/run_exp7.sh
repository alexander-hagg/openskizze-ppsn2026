#!/bin/bash

# Experiment 7: Multi-Scale U-Net Comparison
# Master script for running the full experiment

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
    single      Train single-scale U-Nets (13 jobs, array 0-12)
    multi       Train multi-scale U-Net (1 job)
    both        Train both (14 jobs total)
    analyze     Analyze results
    all         Run all phases sequentially
    status      Check job status

Options:
    --dry-run   Print commands without executing

Examples:
    # Run full experiment
    bash hpc/exp7_multiscale_unet/run_exp7.sh all
    
    # Run individual phases
    bash hpc/exp7_multiscale_unet/run_exp7.sh single
    bash hpc/exp7_multiscale_unet/run_exp7.sh multi
    
    # Check status
    bash hpc/exp7_multiscale_unet/run_exp7.sh status
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

# Phase 1: Train single-scale models
train_single() {
    echo "================================================================"
    echo "TRAINING SINGLE-SCALE U-NETS"
    echo "================================================================"
    echo "Submitting 3 jobs (all parcel sizes: 60m to 240m)..."
    echo ""
    
    run_cmd sbatch "$SCRIPT_DIR/submit_train_single.sh"
    
    echo ""
    echo "Jobs submitted! Monitor with: squeue -u $USER"
    echo "Expected completion time: ~2-6 hours per model (varies by size)"
}

# Phase 2: Train multi-scale model
train_multi() {
    echo "================================================================"
    echo "TRAINING MULTI-SCALE U-NET"
    echo "================================================================"
    echo "Submitting 1 job (mixed 60m, 120m, 240m)..."
    echo ""
    
    run_cmd sbatch "$SCRIPT_DIR/submit_train_multi.sh"
    
    echo ""
    echo "Job submitted! Monitor with: squeue -u $USER"
    echo "Expected completion time: ~4-6 hours"
}

# Phase 3: Analyze results
analyze_results() {
    echo "================================================================"
    echo "ANALYZING RESULTS"
    echo "================================================================"
    echo ""
    
    # Check if results exist
    RESULTS_FILE="results/exp7_multiscale_unet/training_results.json"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "ERROR: Results file not found: $RESULTS_FILE"
        echo "Please run training phases first!"
        exit 1
    fi
    
    run_cmd python experiments/exp7_multiscale_unet/analyze_results.py \
        --results-file "$RESULTS_FILE" \
        --output-dir results/exp7_multiscale_unet/analysis
    
    echo ""
    echo "Analysis complete!"
}

# Check job status
check_status() {
    echo "================================================================"
    echo "JOB STATUS"
    echo "================================================================"
    echo ""
    
    echo "Single-scale U-Net jobs:"
    squeue -u $USER --name=exp7_single --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" || echo "No jobs found"
    
    echo ""
    echo "Multi-scale U-Net job:"
    squeue -u $USER --name=exp7_multi --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" || echo "No jobs found"
    
    echo ""
    echo "Recent completions (last 24h):"
    sacct -u $USER --starttime=$(date -d '1 day ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,MaxRSS | grep exp7 || echo "No completed jobs"
}

# Main execution
case $PHASE in
    single)
        train_single
        ;;
    multi)
        train_multi
        ;;
    both)
        train_single
        train_multi
        ;;
    analyze)
        analyze_results
        ;;
    all)
        train_single
        echo ""
        echo "Waiting for single-scale jobs to complete before starting multi-scale..."
        echo "You can manually submit multi-scale after single-scale jobs finish with:"
        echo "  bash hpc/exp7_multiscale_unet/run_exp7.sh multi"
        ;;
    status)
        check_status
        ;;
    help|*)
        usage
        ;;
esac
