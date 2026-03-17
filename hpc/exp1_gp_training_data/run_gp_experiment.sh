#!/bin/bash
# ============================================================================
# GP EXPERIMENT: Master Orchestration Script
# ============================================================================
#
# This script runs the complete GP training data comparison experiment:
#   Phase 1A: Generate SAIL (optimized) training data
#   Phase 1B: Generate Random (Sobol) training data  
#   Phase 2:  Prepare balanced training datasets
#   Phase 3:  Train GP models on each dataset
#   Phase 4:  Cross-domain evaluation
#
# Usage:
#   bash hpc/run_gp_experiment.sh [phase]
#
# Examples:
#   bash hpc/run_gp_experiment.sh all      # Run all phases
#   bash hpc/run_gp_experiment.sh 1        # Phase 1 only (data gen)
#   bash hpc/run_gp_experiment.sh 2        # Phase 2 only (prepare)
#   bash hpc/run_gp_experiment.sh 3        # Phase 3 only (train)
#   bash hpc/run_gp_experiment.sh 4        # Phase 4 only (evaluate)
#   bash hpc/run_gp_experiment.sh status   # Check job status
#   bash hpc/run_gp_experiment.sh dry      # Dry run (validate without running)
#
# ============================================================================

set -e

PHASE=${1:-status}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "GP Training Data Comparison Experiment"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo "Phase: $PHASE"
echo "Date: $(date)"
echo ""

# Create directories
mkdir -p logs
mkdir -p results/exp1_gp_training_data/sail_data
mkdir -p results/exp1_gp_training_data/random_data
mkdir -p results/exp1_gp_training_data/training_datasets
mkdir -p results/exp1_gp_training_data/gp_experiment
mkdir -p results/exp1_gp_training_data/gp_evaluation

run_phase_1() {
    echo "=============================================="
    echo "PHASE 1: Generate Training Data"
    echo "=============================================="
    
    echo ""
    echo "Phase 1A: Submitting SAIL data generation jobs..."
    SAIL_JOB=$(sbatch --parsable hpc/exp1_gp_training_data/submit_gp_exp_sail_data.sh)
    echo "  SAIL job ID: $SAIL_JOB (39 array tasks)"
    
    echo ""
    echo "Phase 1B: Submitting random data generation jobs..."
    RANDOM_JOB=$(sbatch --parsable hpc/exp1_gp_training_data/submit_gp_exp_random_data.sh)
    echo "  Random job ID: $RANDOM_JOB (13 array tasks)"
    
    echo ""
    echo "Jobs submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/sail_data_*.out"
    echo "  tail -f logs/random_data_*.out"
    echo ""
    echo "After completion, run: bash hpc/exp1_gp_training_data/run_gp_experiment.sh 2"
}

run_phase_2() {
    echo "=============================================="
    echo "PHASE 2: Prepare Training Datasets"
    echo "=============================================="
    
    # Check if Phase 1 data exists
    SAIL_COUNT=$(ls results/exp1_gp_training_data/sail_data/*.npz 2>/dev/null | wc -l)
    RANDOM_COUNT=$(ls results/exp1_gp_training_data/random_data/*.npz 2>/dev/null | wc -l)
    
    echo "Found $SAIL_COUNT SAIL data files"
    echo "Found $RANDOM_COUNT random data files"
    
    if [ "$SAIL_COUNT" -lt 12 ]; then
        echo "WARNING: Expected 12 SAIL files, found $SAIL_COUNT"
        echo "Phase 1A may not be complete."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    if [ "$RANDOM_COUNT" -lt 4 ]; then
        echo "WARNING: Expected 4 random files, found $RANDOM_COUNT"
        echo "Phase 1B may not be complete."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo ""
    echo "Running dataset preparation..."
    python experiments/exp1_gp_training_data/prepare_training_datasets.py \
        --sail-dir results/exp1_gp_training_data/sail_data \
        --random-dir results/exp1_gp_training_data/random_data \
        --output-dir results/exp1_gp_training_data/training_datasets \
        --val-fraction 0.2 \
        --seed 42
    
    echo ""
    echo "Datasets prepared. Run: bash hpc/exp1_gp_training_data/run_gp_experiment.sh 3"
}

run_phase_3() {
    echo "=============================================="
    echo "PHASE 3: Train GP Models"
    echo "=============================================="
    
    # Check if Phase 2 data exists
    if [ ! -f "results/exp1_gp_training_data/training_datasets/dataset_optimized.npz" ]; then
        echo "ERROR: Training datasets not found."
        echo "Run Phase 2 first: bash hpc/exp1_gp_training_data/run_gp_experiment.sh 2"
        exit 1
    fi
    
    echo "Submitting GP training jobs (9 individual jobs)..."
    bash hpc/exp1_gp_training_data/submit_all_gp_training.sh
    
    echo ""
    echo "Jobs submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/gp_train_*.out"
    echo ""
    echo "After completion, run: bash hpc/exp1_gp_training_data/run_gp_experiment.sh 4"
}

run_phase_4() {
    echo "=============================================="
    echo "PHASE 4: Evaluate GP Models"
    echo "=============================================="
    
    # Check if Phase 3 models exist
    MODEL_COUNT=$(ls results/exp1_gp_training_data/gp_experiment/*.pth 2>/dev/null | wc -l)
    echo "Found $MODEL_COUNT trained models"
    
    if [ "$MODEL_COUNT" -lt 9 ]; then
        echo "WARNING: Expected 9 models, found $MODEL_COUNT"
        echo "Phase 3 may not be complete."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo "Submitting evaluation job..."
    EVAL_JOB=$(sbatch --parsable hpc/exp1_gp_training_data/submit_gp_exp_evaluation.sh)
    echo "  Evaluation job ID: $EVAL_JOB"
    
    echo ""
    echo "Job submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/gp_eval_*.out"
}

show_status() {
    echo "=============================================="
    echo "EXPERIMENT STATUS"
    echo "=============================================="
    
    echo ""
    echo "Phase 1A - SAIL Data:"
    SAIL_COUNT=$(ls results/exp1_gp_training_data/sail_data/*.npz 2>/dev/null | wc -l)
    echo "  Files: $SAIL_COUNT / 12 expected"
    if [ "$SAIL_COUNT" -gt 0 ]; then
        echo "  Sizes with data:"
        for size in 60 120 240; do
            count=$(ls results/exp1_gp_training_data/sail_data/sail_${size}x${size}_rep*.npz 2>/dev/null | wc -l)
            if [ "$count" -gt 0 ]; then
                echo "    ${size}m: $count/3 replicates"
            fi
        done
    fi
    
    echo ""
    echo "Phase 1B - Random Data:"
    RANDOM_COUNT=$(ls results/exp1_gp_training_data/random_data/*.npz 2>/dev/null | wc -l)
    echo "  Files: $RANDOM_COUNT / 3 expected"
    if [ "$RANDOM_COUNT" -gt 0 ]; then
        echo "  Sizes with data:"
        for size in 60 120 240; do
            if ls results/exp1_gp_training_data/random_data/*_${size}m_*.npz 1>/dev/null 2>&1; then
                echo "    ${size}m: ✓"
            fi
        done
    fi
    
    echo ""
    echo "Phase 2 - Training Datasets:"
    for ds in optimized random combined; do
        if [ -f "results/training_datasets/dataset_${ds}.npz" ]; then
            echo "  ✓ dataset_${ds}.npz"
        else
            echo "  ✗ dataset_${ds}.npz (missing)"
        fi
    done
    
    echo ""
    echo "Phase 3 - Trained Models:"
    MODEL_COUNT=$(ls results/gp_experiment/*.pth 2>/dev/null | wc -l)
    echo "  Models: $MODEL_COUNT / 9 expected"
    if [ -d "results/gp_experiment" ]; then
        ls results/gp_experiment/*.pth 2>/dev/null | head -10
    fi
    
    echo ""
    echo "Phase 4 - Evaluation Results:"
    if [ -f "results/gp_evaluation/cross_domain_results.csv" ]; then
        echo "  ✓ cross_domain_results.csv"
        echo "  ✓ summary_table.csv"
        
        echo ""
        echo "Quick summary:"
        head -5 results/gp_evaluation/cross_domain_results.csv
    else
        echo "  ✗ Results not yet generated"
    fi
    
    echo ""
    echo "Active Jobs:"
    squeue -u $USER 2>/dev/null | grep -E "gp_exp|sail|random" || echo "  No active jobs"
}

run_dry_run() {
    echo "=============================================="
    echo "DRY RUN - Validating Experiment Setup"
    echo "=============================================="
    
    echo ""
    echo "KLAM_21 Configuration:"
    echo "--------------------------------------------"
    if [ -f "domain_description/cfg.yml" ]; then
        echo "  Wind Speed:     $(grep 'wind_speed:' domain_description/cfg.yml | awk '{print $2}') m/s"
        echo "  Wind Direction: $(grep 'wind_direction:' domain_description/cfg.yml | awk '{print $2}')°"
        echo "  Sim Duration:   $(grep 'sim_duration:' domain_description/cfg.yml | awk '{print $2}') seconds"
    else
        echo "  ⚠ cfg.yml not found!"
    fi
    
    echo ""
    if [ -f "domain_description/evaluation_klam.py" ]; then
        SLOPE_DEG=$(grep -A 1 "slope_angle_deg = " domain_description/evaluation_klam.py | grep "slope_angle_deg = " | head -1 | sed 's/.*= //' | sed 's/ .*//')
        SLOPE_TYPE=$(grep -B 2 "slope_angle_deg = " domain_description/evaluation_klam.py | grep "# Create terrain" | head -1)
        echo "  Terrain Slope:  ${SLOPE_DEG}°"
        echo "  Slope Type:     ${SLOPE_TYPE#*# Create terrain }"
    else
        echo "  ⚠ evaluation_klam.py not found!"
    fi
    
    echo ""
    echo "Experiment Configuration:"
    echo "--------------------------------------------"
    echo "  Parcel Sizes:   60, 120, 240 m (3 sizes)"
    echo "  SAIL Replicates: 3 per size (9 total)"
    echo "  Random Samples:  ~2000 per size (3 total)"
    echo "  Grid Resolution: 3m per cell (xy_scale)"
    echo "  Max Height:      5 floors × 3m = 15m (z_scale)"
    
    echo ""
    echo "Checking Python scripts..."
    
    # Test SAIL data generator
    echo "  Testing generate_sail_data.py..."
    python experiments/exp1_gp_training_data/generate_sail_data.py --parcel-size 60 --replicate 1 \
        --output-dir results/sail_data --dry-run 2>&1 | sed 's/^/    /'
    SAIL_EXIT=$?
    
    echo ""
    echo "  Testing generate_random_data.py..."
    python experiments/exp1_gp_training_data/generate_random_data.py --parcel-size 60 \
        --output-dir results/random_data --dry-run 2>&1 | sed 's/^/    /'
    RANDOM_EXIT=$?
    
    echo ""
    echo "SKIP STATUS CHECK (skip-existing is ON by default)"
    echo "=============================================="
    
    echo ""
    echo "SAIL Data (_klam.npz = real KLAM evaluation):"
    SAIL_WOULD_SKIP=0
    SAIL_WOULD_RUN=0
    SAIL_HAS_SOURCE=0
    SAIL_WOULD_RESUME=0
    for size in 60 120 240; do
        for rep in 1 2 3; do
            klam_file="results/exp1_gp_training_data/sail_data/sail_${size}x${size}_rep${rep}_klam.npz"
            old_file="results/exp1_gp_training_data/sail_data/sail_${size}x${size}_rep${rep}.npz"
            run_dir="results/exp1_gp_training_data/sail_data/sail_${size}x${size}_rep${rep}"
            
            if [ -f "$klam_file" ]; then
                # Final _klam.npz exists - will skip
                if python -c "import numpy as np; np.load('$klam_file')" 2>/dev/null; then
                    echo "  ✓ SKIP: ${size}m rep${rep} (_klam.npz exists, valid)"
                    SAIL_WOULD_SKIP=$((SAIL_WOULD_SKIP + 1))
                else
                    echo "  ⚠ RUN:  ${size}m rep${rep} (_klam.npz CORRUPTED)"
                    SAIL_WOULD_RUN=$((SAIL_WOULD_RUN + 1))
                fi
            elif [ -f "$old_file" ]; then
                # Old surrogate .npz exists - will load and evaluate with KLAM
                if python -c "import numpy as np; np.load('$old_file')" 2>/dev/null; then
                    echo "  → EVAL: ${size}m rep${rep} (has old .npz, needs KLAM eval)"
                    SAIL_WOULD_RUN=$((SAIL_WOULD_RUN + 1))
                    SAIL_HAS_SOURCE=$((SAIL_HAS_SOURCE + 1))
                else
                    echo "  ⚠ RUN:  ${size}m rep${rep} (old .npz CORRUPTED)"
                    SAIL_WOULD_RUN=$((SAIL_WOULD_RUN + 1))
                fi
            elif [ -d "$run_dir" ]; then
                # Check for archive files with timestamp prefix
                archive_files=$(ls "$run_dir"/*FinalQD_archive.pkl 2>/dev/null | wc -l)
                if [ "$archive_files" -gt 0 ]; then
                    echo "  ↻ RESUME: ${size}m rep${rep} (has incomplete archive, will resume SAIL)"
                    SAIL_WOULD_RUN=$((SAIL_WOULD_RUN + 1))
                    SAIL_WOULD_RESUME=$((SAIL_WOULD_RESUME + 1))
                else
                    echo "  → SAIL: ${size}m rep${rep} (run dir exists but no archive)"
                    SAIL_WOULD_RUN=$((SAIL_WOULD_RUN + 1))
                fi
            else
                echo "  → SAIL: ${size}m rep${rep} (no data, needs full SAIL run)"
                SAIL_WOULD_RUN=$((SAIL_WOULD_RUN + 1))
            fi
        done
    done
    echo "  Summary: Would skip $SAIL_WOULD_SKIP, would run $SAIL_WOULD_RUN (of 9 total)"
    echo "           $SAIL_HAS_SOURCE have existing genomes (KLAM eval only, no SAIL needed)"
    echo "           $SAIL_WOULD_RESUME have incomplete archives (will resume SAIL)"
    
    echo ""
    echo "Random Data:"
    RANDOM_WOULD_SKIP=0
    RANDOM_WOULD_RUN=0
    for size in 60 120 240; do
        # Find the file with pattern matching (seed may vary)
        file=$(ls results/exp1_gp_training_data/random_data/random_sobol_${size}m_*.npz 2>/dev/null | head -1)
        if [ -n "$file" ] && [ -f "$file" ]; then
            if python -c "import numpy as np; np.load('$file')" 2>/dev/null; then
                echo "  ✓ SKIP: ${size}m (exists, valid)"
                RANDOM_WOULD_SKIP=$((RANDOM_WOULD_SKIP + 1))
            else
                echo "  ⚠ RUN:  ${size}m (exists, CORRUPTED)"
                RANDOM_WOULD_RUN=$((RANDOM_WOULD_RUN + 1))
            fi
        else
            echo "  → RUN:  ${size}m (not found)"
            RANDOM_WOULD_RUN=$((RANDOM_WOULD_RUN + 1))
        fi
    done
    echo "  Summary: Would skip $RANDOM_WOULD_SKIP, would run $RANDOM_WOULD_RUN (of 4 total)"
    
    echo ""
    echo "=============================================="
    echo "DRY RUN SUMMARY"
    echo "=============================================="
    
    if [ $SAIL_EXIT -eq 0 ]; then
        echo "  ✓ SAIL data generator: OK"
    else
        echo "  ✗ SAIL data generator: FAILED (exit code $SAIL_EXIT)"
    fi
    
    if [ $RANDOM_EXIT -eq 0 ]; then
        echo "  ✓ Random data generator: OK"
    else
        echo "  ✗ Random data generator: FAILED (exit code $RANDOM_EXIT)"
    fi
    
    echo ""
    TOTAL_WOULD_SKIP=$((SAIL_WOULD_SKIP + RANDOM_WOULD_SKIP))
    TOTAL_WOULD_RUN=$((SAIL_WOULD_RUN + RANDOM_WOULD_RUN))
    echo "  Phase 1 jobs to skip: $TOTAL_WOULD_SKIP"
    echo "  Phase 1 jobs to run:  $TOTAL_WOULD_RUN"
    
    echo ""
    if [ $SAIL_EXIT -eq 0 ] && [ $RANDOM_EXIT -eq 0 ]; then
        echo "✓ All validations passed! Ready to run experiment."
        echo ""
        echo "To run the full experiment:"
        echo "  bash hpc/run_gp_experiment.sh 1   # Submit Phase 1 jobs"
        if [ $TOTAL_WOULD_RUN -eq 0 ]; then
            echo ""
            echo "NOTE: All Phase 1 data already exists! You can skip to Phase 2:"
            echo "  bash hpc/run_gp_experiment.sh 2"
        fi
    else
        echo "✗ Some validations failed. Fix errors before running."
    fi
}

case $PHASE in
    1)
        run_phase_1
        ;;
    2)
        run_phase_2
        ;;
    3)
        run_phase_3
        ;;
    4)
        run_phase_4
        ;;
    all)
        run_phase_1
        echo ""
        echo "Phase 1 jobs submitted."
        echo "Wait for Phase 1 to complete, then run:"
        echo "  bash hpc/run_gp_experiment.sh 2"
        echo "  bash hpc/run_gp_experiment.sh 3"
        echo "  bash hpc/run_gp_experiment.sh 4"
        ;;
    status)
        show_status
        ;;
    dry|dry-run|dryrun)
        run_dry_run
        ;;
    *)
        echo "Usage: bash hpc/run_gp_experiment.sh [phase]"
        echo ""
        echo "Phases:"
        echo "  1       - Phase 1: Generate training data (SAIL + Random)"
        echo "  2       - Phase 2: Prepare balanced datasets"
        echo "  3       - Phase 3: Train GP models"
        echo "  4       - Phase 4: Cross-domain evaluation"
        echo "  all     - Submit all phases (with pauses)"
        echo "  status  - Show experiment status"
        echo "  dry     - Dry run: validate scripts without running simulations"
        ;;
esac

echo ""
echo "=============================================="
