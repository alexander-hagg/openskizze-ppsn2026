#!/bin/bash
#SBATCH --job-name=exp6_analyze
#SBATCH --output=logs/exp6_analyze_%j.out
#SBATCH --error=logs/exp6_analyze_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=any

# Experiment 6: Analyze QD comparison results

set -e

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
source ~/.bashrc
conda activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

# Directories
VALIDATION_DIR="results/exp6_qd_comparison/validation"
OUTPUT_DIR="results/exp6_qd_comparison/analysis"

echo "Analyzing validation results from: $VALIDATION_DIR"

# Run analysis
python experiments/exp6_qd_comparison/analyze_qd_comparison.py \
    --results-dir $VALIDATION_DIR \
    --output-dir $OUTPUT_DIR

echo "Job completed at $(date)"
