#!/bin/bash
#SBATCH --job-name=gp_exp_eval
#SBATCH --output=logs/gp_eval_%j.out
#SBATCH --error=logs/gp_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# GP EXPERIMENT: Phase 4 - Evaluate GP Models
# ============================================================================
#
# Runs cross-domain evaluation on all trained GP models.
#
# Prerequisites:
#   1-3. Phases 1-3 must be complete
#
# Usage:
#   sbatch hpc/submit_gp_exp_evaluation.sh
#
# Expected outputs:
#   - results/gp_evaluation/cross_domain_results.csv
#   - results/gp_evaluation/summary_table.csv
#   - results/gp_evaluation/cross_domain_evaluation.png
#   - results/gp_evaluation/generalization_analysis.png
# ============================================================================

echo "=========================================="
echo "GP EXPERIMENT: Cross-Domain Evaluation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# Create output directory
mkdir -p results/exp1_gp_training_data/gp_evaluation
mkdir -p logs

# Activate your conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Check GPU
nvidia-smi || echo "No GPU found, using CPU"

# Run evaluation
python experiments/exp1_gp_training_data/evaluate_gp_experiment.py \
    --models-dir results/exp1_gp_training_data/gp_experiment \
    --data-dir results/exp1_gp_training_data/training_datasets \
    --output-dir results/exp1_gp_training_data/gp_evaluation

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation complete"
    echo ""
    echo "Results saved to:"
    ls -lh results/gp_evaluation/
else
    echo "✗ FAILED"
fi

echo "=========================================="
exit $EXIT_CODE
