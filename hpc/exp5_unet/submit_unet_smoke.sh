#!/bin/bash
#SBATCH --job-name=unet_smoke
#SBATCH --output=logs/unet_smoke_%j.out
#SBATCH --error=logs/unet_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# U-NET Smoke Test on GPU Node
# ============================================================================
# Quick test to verify U-NET training works before submitting full experiment.
#
# Usage:
#   sbatch hpc/submit_unet_smoke.sh
# ============================================================================

echo "======================================================================"
echo "U-NET KLAM_21 Smoke Test"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Create logs directory
mkdir -p logs
mkdir -p results/exp5_unet/unet_experiment

# Configuration
SAIL_DATA_DIR="${SAIL_DATA_DIR:-results/exp1_gp_training_data/sail_data}"
PARCEL_SIZE=60
EPOCHS=10
BATCH_SIZE=16

echo "Configuration:"
echo "  Data dir:      $SAIL_DATA_DIR"
echo "  Parcel size:   $PARCEL_SIZE"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo ""

# Check GPU
echo "----------------------------------------------------------------------"
echo "GPU Check"
echo "----------------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Test 1: Model architecture
echo "----------------------------------------------------------------------"
echo "Test 1: U-NET Architecture"
echo "----------------------------------------------------------------------"
python -c "
from experiments.models.unet import UNet, UNetConfig, get_loss_function
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create model
config = UNetConfig(input_height=66, input_width=94)
model = UNet(config).to(device)
print(model.summary())

# Test forward pass
x = torch.randn(2, 3, 66, 94).to(device)
y = model(x)
print(f'Input:  {x.shape}')
print(f'Output: {y.shape}')
assert y.shape == (2, 6, 66, 94), 'Output shape mismatch!'

# Test losses
target = torch.randn(2, 6, 66, 94).to(device)
mse = get_loss_function('mse')
mse_grad = get_loss_function('mse_grad', 0.5)
print(f'MSE loss: {mse(y, target):.4f}')
print(f'MSE+Grad loss: {mse_grad(y, target):.4f}')
print('✓ Model architecture test passed!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Model architecture test failed!"
    exit 1
fi
echo ""

# Test 2: Data loading
echo "----------------------------------------------------------------------"
echo "Test 2: Data Loading"
echo "----------------------------------------------------------------------"
python -c "
import numpy as np
from pathlib import Path

data_dir = Path('$SAIL_DATA_DIR')
f = data_dir / 'sail_60x60_rep1_spatial.npz'
print(f'Loading {f.name}...')
data = np.load(f)

print('Keys:', list(data.keys()))
print(f'terrain shape: {data[\"terrain\"].shape}')
print(f'buildings shape: {data[\"buildings\"].shape}')
print(f'landuse shape: {data[\"landuse\"].shape}')
print(f'Ex shape: {data[\"Ex\"].shape}')
print(f'timestamps: {data[\"timestamps\"]}')
print('✓ Data loading test passed!')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Data loading test failed!"
    exit 1
fi
echo ""

# Test 3: Full training smoke test (MSE)
echo "----------------------------------------------------------------------"
echo "Test 3: Training Smoke Test - MSE Loss ($EPOCHS epochs)"
echo "----------------------------------------------------------------------"
python experiments/exp5_unet/train_unet_klam.py \
    --data-type sail \
    --data-dir "$SAIL_DATA_DIR" \
    --parcel-size "$PARCEL_SIZE" \
    --loss mse \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed 42 \
    --output-dir results/unet_experiment/smoke_test_mse

if [ $? -ne 0 ]; then
    echo "ERROR: MSE training smoke test failed!"
    exit 1
fi
echo ""

# Test 4: Full training smoke test (MSE+Grad)
echo "----------------------------------------------------------------------"
echo "Test 4: Training Smoke Test - MSE+Gradient Loss ($EPOCHS epochs)"
echo "----------------------------------------------------------------------"
python experiments/exp5_unet/train_unet_klam.py \
    --data-type sail \
    --data-dir "$SAIL_DATA_DIR" \
    --parcel-size "$PARCEL_SIZE" \
    --loss mse_grad \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed 42 \
    --output-dir results/exp5_unet/unet_experiment/smoke_test_mse_grad

if [ $? -ne 0 ]; then
    echo "ERROR: MSE+Gradient training smoke test failed!"
    exit 1
fi
echo ""

# Summary
echo "======================================================================"
echo "Smoke Test Results"
echo "======================================================================"

echo ""
echo "MSE Loss Results:"
cat results/exp5_unet/unet_experiment/smoke_test_mse/results.json | python -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Best epoch: {data['best_epoch']}\")
print(f\"  Best val loss: {data['best_val_loss']:.6f}\")
print(f\"  Test R² (overall): {data['test_metrics']['overall']['r2']:.4f}\")
for var in ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']:
    print(f\"    {var}: R²={data['test_metrics'][var]['r2']:.3f}\")
"

echo ""
echo "MSE+Gradient Loss Results:"
cat results/exp5_unet/unet_experiment/smoke_test_mse_grad/results.json | python -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Best epoch: {data['best_epoch']}\")
print(f\"  Best val loss: {data['best_val_loss']:.6f}\")
print(f\"  Test R² (overall): {data['test_metrics']['overall']['r2']:.4f}\")
for var in ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']:
    print(f\"    {var}: R²={data['test_metrics'][var]['r2']:.3f}\")
"

echo ""
echo "======================================================================"
echo "All Smoke Tests Passed!"
echo "======================================================================"
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Review results above"
echo "  2. If satisfactory, submit full experiment:"
echo "     sbatch hpc/submit_unet_training.sh"
echo ""
