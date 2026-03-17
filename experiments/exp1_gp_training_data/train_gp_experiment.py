#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Phase 3: Train GP Experiment

This script trains SVGP models for the experiment, supporting:
1. Training on Optimized (SAIL), Random (Sobol), or Combined datasets
2. Consistent hyperparameters across all runs
3. Saving models with experiment metadata

Usage:
    python experiments/train_gp_experiment.py --dataset optimized
    python experiments/train_gp_experiment.py --dataset random
    python experiments/train_gp_experiment.py --dataset combined
    
For HPC SLURM jobs:
    python experiments/train_gp_experiment.py --dataset $DATASET --replicate $REP
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import time

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SVGPModel(ApproximateGP):
    """
    Sparse Variational GP for KLAM_21 surrogate.
    
    Uses inducing points for scalability and Matern 2.5 kernel with ARD.
    """
    
    def __init__(self, inducing_points, input_dim=62):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=input_dim,
            )
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GP experiment on prepared datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['optimized', 'random', 'combined'],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/training_datasets",
        help="Directory containing prepared datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/gp_experiment",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Replicate number (for multiple runs)"
    )
    parser.add_argument(
        "--num-inducing",
        type=int,
        default=500,
        help="Number of inducing points (default: 500)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size (default: 1024)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    return parser.parse_args()


def load_dataset(data_dir: Path, dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Load prepared dataset.
    
    Returns dict with train/val splits and all arrays.
    """
    dataset_path = data_dir / f"dataset_{dataset_name}.npz"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = np.load(dataset_path)
    
    result = {
        'train': {
            'genomes': data['train_genomes'],
            'widths': data['train_widths'],
            'heights': data['train_heights'],
            'objectives': data['train_objectives'],
            'features': data['train_features'],
            'parcel_sizes': data['train_parcel_sizes'],
        },
        'val': {
            'genomes': data['val_genomes'],
            'widths': data['val_widths'],
            'heights': data['val_heights'],
            'objectives': data['val_objectives'],
            'features': data['val_features'],
            'parcel_sizes': data['val_parcel_sizes'],
        }
    }
    
    logger.info(f"Loaded dataset: {dataset_name}")
    logger.info(f"  Train: {len(result['train']['objectives'])} samples")
    logger.info(f"  Val: {len(result['val']['objectives'])} samples")
    
    return result


def prepare_inputs(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare inputs for GP training.
    
    Input format: [genome (60), parcel_width (1), parcel_height (1)] = 62 dims
    
    Note: We use parcel_size for both width and height (square parcels).
    """
    genomes = data['genomes']
    parcel_sizes = data['parcel_sizes'].reshape(-1, 1)
    
    # X: [genome, width, height] - width=height=parcel_size for square parcels
    X = np.column_stack([genomes, parcel_sizes, parcel_sizes]).astype(np.float32)
    y = data['objectives'].astype(np.float32)
    
    return X, y


def train_svgp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_inducing: int = 500,
    num_epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 0.01,
    seed: int = 42,
    use_gpu: bool = True
) -> Dict:
    """
    Train SVGP model.
    
    Returns dict with model, likelihood, losses, and metrics.
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Inducing points: {num_inducing}")
    logger.info(f"Batch size: {batch_size}")
    
    input_dim = X_train.shape[1]
    
    # Convert to tensors
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    
    # Normalize inputs
    train_x_mean = train_x.mean(dim=0)
    train_x_std = train_x.std(dim=0)
    train_x_std[train_x_std < 1e-6] = 1.0
    train_x_normalized = (train_x - train_x_mean) / train_x_std
    
    # Normalize outputs
    train_y_mean = train_y.mean()
    train_y_std = train_y.std()
    train_y_normalized = (train_y - train_y_mean) / train_y_std
    
    # DataLoader
    train_dataset = TensorDataset(train_x_normalized, train_y_normalized)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize inducing points
    inducing_idx = np.random.choice(
        len(train_x_normalized), 
        min(num_inducing, len(train_x_normalized)), 
        replace=False
    )
    inducing_points = train_x_normalized[inducing_idx].to(device)
    
    # Model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SVGPModel(inducing_points, input_dim=input_dim).to(device)
    
    model.train()
    likelihood.train()
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Loss
    mll = VariationalELBO(likelihood, model, num_data=len(train_y))
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    losses = []
    best_loss = float('inf')
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        scheduler.step()
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict()
            }
        
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                       f"Time: {elapsed:.1f}s")
    
    # Restore best model
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    model.eval()
    likelihood.eval()
    
    # Store normalization parameters
    model.train_x_mean = train_x_mean.to(device)
    model.train_x_std = train_x_std.to(device)
    model.train_y_mean = train_y_mean.to(device)
    model.train_y_std = train_y_std.to(device)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, likelihood, X_val, y_val)
    
    return {
        'model': model,
        'likelihood': likelihood,
        'losses': losses,
        'val_metrics': val_metrics,
        'training_time': training_time,
        'best_loss': best_loss
    }


def evaluate_model(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024
) -> Dict:
    """
    Evaluate model and compute all metrics.
    """
    device = next(model.parameters()).device
    
    test_x = torch.tensor(X, dtype=torch.float32).to(device)
    test_y = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Normalize inputs
    test_x_normalized = (test_x - model.train_x_mean) / model.train_x_std
    
    # Predict in batches
    pred_means = []
    pred_stds = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(test_x_normalized), batch_size):
            batch_x = test_x_normalized[i:i+batch_size]
            pred = likelihood(model(batch_x))
            
            # Un-normalize predictions
            pred_mean = pred.mean * model.train_y_std + model.train_y_mean
            pred_std = pred.stddev * model.train_y_std
            
            pred_means.append(pred_mean.cpu())
            pred_stds.append(pred_std.cpu())
    
    pred_mean = torch.cat(pred_means).numpy()
    pred_std = torch.cat(pred_stds).numpy()
    y_np = test_y.cpu().numpy()
    
    # Metrics
    mse = np.mean((pred_mean - y_np)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - y_np))
    r2 = 1 - mse / np.var(y_np)
    
    # Correlation
    pearson_r, pearson_p = pearsonr(pred_mean, y_np)
    spearman_rho, spearman_p = spearmanr(pred_mean, y_np)
    
    # Calibration
    ci_lower = pred_mean - 1.96 * pred_std
    ci_upper = pred_mean + 1.96 * pred_std
    in_ci = (y_np >= ci_lower) & (y_np <= ci_upper)
    calibration = in_ci.mean()
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'calibration_95ci': float(calibration),
        'n_samples': len(y_np),
        'predictions': pred_mean,
        'uncertainties': pred_std,
        'true_values': y_np
    }
    
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  Pearson r: {pearson_r:.4f}")
    logger.info(f"  Spearman ρ: {spearman_rho:.4f}")
    logger.info(f"  Calibration (95% CI): {calibration:.1%}")
    
    return metrics


def save_experiment_results(
    output_dir: Path,
    dataset_name: str,
    replicate: int,
    model: SVGPModel,
    likelihood,
    results: Dict,
    hyperparams: Dict
):
    """
    Save model and results for experiment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model filename
    model_name = f"gp_{dataset_name}_rep{replicate}"
    model_path = output_dir / f"{model_name}.pth"
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'train_x_mean': model.train_x_mean.cpu(),
        'train_x_std': model.train_x_std.cpu(),
        'train_y_mean': model.train_y_mean.cpu(),
        'train_y_std': model.train_y_std.cpu(),
    }, model_path)
    
    logger.info(f"Saved model to {model_path}")
    
    # Save metrics (without arrays for JSON)
    metrics_for_json = {
        k: v for k, v in results['val_metrics'].items()
        if not isinstance(v, np.ndarray)
    }
    
    metrics = {
        'dataset': dataset_name,
        'replicate': replicate,
        'hyperparams': hyperparams,
        'training_time': results['training_time'],
        'best_loss': results['best_loss'],
        'val_metrics': metrics_for_json,
        'losses': results['losses']
    }
    
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save predictions for later analysis
    predictions_path = output_dir / f"{model_name}_predictions.npz"
    np.savez(
        predictions_path,
        predictions=results['val_metrics']['predictions'],
        uncertainties=results['val_metrics']['uncertainties'],
        true_values=results['val_metrics']['true_values']
    )
    
    logger.info(f"Saved predictions to {predictions_path}")


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 3: GP Experiment Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Replicate: {args.replicate}")
    logger.info(f"Seed: {args.seed + args.replicate}")
    
    # Set seed for this replicate
    seed = args.seed + args.replicate
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = load_dataset(data_dir, args.dataset)
    
    # Prepare inputs
    logger.info("\nPreparing inputs...")
    X_train, y_train = prepare_inputs(dataset['train'])
    X_val, y_val = prepare_inputs(dataset['val'])
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Train objective range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    logger.info(f"Val objective range: [{y_val.min():.2f}, {y_val.max():.2f}]")
    
    # Hyperparameters
    hyperparams = {
        'num_inducing': args.num_inducing,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': seed,
        'input_dim': X_train.shape[1]
    }
    
    # Train model
    logger.info("\nTraining SVGP model...")
    results = train_svgp(
        X_train, y_train,
        X_val, y_val,
        num_inducing=args.num_inducing,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed,
        use_gpu=not args.no_gpu
    )
    
    # Save results
    logger.info("\nSaving results...")
    save_experiment_results(
        output_dir=output_dir,
        dataset_name=args.dataset,
        replicate=args.replicate,
        model=results['model'],
        likelihood=results['likelihood'],
        results=results,
        hyperparams=hyperparams
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
