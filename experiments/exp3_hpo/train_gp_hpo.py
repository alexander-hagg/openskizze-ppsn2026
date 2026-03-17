#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GP Hyperparameter Optimization Training Script

This script trains SVGP models with various hyperparameter configurations
for systematic comparison. Features:
- Early stopping with patience
- Learning rate warmup
- K-means inducing point initialization (optional)
- Proper train/val/test split

Hyperparameters explored:
- Number of inducing points: 500, 1000, 2000
- K-means initialization: True/False

Fixed settings:
- Max epochs: 200
- Early stopping patience: 20 epochs (validation loss)
- LR warmup: 10 epochs (linear)
- Batch size: 1024
- Learning rate: 0.01
- ARD Matern 2.5 kernel (62 lengthscales)

Usage:
    python experiments/train_gp_hpo.py \\
        --dataset optimized \\
        --num-inducing 1000 \\
        --kmeans-init \\
        --replicate 1

For SLURM array jobs:
    See hpc/submit_gp_hpo.sh
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Definition
# ============================================================================

class SVGPModel(ApproximateGP):
    """
    Sparse Variational GP for KLAM_21 surrogate.
    
    Uses inducing points for scalability and Matern 2.5 kernel with ARD.
    """
    
    def __init__(self, inducing_points: torch.Tensor, input_dim: int = 62):
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


# ============================================================================
# Learning Rate Scheduler with Warmup
# ============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self) -> float:
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def get_last_lr(self) -> float:
        return self._get_lr()


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Early stopping based on validation loss.
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.should_stop = False
        
    def __call__(
        self, 
        val_loss: float, 
        epoch: int, 
        model_state: Dict,
        likelihood_state: Dict
    ) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {
                'model': {k: v.cpu().clone() for k, v in model_state.items()},
                'likelihood': {k: v.cpu().clone() for k, v in likelihood_state.items()}
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GP with hyperparameter optimization"
    )
    
    # Dataset
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
    
    # Hyperparameters to optimize
    parser.add_argument(
        "--num-inducing",
        type=int,
        required=True,
        # choices=[100, 500, 1000, 2500, 5000],
        help="Number of inducing points"
    )
    parser.add_argument(
        "--kmeans-init",
        action="store_true",
        help="Use K-means initialization for inducing points"
    )
    
    # Fixed hyperparameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20)"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Learning rate warmup epochs (default: 10)"
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
        help="Base learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of data for test set (default: 0.15)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of remaining data for validation (default: 0.15 of remaining = ~12.75% total)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparameterization",
        help="Output directory for results"
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Replicate number (1-3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (actual seed = base + replicate)"
    )
    
    # GPU
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    return parser.parse_args()


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_dataset(data_dir: Path, dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Load prepared dataset and combine train+val for re-splitting.
    """
    dataset_path = data_dir / f"dataset_{dataset_name}.npz"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = np.load(dataset_path)
    
    # Combine train and val for proper re-splitting
    result = {
        'genomes': np.vstack([data['train_genomes'], data['val_genomes']]),
        'widths': np.concatenate([data['train_widths'], data['val_widths']]),
        'heights': np.concatenate([data['train_heights'], data['val_heights']]),
        'objectives': np.concatenate([data['train_objectives'], data['val_objectives']]),
        'features': np.vstack([data['train_features'], data['val_features']]),
        'parcel_sizes': np.concatenate([data['train_parcel_sizes'], data['val_parcel_sizes']]),
    }
    
    logger.info(f"Loaded dataset: {dataset_name}")
    logger.info(f"  Total samples: {len(result['objectives'])}")
    
    return result


def prepare_train_val_test_split(
    data: Dict[str, np.ndarray],
    test_fraction: float,
    val_fraction: float,
    seed: int
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train/val/test sets.
    
    First splits off test set, then splits remaining into train/val.
    """
    n_total = len(data['objectives'])
    indices = np.arange(n_total)
    
    # Split off test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed
    )
    
    # Split remaining into train/val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction / (1 - test_fraction),  # Adjust fraction
        random_state=seed
    )
    
    def extract_split(idx):
        return {k: v[idx] for k, v in data.items()}
    
    train_data = extract_split(train_idx)
    val_data = extract_split(val_idx)
    test_data = extract_split(test_idx)
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_idx)} samples ({100*len(train_idx)/n_total:.1f}%)")
    logger.info(f"  Val:   {len(val_idx)} samples ({100*len(val_idx)/n_total:.1f}%)")
    logger.info(f"  Test:  {len(test_idx)} samples ({100*len(test_idx)/n_total:.1f}%)")
    
    return train_data, val_data, test_data


def prepare_inputs(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare inputs for GP training.
    
    Input format: [genome (60), parcel_width (1), parcel_height (1)] = 62 dims
    """
    genomes = data['genomes']
    parcel_sizes = data['parcel_sizes'].reshape(-1, 1)
    
    X = np.column_stack([genomes, parcel_sizes, parcel_sizes]).astype(np.float32)
    y = data['objectives'].astype(np.float32)
    
    return X, y


# ============================================================================
# Inducing Point Initialization
# ============================================================================

def initialize_inducing_points(
    X_train: torch.Tensor,
    num_inducing: int,
    use_kmeans: bool,
    seed: int
) -> torch.Tensor:
    """
    Initialize inducing points either randomly or via K-means.
    """
    n_train = len(X_train)
    num_inducing = min(num_inducing, n_train)
    
    if use_kmeans:
        logger.info(f"Initializing {num_inducing} inducing points with MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(
            n_clusters=num_inducing,
            random_state=seed,
            batch_size=min(1024, n_train),
            n_init=3,
            max_iter=100
        )
        kmeans.fit(X_train.cpu().numpy())
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        logger.info(f"  K-means inertia: {kmeans.inertia_:.2f}")
    else:
        logger.info(f"Initializing {num_inducing} inducing points randomly...")
        np.random.seed(seed)
        idx = np.random.choice(n_train, num_inducing, replace=False)
        inducing_points = X_train[idx].clone()
    
    return inducing_points


# ============================================================================
# Training
# ============================================================================

def train_svgp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_inducing: int = 500,
    use_kmeans: bool = False,
    num_epochs: int = 200,
    patience: int = 20,
    warmup_epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.01,
    seed: int = 42,
    use_gpu: bool = True
) -> Dict:
    """
    Train SVGP model with early stopping and warmup.
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    input_dim = X_train.shape[1]
    
    # Convert to tensors
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)
    
    # Normalize inputs
    train_x_mean = train_x.mean(dim=0)
    train_x_std = train_x.std(dim=0)
    train_x_std[train_x_std < 1e-6] = 1.0
    train_x_normalized = (train_x - train_x_mean) / train_x_std
    val_x_normalized = (val_x - train_x_mean) / train_x_std
    
    # Normalize outputs
    train_y_mean = train_y.mean()
    train_y_std = train_y.std()
    train_y_normalized = (train_y - train_y_mean) / train_y_std
    val_y_normalized = (val_y - train_y_mean) / train_y_std
    
    # DataLoader
    train_dataset = TensorDataset(train_x_normalized, train_y_normalized)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize inducing points
    inducing_points = initialize_inducing_points(
        train_x_normalized, num_inducing, use_kmeans, seed
    ).to(device)
    
    # Model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SVGPModel(inducing_points, input_dim=input_dim).to(device)
    
    # Move validation data to device
    val_x_normalized = val_x_normalized.to(device)
    val_y_normalized = val_y_normalized.to(device)
    
    model.train()
    likelihood.train()
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=learning_rate)
    
    # Scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs, num_epochs, learning_rate
    )
    
    # Loss
    mll = VariationalELBO(likelihood, model, num_data=len(train_y))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training loop
    logger.info(f"Starting training (max {num_epochs} epochs, patience {patience})...")
    train_losses = []
    val_losses = []
    learning_rates = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        likelihood.train()
        
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
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            val_output = model(val_x_normalized)
            val_loss = -mll(val_output, val_y_normalized).item()
        
        val_losses.append(val_loss)
        learning_rates.append(scheduler.get_last_lr())
        
        # Learning rate step
        scheduler.step()
        
        # Early stopping check
        if early_stopping(val_loss, epoch, model.state_dict(), likelihood.state_dict()):
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"  Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {learning_rates[-1]:.5f} | Time: {elapsed:.1f}s"
            )
    
    # Restore best model
    if early_stopping.best_state is not None:
        model.load_state_dict({
            k: v.to(device) for k, v in early_stopping.best_state['model'].items()
        })
        likelihood.load_state_dict({
            k: v.to(device) for k, v in early_stopping.best_state['likelihood'].items()
        })
        logger.info(f"Restored best model from epoch {early_stopping.best_epoch + 1}")
    
    model.eval()
    likelihood.eval()
    
    # Store normalization parameters
    model.train_x_mean = train_x_mean.to(device)
    model.train_x_std = train_x_std.to(device)
    model.train_y_mean = train_y_mean.to(device)
    model.train_y_std = train_y_std.to(device)
    
    training_time = time.time() - start_time
    epochs_trained = len(train_losses)
    
    logger.info(f"Training completed: {epochs_trained} epochs in {training_time:.1f}s")
    
    return {
        'model': model,
        'likelihood': likelihood,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'training_time': training_time,
        'epochs_trained': epochs_trained,
        'best_epoch': early_stopping.best_epoch + 1,
        'best_val_loss': early_stopping.best_loss,
    }


# ============================================================================
# Evaluation
# ============================================================================

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
    
    model.eval()
    likelihood.eval()
    
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
    
    # Handle edge case of zero variance
    y_var = np.var(y_np)
    r2 = 1 - mse / y_var if y_var > 1e-10 else 0.0
    
    # Correlation
    try:
        pearson_r, pearson_p = pearsonr(pred_mean, y_np)
        spearman_rho, spearman_p = spearmanr(pred_mean, y_np)
    except:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_rho, spearman_p = np.nan, np.nan
    
    # Calibration (95% CI)
    ci_lower = pred_mean - 1.96 * pred_std
    ci_upper = pred_mean + 1.96 * pred_std
    in_ci = (y_np >= ci_lower) & (y_np <= ci_upper)
    calibration = float(in_ci.mean())
    
    # Mean uncertainty
    mean_uncertainty = float(pred_std.mean())
    
    return {
        'n_samples': len(y_np),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else None,
        'pearson_p': float(pearson_p) if not np.isnan(pearson_p) else None,
        'spearman_rho': float(spearman_rho) if not np.isnan(spearman_rho) else None,
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
        'calibration_95ci': calibration,
        'mean_uncertainty': mean_uncertainty,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Compute actual seed
    seed = args.seed + args.replicate
    
    # Generate config name
    kmeans_str = "kmeans" if args.kmeans_init else "random"
    config_name = f"ind{args.num_inducing}_{kmeans_str}"
    run_name = f"{args.dataset}_{config_name}_rep{args.replicate}"
    
    logger.info("=" * 60)
    logger.info("GP Hyperparameter Optimization")
    logger.info("=" * 60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Inducing points: {args.num_inducing}")
    logger.info(f"K-means init: {args.kmeans_init}")
    logger.info(f"Replicate: {args.replicate}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Path(args.data_dir)
    data = load_dataset(data_dir, args.dataset)
    
    # Split into train/val/test
    train_data, val_data, test_data = prepare_train_val_test_split(
        data, args.test_fraction, args.val_fraction, seed
    )
    
    # Prepare inputs
    X_train, y_train = prepare_inputs(train_data)
    X_val, y_val = prepare_inputs(val_data)
    X_test, y_test = prepare_inputs(test_data)
    
    logger.info(f"\nInput shape: {X_train.shape[1]} dimensions")
    logger.info(f"Output range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # Train model
    use_gpu = not args.no_gpu
    
    result = train_svgp(
        X_train, y_train,
        X_val, y_val,
        num_inducing=args.num_inducing,
        use_kmeans=args.kmeans_init,
        num_epochs=args.num_epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed,
        use_gpu=use_gpu
    )
    
    model = result['model']
    likelihood = result['likelihood']
    
    # Evaluate on all sets
    logger.info("\nEvaluating on all data splits...")
    
    train_metrics = evaluate_model(model, likelihood, X_train, y_train)
    val_metrics = evaluate_model(model, likelihood, X_val, y_val)
    test_metrics = evaluate_model(model, likelihood, X_test, y_test)
    
    logger.info(f"\nResults:")
    logger.info(f"  Train: R²={train_metrics['r2']:.4f}, RMSE={train_metrics['rmse']:.4f}")
    logger.info(f"  Val:   R²={val_metrics['r2']:.4f}, RMSE={val_metrics['rmse']:.4f}")
    logger.info(f"  Test:  R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}")
    
    # Save model
    model_path = output_dir / f"model_{run_name}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'train_x_mean': model.train_x_mean.cpu(),
        'train_x_std': model.train_x_std.cpu(),
        'train_y_mean': model.train_y_mean.cpu(),
        'train_y_std': model.train_y_std.cpu(),
    }, model_path)
    logger.info(f"\nSaved model to {model_path}")
    
    # Save results
    results = {
        'run_name': run_name,
        'config': {
            'dataset': args.dataset,
            'num_inducing': args.num_inducing,
            'kmeans_init': args.kmeans_init,
            'replicate': args.replicate,
            'seed': seed,
            'num_epochs': args.num_epochs,
            'patience': args.patience,
            'warmup_epochs': args.warmup_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'test_fraction': args.test_fraction,
            'val_fraction': args.val_fraction,
        },
        'training': {
            'epochs_trained': result['epochs_trained'],
            'best_epoch': result['best_epoch'],
            'best_val_loss': result['best_val_loss'],
            'training_time': result['training_time'],
            'train_losses': result['train_losses'],
            'val_losses': result['val_losses'],
            'learning_rates': result['learning_rates'],
        },
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
        'data_splits': {
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_test': len(y_test),
        }
    }
    
    results_path = output_dir / f"results_{run_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("HPO Run Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
