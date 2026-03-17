#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Phase 4: Evaluate GP Experiment

This script evaluates trained GP models with cross-domain testing:
1. Each model evaluated on all test sets (Optimized, Random, Combined)
2. Comprehensive metrics: RMSE, MAE, R², Pearson, Spearman, Calibration
3. Analysis by parcel size to understand generalization
4. Generate summary tables and plots

Usage:
    python experiments/evaluate_gp_experiment.py
    python experiments/evaluate_gp_experiment.py --models-dir results/gp_experiment
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Parcel sizes
PARCEL_SIZES = [60, 120, 240]


class SVGPModel(ApproximateGP):
    """SVGP model (must match training definition)."""
    
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
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GP models with cross-domain testing"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="results/gp_experiment",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/training_datasets",
        help="Directory containing test datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/gp_evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU"
    )
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> Tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """Load trained SVGP model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get number of inducing points and input dim from state dict
    inducing_key = 'variational_strategy.inducing_points'
    inducing_points = checkpoint['model_state_dict'][inducing_key]
    num_inducing = inducing_points.shape[0]
    input_dim = inducing_points.shape[1]
    
    # Create model structure
    model = SVGPModel(inducing_points.to(device), input_dim=input_dim).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    # Load normalization parameters
    model.train_x_mean = checkpoint['train_x_mean'].to(device)
    model.train_x_std = checkpoint['train_x_std'].to(device)
    model.train_y_mean = checkpoint['train_y_mean'].to(device)
    model.train_y_std = checkpoint['train_y_std'].to(device)
    
    model.eval()
    likelihood.eval()
    
    return model, likelihood


def load_test_data(data_dir: Path, dataset_name: str) -> Dict[str, np.ndarray]:
    """Load validation set from prepared dataset."""
    dataset_path = data_dir / f"dataset_{dataset_name}.npz"
    data = np.load(dataset_path)
    
    return {
        'genomes': data['val_genomes'],
        'widths': data['val_widths'],
        'heights': data['val_heights'],
        'objectives': data['val_objectives'],
        'features': data['val_features'],
        'parcel_sizes': data['val_parcel_sizes']
    }


def prepare_inputs(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare inputs for GP prediction."""
    genomes = data['genomes']
    parcel_sizes = data['parcel_sizes'].reshape(-1, 1)
    
    X = np.column_stack([genomes, parcel_sizes, parcel_sizes]).astype(np.float32)
    y = data['objectives'].astype(np.float32)
    
    return X, y


def evaluate_model_on_dataset(
    model: SVGPModel,
    likelihood,
    X: np.ndarray,
    y: np.ndarray,
    parcel_sizes: np.ndarray,
    batch_size: int = 1024
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Returns comprehensive metrics including per-parcel-size breakdown.
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
            pred_mean = pred.mean * model.train_y_std + model.train_y_mean
            pred_std = pred.stddev * model.train_y_std
            pred_means.append(pred_mean.cpu())
            pred_stds.append(pred_std.cpu())
    
    pred_mean = torch.cat(pred_means).numpy()
    pred_std = torch.cat(pred_stds).numpy()
    y_np = test_y.cpu().numpy()
    
    # Overall metrics
    mse = np.mean((pred_mean - y_np)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - y_np))
    r2 = 1 - mse / np.var(y_np)
    
    pearson_r, pearson_p = pearsonr(pred_mean, y_np)
    spearman_rho, spearman_p = spearmanr(pred_mean, y_np)
    
    # Calibration
    ci_lower = pred_mean - 1.96 * pred_std
    ci_upper = pred_mean + 1.96 * pred_std
    in_ci = (y_np >= ci_lower) & (y_np <= ci_upper)
    calibration = in_ci.mean()
    
    # Mean uncertainty
    mean_uncertainty = pred_std.mean()
    
    overall = {
        'n_samples': len(y_np),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'calibration_95ci': float(calibration),
        'mean_uncertainty': float(mean_uncertainty)
    }
    
    # Per-parcel-size metrics
    per_size = {}
    unique_sizes = np.unique(parcel_sizes)
    
    for size in unique_sizes:
        mask = parcel_sizes == size
        if mask.sum() < 5:
            continue
            
        y_size = y_np[mask]
        pred_size = pred_mean[mask]
        std_size = pred_std[mask]
        
        mse_size = np.mean((pred_size - y_size)**2)
        rmse_size = np.sqrt(mse_size)
        r2_size = 1 - mse_size / np.var(y_size) if np.var(y_size) > 0 else 0
        
        ci_lower_size = pred_size - 1.96 * std_size
        ci_upper_size = pred_size + 1.96 * std_size
        calib_size = ((y_size >= ci_lower_size) & (y_size <= ci_upper_size)).mean()
        
        per_size[int(size)] = {
            'n_samples': int(mask.sum()),
            'rmse': float(rmse_size),
            'r2': float(r2_size),
            'calibration': float(calib_size)
        }
    
    return {
        'overall': overall,
        'per_size': per_size,
        'predictions': pred_mean,
        'uncertainties': pred_std,
        'true_values': y_np,
        'parcel_sizes': parcel_sizes
    }


def discover_models(models_dir: Path) -> List[Dict]:
    """Discover trained models and their metadata."""
    models = []
    
    for model_file in sorted(models_dir.glob("gp_*.pth")):
        # Parse model name: gp_<dataset>_rep<N>.pth
        name = model_file.stem
        parts = name.split('_')
        
        if len(parts) >= 3 and parts[0] == 'gp':
            dataset = parts[1]
            replicate = int(parts[2].replace('rep', ''))
            
            # Look for corresponding metrics file
            metrics_file = models_dir / f"{name}_metrics.json"
            training_metrics = {}
            if metrics_file.exists():
                with open(metrics_file) as f:
                    training_metrics = json.load(f)
            
            models.append({
                'path': model_file,
                'name': name,
                'dataset': dataset,
                'replicate': replicate,
                'training_metrics': training_metrics
            })
    
    logger.info(f"Discovered {len(models)} trained models")
    return models


def run_cross_domain_evaluation(
    models: List[Dict],
    data_dir: Path,
    device: torch.device
) -> pd.DataFrame:
    """
    Evaluate all models on all datasets.
    
    Returns DataFrame with cross-domain evaluation results.
    """
    datasets = ['optimized', 'random', 'combined']
    
    # Load all test datasets
    test_data = {}
    for ds in datasets:
        data = load_test_data(data_dir, ds)
        X, y = prepare_inputs(data)
        test_data[ds] = {
            'X': X,
            'y': y,
            'parcel_sizes': data['parcel_sizes']
        }
        logger.info(f"Loaded {ds} test set: {len(y)} samples")
    
    # Evaluate each model on each dataset
    results = []
    
    for model_info in models:
        logger.info(f"\nEvaluating {model_info['name']}...")
        
        # Load model
        model, likelihood = load_model(model_info['path'], device)
        
        for eval_dataset in datasets:
            eval_data = test_data[eval_dataset]
            
            metrics = evaluate_model_on_dataset(
                model, likelihood,
                eval_data['X'], eval_data['y'],
                eval_data['parcel_sizes']
            )
            
            result = {
                'model_name': model_info['name'],
                'train_dataset': model_info['dataset'],
                'replicate': model_info['replicate'],
                'eval_dataset': eval_dataset,
                **metrics['overall']
            }
            results.append(result)
            
            logger.info(f"  → {eval_dataset}: R²={metrics['overall']['r2']:.4f}, "
                       f"RMSE={metrics['overall']['rmse']:.4f}")
    
    return pd.DataFrame(results)


def create_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table averaging across replicates."""
    summary = results_df.groupby(['train_dataset', 'eval_dataset']).agg({
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'pearson_r': ['mean', 'std'],
        'spearman_rho': ['mean', 'std'],
        'calibration_95ci': ['mean', 'std'],
        'n_samples': 'first'
    }).round(4)
    
    return summary


def plot_cross_domain_results(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization of cross-domain evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Consistent font settings for publication (target ~7pt at LNCS textwidth ≈ 4.8")
    # figsize=10" wide → scale = 4.8/10 = 0.48 → 15pt * 0.48 ≈ 7pt at print
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 15,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    })
    
    # 1. R² heatmap
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    train_datasets = ['optimized', 'random', 'combined']
    eval_datasets = ['optimized', 'random', 'combined']
    
    # Average across replicates
    avg_results = results_df.groupby(['train_dataset', 'eval_dataset'])['r2'].mean().unstack()
    
    # Reorder
    avg_results = avg_results.reindex(train_datasets)[eval_datasets]
    
    # Heatmap
    ax = axes[0]
    im = ax.imshow(avg_results.values, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(eval_datasets)))
    ax.set_yticks(range(len(train_datasets)))
    ax.set_xticklabels(['Opt', 'Rand', 'Comb'])
    ax.set_yticklabels(['Opt', 'Rand', 'Comb'])
    ax.set_xlabel('Evaluation Dataset')
    ax.set_ylabel('Training Dataset')
    ax.set_title('R² (higher is better)')
    
    # Add values
    for i in range(len(train_datasets)):
        for j in range(len(eval_datasets)):
            val = avg_results.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=14)
    
    plt.colorbar(im, ax=ax)
    
    # 2. RMSE comparison
    ax = axes[1]
    avg_rmse = results_df.groupby(['train_dataset', 'eval_dataset'])['rmse'].mean().unstack()
    avg_rmse = avg_rmse.reindex(train_datasets)[eval_datasets]
    
    x = np.arange(len(train_datasets))
    width = 0.25
    
    for i, eval_ds in enumerate(eval_datasets):
        offset = (i - 1) * width
        ax.bar(x + offset, avg_rmse[eval_ds], width, label=f'Eval: {eval_ds}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Optimized', 'Random', 'Combined'])
    ax.set_xlabel('Training Dataset')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE by Training/Evaluation Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Correlation comparison
    ax = axes[2]
    avg_pearson = results_df.groupby(['train_dataset', 'eval_dataset'])['pearson_r'].mean().unstack()
    avg_pearson = avg_pearson.reindex(train_datasets)[eval_datasets]
    
    for i, eval_ds in enumerate(eval_datasets):
        offset = (i - 1) * width
        ax.bar(x + offset, avg_pearson[eval_ds], width, label=f'Eval: {eval_ds}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Optimized', 'Random', 'Combined'])
    ax.set_xlabel('Training Dataset')
    ax.set_ylabel('Pearson r')
    ax.set_title('Correlation by Training/Evaluation Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_domain_evaluation.png', dpi=300)
    plt.savefig(output_dir / 'cross_domain_evaluation.pdf')
    plt.close()
    
    logger.info(f"Saved cross-domain plot to {output_dir / 'cross_domain_evaluation.png/.pdf'}")
    
    # 4. Diagonal vs off-diagonal analysis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Diagonal = same train/eval, off-diagonal = different
    diagonal_r2 = []
    offdiag_r2 = []
    
    for _, row in results_df.iterrows():
        if row['train_dataset'] == row['eval_dataset']:
            diagonal_r2.append(row['r2'])
        else:
            offdiag_r2.append(row['r2'])
    
    positions = [1, 2]
    bp = ax.boxplot([diagonal_r2, offdiag_r2], positions=positions, widths=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Same Domain\n(Train=Eval)', 'Cross Domain\n(Train≠Eval)'])
    ax.set_ylabel('R²')
    ax.set_title('Generalization Analysis: Same vs Cross Domain')
    ax.grid(True, alpha=0.3)
    
    # Add means
    ax.scatter([1], [np.mean(diagonal_r2)], color='red', s=100, zorder=5, label=f'Mean: {np.mean(diagonal_r2):.3f}')
    ax.scatter([2], [np.mean(offdiag_r2)], color='blue', s=100, zorder=5, label=f'Mean: {np.mean(offdiag_r2):.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generalization_analysis.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved generalization plot to {output_dir / 'generalization_analysis.png'}")


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 4: GP Experiment Evaluation")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if not args.no_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover models
    logger.info("\nDiscovering trained models...")
    models = discover_models(models_dir)
    
    if not models:
        logger.error(f"No models found in {models_dir}")
        return
    
    # Run cross-domain evaluation
    logger.info("\nRunning cross-domain evaluation...")
    results_df = run_cross_domain_evaluation(models, data_dir, device)
    
    # Save raw results
    results_path = output_dir / 'cross_domain_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nSaved results to {results_path}")
    
    # Create summary table
    logger.info("\nCreating summary table...")
    summary = create_summary_table(results_df)
    summary_path = output_dir / 'summary_table.csv'
    summary.to_csv(summary_path)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-DOMAIN EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    print("\n" + summary.to_string())
    
    # Key findings
    logger.info("\n" + "=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)
    
    # Best model for each evaluation set
    for eval_ds in ['optimized', 'random', 'combined']:
        subset = results_df[results_df['eval_dataset'] == eval_ds]
        best = subset.loc[subset['r2'].idxmax()]
        logger.info(f"\nBest model for {eval_ds} data:")
        logger.info(f"  Trained on: {best['train_dataset']}")
        logger.info(f"  R²: {best['r2']:.4f}, Pearson: {best['pearson_r']:.4f}")
    
    # Generalization gap
    diagonal = results_df[results_df['train_dataset'] == results_df['eval_dataset']]['r2'].mean()
    offdiag = results_df[results_df['train_dataset'] != results_df['eval_dataset']]['r2'].mean()
    logger.info(f"\nGeneralization gap:")
    logger.info(f"  Same domain R²: {diagonal:.4f}")
    logger.info(f"  Cross domain R²: {offdiag:.4f}")
    logger.info(f"  Gap: {diagonal - offdiag:.4f}")
    
    # Create plots
    logger.info("\nCreating visualizations...")
    plot_cross_domain_results(results_df, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
