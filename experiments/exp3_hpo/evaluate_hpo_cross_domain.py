#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Cross-Domain Evaluation for HPO Models

This script evaluates HPO-trained GP models on all datasets to assess
generalization performance. For each trained model, it computes metrics
on optimized, random, and combined test sets.

Usage:
    python experiments/evaluate_hpo_cross_domain.py \\
        --models-dir results/hyperparameterization \\
        --data-dir results/training_datasets \\
        --output-dir results/hyperparameterization/cross_domain
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp3_hpo.train_gp_hpo import SVGPModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metric configurations for multi-metric plotting
# key: column name in results_df
# label: display name for plots
# vmin/vmax: heatmap color scale bounds
# cmap: colormap (reversed for error metrics where lower=better)
# fmt: number format string
# higher_better: True if higher values are better (affects best config selection)
METRIC_CONFIGS = {
    'r2': {
        'key': 'r2',
        'label': 'R²',
        'vmin': 0.0,
        'vmax': 1.0,
        'cmap': 'RdYlGn',
        'fmt': '.3f',
        'higher_better': True,
    },
    'rmse': {
        'key': 'rmse',
        'label': 'RMSE',
        'vmin': 0.0,
        'vmax': 5.0,
        'cmap': 'RdYlGn_r',  # Reversed: green=low (good)
        'fmt': '.3f',
        'higher_better': False,
    },
    'mae': {
        'key': 'mae',
        'label': 'MAE',
        'vmin': 0.0,
        'vmax': 3.0,
        'cmap': 'RdYlGn_r',  # Reversed: green=low (good)
        'fmt': '.3f',
        'higher_better': False,
    },
    'pearson_r': {
        'key': 'pearson_r',
        'label': 'Pearson r',
        'vmin': -1.0,
        'vmax': 1.0,
        'cmap': 'RdYlGn',
        'fmt': '.3f',
        'higher_better': True,
    },
    'spearman_rho': {
        'key': 'spearman_rho',
        'label': 'Spearman ρ',
        'vmin': -1.0,
        'vmax': 1.0,
        'cmap': 'RdYlGn',
        'fmt': '.3f',
        'higher_better': True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation for HPO models"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="results/hyperparameterization",
        help="Directory containing trained HPO models"
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
        default=None,
        help="Output directory (default: models-dir/cross_domain)"
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction for test set (must match HPO training)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (must match HPO training)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU"
    )
    return parser.parse_args()


def load_dataset(data_dir: Path, dataset_name: str) -> Dict[str, np.ndarray]:
    """Load and combine train+val data."""
    dataset_path = data_dir / f"dataset_{dataset_name}.npz"
    data = np.load(dataset_path)
    
    return {
        'genomes': np.vstack([data['train_genomes'], data['val_genomes']]),
        'widths': np.concatenate([data['train_widths'], data['val_widths']]),
        'heights': np.concatenate([data['train_heights'], data['val_heights']]),
        'objectives': np.concatenate([data['train_objectives'], data['val_objectives']]),
        'features': np.vstack([data['train_features'], data['val_features']]),
        'parcel_sizes': np.concatenate([data['train_parcel_sizes'], data['val_parcel_sizes']]),
    }


def get_test_split(
    data: Dict[str, np.ndarray],
    test_fraction: float,
    seed: int
) -> Dict[str, np.ndarray]:
    """Extract test set using same split as training."""
    n_total = len(data['objectives'])
    indices = np.arange(n_total)
    
    # Same split logic as train_gp_hpo.py
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed
    )
    
    return {k: v[test_idx] for k, v in data.items()}


def prepare_inputs(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare inputs for GP evaluation."""
    genomes = data['genomes']
    parcel_sizes = data['parcel_sizes'].reshape(-1, 1)
    X = np.column_stack([genomes, parcel_sizes, parcel_sizes]).astype(np.float32)
    y = data['objectives'].astype(np.float32)
    return X, y


def load_model(model_path: Path, device: torch.device) -> Tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get inducing points size from state dict
    inducing_key = 'variational_strategy.inducing_points'
    inducing_points = checkpoint['model_state_dict'][inducing_key]
    num_inducing = inducing_points.size(0)
    input_dim = inducing_points.size(1)
    
    # Create model and likelihood
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


def evaluate_model(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024
) -> Dict:
    """Evaluate model on given data."""
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
    
    # Metrics
    mse = np.mean((pred_mean - y_np)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - y_np))
    y_var = np.var(y_np)
    r2 = 1 - mse / y_var if y_var > 1e-10 else 0.0
    
    # Correlation
    try:
        pearson_r, pearson_p = pearsonr(pred_mean, y_np)
        spearman_rho, spearman_p = spearmanr(pred_mean, y_np)
    except:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_rho, spearman_p = np.nan, np.nan
    
    # Calibration
    ci_lower = pred_mean - 1.96 * pred_std
    ci_upper = pred_mean + 1.96 * pred_std
    in_ci = (y_np >= ci_lower) & (y_np <= ci_upper)
    calibration = float(in_ci.mean())
    
    return {
        'n_samples': len(y_np),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else None,
        'spearman_rho': float(spearman_rho) if not np.isnan(spearman_rho) else None,
        'calibration_95ci': calibration,
        'mean_uncertainty': float(pred_std.mean()),
    }


def parse_model_name(model_path: Path) -> Dict:
    """Parse model filename to extract config."""
    # Format: model_<dataset>_ind<N>_<init>_rep<R>.pth
    name = model_path.stem.replace('model_', '')
    parts = name.split('_')
    
    # Find dataset, inducing, init, replicate
    dataset = parts[0]
    num_inducing = int(parts[1].replace('ind', ''))
    kmeans_init = parts[2] == 'kmeans'
    replicate = int(parts[3].replace('rep', ''))
    
    return {
        'dataset': dataset,
        'num_inducing': num_inducing,
        'kmeans_init': kmeans_init,
        'replicate': replicate,
        'config': f"ind{num_inducing}_{'kmeans' if kmeans_init else 'random'}",
    }


def plot_cross_domain_heatmap(results_df: pd.DataFrame, output_dir: Path):
    """Plot cross-domain heatmaps for best configs across all metrics."""
    for metric_name, cfg in METRIC_CONFIGS.items():
        metric_key = cfg['key']
        
        # Skip if metric has None values (e.g., pearson_r can be None)
        if results_df[metric_key].isna().all():
            logger.warning(f"Skipping {metric_name}: all values are NaN")
            continue
        
        # Get best config per training dataset based on same-domain performance
        # For error metrics (lower=better), use nsmallest; for others, use nlargest
        if cfg['higher_better']:
            best_configs = results_df.groupby('train_dataset').apply(
                lambda x: x[x['eval_dataset'] == x['train_dataset']].nlargest(1, metric_key)['config'].values[0]
                if len(x[x['eval_dataset'] == x['train_dataset']]) > 0 else x['config'].iloc[0]
            ).to_dict()
        else:
            best_configs = results_df.groupby('train_dataset').apply(
                lambda x: x[x['eval_dataset'] == x['train_dataset']].nsmallest(1, metric_key)['config'].values[0]
                if len(x[x['eval_dataset'] == x['train_dataset']]) > 0 else x['config'].iloc[0]
            ).to_dict()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, (train_ds, best_config) in zip(axes, best_configs.items()):
            # Filter to best config for this training dataset
            df_config = results_df[
                (results_df['train_dataset'] == train_ds) & 
                (results_df['config'] == best_config)
            ]
            
            # Pivot for heatmap
            pivot = df_config.groupby('eval_dataset')[metric_key].mean()
            
            # Create single-row heatmap
            data = [[pivot.get('optimized', 0), pivot.get('random', 0), pivot.get('combined', 0)]]
            
            sns.heatmap(
                data, annot=True, fmt=cfg['fmt'], cmap=cfg['cmap'],
                vmin=cfg['vmin'], vmax=cfg['vmax'], ax=ax,
                xticklabels=['Opt', 'Rand', 'Comb'],
                yticklabels=[f'{train_ds[:3]}→'],
                cbar_kws={'label': cfg['label']}
            )
            ax.set_title(f'Train: {train_ds.capitalize()}\n({best_config})')
            ax.set_xlabel('Evaluation Dataset')
        
        plt.tight_layout()
        filename = f'cross_domain_heatmap_best_{metric_name}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}")


def plot_full_cross_domain_matrix(results_df: pd.DataFrame, output_dir: Path):
    """Plot full cross-domain matrix averaged over configs for all metrics."""
    for metric_name, cfg in METRIC_CONFIGS.items():
        metric_key = cfg['key']
        
        # Skip if metric has None values
        if results_df[metric_key].isna().all():
            logger.warning(f"Skipping {metric_name}: all values are NaN")
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Average metric over configs and replicates
        pivot = results_df.pivot_table(
            values=metric_key,
            index='train_dataset',
            columns='eval_dataset',
            aggfunc='mean'
        )
        
        # Reorder
        order = ['optimized', 'random', 'combined']
        pivot = pivot.reindex(index=order, columns=order)
        
        sns.heatmap(
            pivot, annot=True, fmt=cfg['fmt'], cmap=cfg['cmap'],
            vmin=cfg['vmin'], vmax=cfg['vmax'], ax=ax,
            cbar_kws={'label': cfg['label']}
        )
        ax.set_title(f'Cross-Domain {cfg["label"]} (averaged over all HPO configs)')
        ax.set_xlabel('Evaluation Dataset')
        ax.set_ylabel('Training Dataset')
        
        plt.tight_layout()
        filename = f'cross_domain_matrix_avg_{metric_name}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}")


def plot_cross_domain_by_config(results_df: pd.DataFrame, output_dir: Path):
    """Plot cross-domain performance for each config across all metrics."""
    configs = sorted(results_df['config'].unique())
    
    for metric_name, cfg in METRIC_CONFIGS.items():
        metric_key = cfg['key']
        
        # Skip if metric has None values
        if results_df[metric_key].isna().all():
            logger.warning(f"Skipping {metric_name}: all values are NaN")
            continue
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for ax, config in zip(axes, configs):
            df_cfg = results_df[results_df['config'] == config]
            
            pivot = df_cfg.pivot_table(
                values=metric_key,
                index='train_dataset',
                columns='eval_dataset',
                aggfunc='mean'
            )
            
            order = ['optimized', 'random', 'combined']
            pivot = pivot.reindex(index=order, columns=order)
            
            sns.heatmap(
                pivot, annot=True, fmt=cfg['fmt'], cmap=cfg['cmap'],
                vmin=cfg['vmin'], vmax=cfg['vmax'], ax=ax,
                cbar=False
            )
            ax.set_title(config)
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        # Hide unused
        for ax in axes[len(configs):]:
            ax.axis('off')
        
        plt.suptitle(f'Cross-Domain {cfg["label"]} by Configuration', fontsize=14)
        plt.tight_layout()
        filename = f'cross_domain_by_config_{metric_name}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}")


def plot_generalization_gap(results_df: pd.DataFrame, output_dir: Path):
    """Plot generalization gap (same-domain vs cross-domain) for all metrics."""
    same_domain = results_df[results_df['train_dataset'] == results_df['eval_dataset']]
    cross_domain = results_df[results_df['train_dataset'] != results_df['eval_dataset']]
    
    for metric_name, cfg in METRIC_CONFIGS.items():
        metric_key = cfg['key']
        
        # Skip if metric has None values
        same_vals = same_domain[metric_key].dropna().values
        cross_vals = cross_domain[metric_key].dropna().values
        
        if len(same_vals) == 0 or len(cross_vals) == 0:
            logger.warning(f"Skipping {metric_name}: insufficient data")
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot
        data = [same_vals, cross_vals]
        positions = [1, 2]
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        
        # Color scheme: for error metrics, swap colors (lower cross-domain error = green)
        if cfg['higher_better']:
            colors = ['#2ecc71', '#e74c3c']  # green=same (good), red=cross (worse)
        else:
            colors = ['#2ecc71', '#e74c3c']  # Keep same visual convention
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean points
        means = [np.mean(d) for d in data]
        ax.scatter(positions, means, color='blue', s=100, zorder=5, label='Mean')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Same Domain\n(Train=Eval)', 'Cross Domain\n(Train≠Eval)'])
        ax.set_ylabel(f'Test {cfg["label"]}')
        ax.set_title(f'Generalization Gap ({cfg["label"]}): Same vs Cross Domain')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add stats - gap interpretation depends on metric direction
        same_mean = np.mean(data[0])
        cross_mean = np.mean(data[1])
        if cfg['higher_better']:
            gap = same_mean - cross_mean  # Positive gap = same domain better
            gap_label = f'Gap: {gap:.3f}'
        else:
            gap = cross_mean - same_mean  # Positive gap = cross domain worse (higher error)
            gap_label = f'Gap: +{gap:.3f}' if gap > 0 else f'Gap: {gap:.3f}'
        
        ax.text(0.95, 0.95, gap_label, transform=ax.transAxes, 
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = f'generalization_gap_{metric_name}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}")


def main():
    args = parse_args()
    
    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else models_dir / 'cross_domain'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if not args.no_gpu and torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("HPO Cross-Domain Evaluation")
    logger.info("=" * 60)
    logger.info(f"Models dir: {models_dir}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")
    
    # Load all test sets
    logger.info("\nLoading test sets...")
    datasets = ['optimized', 'random', 'combined']
    test_sets = {}
    
    for ds_name in datasets:
        data = load_dataset(data_dir, ds_name)
        # Use seed + replicate as in training
        test_data = get_test_split(data, args.test_fraction, args.seed + 1)  # rep 1 seed
        X_test, y_test = prepare_inputs(test_data)
        test_sets[ds_name] = (X_test, y_test)
        logger.info(f"  {ds_name}: {len(y_test)} test samples")
    
    # Find all models
    model_files = list(models_dir.glob("model_*.pth"))
    logger.info(f"\nFound {len(model_files)} trained models")
    
    if not model_files:
        logger.error("No models found!")
        return
    
    # Evaluate each model on all datasets
    results = []
    
    for model_path in sorted(model_files):
        model_info = parse_model_name(model_path)
        logger.info(f"\nEvaluating: {model_path.name}")
        
        try:
            model, likelihood = load_model(model_path, device)
        except Exception as e:
            logger.error(f"  Failed to load: {e}")
            continue
        
        for eval_ds in datasets:
            X_test, y_test = test_sets[eval_ds]
            metrics = evaluate_model(model, likelihood, X_test, y_test)
            
            results.append({
                'model_name': model_path.stem,
                'train_dataset': model_info['dataset'],
                'num_inducing': model_info['num_inducing'],
                'kmeans_init': model_info['kmeans_init'],
                'config': model_info['config'],
                'replicate': model_info['replicate'],
                'eval_dataset': eval_ds,
                **metrics
            })
            
            logger.info(f"  → {eval_ds}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv(output_dir / 'cross_domain_results.csv', index=False)
    logger.info(f"\nSaved: cross_domain_results.csv")
    
    # Create summary table
    summary = results_df.pivot_table(
        values=['r2', 'rmse', 'calibration_95ci'],
        index=['train_dataset', 'config'],
        columns='eval_dataset',
        aggfunc=['mean', 'std']
    ).round(4)
    summary.to_csv(output_dir / 'cross_domain_summary.csv')
    logger.info("Saved: cross_domain_summary.csv")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_full_cross_domain_matrix(results_df, output_dir)
    plot_cross_domain_by_config(results_df, output_dir)
    plot_cross_domain_heatmap(results_df, output_dir)
    plot_generalization_gap(results_df, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-DOMAIN SUMMARY")
    logger.info("=" * 60)
    
    # Best cross-domain performance per training dataset
    for train_ds in datasets:
        df_train = results_df[results_df['train_dataset'] == train_ds]
        
        # Same-domain performance
        same = df_train[df_train['eval_dataset'] == train_ds]['r2'].mean()
        
        # Cross-domain performance
        cross = df_train[df_train['eval_dataset'] != train_ds].groupby('eval_dataset')['r2'].mean()
        
        logger.info(f"\nTrained on {train_ds.upper()}:")
        logger.info(f"  Same-domain (→{train_ds}): R²={same:.4f}")
        for eval_ds, r2 in cross.items():
            logger.info(f"  Cross-domain (→{eval_ds}): R²={r2:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Cross-domain evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
