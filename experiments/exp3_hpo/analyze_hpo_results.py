#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GP Hyperparameter Optimization Analysis Script

This script analyzes results from the HPO experiment, generating:
1. Summary statistics across all configurations
2. Statistical comparisons between hyperparameter settings
3. Visualizations (heatmaps, bar plots, learning curves)
4. Best configuration recommendations

Usage:
    python experiments/analyze_hpo_results.py --results-dir results/hyperparameterization
    python experiments/analyze_hpo_results.py --results-dir results/hyperparameterization --output-dir results/hpo_analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

try:
    import torch
    import gpytorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/GPyTorch not available. Inference timing analysis will be skipped.")

# Import model class from train_gp_hpo
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from experiments.exp3_hpo.train_gp_hpo import SVGPModel
    

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all HPO results into a DataFrame.
    """
    results_files = list(results_dir.glob("results_*.json"))
    
    if not results_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    logger.info(f"Found {len(results_files)} result files")
    
    records = []
    for f in results_files:
        with open(f) as fp:
            data = json.load(fp)
        
        record = {
            'run_name': data['run_name'],
            'dataset': data['config']['dataset'],
            'num_inducing': data['config']['num_inducing'],
            'kmeans_init': data['config']['kmeans_init'],
            'replicate': data['config']['replicate'],
            'seed': data['config']['seed'],
            
            # Training info
            'epochs_trained': data['training']['epochs_trained'],
            'best_epoch': data['training']['best_epoch'],
            'best_val_loss': data['training']['best_val_loss'],
            'training_time': data['training']['training_time'],
            
            # Data splits
            'n_train': data['data_splits']['n_train'],
            'n_val': data['data_splits']['n_val'],
            'n_test': data['data_splits']['n_test'],
        }
        
        # Add metrics for each split
        for split in ['train', 'val', 'test']:
            metrics = data['metrics'][split]
            for metric, value in metrics.items():
                record[f'{split}_{metric}'] = value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add config string for grouping
    df['config'] = df.apply(
        lambda r: f"ind{r['num_inducing']}_{'kmeans' if r['kmeans_init'] else 'random'}",
        axis=1
    )
    
    logger.info(f"Loaded {len(df)} runs")
    logger.info(f"Datasets: {df['dataset'].unique().tolist()}")
    logger.info(f"Configs: {df['config'].unique().tolist()}")
    
    return df


def load_learning_curves(results_dir: Path, run_names: List[str] = None) -> Dict[str, Dict]:
    """
    Load learning curves for specific runs.
    """
    curves = {}
    
    for f in results_dir.glob("results_*.json"):
        with open(f) as fp:
            data = json.load(fp)
        
        run_name = data['run_name']
        if run_names is None or run_name in run_names:
            curves[run_name] = {
                'train_losses': data['training']['train_losses'],
                'val_losses': data['training']['val_losses'],
                'learning_rates': data['training']['learning_rates'],
                'best_epoch': data['training']['best_epoch'],
            }
    
    return curves


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by config and dataset.
    """
    metrics = ['test_r2', 'test_rmse', 'test_mae', 'test_calibration_95ci',
               'epochs_trained', 'training_time']
    
    summary = df.groupby(['dataset', 'config']).agg({
        **{m: ['mean', 'std', 'min', 'max'] for m in metrics},
        'run_name': 'count'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.rename(columns={'run_name_count': 'n_replicates'})
    
    return summary.reset_index()


def perform_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical tests comparing hyperparameter settings.
    """
    results = []
    
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        
        # Compare inducing points (within same init method)
        for kmeans in [True, False]:
            df_km = df_ds[df_ds['kmeans_init'] == kmeans]
            
            inducing_values = sorted(df_km['num_inducing'].unique())
            if len(inducing_values) >= 2:
                for i in range(len(inducing_values)):
                    for j in range(i + 1, len(inducing_values)):
                        ind1, ind2 = inducing_values[i], inducing_values[j]
                        
                        r2_1 = df_km[df_km['num_inducing'] == ind1]['test_r2'].values
                        r2_2 = df_km[df_km['num_inducing'] == ind2]['test_r2'].values
                        
                        if len(r2_1) >= 2 and len(r2_2) >= 2:
                            stat, pval = stats.mannwhitneyu(r2_1, r2_2, alternative='two-sided')
                            
                            results.append({
                                'dataset': dataset,
                                'comparison': f"ind{ind1} vs ind{ind2}",
                                'condition': 'kmeans' if kmeans else 'random',
                                'metric': 'test_r2',
                                'group1_mean': r2_1.mean(),
                                'group2_mean': r2_2.mean(),
                                'statistic': stat,
                                'p_value': pval,
                                'significant': pval < 0.05,
                            })
        
        # Compare init methods (within same inducing points)
        for num_ind in df_ds['num_inducing'].unique():
            df_ind = df_ds[df_ds['num_inducing'] == num_ind]
            
            r2_kmeans = df_ind[df_ind['kmeans_init'] == True]['test_r2'].values
            r2_random = df_ind[df_ind['kmeans_init'] == False]['test_r2'].values
            
            if len(r2_kmeans) >= 2 and len(r2_random) >= 2:
                stat, pval = stats.mannwhitneyu(r2_kmeans, r2_random, alternative='two-sided')
                
                results.append({
                    'dataset': dataset,
                    'comparison': "kmeans vs random",
                    'condition': f"ind{num_ind}",
                    'metric': 'test_r2',
                    'group1_mean': r2_kmeans.mean(),
                    'group2_mean': r2_random.mean(),
                    'statistic': stat,
                    'p_value': pval,
                    'significant': pval < 0.05,
                })
    
    return pd.DataFrame(results)


def find_best_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find best configuration for each dataset based on test R².
    """
    # Average over replicates
    avg_df = df.groupby(['dataset', 'config', 'num_inducing', 'kmeans_init']).agg({
        'test_r2': 'mean',
        'test_rmse': 'mean',
        'test_calibration_95ci': 'mean',
        'training_time': 'mean',
        'epochs_trained': 'mean',
    }).reset_index()
    
    # Find best for each dataset
    best_idx = avg_df.groupby('dataset')['test_r2'].idxmax()
    best_df = avg_df.loc[best_idx].copy()
    best_df['rank'] = 1
    
    return best_df


# ============================================================================
# Visualization
# ============================================================================

def plot_heatmap_by_dataset(df: pd.DataFrame, output_dir: Path):
    """
    Create heatmaps of test R² for each dataset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['optimized', 'random', 'combined']
    
    # Compute global min/max for consistent color scaling
    all_r2_values = df['test_r2'].values
    vmin = max(0.0, np.floor(np.min(all_r2_values) * 10) / 10)  # Round down to 0.1
    vmax = min(1.0, np.ceil(np.max(all_r2_values) * 10) / 10)   # Round up to 0.1
    
    for ax, dataset in zip(axes, datasets):
        df_ds = df[df['dataset'] == dataset]
        
        if df_ds.empty:
            ax.set_title(f'{dataset.capitalize()}\n(No data)')
            ax.axis('off')
            continue
        
        # Pivot for heatmap
        pivot = df_ds.groupby(['num_inducing', 'kmeans_init'])['test_r2'].mean().unstack()
        pivot.columns = ['Random Init', 'K-means Init']
        pivot.index = [f'{x}' for x in pivot.index]
        
        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=vmin, vmax=vmax, ax=ax,
            cbar_kws={'label': 'Test R²'}
        )
        ax.set_title(f'{dataset.capitalize()} Dataset')
        ax.set_xlabel('Initialization Method')
        ax.set_ylabel('Inducing Points')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_test_r2_by_dataset.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_test_r2_by_dataset.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved heatmap: heatmap_test_r2_by_dataset.png")


def plot_bar_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create bar plots comparing configurations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Aggregate data
    agg_df = df.groupby(['dataset', 'config']).agg({
        'test_r2': ['mean', 'std'],
        'test_rmse': ['mean', 'std'],
        'training_time': ['mean', 'std'],
        'epochs_trained': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names properly
    new_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            if col[1] == '':
                new_cols.append(col[0])
            else:
                new_cols.append(f'{col[0]}_{col[1]}')
        else:
            new_cols.append(col)
    agg_df.columns = new_cols
    
    metrics = [
        ('test_r2', 'Test R² (higher is better)', True),
        ('test_rmse', 'Test RMSE (lower is better)', False),
        ('training_time', 'Training Time (seconds)', False),
        ('epochs_trained', 'Epochs Until Convergence', False),
    ]
    
    # Collect y-limits for consistent scaling
    y_limits = {}
    configs = sorted(agg_df['config'].unique())
    datasets = ['optimized', 'random', 'combined']
    
    for metric, _, _ in metrics:
        all_vals = []
        for dataset in datasets:
            df_ds = agg_df[agg_df['dataset'] == dataset]
            for c in configs:
                if len(df_ds[df_ds['config'] == c]) > 0:
                    mean = df_ds[df_ds['config'] == c][f'{metric}_mean'].values[0]
                    std = df_ds[df_ds['config'] == c][f'{metric}_std'].values[0]
                    all_vals.extend([mean - std, mean + std])
        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.1
            y_limits[metric] = (max(0, min(all_vals) - margin), max(all_vals) + margin)
    
    for ax, (metric, title, higher_better) in zip(axes.flat, metrics):
        # Create grouped bar plot
        x = np.arange(len(configs))
        width = 0.25
        
        for i, dataset in enumerate(datasets):
            df_ds = agg_df[agg_df['dataset'] == dataset]
            means = [df_ds[df_ds['config'] == c][f'{metric}_mean'].values[0] 
                    if len(df_ds[df_ds['config'] == c]) > 0 else 0 for c in configs]
            stds = [df_ds[df_ds['config'] == c][f'{metric}_std'].values[0] 
                   if len(df_ds[df_ds['config'] == c]) > 0 else 0 for c in configs]
            
            ax.bar(x + i * width, means, width, yerr=stds, label=dataset.capitalize(),
                   capsize=3, alpha=0.8)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Set consistent y-limits
        if metric in y_limits:
            ax.set_ylim(y_limits[metric])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bar_comparison_metrics.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'bar_comparison_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved bar plots: bar_comparison_metrics.png")


def plot_learning_curves(curves: Dict[str, Dict], output_dir: Path, max_curves: int = 6):
    """
    Plot learning curves for selected runs (legacy function for backward compatibility).
    """
    if not curves:
        logger.warning("No learning curves to plot")
        return
    
    # Select a subset if too many
    run_names = list(curves.keys())[:max_curves]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Compute global y-limits for consistent scaling
    all_losses = []
    for run_name in run_names:
        data = curves[run_name]
        all_losses.extend(data['train_losses'])
        all_losses.extend(data['val_losses'])
    
    if all_losses:
        y_min = min(all_losses)
        y_max = max(all_losses)
        margin = (y_max - y_min) * 0.05
        y_limits = (max(0, y_min - margin), y_max + margin)
    else:
        y_limits = None
    
    for ax, run_name in zip(axes, run_names):
        data = curves[run_name]
        epochs = np.arange(1, len(data['train_losses']) + 1)
        
        ax.plot(epochs, data['train_losses'], label='Train Loss', alpha=0.7)
        ax.plot(epochs, data['val_losses'], label='Val Loss', alpha=0.7)
        ax.axvline(data['best_epoch'], color='red', linestyle='--', 
                   label=f"Best (epoch {data['best_epoch']})", alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(run_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Set consistent y-limits
        if y_limits:
            ax.set_ylim(y_limits)
    
    # Hide unused subplots
    for ax in axes[len(run_names):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved learning curves: learning_curves.png")


def plot_all_learning_curves(curves: Dict[str, Dict], output_dir: Path):
    """
    Plot learning curves for ALL runs, organized by dataset/config/replicate.
    
    Creates a comprehensive grid with:
    - Rows: configurations (inducing points × init method)
    - Columns: datasets (optimized, random, combined)
    - Separate plots per replicate within each cell
    """
    if not curves:
        logger.warning("No learning curves to plot")
        return
    
    # Parse run names to organize by dataset/config
    organized = {}  # {dataset: {config: {replicate: data}}}
    
    for run_name, data in curves.items():
        # Parse: optimized_ind500_kmeans_rep1
        parts = run_name.split('_')
        dataset = parts[0]
        num_ind = int(parts[1].replace('ind', ''))
        init_method = parts[2]  # 'kmeans' or 'random'
        replicate = int(parts[3].replace('rep', ''))
        config = f"ind{num_ind}_{init_method}"
        
        if dataset not in organized:
            organized[dataset] = {}
        if config not in organized[dataset]:
            organized[dataset][config] = {}
        organized[dataset][config][replicate] = data
    
    # Get unique datasets and configs
    datasets = ['optimized', 'random', 'combined']
    all_configs = set()
    for ds in organized.values():
        all_configs.update(ds.keys())
    configs = sorted(all_configs, key=lambda x: (int(x.split('_')[0].replace('ind', '')), x.split('_')[1]))
    
    # Compute global y-limits
    all_losses = []
    for ds_data in organized.values():
        for cfg_data in ds_data.values():
            for rep_data in cfg_data.values():
                all_losses.extend(rep_data['train_losses'])
                all_losses.extend(rep_data['val_losses'])
    
    if all_losses:
        y_min = min(all_losses)
        y_max = max(all_losses)
        margin = (y_max - y_min) * 0.05
        y_limits = (max(0, y_min - margin), y_max + margin)
    else:
        y_limits = (0, 2)
    
    # Create figure: rows = configs, cols = datasets
    n_rows = len(configs)
    n_cols = len(datasets)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    
    for row, config in enumerate(configs):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            
            if dataset in organized and config in organized[dataset]:
                cfg_data = organized[dataset][config]
                
                # Plot each replicate with different alpha
                colors = plt.cm.tab10.colors
                for rep, rep_data in sorted(cfg_data.items()):
                    epochs = np.arange(1, len(rep_data['train_losses']) + 1)
                    color_idx = (rep - 1) % len(colors)
                    
                    ax.plot(epochs, rep_data['train_losses'], 
                           color=colors[color_idx], alpha=0.5, linewidth=1)
                    ax.plot(epochs, rep_data['val_losses'], 
                           color=colors[color_idx], alpha=0.8, linewidth=1.5,
                           label=f"Rep {rep} (best: {rep_data['best_epoch']})")
                    ax.axvline(rep_data['best_epoch'], color=colors[color_idx], 
                              linestyle='--', alpha=0.5, linewidth=0.8)
                
                ax.legend(fontsize=6, loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
            
            ax.set_ylim(y_limits)
            ax.grid(alpha=0.3)
            
            # Labels
            if row == 0:
                ax.set_title(f'{dataset.capitalize()}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{config}\nLoss', fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Epoch')
    
    plt.suptitle('Learning Curves: All Configurations × Datasets × Replicates', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_all.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'learning_curves_all.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved all learning curves: learning_curves_all.png")
    
    # Also create per-dataset summary plots
    for dataset in datasets:
        if dataset not in organized:
            continue
        
        ds_configs = sorted(organized[dataset].keys(), 
                           key=lambda x: (int(x.split('_')[0].replace('ind', '')), x.split('_')[1]))
        n_configs = len(ds_configs)
        
        # Determine grid size
        n_cols_ds = min(4, n_configs)
        n_rows_ds = (n_configs + n_cols_ds - 1) // n_cols_ds
        
        fig, axes = plt.subplots(n_rows_ds, n_cols_ds, figsize=(4 * n_cols_ds, 3 * n_rows_ds), squeeze=False)
        axes_flat = axes.flatten()
        
        for idx, config in enumerate(ds_configs):
            ax = axes_flat[idx]
            cfg_data = organized[dataset][config]
            
            colors = plt.cm.tab10.colors
            for rep, rep_data in sorted(cfg_data.items()):
                epochs = np.arange(1, len(rep_data['train_losses']) + 1)
                color_idx = (rep - 1) % len(colors)
                
                ax.plot(epochs, rep_data['train_losses'], 
                       color=colors[color_idx], alpha=0.5, linewidth=1, label=f'Train r{rep}')
                ax.plot(epochs, rep_data['val_losses'], 
                       color=colors[color_idx], alpha=0.9, linewidth=1.5, label=f'Val r{rep}')
                ax.axvline(rep_data['best_epoch'], color=colors[color_idx], 
                          linestyle='--', alpha=0.5, linewidth=0.8)
            
            ax.set_title(config, fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_ylim(y_limits)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc='upper right')
        
        # Hide unused subplots
        for idx in range(len(ds_configs), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.suptitle(f'Learning Curves: {dataset.capitalize()} Dataset', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'learning_curves_{dataset}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved learning curves: learning_curves_{dataset}.png")


def plot_inducing_effect(df: pd.DataFrame, output_dir: Path):
    """
    Plot effect of number of inducing points on performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fixed y-limits for R² plot: clip to [0.90, 1.0] to focus on the
    # interesting range. The ran-km@100 outlier (R²≈0.65) is annotated
    # separately rather than stretching the entire axis.
    r2_limits = (0.90, 1.005)
    
    # Track outliers that fall below r2_limits for annotation
    outlier_annotations = []
    
    # Effect on R²
    ax = axes[0]
    for dataset in ['optimized', 'random', 'combined']:
        for kmeans in [True, False]:
            df_sub = df[(df['dataset'] == dataset) & (df['kmeans_init'] == kmeans)]
            if df_sub.empty:
                continue
            
            agg = df_sub.groupby('num_inducing')['test_r2'].agg(['mean', 'std'])
            linestyle = '-' if kmeans else '--'
            label = f"{dataset[:3]}-{'km' if kmeans else 'rd'}"
            
            # Check for points below the clipped y-axis
            for n_ind, row in agg.iterrows():
                if row['mean'] < r2_limits[0]:
                    outlier_annotations.append((n_ind, row['mean'], label))
            
            ax.errorbar(agg.index, agg['mean'], yerr=agg['std'], 
                       marker='o', linestyle=linestyle, label=label, capsize=3)
    
    # Annotate off-chart outliers
    for n_ind, r2_val, label in outlier_annotations:
        ax.annotate(
            f'{label}@{n_ind}: R²={r2_val:.2f}',
            xy=(n_ind, r2_limits[0]),
            xytext=(n_ind + 300, r2_limits[0] + 0.015),
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='red', alpha=0.9),
        )
    
    ax.set_xlabel('Number of Inducing Points')
    ax.set_ylabel('Test R²')
    ax.set_title('Effect of Inducing Points on R²')
    ax.set_ylim(r2_limits)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # Effect on training time
    ax = axes[1]
    for dataset in ['optimized', 'random', 'combined']:
        for kmeans in [True, False]:
            df_sub = df[(df['dataset'] == dataset) & (df['kmeans_init'] == kmeans)]
            if df_sub.empty:
                continue
            
            agg = df_sub.groupby('num_inducing')['training_time'].agg(['mean', 'std'])
            linestyle = '-' if kmeans else '--'
            label = f"{dataset[:3]}-{'km' if kmeans else 'rd'}"
            
            ax.errorbar(agg.index, agg['mean'], yerr=agg['std'], 
                       marker='o', linestyle=linestyle, label=label, capsize=3)
    
    ax.set_xlabel('Number of Inducing Points')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Effect of Inducing Points on Training Time')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inducing_points_effect.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'inducing_points_effect.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved inducing points effect: inducing_points_effect.png/.pdf")


def plot_kmeans_effect(df: pd.DataFrame, output_dir: Path):
    """
    Plot effect of K-means initialization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Compute global y-limits for consistent scaling across all datasets
    all_vals = []
    for dataset in ['optimized', 'random', 'combined']:
        df_ds = df[df['dataset'] == dataset]
        if not df_ds.empty:
            agg = df_ds.groupby(['num_inducing', 'kmeans_init'])['test_r2'].agg(['mean', 'std']).reset_index()
            all_vals.extend((agg['mean'] - agg['std']).tolist())
            all_vals.extend((agg['mean'] + agg['std']).tolist())
    
    if all_vals:
        margin = (max(all_vals) - min(all_vals)) * 0.05
        y_limits = (max(0, min(all_vals) - margin), min(1.0, max(all_vals) + margin))
    else:
        y_limits = (0, 1)
    
    for ax, dataset in zip(axes, ['optimized', 'random', 'combined']):
        df_ds = df[df['dataset'] == dataset]
        
        if df_ds.empty:
            ax.set_title(f'{dataset.capitalize()} (No data)')
            ax.set_ylim(y_limits)
            continue
        
        # Group by inducing points and kmeans
        agg = df_ds.groupby(['num_inducing', 'kmeans_init'])['test_r2'].agg(['mean', 'std']).reset_index()
        
        x = np.arange(len(agg['num_inducing'].unique()))
        width = 0.35
        
        for i, (kmeans, label) in enumerate([(False, 'Random Init'), (True, 'K-means Init')]):
            data = agg[agg['kmeans_init'] == kmeans]
            ax.bar(x + i * width, data['mean'], width, yerr=data['std'],
                   label=label, capsize=3, alpha=0.8)
        
        ax.set_xlabel('Inducing Points')
        ax.set_ylabel('Test R²')
        ax.set_title(f'{dataset.capitalize()} Dataset')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(sorted(agg['num_inducing'].unique()))
        ax.set_ylim(y_limits)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kmeans_effect.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'kmeans_effect.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved K-means effect: kmeans_effect.png")


def plot_calibration_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Plot uncertainty calibration analysis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Compute global axis limits for consistent scaling
    all_r2 = df['test_r2'].values
    all_cal = df['test_calibration_95ci'].values
    
    r2_margin = (np.max(all_r2) - np.min(all_r2)) * 0.05
    cal_margin = (np.max(all_cal) - np.min(all_cal)) * 0.05
    
    x_limits = (max(0, np.min(all_r2) - r2_margin), min(1.0, np.max(all_r2) + r2_margin))
    y_limits = (max(0, np.min(all_cal) - cal_margin), min(1.0, np.max(all_cal) + cal_margin))
    
    for ax, dataset in zip(axes, ['optimized', 'random', 'combined']):
        df_ds = df[df['dataset'] == dataset]
        
        if df_ds.empty:
            ax.set_title(f'{dataset.capitalize()} (No data)')
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            continue
        
        # Scatter: R² vs Calibration
        for kmeans in [True, False]:
            df_km = df_ds[df_ds['kmeans_init'] == kmeans]
            marker = 'o' if kmeans else 's'
            label = 'K-means' if kmeans else 'Random'
            
            for ind in df_km['num_inducing'].unique():
                df_ind = df_km[df_km['num_inducing'] == ind]
                ax.scatter(df_ind['test_r2'], df_ind['test_calibration_95ci'],
                          marker=marker, s=50 + ind/10, alpha=0.7,
                          label=f"{label} ({ind})")
        
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Ideal (0.95)')
        ax.set_xlabel('Test R²')
        ax.set_ylabel('95% CI Calibration')
        ax.set_title(f'{dataset.capitalize()} Dataset')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved calibration analysis: calibration_analysis.png")


def analyze_inference_timing(
    df: pd.DataFrame,
    results_dir: Path,
    output_dir: Path,
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000, 100000],
    n_warmup: int = 10,
    n_iterations: int = 100
) -> Optional[pd.DataFrame]:
    """
    Analyze inference timing for different inducing point configurations.
    
    Loads trained models (without K-means, optimized dataset) and measures
    prediction speed for different batch sizes.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Skipping inference timing analysis.")
        return None
    
    # Check if results already exist
    timing_csv_path = output_dir / 'inference_timing.csv'
    if timing_csv_path.exists():
        logger.info("\nFound existing inference timing results, loading from file...")
        timing_df = pd.read_csv(timing_csv_path)
        logger.info(f"Loaded {len(timing_df)} timing measurements")
        # Regenerate plots with existing data
        plot_inference_timing(timing_df, output_dir)
        return timing_df
    
    logger.info("\nAnalyzing inference timing...")
    
    
    # Find models for optimized dataset, no kmeans
    target_df = df[
        (df['dataset'] == 'optimized') & 
        (df['kmeans_init'] == False)
    ].copy()
    
    if target_df.empty:
        logger.warning("No models found for optimized dataset without K-means.")
        return None
    
    # Get one model per inducing points count
    models_to_test = target_df.groupby('num_inducing').first().reset_index()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timing_results = []
    
    for _, row in models_to_test.iterrows():
        num_inducing = row['num_inducing']
        run_name = row['run_name']
        model_path = results_dir / f"model_{run_name}.pth"
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            continue
        
        logger.info(f"\n  Testing {num_inducing} inducing points...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Get normalization parameters
            train_x_mean = checkpoint['train_x_mean'].to(device)
            train_x_std = checkpoint['train_x_std'].to(device)
            
            # Create dummy inducing points to reconstruct model
            inducing_points = torch.randn(num_inducing, 62, device=device)
            
            # Create model and likelihood
            model = SVGPModel(inducing_points).to(device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            
            # Load state dicts
            model.load_state_dict(checkpoint['model_state_dict'])
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            
            # Set to eval mode
            model.eval()
            likelihood.eval()
            
            # Test different batch sizes
            for batch_size in batch_sizes:
                # Create random test data
                test_x = torch.randn(batch_size, 62, device=device)
                test_x_norm = (test_x - train_x_mean) / (train_x_std + 1e-6)
                
                # Warmup
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    for _ in range(n_warmup):
                        _ = likelihood(model(test_x_norm))
                
                # Time inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times = []
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    for _ in range(n_iterations):
                        start = time.perf_counter()
                        _ = likelihood(model(test_x_norm))
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                
                # Compute statistics
                times = np.array(times) * 1000  # Convert to ms
                mean_time = np.mean(times)
                std_time = np.std(times)
                per_sample_time = mean_time / batch_size
                
                timing_results.append({
                    'num_inducing': num_inducing,
                    'batch_size': batch_size,
                    'mean_time_ms': mean_time,
                    'std_time_ms': std_time,
                    'per_sample_ms': per_sample_time,
                    'throughput': 1000 * batch_size / mean_time,  # samples/sec
                })
                
                logger.info(f"    Batch {batch_size:4d}: {mean_time:6.2f} ± {std_time:5.2f} ms "
                          f"({per_sample_time:5.2f} ms/sample)")
            
        except Exception as e:
            logger.error(f"Error testing model {run_name}: {e}")
            continue
    
    if not timing_results:
        logger.warning("No timing results collected.")
        return None
    
    timing_df = pd.DataFrame(timing_results)
    timing_df.to_csv(output_dir / 'inference_timing.csv', index=False)
    logger.info(f"\nSaved timing results: inference_timing.csv")
    
    # Plot timing results
    plot_inference_timing(timing_df, output_dir)
    
    return timing_df


def plot_inference_timing(timing_df: pd.DataFrame, output_dir: Path):
    """
    Plot inference timing analysis results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Time vs Batch Size for different inducing points
    ax = axes[0]
    for num_ind in sorted(timing_df['num_inducing'].unique()):
        data = timing_df[timing_df['num_inducing'] == num_ind]
        ax.errorbar(data['batch_size'], data['mean_time_ms'], 
                   yerr=data['std_time_ms'], marker='o', 
                   label=f"{num_ind} inducing", capsize=3)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time vs Batch Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    # Plot 2: Throughput vs Batch Size
    ax = axes[1]
    for num_ind in sorted(timing_df['num_inducing'].unique()):
        data = timing_df[timing_df['num_inducing'] == num_ind]
        ax.plot(data['batch_size'], data['throughput'], 
               marker='o', label=f"{num_ind} inducing")
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (samples/s)')
    ax.set_title('Throughput vs Batch Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    # Plot 3: Per-sample time vs Inducing Points
    ax = axes[2]
    
    # For each batch size, show per-sample time
    batch_sizes = sorted(timing_df['batch_size'].unique())
    x = np.arange(len(timing_df['num_inducing'].unique()))
    width = 0.2
    
    for i, batch_size in enumerate(batch_sizes):
        data = timing_df[timing_df['batch_size'] == batch_size]
        data = data.sort_values('num_inducing')
        offset = (i - len(batch_sizes)/2 + 0.5) * width
        ax.bar(x + offset, data['per_sample_ms'], width, 
               label=f"Batch {batch_size}", alpha=0.8)
    
    ax.set_xlabel('Number of Inducing Points')
    ax.set_ylabel('Per-Sample Inference Time (ms)')
    ax.set_title('Per-Sample Time vs Model Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(timing_df['num_inducing'].unique()))
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_timing.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'inference_timing.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("Saved inference timing plots: inference_timing.png")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    stat_tests: pd.DataFrame,
    best_configs: pd.DataFrame,
    output_dir: Path,
    timing_df: Optional[pd.DataFrame] = None
):
    """
    Generate a text report with all findings.
    """
    report_lines = [
        "=" * 70,
        "GP HYPERPARAMETER OPTIMIZATION ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Total runs analyzed: {len(df)}",
        f"Datasets: {', '.join(df['dataset'].unique())}",
        f"Configurations: {', '.join(df['config'].unique())}",
        f"Replicates per config: {df.groupby(['dataset', 'config']).size().min()}-{df.groupby(['dataset', 'config']).size().max()}",
        "",
        "=" * 70,
        "BEST CONFIGURATIONS (by Test R²)",
        "=" * 70,
    ]
    
    for _, row in best_configs.iterrows():
        report_lines.extend([
            "",
            f"Dataset: {row['dataset'].upper()}",
            f"  Best config: {row['config']}",
            f"  Inducing points: {row['num_inducing']}",
            f"  K-means init: {row['kmeans_init']}",
            f"  Test R²: {row['test_r2']:.4f}",
            f"  Test RMSE: {row['test_rmse']:.4f}",
            f"  Calibration: {row['test_calibration_95ci']:.4f}",
            f"  Avg training time: {row['training_time']:.1f}s",
        ])
    
    report_lines.extend([
        "",
        "=" * 70,
        "SUMMARY STATISTICS",
        "=" * 70,
        "",
    ])
    
    # Add summary table
    summary_str = summary.to_string(index=False)
    report_lines.extend(summary_str.split('\n'))
    
    report_lines.extend([
        "",
        "=" * 70,
        "STATISTICAL TESTS",
        "=" * 70,
        "",
    ])
    
    if len(stat_tests) > 0:
        for _, row in stat_tests.iterrows():
            sig_marker = "*" if row['significant'] else ""
            report_lines.append(
                f"{row['dataset']:10s} | {row['comparison']:20s} | "
                f"{row['condition']:10s} | p={row['p_value']:.4f}{sig_marker} | "
                f"{row['group1_mean']:.3f} vs {row['group2_mean']:.3f}"
            )
    else:
        report_lines.append("No statistical tests performed (insufficient data)")
    
    report_lines.extend([
        "",
        "=" * 70,
        "RECOMMENDATIONS",
        "=" * 70,
        "",
    ])
    
    # Generate recommendations based on results
    for dataset in df['dataset'].unique():
        best = best_configs[best_configs['dataset'] == dataset].iloc[0]
        report_lines.append(f"{dataset.upper()}:")
        report_lines.append(f"  Use {best['num_inducing']} inducing points with "
                          f"{'K-means' if best['kmeans_init'] else 'random'} initialization")
        report_lines.append(f"  Expected Test R²: {best['test_r2']:.3f}")
        report_lines.append("")
    
    # Check if more inducing points help
    avg_by_ind = df.groupby('num_inducing')['test_r2'].mean()
    if len(avg_by_ind) > 1:
        if avg_by_ind.idxmax() == avg_by_ind.index.max():
            report_lines.append("NOTE: More inducing points may further improve performance.")
        else:
            report_lines.append(f"NOTE: Optimal inducing points appears to be {avg_by_ind.idxmax()}.")
    
    # Check K-means effect
    avg_by_km = df.groupby('kmeans_init')['test_r2'].mean()
    if len(avg_by_km) == 2:
        better = 'K-means' if avg_by_km[True] > avg_by_km[False] else 'Random'
        diff = abs(avg_by_km[True] - avg_by_km[False])
        report_lines.append(f"NOTE: {better} initialization is on average {diff:.3f} R² better.")
    
    # Add inference timing section if available
    if timing_df is not None and not timing_df.empty:
        report_lines.extend([
            "",
            "=" * 70,
            "INFERENCE TIMING ANALYSIS",
            "=" * 70,
            "",
        ])
        
        for num_ind in sorted(timing_df['num_inducing'].unique()):
            data = timing_df[timing_df['num_inducing'] == num_ind]
            report_lines.append(f"Inducing Points: {num_ind}")
            for _, row in data.iterrows():
                report_lines.append(
                    f"  Batch {int(row['batch_size']):6d}: {row['mean_time_ms']:6.2f} ± {row['std_time_ms']:5.2f} ms "
                    f"({row['per_sample_ms']:5.2f} ms/sample, {row['throughput']:.0f} samples/s)"
                )
            report_lines.append("")
    
    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])
    
    # Write report
    report_path = output_dir / 'hpo_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved report: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze GP hyperparameter optimization results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/hyperparameterization",
        help="Directory containing HPO results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results-dir/analysis)"
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip inference timing analysis"
    )
    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations for timing (default: 10)"
    )
    parser.add_argument(
        "--timing-iterations",
        type=int,
        default=100,
        help="Number of timing iterations (default: 100)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GP HPO Analysis")
    logger.info("=" * 60)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Load data
    logger.info("\nLoading results...")
    df = load_all_results(results_dir)
    
    # Save raw data
    df.to_csv(output_dir / 'all_results.csv', index=False)
    logger.info(f"Saved raw data: all_results.csv")
    
    # Compute statistics
    logger.info("\nComputing statistics...")
    summary = compute_summary_statistics(df)
    summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
    
    stat_tests = perform_statistical_tests(df)
    stat_tests.to_csv(output_dir / 'statistical_tests.csv', index=False)
    
    best_configs = find_best_configurations(df)
    best_configs.to_csv(output_dir / 'best_configurations.csv', index=False)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_heatmap_by_dataset(df, output_dir)
    plot_bar_comparison(df, output_dir)
    plot_inducing_effect(df, output_dir)
    plot_kmeans_effect(df, output_dir)
    plot_calibration_analysis(df, output_dir)
    
    # Load and plot ALL learning curves
    logger.info("\nLoading all learning curves...")
    all_curves = load_learning_curves(results_dir, run_names=None)  # Load ALL
    if all_curves:
        logger.info(f"Loaded {len(all_curves)} learning curves")
        plot_all_learning_curves(all_curves, output_dir)
        
        # Also plot the legacy 6-curve summary for best runs
        best_run_names = best_configs['run_name'].tolist() if 'run_name' in best_configs.columns else []
        if not best_run_names:
            best_run_names = df.groupby('dataset').apply(
                lambda x: x.loc[x['test_r2'].idxmax(), 'run_name']
            ).tolist()
        best_curves = {k: v for k, v in all_curves.items() if k in best_run_names}
        if best_curves:
            plot_learning_curves(best_curves, output_dir)
    
    # Inference timing analysis
    timing_df = None
    if not args.skip_timing:
        logger.info("\nPerforming inference timing analysis...")
        timing_df = analyze_inference_timing(
            df, results_dir, output_dir,
            n_warmup=args.timing_warmup,
            n_iterations=args.timing_iterations
        )
    else:
        logger.info("\nSkipping inference timing analysis (--skip-timing flag set)")
    
    # Generate report
    logger.info("\nGenerating report...")
    generate_report(df, summary, stat_tests, best_configs, output_dir, timing_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
