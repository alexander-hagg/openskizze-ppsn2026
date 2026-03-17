#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Analyze U-NET experiment results.

Compares performance across:
- Training data: SAIL vs Random
- Loss function: MSE vs MSE+Gradient
- Multiple replicates

Usage:
    python experiments/analyze_unet_results.py --results-dir results/exp5_unet/unet_experiment
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output variable names
OUTPUT_VARS = ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    rows = []
    
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / 'results.json'
        if not results_file.exists():
            logger.warning(f"No results.json in {run_dir.name}")
            continue
        
        with open(results_file) as f:
            data = json.load(f)
        
        # Parse run name: {data_type}_{loss_type}_seed{seed}
        # Expected formats: sail_mse_seed42, sail_mse_grad_seed42, random_mse_seed42, etc.
        parts = run_dir.name.split('_')
        
        # Find the seed part (starts with 'seed')
        seed_idx = None
        for i, part in enumerate(parts):
            if part.startswith('seed'):
                seed_idx = i
                break
        
        if seed_idx is None:
            logger.warning(f"Could not parse seed from {run_dir.name}, skipping")
            continue
        
        data_type = parts[0]
        loss_type = '_'.join(parts[1:seed_idx])  # Everything between data_type and seed
        seed = int(parts[seed_idx].replace('seed', ''))
        
        row = {
            'run_name': run_dir.name,
            'data_type': data_type,
            'loss_type': loss_type,
            'seed': seed,
            'best_epoch': data['best_epoch'],
            'best_val_loss': data['best_val_loss'],
            'total_time': data['total_time'],
            'n_train': data['data']['n_train'],
            'n_test': data['data']['n_test'],
        }
        
        # Add per-variable metrics
        for var in OUTPUT_VARS:
            if var in data['test_metrics']:
                for metric in ['mse', 'mae', 'r2']:
                    row[f'{var}_{metric}'] = data['test_metrics'][var][metric]
        
        # Add overall metrics
        for metric in ['mse', 'mae', 'r2']:
            row[f'overall_{metric}'] = data['test_metrics']['overall'][metric]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by data_type and loss_type
    grouped = df.groupby(['data_type', 'loss_type'])
    
    # 1. Overall R² comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = []
    means = []
    stds = []
    
    for (data_type, loss_type), group in grouped:
        configs.append(f"{data_type}\n{loss_type}")
        means.append(group['overall_r2'].mean())
        stds.append(group['overall_r2'].std())
    
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#2ecc71', '#27ae60', '#3498db', '#2980b9'])
    ax.set_ylabel('R² (mean ± std)')
    ax.set_title('Overall Test R² by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target R²=0.9')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.3f}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_r2_comparison.png', dpi=150)
    plt.savefig(output_dir / 'overall_r2_comparison.pdf')
    plt.close()
    
    # 2. Per-variable R² heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use consistent color scale across both heatmaps
    for idx, loss_type in enumerate(['mse', 'mse_grad']):
        ax = axes[idx]
        
        # Build matrix: rows = data_type, cols = output variables
        data_types = ['sail', 'random']
        matrix = np.zeros((len(data_types), len(OUTPUT_VARS)))
        
        for i, data_type in enumerate(data_types):
            subset = df[(df['data_type'] == data_type) & (df['loss_type'] == loss_type)]
            for j, var in enumerate(OUTPUT_VARS):
                matrix[i, j] = subset[f'{var}_r2'].mean()
        
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        ax.set_xticks(np.arange(len(OUTPUT_VARS)))
        ax.set_yticks(np.arange(len(data_types)))
        ax.set_xticklabels(OUTPUT_VARS)
        ax.set_yticklabels(data_types)
        ax.set_title(f'R² by Variable ({loss_type})')
        
        # Add text annotations
        for i in range(len(data_types)):
            for j in range(len(OUTPUT_VARS)):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', 
                       color='white' if matrix[i, j] < 0.5 else 'black')
        
        plt.colorbar(im, ax=ax, label='R²')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_variable_r2_heatmap.png', dpi=150)
    plt.savefig(output_dir / 'per_variable_r2_heatmap.pdf')
    plt.close()
    
    # 3. Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = []
    times = []
    stds = []
    
    for (data_type, loss_type), group in grouped:
        configs.append(f"{data_type}\n{loss_type}")
        times.append(group['total_time'].mean() / 60)
        stds.append(group['total_time'].std() / 60)
    
    x = np.arange(len(configs))
    ax.bar(x, times, yerr=stds, capsize=5, color=['#2ecc71', '#27ae60', '#3498db', '#2980b9'])
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time by Configuration')
    ax.set_ylim(0, max(times) * 1.2)  # Consistent y-axis starting at 0
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time.png', dpi=150)
    plt.savefig(output_dir / 'training_time.pdf')
    plt.close()
    
    # 4. Learning curves (if history files exist)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    configs = list(grouped.groups.keys())
    
    # First pass: collect all loss values to determine y-axis range
    all_train_losses = []
    all_val_losses = []
    
    for idx, (data_type, loss_type) in enumerate(configs[:4]):
        ax = axes[idx]
        group = grouped.get_group((data_type, loss_type))
        
        for _, row in group.iterrows():
            # Try to find history file
            for parent in [Path('results/exp5_unet/unet_experiment'), Path('.')]:
                hf = parent / row['run_name'] / 'history.json'
                if hf.exists():
                    with open(hf) as f:
                        history = json.load(f)
                    all_train_losses.extend(history['train_loss'])
                    all_val_losses.extend(history['val_loss'])
                    ax.plot(history['epoch'], history['train_loss'], 
                           alpha=0.7, label=f"Train (seed {row['seed']})")
                    ax.plot(history['epoch'], history['val_loss'], '--',
                           alpha=0.7, label=f"Val (seed {row['seed']})")
                    break
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{data_type} + {loss_type}')
        ax.legend(fontsize=8)
        ax.set_yscale('log')
    
    # Set consistent y-axis range across all subplots
    if all_train_losses and all_val_losses:
        all_losses = all_train_losses + all_val_losses
        y_min = min(all_losses)
        y_max = max(all_losses)
        for ax in axes:
            ax.set_ylim(y_min * 0.5, y_max * 2)  # Add some padding in log space
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150)
    plt.savefig(output_dir / 'learning_curves.pdf')
    plt.close()
    
    logger.info(f"Saved plots to {output_dir}")


def plot_predictions(results_dir: Path, output_dir: Path):
    """Plot ground truth vs predicted samples for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid dependency issues if not needed
    try:
        import sys
        # Add project root to path (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from models.unet import UNet, UNetConfig
        from train_unet_klam import KLAMSpatialDataset
    except ImportError as e:
        logger.warning(f"Could not import U-NET model: {e}, skipping prediction plots")
        return
    
    # Load one run from each configuration
    configs = {}
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / 'results.json'
        model_file = run_dir / 'best_model.pth'
        if not results_file.exists() or not model_file.exists():
            continue
        
        # Parse config
        parts = run_dir.name.split('_')
        seed_idx = None
        for i, part in enumerate(parts):
            if part.startswith('seed'):
                seed_idx = i
                break
        if seed_idx is None:
            continue
        
        data_type = parts[0]
        loss_type = '_'.join(parts[1:seed_idx])
        config_key = f"{data_type}_{loss_type}"
        
        # Use first replicate for each config
        if config_key not in configs:
            configs[config_key] = {
                'run_dir': run_dir,
                'data_type': data_type,
                'loss_type': loss_type
            }
    
    if not configs:
        logger.warning("No model files found for prediction plots")
        return
    
    # Output variables
    output_vars = ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']
    
    # Create figures for each output variable UPFRONT
    n_configs = len(configs)
    n_samples = 6
    
    figs = {}
    axes_dict = {}
    for var_name in output_vars:
        fig, axes = plt.subplots(n_configs, n_samples, figsize=(4*n_samples, 3*n_configs))
        if n_configs == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Ground Truth vs Predictions: {var_name}', fontsize=28, y=0.995)
        figs[var_name] = fig
        axes_dict[var_name] = axes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process each config ONCE
    for config_idx, (config_key, config_info) in enumerate(sorted(configs.items())):
        run_dir = config_info['run_dir']
        data_type = config_info['data_type']
        loss_type = config_info['loss_type']
        
        logger.info(f"Generating predictions for {config_key}")
        
        # Load model ONCE
        with open(run_dir / 'results.json') as f:
            results = json.load(f)
        
        checkpoint = torch.load(run_dir / 'best_model.pth', map_location=device)
        model_config = checkpoint['config']
        
        config = UNetConfig(**model_config)
        model = UNet(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load dataset ONCE
        data_files = [Path(f) for f in results['data']['files']]
        existing_files = [f for f in data_files if f.exists()]
        
        if not existing_files:
            logger.warning(f"Data files not found for {config_key}, skipping predictions")
            continue
        
        dataset_full = KLAMSpatialDataset(existing_files)
        train_size = results['data']['n_train']
        test_size = results['data']['n_test']
        val_size = results['data']['n_val']
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset_full, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(results['training']['seed'])
        )
        
        # Get 3 samples from train, 3 from test
        train_indices = [0, train_size // 2, train_size - 1]
        test_indices = [0, test_size // 2, test_size - 1]
        
        sample_datasets = [
            (train_dataset, train_indices, "Train"),
            (test_dataset, test_indices, "Test")
        ]
        
        sample_idx = 0
        with torch.no_grad():
            for dataset, indices, split_name in sample_datasets:
                for idx in indices:
                    inputs, targets = dataset[idx]
                    inputs = inputs.unsqueeze(0).to(device)
                    targets = targets.numpy()
                    
                    # Predict ONCE for all variables
                    outputs = model(inputs).cpu().numpy()[0]
                    
                    # Plot ALL variables for this sample
                    for var_idx, var_name in enumerate(output_vars):
                        ax = axes_dict[var_name][config_idx, sample_idx]
                        
                        # Get current variable channel
                        gt = targets[var_idx]  # (H, W)
                        pred = outputs[var_idx]  # (H, W)
                        
                        # Plot side by side
                        combined = np.concatenate([gt, pred], axis=1)
                        im = ax.imshow(combined, cmap='viridis', aspect='equal', origin='lower')
                        
                        # Add vertical line to separate GT and Pred
                        ax.axvline(x=gt.shape[1] - 0.5, color='white', linewidth=2)
                        
                        # Title
                        if config_idx == 0:
                            ax.set_title(f"{split_name} #{idx}", fontsize=18)
                        if sample_idx == 0:
                            ax.set_ylabel(f"{data_type}\n{loss_type}", fontsize=16, rotation=0, 
                                         ha='right', va='center', labelpad=20)
                        
                        # Labels at the top
                        if config_idx == 0:
                            y_pos = combined.shape[0] + combined.shape[0] * 0.05
                            ax.text(gt.shape[1]//2, y_pos, 'GT', ha='center', fontsize=16, 
                                   weight='bold')
                            ax.text(gt.shape[1] + gt.shape[1]//2, y_pos, 'Pred', ha='center', 
                                   fontsize=16, weight='bold')
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xlim(-0.5, combined.shape[1] - 0.5)
                        ax.set_ylim(-0.5, combined.shape[0] - 0.5)
                    
                    sample_idx += 1
    
    # Save all figures at the end
    for var_name in output_vars:
        fig = figs[var_name]
        fig.tight_layout()
        fig.savefig(output_dir / f'predictions_{var_name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / f'predictions_{var_name}.pdf', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved predictions_{var_name}.png and .pdf")
    
    logger.info(f"Saved all prediction comparisons to {output_dir}")


def analyze_data_similarity(results_dir: Path, output_dir: Path):
    """Analyze similarity between train/val/test splits to check for data leakage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Analyzing data similarity between splits...")
    
    # Import here to avoid dependency issues
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train_unet_klam import KLAMSpatialDataset
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        logger.warning(f"Could not import required modules: {e}, skipping similarity analysis")
        return
    
    # Load one configuration to analyze
    run_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir() and (d / 'results.json').exists()]
    if not run_dirs:
        logger.warning("No valid run directories found for similarity analysis")
        return
    
    # Use first run for analysis
    run_dir = run_dirs[0]
    logger.info(f"Using {run_dir.name} for similarity analysis")
    
    with open(run_dir / 'results.json') as f:
        results = json.load(f)
    
    # Load dataset
    data_files = [Path(f) for f in results['data']['files']]
    existing_files = [f for f in data_files if f.exists()]
    
    if not existing_files:
        logger.warning("Data files not found, skipping similarity analysis")
        return
    
    dataset_full = KLAMSpatialDataset(existing_files)
    train_size = results['data']['n_train']
    val_size = results['data']['n_val']
    test_size = results['data']['n_test']
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_full, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(results['training']['seed'])
    )
    
    logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Extract building layouts (inputs) for similarity analysis
    logger.info("Extracting building layouts...")
    
    def extract_features(dataset, max_samples=500):
        """Extract flattened building layouts."""
        features = []
        n_samples = min(len(dataset), max_samples)
        for i in range(n_samples):
            inputs, _ = dataset[i]
            # Use buildings channel (index 1) and flatten
            buildings = inputs[1].numpy().flatten()
            features.append(buildings)
        return np.array(features)
    
    train_features = extract_features(train_dataset)
    val_features = extract_features(val_dataset)
    test_features = extract_features(test_dataset)
    
    logger.info(f"Extracted features - Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")
    
    # Compute pairwise similarities
    logger.info("Computing pairwise similarities...")
    
    # Train vs Test
    train_test_sim = cosine_similarity(test_features, train_features)
    max_train_test_sim = train_test_sim.max(axis=1)  # For each test sample, max similarity to any train sample
    
    # Val vs Test
    val_test_sim = cosine_similarity(test_features, val_features)
    max_val_test_sim = val_test_sim.max(axis=1)
    
    # Val vs Train
    val_train_sim = cosine_similarity(val_features, train_features)
    max_val_train_sim = val_train_sim.max(axis=1)
    
    # Self-similarity for reference (within test set)
    test_test_sim = cosine_similarity(test_features, test_features)
    # Remove diagonal (self-similarity = 1.0)
    np.fill_diagonal(test_test_sim, -1)
    max_test_test_sim = test_test_sim.max(axis=1)
    
    logger.info("Similarity statistics:")
    logger.info(f"  Test vs Train - Mean: {max_train_test_sim.mean():.3f}, Max: {max_train_test_sim.max():.3f}")
    logger.info(f"  Test vs Val   - Mean: {max_val_test_sim.mean():.3f}, Max: {max_val_test_sim.max():.3f}")
    logger.info(f"  Val vs Train  - Mean: {max_val_train_sim.mean():.3f}, Max: {max_val_train_sim.max():.3f}")
    logger.info(f"  Test vs Test  - Mean: {max_test_test_sim.mean():.3f}, Max: {max_test_test_sim.max():.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histogram of maximum similarities
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(max_train_test_sim, bins=bins, alpha=0.7, label='Test→Train', color='#3498db')
    ax.hist(max_val_test_sim, bins=bins, alpha=0.7, label='Test→Val', color='#e74c3c')
    ax.hist(max_val_train_sim, bins=bins, alpha=0.7, label='Val→Train', color='#2ecc71')
    ax.hist(max_test_test_sim, bins=bins, alpha=0.7, label='Test→Test', color='#95a5a6')
    ax.axvline(x=0.99, color='red', linestyle='--', linewidth=2, label='Duplicate threshold (0.99)')
    ax.set_xlabel('Maximum Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Maximum Similarities\n(Data Leakage Check)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. CDF plot
    ax = axes[0, 1]
    sorted_train_test = np.sort(max_train_test_sim)
    sorted_val_test = np.sort(max_val_test_sim)
    sorted_val_train = np.sort(max_val_train_sim)
    sorted_test_test = np.sort(max_test_test_sim)
    
    cdf_x = np.linspace(0, 1, 100)
    ax.plot(sorted_train_test, np.linspace(0, 1, len(sorted_train_test)), 
            label='Test→Train', linewidth=2, color='#3498db')
    ax.plot(sorted_val_test, np.linspace(0, 1, len(sorted_val_test)), 
            label='Test→Val', linewidth=2, color='#e74c3c')
    ax.plot(sorted_val_train, np.linspace(0, 1, len(sorted_val_train)), 
            label='Val→Train', linewidth=2, color='#2ecc71')
    ax.plot(sorted_test_test, np.linspace(0, 1, len(sorted_test_test)), 
            label='Test→Test', linewidth=2, color='#95a5a6', linestyle='--')
    ax.axvline(x=0.99, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Maximum Cosine Similarity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF of Maximum Similarities')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    
    # 3. Summary statistics table
    ax = axes[1, 0]
    ax.axis('off')
    
    stats_data = [
        ['Split Comparison', 'Mean', 'Median', 'Max', '> 0.99', '> 0.95'],
        ['Test → Train', 
         f'{max_train_test_sim.mean():.3f}',
         f'{np.median(max_train_test_sim):.3f}',
         f'{max_train_test_sim.max():.3f}',
         f'{(max_train_test_sim > 0.99).sum()}',
         f'{(max_train_test_sim > 0.95).sum()}'],
        ['Test → Val',
         f'{max_val_test_sim.mean():.3f}',
         f'{np.median(max_val_test_sim):.3f}',
         f'{max_val_test_sim.max():.3f}',
         f'{(max_val_test_sim > 0.99).sum()}',
         f'{(max_val_test_sim > 0.95).sum()}'],
        ['Val → Train',
         f'{max_val_train_sim.mean():.3f}',
         f'{np.median(max_val_train_sim):.3f}',
         f'{max_val_train_sim.max():.3f}',
         f'{(max_val_train_sim > 0.99).sum()}',
         f'{(max_val_train_sim > 0.95).sum()}'],
        ['Test → Test',
         f'{max_test_test_sim.mean():.3f}',
         f'{np.median(max_test_test_sim):.3f}',
         f'{max_test_test_sim.max():.3f}',
         f'{(max_test_test_sim > 0.99).sum()}',
         f'{(max_test_test_sim > 0.95).sum()}'],
    ]
    
    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Similarity Statistics Summary\n(Values > 0.99 indicate potential duplicates)', 
                 fontsize=11, pad=20)
    
    # 4. Interpretation guide
    ax = axes[1, 1]
    ax.axis('off')
    
    interpretation = [
        "Data Leakage Assessment:",
        "",
        "✓ Low similarity (< 0.95): Good separation, no leakage",
        "⚠ High similarity (0.95-0.99): Similar samples, monitor",
        "✗ Very high similarity (> 0.99): Likely duplicates, data leakage!",
        "",
        "Expected patterns:",
        "• Test→Test similarity: Baseline for comparison",
        "• Test→Train/Val should be lower than Test→Test",
        "• If Test→Train ≈ Test→Test: Severe data leakage",
        "",
        f"Analysis Results:",
        f"• Samples with similarity > 0.99:",
        f"  - Test→Train: {(max_train_test_sim > 0.99).sum()} / {len(max_train_test_sim)}",
        f"  - Test→Val: {(max_val_test_sim > 0.99).sum()} / {len(max_val_test_sim)}",
        f"  - Val→Train: {(max_val_train_sim > 0.99).sum()} / {len(max_val_train_sim)}",
        "",
        "Conclusion:",
    ]
    
    # Add conclusion based on analysis
    n_duplicates = (max_train_test_sim > 0.99).sum()
    if n_duplicates == 0:
        interpretation.append("✓ No duplicates detected - Clean split!")
    elif n_duplicates < len(max_train_test_sim) * 0.01:
        interpretation.append(f"⚠ {n_duplicates} potential duplicates (<1%) - Acceptable")
    else:
        interpretation.append(f"✗ {n_duplicates} duplicates ({n_duplicates/len(max_train_test_sim)*100:.1f}%) - Data leakage!")
    
    ax.text(0.05, 0.95, '\n'.join(interpretation), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_similarity_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'data_similarity_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity analysis to {output_dir}")
    
    # Save detailed statistics to JSON
    similarity_stats = {
        'dataset_sizes': {
            'train': train_size,
            'val': val_size,
            'test': test_size,
        },
        'test_vs_train': {
            'mean': float(max_train_test_sim.mean()),
            'median': float(np.median(max_train_test_sim)),
            'max': float(max_train_test_sim.max()),
            'min': float(max_train_test_sim.min()),
            'n_above_0.99': int((max_train_test_sim > 0.99).sum()),
            'n_above_0.95': int((max_train_test_sim > 0.95).sum()),
            'n_above_0.90': int((max_train_test_sim > 0.90).sum()),
        },
        'test_vs_val': {
            'mean': float(max_val_test_sim.mean()),
            'median': float(np.median(max_val_test_sim)),
            'max': float(max_val_test_sim.max()),
            'min': float(max_val_test_sim.min()),
            'n_above_0.99': int((max_val_test_sim > 0.99).sum()),
            'n_above_0.95': int((max_val_test_sim > 0.95).sum()),
            'n_above_0.90': int((max_val_test_sim > 0.90).sum()),
        },
        'val_vs_train': {
            'mean': float(max_val_train_sim.mean()),
            'median': float(np.median(max_val_train_sim)),
            'max': float(max_val_train_sim.max()),
            'min': float(max_val_train_sim.min()),
            'n_above_0.99': int((max_val_train_sim > 0.99).sum()),
            'n_above_0.95': int((max_val_train_sim > 0.95).sum()),
            'n_above_0.90': int((max_val_train_sim > 0.90).sum()),
        },
        'test_vs_test_baseline': {
            'mean': float(max_test_test_sim.mean()),
            'median': float(np.median(max_test_test_sim)),
            'max': float(max_test_test_sim.max()),
        }
    }
    
    with open(output_dir / 'similarity_statistics.json', 'w') as f:
        json.dump(similarity_stats, f, indent=2)
    
    logger.info(f"Saved similarity statistics to {output_dir / 'similarity_statistics.json'}")


def generate_summary(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics and markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    summary = df.groupby(['data_type', 'loss_type']).agg({
        'overall_r2': ['mean', 'std'],
        'overall_mse': ['mean', 'std'],
        'overall_mae': ['mean', 'std'],
        'best_epoch': 'mean',
        'total_time': 'mean',
    }).round(4)
    
    summary.to_csv(output_dir / 'summary_statistics.csv')
    
    # Generate markdown report
    report = [
        "# U-NET Experiment Results",
        "",
        "## Overview",
        "",
        f"- Total runs: {len(df)}",
        f"- Data types: {df['data_type'].unique().tolist()}",
        f"- Loss types: {df['loss_type'].unique().tolist()}",
        f"- Replicates per config: {df.groupby(['data_type', 'loss_type']).size().mean():.0f}",
        "",
        "## Summary Statistics",
        "",
        "| Data Type | Loss Type | R² (mean±std) | MSE (mean±std) | Time (min) |",
        "|-----------|-----------|---------------|----------------|------------|",
    ]
    
    for (data_type, loss_type), group in df.groupby(['data_type', 'loss_type']):
        r2_mean = group['overall_r2'].mean()
        r2_std = group['overall_r2'].std()
        mse_mean = group['overall_mse'].mean()
        mse_std = group['overall_mse'].std()
        time_mean = group['total_time'].mean() / 60
        
        report.append(
            f"| {data_type} | {loss_type} | {r2_mean:.3f}±{r2_std:.3f} | "
            f"{mse_mean:.4f}±{mse_std:.4f} | {time_mean:.1f} |"
        )
    
    report.extend([
        "",
        "## Per-Variable R²",
        "",
        "| Data | Loss | Ex | Hx | uq | vq | uz | vz |",
        "|------|------|----|----|----|----|----|----|",
    ])
    
    for (data_type, loss_type), group in df.groupby(['data_type', 'loss_type']):
        row = f"| {data_type} | {loss_type} |"
        for var in OUTPUT_VARS:
            r2 = group[f'{var}_r2'].mean()
            row += f" {r2:.2f} |"
        report.append(row)
    
    report.extend([
        "",
        "## Key Findings",
        "",
        "1. **Best Configuration**: ",
        f"   - Data: {df.loc[df['overall_r2'].idxmax(), 'data_type']}",
        f"   - Loss: {df.loc[df['overall_r2'].idxmax(), 'loss_type']}",
        f"   - R²: {df['overall_r2'].max():.3f}",
        "",
        "2. **Gradient Loss Effect**:",
    ])
    
    # Compare MSE vs MSE+Grad
    for data_type in df['data_type'].unique():
        mse_r2 = df[(df['data_type'] == data_type) & (df['loss_type'] == 'mse')]['overall_r2'].mean()
        grad_r2 = df[(df['data_type'] == data_type) & (df['loss_type'] == 'mse_grad')]['overall_r2'].mean()
        diff = grad_r2 - mse_r2
        report.append(f"   - {data_type}: MSE+Grad {'improves' if diff > 0 else 'worsens'} by {abs(diff):.3f}")
    
    report.extend([
        "",
        "3. **Training Data Effect**:",
    ])
    
    for loss_type in df['loss_type'].unique():
        sail_r2 = df[(df['data_type'] == 'sail') & (df['loss_type'] == loss_type)]['overall_r2'].mean()
        random_r2 = df[(df['data_type'] == 'random') & (df['loss_type'] == loss_type)]['overall_r2'].mean()
        diff = sail_r2 - random_r2
        report.append(f"   - {loss_type}: SAIL {'better' if diff > 0 else 'worse'} than Random by {abs(diff):.3f}")
    
    report.extend([
        "",
        "## Figures",
        "",
        "- `overall_r2_comparison.png/.pdf` - Bar chart of R² by configuration",
        "- `per_variable_r2_heatmap.png/.pdf` - Heatmap of R² per output variable",
        "- `learning_curves.png/.pdf` - Training/validation loss curves",
        "- `training_time.png/.pdf` - Training time comparison",
        "- `data_similarity_analysis.png/.pdf` - Data leakage check (similarity between splits)",
        "- `predictions_Ex.png/.pdf` - Ground truth vs predicted samples for Ex (cold air content)",
        "- `predictions_Hx.png/.pdf` - Ground truth vs predicted samples for Hx (cold air height)",
        "- `predictions_uq.png/.pdf` - Ground truth vs predicted samples for uq (u-velocity @ 2m)",
        "- `predictions_vq.png/.pdf` - Ground truth vs predicted samples for vq (v-velocity @ 2m)",
        "- `predictions_uz.png/.pdf` - Ground truth vs predicted samples for uz (column-avg u-velocity)",
        "- `predictions_vz.png/.pdf` - Ground truth vs predicted samples for vz (column-avg v-velocity)",
        "",
        "## Data Files",
        "",
        "- `all_results.csv` - Raw results for all runs",
        "- `summary_statistics.csv` - Aggregated statistics by configuration",
        "- `similarity_statistics.json` - Detailed similarity analysis between splits",
    ])
    
    with open(output_dir / 'SUMMARY.md', 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Saved summary to {output_dir / 'SUMMARY.md'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze U-NET experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/exp5_unet/unet_experiment",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results_dir/analysis)"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    
    logger.info(f"Loading results from {results_dir}")
    df = load_results(results_dir)
    
    if len(df) == 0:
        logger.error("No results found!")
        return
    
    logger.info(f"Found {len(df)} runs")
    
    # Save raw data
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Generate plots
    logger.info("Generating plots...")
    plot_comparison(df, output_dir)
    
    # Generate prediction visualizations
    logger.info("Generating prediction visualizations...")
    plot_predictions(results_dir, output_dir)
    
    # Analyze data similarity
    logger.info("Analyzing data similarity...")
    analyze_data_similarity(results_dir, output_dir)
    
    # Generate summary
    logger.info("Generating summary...")
    generate_summary(df, output_dir)
    
    # Print quick summary
    print("\n" + "=" * 60)
    print("Quick Summary")
    print("=" * 60)
    print(df.groupby(['data_type', 'loss_type'])['overall_r2'].agg(['mean', 'std']).round(3))
    print("=" * 60)


if __name__ == "__main__":
    main()
