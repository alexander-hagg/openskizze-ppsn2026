#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Analyze Experiment 7 Results: Multi-Scale vs Single-Scale U-Nets

Creates comparison plots and tables:
- R² comparison per parcel size
- Training time comparison
- Model size comparison
- Per-field accuracy breakdown
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results(results_file: Path) -> Dict:
    """Load training results JSON."""
    logger.info(f"Loading results from {results_file}")
    with open(results_file) as f:
        return json.load(f)


def create_comparison_table(results: Dict, output_dir: Path):
    """Create comparison table of single vs multi-scale models."""
    logger.info("Creating comparison table...")
    
    rows = []
    
    # Single-scale models
    if 'single_scale' in results:
        for result in results['single_scale']:
            size = result['parcel_size']
            metrics = result['best_metrics']
            rows.append({
                'Model': f'Single-{size}m',
                'Parcel Size': f'{size}m',
                'R²': metrics['r2'],
                'MSE': metrics['mse'],
                'MAE': metrics['mae'],
                'Training Time (min)': result['training_time'] / 60,
            })
    
    # Multi-scale model
    if 'multi_scale' in results:
        multi_result = results['multi_scale']
        for size, metrics in multi_result['best_metrics_per_size'].items():
            rows.append({
                'Model': 'Multi-Scale',
                'Parcel Size': f'{size}m',
                'R²': metrics['r2'],
                'MSE': metrics['mse'],
                'MAE': metrics['mae'],
                'Training Time (min)': multi_result['training_time'] / 60,
            })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Saved table to {csv_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def plot_r2_comparison(results: Dict, output_dir: Path):
    """Plot R² comparison across parcel sizes."""
    logger.info("Creating R² comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parcel_sizes = []
    single_r2 = []
    multi_r2 = []
    
    # Extract single-scale results
    if 'single_scale' in results:
        single_dict = {r['parcel_size']: r['best_metrics']['r2'] 
                      for r in results['single_scale']}
        parcel_sizes = sorted(single_dict.keys())
        single_r2 = [single_dict[size] for size in parcel_sizes]
    
    # Extract multi-scale results
    if 'multi_scale' in results:
        multi_dict = {int(size): metrics['r2'] 
                     for size, metrics in results['multi_scale']['best_metrics_per_size'].items()}
        multi_r2 = [multi_dict.get(size, np.nan) for size in parcel_sizes]
    
    x = np.arange(len(parcel_sizes))
    width = 0.35
    
    ax.bar(x - width/2, single_r2, width, label='Single-Scale', alpha=0.8)
    ax.bar(x + width/2, multi_r2, width, label='Multi-Scale', alpha=0.8)
    
    ax.set_xlabel('Parcel Size (m)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Single-Scale vs Multi-Scale U-Net: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}m' for s in parcel_sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (s, m) in enumerate(zip(single_r2, multi_r2)):
        ax.text(i - width/2, s + 0.01, f'{s:.3f}', ha='center', va='bottom', fontsize=9)
        if not np.isnan(m):
            ax.text(i + width/2, m + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / 'r2_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    plt.close()


def plot_per_field_accuracy(results: Dict, output_dir: Path):
    """Plot per-field R² breakdown."""
    logger.info("Creating per-field accuracy plot...")
    
    field_names = ['uq', 'vq', 'uz', 'vz', 'Ex', 'Hx']
    
    # Collect data for one parcel size (e.g., 51m)
    target_size = 51
    
    single_r2_per_field = []
    multi_r2_per_field = []
    
    # Single-scale
    if 'single_scale' in results:
        for result in results['single_scale']:
            if result['parcel_size'] == target_size:
                metrics = result['best_metrics']
                single_r2_per_field = [metrics[f'r2_{field}'] for field in field_names]
                break
    
    # Multi-scale
    if 'multi_scale' in results:
        metrics = results['multi_scale']['best_metrics_per_size'].get(str(target_size), {})
        multi_r2_per_field = [metrics.get(f'r2_{field}', np.nan) for field in field_names]
    
    if not single_r2_per_field and not multi_r2_per_field:
        logger.warning(f"No data found for parcel size {target_size}m")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(field_names))
    width = 0.35
    
    if single_r2_per_field:
        ax.bar(x - width/2, single_r2_per_field, width, label='Single-Scale', alpha=0.8)
    if multi_r2_per_field:
        ax.bar(x + width/2, multi_r2_per_field, width, label='Multi-Scale', alpha=0.8)
    
    ax.set_xlabel('Output Field', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'Per-Field Accuracy Comparison ({target_size}m Parcel)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(field_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plot_path = output_dir / f'per_field_accuracy_{target_size}m.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    plt.close()


def plot_training_time(results: Dict, output_dir: Path):
    """Plot training time comparison."""
    logger.info("Creating training time plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = []
    times = []
    
    if 'single_scale' in results:
        for result in results['single_scale']:
            models.append(f"Single-{result['parcel_size']}m")
            times.append(result['training_time'] / 60)  # Convert to minutes
    
    if 'multi_scale' in results:
        models.append('Multi-Scale\n(27m+51m+69m)')
        times.append(results['multi_scale']['training_time'] / 60)
    
    colors = ['#1f77b4'] * (len(times) - 1) + ['#ff7f0e']
    bars = ax.bar(range(len(models)), times, color=colors, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time_val:.1f} min\n({time_val/60:.1f} hr)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_time.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    plt.close()


def create_summary_report(results: Dict, output_dir: Path):
    """Create text summary report."""
    logger.info("Creating summary report...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EXPERIMENT 7: MULTI-SCALE U-NET COMPARISON - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Single-scale results
    if 'single_scale' in results:
        report_lines.append("SINGLE-SCALE U-NETS:")
        report_lines.append("-" * 40)
        for result in results['single_scale']:
            size = result['parcel_size']
            metrics = result['best_metrics']
            time_min = result['training_time'] / 60
            
            report_lines.append(f"  {size}m Parcel:")
            report_lines.append(f"    R² Score: {metrics['r2']:.4f}")
            report_lines.append(f"    MSE: {metrics['mse']:.6f}")
            report_lines.append(f"    MAE: {metrics['mae']:.6f}")
            report_lines.append(f"    Training Time: {time_min:.1f} min")
            report_lines.append("")
    
    # Multi-scale results
    if 'multi_scale' in results:
        report_lines.append("MULTI-SCALE U-NET:")
        report_lines.append("-" * 40)
        multi_result = results['multi_scale']
        time_min = multi_result['training_time'] / 60
        avg_r2 = multi_result['best_avg_r2']
        
        report_lines.append(f"  Average R² Score: {avg_r2:.4f}")
        report_lines.append(f"  Training Time: {time_min:.1f} min")
        report_lines.append("")
        report_lines.append("  Per-Size Performance:")
        
        for size, metrics in multi_result['best_metrics_per_size'].items():
            report_lines.append(f"    {size}m: R² = {metrics['r2']:.4f}, MSE = {metrics['mse']:.6f}")
        report_lines.append("")
    
    # Comparison
    if 'single_scale' in results and 'multi_scale' in results:
        report_lines.append("COMPARISON:")
        report_lines.append("-" * 40)
        
        single_avg_r2 = np.mean([r['best_metrics']['r2'] for r in results['single_scale']])
        multi_avg_r2 = results['multi_scale']['best_avg_r2']
        
        report_lines.append(f"  Single-Scale Average R²: {single_avg_r2:.4f}")
        report_lines.append(f"  Multi-Scale Average R²: {multi_avg_r2:.4f}")
        report_lines.append(f"  Performance Ratio: {(multi_avg_r2/single_avg_r2)*100:.1f}%")
        report_lines.append("")
        
        single_total_time = sum(r['training_time'] for r in results['single_scale']) / 60
        multi_total_time = results['multi_scale']['training_time'] / 60
        
        report_lines.append(f"  Single-Scale Total Training: {single_total_time:.1f} min")
        report_lines.append(f"  Multi-Scale Training: {multi_total_time:.1f} min")
        report_lines.append(f"  Time Ratio: {(multi_total_time/single_total_time)*100:.1f}%")
        report_lines.append("")
    
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Saved report to {report_path}")
    
    # Print to console
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 7 Results")
    
    parser.add_argument('--results-file', type=str,
                       default='results/exp7_multiscale_unet/training_results.json',
                       help='Path to training results JSON')
    parser.add_argument('--output-dir', type=str,
                       default='results/exp7_multiscale_unet/analysis',
                       help='Output directory for plots and tables')
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    # Load results
    results = load_results(results_file)
    
    # Create analysis outputs
    create_comparison_table(results, output_dir)
    plot_r2_comparison(results, output_dir)
    plot_per_field_accuracy(results, output_dir)
    plot_training_time(results, output_dir)
    create_summary_report(results, output_dir)
    
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
