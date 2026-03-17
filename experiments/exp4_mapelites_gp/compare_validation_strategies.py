#!/usr/bin/env python3
"""
Compare validation results between TOP and RANDOM sampling strategies.

This tests whether Exp 4 failure is due to:
1. Selection bias - GP is only wrong for top-scoring solutions
2. False optimum - Entire archive is in a region where GP is systematically wrong
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_validation_results(base_dir: Path, strategy: str):
    """Load validation metrics and results for a strategy."""
    val_dir = base_dir / f"validation_{strategy}"
    
    if not val_dir.exists():
        return None, None
    
    # Load metrics
    metrics_file = val_dir / "validation_metrics.yaml"
    if not metrics_file.exists():
        return None, None
    
    with open(metrics_file) as f:
        metrics = yaml.safe_load(f)
    
    # Load results CSV
    results_file = val_dir / "validation_results.csv"
    if results_file.exists():
        results = pd.read_csv(results_file)
    else:
        results = None
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(
        description="Compare TOP vs RANDOM validation strategies"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="results/exp4_mapelites_gp/mapelites_gp/emit64_batch16_rep1",
        help="Run directory containing validation_top and validation_random subdirs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for comparison plots"
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = run_dir / "validation_comparison"
        output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("VALIDATION STRATEGY COMPARISON")
    print("="*70)
    print(f"\nRun directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load both validations
    top_metrics, top_results = load_validation_results(run_dir, "top")
    random_metrics, random_results = load_validation_results(run_dir, "random")
    
    if top_metrics is None and random_metrics is None:
        print("\n❌ ERROR: No validation results found!")
        print("   Run validations first with:")
        print("   bash hpc/exp4_mapelites_gp/submit_validate_both_strategies.sh")
        return 1
    
    # ========================================================================
    # Compare Metrics
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON: Overall Metrics")
    print("="*70)
    
    metrics_comparison = {}
    
    for metric in ['r2', 'rmse', 'mae', 'spearman_rho', 'pearson_r']:
        top_val = top_metrics.get(metric, None) if top_metrics else None
        random_val = random_metrics.get(metric, None) if random_metrics else None
        
        metrics_comparison[metric] = {
            'top': top_val,
            'random': random_val
        }
    
    # Print table
    print(f"\n{'Metric':<20} {'TOP 50':>12} {'RANDOM 50':>12} {'Difference':>12}")
    print("-" * 70)
    
    for metric, values in metrics_comparison.items():
        top_val = values['top']
        random_val = values['random']
        
        if top_val is not None and random_val is not None:
            diff = random_val - top_val
            print(f"{metric:<20} {top_val:12.4f} {random_val:12.4f} {diff:+12.4f}")
        else:
            top_str = f"{top_val:.4f}" if top_val is not None else "N/A"
            random_str = f"{random_val:.4f}" if random_val is not None else "N/A"
            print(f"{metric:<20} {top_str:>12} {random_str:>12} {'N/A':>12}")
    
    # ========================================================================
    # Compare Distributions
    # ========================================================================
    if top_results is not None and random_results is not None:
        print("\n" + "="*70)
        print("COMPARISON: Prediction Distributions")
        print("="*70)
        
        print(f"\n{'Statistic':<20} {'TOP 50':>12} {'RANDOM 50':>12}")
        print("-" * 50)
        
        # GP predictions
        print("\nGP Predictions:")
        print(f"  Mean           {top_results['gp_objective'].mean():12.2f} {random_results['gp_objective'].mean():12.2f}")
        print(f"  Std            {top_results['gp_objective'].std():12.2f} {random_results['gp_objective'].std():12.2f}")
        print(f"  Min            {top_results['gp_objective'].min():12.2f} {random_results['gp_objective'].min():12.2f}")
        print(f"  Max            {top_results['gp_objective'].max():12.2f} {random_results['gp_objective'].max():12.2f}")
        print(f"  Range          {top_results['gp_objective'].max()-top_results['gp_objective'].min():12.2f} {random_results['gp_objective'].max()-random_results['gp_objective'].min():12.2f}")
        
        # KLAM true values
        print("\nKLAM True Values:")
        print(f"  Mean           {top_results['klam_objective'].mean():12.2f} {random_results['klam_objective'].mean():12.2f}")
        print(f"  Std            {top_results['klam_objective'].std():12.2f} {random_results['klam_objective'].std():12.2f}")
        print(f"  Min            {top_results['klam_objective'].min():12.2f} {random_results['klam_objective'].min():12.2f}")
        print(f"  Max            {top_results['klam_objective'].max():12.2f} {random_results['klam_objective'].max():12.2f}")
        print(f"  Range          {top_results['klam_objective'].max()-top_results['klam_objective'].min():12.2f} {random_results['klam_objective'].max()-random_results['klam_objective'].min():12.2f}")
        
        # ====================================================================
        # Create Comparison Plots
        # ====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: GP vs KLAM scatter (TOP)
        ax = axes[0, 0]
        ax.scatter(top_results['klam_objective'], top_results['gp_objective'], 
                  alpha=0.6, s=50)
        ax.plot([top_results['klam_objective'].min(), top_results['klam_objective'].max()],
               [top_results['klam_objective'].min(), top_results['klam_objective'].max()],
               'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('KLAM True Objective')
        ax.set_ylabel('GP Predicted Objective')
        ax.set_title(f'TOP 50 Strategy\nR²={top_metrics["r2"]:.3f}, ρ={top_metrics["spearman_rho"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: GP vs KLAM scatter (RANDOM)
        ax = axes[0, 1]
        ax.scatter(random_results['klam_objective'], random_results['gp_objective'],
                  alpha=0.6, s=50, color='orange')
        ax.plot([random_results['klam_objective'].min(), random_results['klam_objective'].max()],
               [random_results['klam_objective'].min(), random_results['klam_objective'].max()],
               'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('KLAM True Objective')
        ax.set_ylabel('GP Predicted Objective')
        ax.set_title(f'RANDOM 50 Strategy\nR²={random_metrics["r2"]:.3f}, ρ={random_metrics["spearman_rho"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Residuals comparison
        ax = axes[1, 0]
        top_results['residual'] = top_results['gp_objective'] - top_results['klam_objective']
        random_results['residual'] = random_results['gp_objective'] - random_results['klam_objective']
        
        positions = [1, 2]
        bp = ax.boxplot([top_results['residual'], random_results['residual']],
                        positions=positions, widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('orange')
        ax.axhline(y=0, color='r', linestyle='--', lw=2, label='No error')
        ax.set_xticks(positions)
        ax.set_xticklabels(['TOP 50', 'RANDOM 50'])
        ax.set_ylabel('Residual (GP - KLAM)')
        ax.set_title('Prediction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribution histograms
        ax = axes[1, 1]
        bins = np.linspace(min(top_results['gp_objective'].min(), random_results['gp_objective'].min()),
                          max(top_results['gp_objective'].max(), random_results['gp_objective'].max()),
                          30)
        ax.hist(top_results['gp_objective'], bins=bins, alpha=0.5, label='TOP: GP predictions', color='blue')
        ax.hist(top_results['klam_objective'], bins=bins, alpha=0.5, label='TOP: KLAM true', color='lightblue')
        ax.hist(random_results['gp_objective'], bins=bins, alpha=0.5, label='RANDOM: GP predictions', color='orange')
        ax.hist(random_results['klam_objective'], bins=bins, alpha=0.5, label='RANDOM: KLAM true', color='wheat')
        ax.set_xlabel('Objective Value')
        ax.set_ylabel('Count')
        ax.set_title('Objective Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot: {output_dir / 'validation_comparison.png'}")
    
    # ========================================================================
    # Diagnosis
    # ========================================================================
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if top_metrics and random_metrics:
        top_r2 = top_metrics['r2']
        random_r2 = random_metrics['r2']
        
        print(f"\nR² scores:")
        print(f"  TOP 50:    {top_r2:7.4f}")
        print(f"  RANDOM 50: {random_r2:7.4f}")
        print(f"  Difference: {random_r2 - top_r2:+7.4f}")
        
        if top_r2 < 0 and random_r2 > 0.8:
            print("\n✓ SELECTION BIAS CONFIRMED")
            print("  - GP is accurate for random samples (R² > 0.8)")
            print("  - GP fails catastrophically for top samples (R² < 0)")
            print("  → Only the highest-scoring region is overestimated")
            print("  → Most of the archive is probably accurate")
        elif top_r2 < 0 and random_r2 < 0.5:
            print("\n✗ FALSE OPTIMUM CONFIRMED")
            print("  - GP fails for BOTH top and random samples")
            print("  → The entire archive is in a region where GP is wrong")
            print("  → MAP-Elites converged to a false peak")
        elif top_r2 > 0.8 and random_r2 > 0.8:
            print("\n? UNEXPECTED: Both strategies show good accuracy")
            print("  - This contradicts the original validation results")
            print("  - May need to check data consistency")
        else:
            print("\n? MIXED RESULTS:")
            print("  - GP performance varies between strategies")
            print("  - May indicate partial convergence to false optimum")
    
    print("\n" + "="*70)
    
    return 0

if __name__ == "__main__":
    exit(main())
