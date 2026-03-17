#!/usr/bin/env python3
"""
Diagnostic script to analyze raw wind speed fields from KLAM_21 results.

This script investigates why average wind speed appears constant across
different GRZ configurations despite visible wakes in the airflow plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_data(data_dir: str):
    """Load experiment results."""
    data_path = Path(data_dir)
    
    results = np.load(data_path / "flux_sensitivity_results.npz", allow_pickle=True)
    spatial = np.load(data_path / "flux_sensitivity_spatial.npz", allow_pickle=True)
    
    return {
        'uq': spatial['uq'],
        'vq': spatial['vq'],
        'config_indices': spatial['config_indices'],
        'labels': results['labels'],
        'fitness': results['fitness'],
        'grz': results['grz'],
        'height_floors': results['height_floors'],
        'orientation_deg': results['orientation_deg'],
        'heightmaps': results['heightmaps'],
        'parcel_cells': int(results['parcel_cells']),
        'xy_scale': float(results['xy_scale']),
    }


def compute_wind_statistics(data: dict):
    """Compute detailed wind statistics for each configuration."""
    
    uq = data['uq']
    vq = data['vq']
    config_indices = data['config_indices']
    labels = data['labels']
    grz = data['grz']
    height_floors = data['height_floors']
    parcel_cells = data['parcel_cells']
    
    # Domain geometry (approximate from experiment setup)
    # env_size_m = 99, xy_scale = 3 -> env_cells_base = 33
    # left_extension = 8 -> env_cells_x = 41
    # offset_cells_x = 16, offset_cells_y = 8
    env_cells_x = uq.shape[2] if len(uq.shape) == 3 else uq[0].shape[1]
    env_cells_y = uq.shape[1] if len(uq.shape) == 3 else uq[0].shape[0]
    
    # Estimate parcel position
    offset_x = (env_cells_x - parcel_cells) // 2 + (env_cells_x - int(99/3)) // 2
    offset_y = (env_cells_y - parcel_cells) // 2
    
    # Calculate correct parcel position based on environment_xy_size=200m
    # env_cells_base = 200 / 3 = 66.67 ≈ 66 cells
    # original_offset = (66 - 17) // 2 = 24
    # left_extension = 24 (100% more upwind)
    # offset_cells_x = 24 + 24 = 48
    # offset_cells_y = 24
    offset_x = 48
    offset_y = 24
    
    print("=" * 80)
    print("WIND SPEED FIELD DIAGNOSTICS")
    print("=" * 80)
    print(f"\nDomain: {env_cells_x} x {env_cells_y} cells")
    print(f"Parcel: {parcel_cells} x {parcel_cells} cells at x=[{offset_x}, {offset_x + parcel_cells}], y=[{offset_y}, {offset_y + parcel_cells}]")
    
    print("\n" + "-" * 80)
    print("PER-CONFIGURATION WIND SPEED ANALYSIS")
    print("-" * 80)
    print(f"{'Config':<18} {'GRZ':>5} {'H':>3} | {'Whole Domain':>12} | {'Parcel':>12} | {'Upwind':>12} | {'Downwind':>12} | {'Wake (min)':>12}")
    print("-" * 80)
    
    stats_list = []
    
    for i, idx in enumerate(config_indices):
        u = uq[i]
        v = vq[i]
        speed = np.sqrt(u**2 + v**2)
        
        label = str(labels[idx])
        g = grz[idx]
        h = height_floors[idx]
        
        # Regions
        whole = speed
        parcel = speed[offset_y:offset_y+parcel_cells, offset_x:offset_x+parcel_cells]
        upwind = speed[offset_y:offset_y+parcel_cells, :offset_x]
        downwind = speed[offset_y:offset_y+parcel_cells, offset_x+parcel_cells:]
        
        # For wake detection, look for minimum in the building/downwind area
        wake_region = speed[offset_y:offset_y+parcel_cells, offset_x:]
        
        stats = {
            'label': label,
            'grz': g,
            'height': h,
            'whole_mean': np.mean(whole),
            'whole_std': np.std(whole),
            'parcel_mean': np.mean(parcel),
            'parcel_min': np.min(parcel),
            'parcel_max': np.max(parcel),
            'upwind_mean': np.mean(upwind) if upwind.size > 0 else np.nan,
            'downwind_mean': np.mean(downwind) if downwind.size > 0 else np.nan,
            'wake_min': np.min(wake_region),
        }
        stats_list.append(stats)
        
        print(f"{label:<18} {g*100:>4.0f}% {h:>3} | "
              f"{stats['whole_mean']:>12.4f} | "
              f"{stats['parcel_mean']:>12.4f} | "
              f"{stats['upwind_mean']:>12.4f} | "
              f"{stats['downwind_mean']:>12.4f} | "
              f"{stats['wake_min']:>12.4f}")
    
    return stats_list


def plot_wind_speed_comparison(data: dict, output_dir: str):
    """Create diagnostic plots comparing wind speed fields."""
    
    uq = data['uq']
    vq = data['vq']
    config_indices = data['config_indices']
    labels = data['labels']
    grz = data['grz']
    heightmaps = data['heightmaps']
    parcel_cells = data['parcel_cells']
    
    # Find representative configs for each GRZ level
    grz_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    selected_configs = []
    
    for g in grz_levels:
        # Find first config at this GRZ level that has spatial data
        for i, idx in enumerate(config_indices):
            if grz[idx] == g:
                selected_configs.append((i, idx, g))
                break
    
    if not selected_configs:
        print("No configs with spatial data found!")
        return
    
    # Create figure with wind speed fields
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, len(selected_configs), height_ratios=[1, 1, 0.5], hspace=0.3)
    
    # Row 1: Wind speed magnitude
    # Row 2: U-component (to see wakes)
    # Row 3: Heightmaps
    
    vmin_speed = 0
    vmax_speed = 1.5
    vmin_u = -0.5
    vmax_u = 1.5
    
    for col, (spatial_idx, config_idx, g) in enumerate(selected_configs):
        u = uq[spatial_idx]
        v = vq[spatial_idx]
        speed = np.sqrt(u**2 + v**2)
        hmap = heightmaps[config_idx]
        label = str(labels[config_idx])
        
        # Wind speed
        ax1 = fig.add_subplot(gs[0, col])
        im1 = ax1.imshow(speed, origin='lower', cmap='Blues', vmin=vmin_speed, vmax=vmax_speed)
        ax1.set_title(f'{label}\nGRZ={g*100:.0f}%', fontsize=10)
        ax1.set_ylabel('Wind Speed (m/s)' if col == 0 else '')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Add parcel outline (correct coordinates)
        offset_x, offset_y = 48, 24
        rect = plt.Rectangle((offset_x, offset_y), parcel_cells, parcel_cells, 
                             fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)
        
        # U-component (shows wakes better)
        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.imshow(u, origin='lower', cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
        ax2.set_ylabel('U-component (m/s)' if col == 0 else '')
        ax2.set_xticks([])
        ax2.set_yticks([])
        rect2 = plt.Rectangle((offset_x, offset_y), parcel_cells, parcel_cells, 
                              fill=False, edgecolor='black', linewidth=2)
        ax2.add_patch(rect2)
        
        # Heightmap
        ax3 = fig.add_subplot(gs[2, col])
        ax3.imshow(hmap, origin='lower', cmap='gray_r', vmin=0, vmax=8)
        ax3.set_ylabel('Buildings' if col == 0 else '')
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # Add colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.35])
    fig.colorbar(im1, cax=cbar_ax1, label='Wind Speed (m/s)')
    
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label='U-component (m/s)')
    
    fig.suptitle('Wind Speed Fields by GRZ Level\n(Red/black boxes = parcel location)', fontsize=14)
    
    output_path = Path(output_dir) / "wind_field_diagnostic.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_regional_statistics(data: dict, output_dir: str):
    """Plot statistics of wind speed in different regions."""
    
    uq = data['uq']
    vq = data['vq']
    config_indices = data['config_indices']
    grz = data['grz']
    parcel_cells = data['parcel_cells']
    
    # Correct parcel position for environment_xy_size=200m
    offset_x, offset_y = 48, 24
    
    # Collect statistics per GRZ level
    grz_levels = sorted(set(grz))
    
    stats_by_grz = {g: {'parcel': [], 'upwind': [], 'downwind': [], 'wake_min': []} for g in grz_levels}
    
    for i, idx in enumerate(config_indices):
        u = uq[i]
        v = vq[i]
        speed = np.sqrt(u**2 + v**2)
        g = grz[idx]
        
        parcel_speed = speed[offset_y:offset_y+parcel_cells, offset_x:offset_x+parcel_cells]
        upwind_speed = speed[offset_y:offset_y+parcel_cells, :offset_x]
        downwind_speed = speed[offset_y:offset_y+parcel_cells, offset_x+parcel_cells:]
        wake_region = speed[offset_y:offset_y+parcel_cells, offset_x:]
        
        stats_by_grz[g]['parcel'].append(np.mean(parcel_speed))
        stats_by_grz[g]['upwind'].append(np.mean(upwind_speed) if upwind_speed.size > 0 else np.nan)
        stats_by_grz[g]['downwind'].append(np.mean(downwind_speed) if downwind_speed.size > 0 else np.nan)
        stats_by_grz[g]['wake_min'].append(np.min(wake_region))
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    grz_pct = [g * 100 for g in grz_levels if g > 0]  # Exclude 0% for comparison
    
    # Plot 1: Mean speed by region
    ax1 = axes[0]
    regions = ['parcel', 'upwind', 'downwind']
    colors = ['blue', 'green', 'orange']
    
    for region, color in zip(regions, colors):
        means = [np.mean(stats_by_grz[g][region]) for g in grz_levels if g > 0]
        ax1.plot(grz_pct, means, 'o-', color=color, label=region.capitalize(), markersize=8)
    
    ax1.set_xlabel('Site Coverage GRZ (%)')
    ax1.set_ylabel('Mean Wind Speed (m/s)')
    ax1.set_title('Mean Wind Speed by Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Minimum speed in wake region
    ax2 = axes[1]
    wake_mins = [np.mean(stats_by_grz[g]['wake_min']) for g in grz_levels if g > 0]
    ax2.plot(grz_pct, wake_mins, 's-', color='red', markersize=8, label='Min in wake region')
    
    # Also plot parcel mean for comparison
    parcel_means = [np.mean(stats_by_grz[g]['parcel']) for g in grz_levels if g > 0]
    ax2.plot(grz_pct, parcel_means, 'o-', color='blue', markersize=8, label='Mean in parcel')
    
    ax2.set_xlabel('Site Coverage GRZ (%)')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_title('Wake Detection: Min vs Mean Speed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Regional Wind Speed Statistics', fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "wind_regional_stats.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_single_config_detail(data: dict, output_dir: str, config_label: str = "GRZ060_H4_O00"):
    """Create detailed diagnostic for a single configuration."""
    
    uq = data['uq']
    vq = data['vq']
    config_indices = data['config_indices']
    labels = data['labels']
    heightmaps = data['heightmaps']
    parcel_cells = data['parcel_cells']
    
    # Find the config
    target_idx = None
    spatial_idx = None
    for i, idx in enumerate(config_indices):
        if str(labels[idx]) == config_label:
            target_idx = idx
            spatial_idx = i
            break
    
    if target_idx is None:
        print(f"Config {config_label} not found in spatial data!")
        return
    
    u = uq[spatial_idx]
    v = vq[spatial_idx]
    speed = np.sqrt(u**2 + v**2)
    hmap = heightmaps[target_idx]
    
    offset_x, offset_y = 48, 24
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.3)
    
    # Full domain wind speed
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(speed, origin='lower', cmap='Blues', vmin=0, vmax=1.5)
    ax1.set_title('Wind Speed (full domain)')
    rect = plt.Rectangle((offset_x, offset_y), parcel_cells, parcel_cells, 
                         fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)
    plt.colorbar(im1, ax=ax1, label='m/s')
    
    # U-component
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(u, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=1.5)
    ax2.set_title('U-component (positive = rightward)')
    rect2 = plt.Rectangle((offset_x, offset_y), parcel_cells, parcel_cells, 
                          fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(rect2)
    plt.colorbar(im2, ax=ax2, label='m/s')
    
    # Heightmap (parcel only)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(hmap, origin='lower', cmap='gray_r', vmin=0, vmax=8)
    ax3.set_title('Buildings (parcel)')
    
    # Cross-section through center of parcel
    center_y = offset_y + parcel_cells // 2
    ax4 = fig.add_subplot(gs[1, :2])
    
    x_coords = np.arange(speed.shape[1])
    ax4.plot(x_coords, speed[center_y, :], 'b-', linewidth=2, label='Wind speed at y=center')
    ax4.plot(x_coords, u[center_y, :], 'r--', linewidth=1.5, label='U-component')
    
    # Mark parcel region
    ax4.axvspan(offset_x, offset_x + parcel_cells, alpha=0.2, color='gray', label='Parcel')
    ax4.axvline(offset_x, color='gray', linestyle=':', linewidth=1)
    ax4.axvline(offset_x + parcel_cells, color='gray', linestyle=':', linewidth=1)
    
    ax4.set_xlabel('X position (cells)')
    ax4.set_ylabel('Wind Speed (m/s)')
    ax4.set_title(f'Cross-section at y={center_y} (center of parcel)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, speed.shape[1])
    
    # Statistics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    parcel_speed = speed[offset_y:offset_y+parcel_cells, offset_x:offset_x+parcel_cells]
    upwind_speed = speed[offset_y:offset_y+parcel_cells, :offset_x]
    downwind_speed = speed[offset_y:offset_y+parcel_cells, offset_x+parcel_cells:]
    
    stats_text = f"""Configuration: {config_label}

Domain Statistics:
  Shape: {speed.shape}
  Mean: {np.mean(speed):.4f} m/s
  Std:  {np.std(speed):.4f} m/s
  Min:  {np.min(speed):.4f} m/s
  Max:  {np.max(speed):.4f} m/s

Parcel Region:
  Mean: {np.mean(parcel_speed):.4f} m/s
  Min:  {np.min(parcel_speed):.4f} m/s
  Max:  {np.max(parcel_speed):.4f} m/s

Upwind Region:
  Mean: {np.mean(upwind_speed):.4f} m/s

Downwind Region:
  Mean: {np.mean(downwind_speed):.4f} m/s
  Min:  {np.min(downwind_speed):.4f} m/s
"""
    ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    fig.suptitle(f'Detailed Wind Field Analysis: {config_label}', fontsize=14)
    
    output_path = Path(output_dir) / f"wind_detail_{config_label}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose wind speed fields from KLAM results")
    parser.add_argument("--data-dir", type=str, default="results/flux_sensitivity_v2",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for diagnostic plots")
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.data_dir
    
    print("Loading data...")
    data = load_data(args.data_dir)
    
    print(f"\nLoaded {len(data['uq'])} configurations with spatial data")
    
    # Compute and print statistics
    stats = compute_wind_statistics(data)
    
    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_wind_speed_comparison(data, output_dir)
    plot_regional_statistics(data, output_dir)
    
    # Detailed plot for a medium-density config
    plot_single_config_detail(data, output_dir, "GRZ060_H4_O00")
    plot_single_config_detail(data, output_dir, "GRZ000_H0_O00")
    plot_single_config_detail(data, output_dir, "GRZ080_H4_O00")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    # Check if wind speed varies
    speeds = [s['parcel_mean'] for s in stats]
    print(f"\nParcel mean wind speed range: {min(speeds):.4f} - {max(speeds):.4f} m/s")
    print(f"Parcel mean wind speed variation: {max(speeds) - min(speeds):.4f} m/s ({(max(speeds) - min(speeds))/np.mean(speeds)*100:.1f}%)")
    
    wake_mins = [s['wake_min'] for s in stats]
    print(f"\nWake minimum speed range: {min(wake_mins):.4f} - {max(wake_mins):.4f} m/s")
    print(f"Wake minimum variation: {max(wake_mins) - min(wake_mins):.4f} m/s")
    
    print("\nDiagnostic plots saved to:", output_dir)


if __name__ == "__main__":
    main()
