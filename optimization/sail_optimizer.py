# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import os
import shutil
import yaml
import psutil
import multiprocessing
import pickle
import importlib
from datetime import datetime
from typing import Dict

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler
from scipy.stats import qmc
from domain_description.gp_utils import train_gp

# NEW: Matplotlib for saving debug plots
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def save_floodfill_visualizations(heightmaps, floodfill_data, config_environment, output_path):
    """
    Saves a grid of plots showing the building layout and the area reached by the flood fill.
    Correctly flips the visualization to show the flood coming from the left.
    """
    n_images = len(heightmaps)
    if n_images == 0 or not floodfill_data:
        return

    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    axes = axes.flatten()

    env_xy = config_environment['environment_xy_size']
    design_xy = config_environment['length_design']
    offset_xy = (env_xy - design_xy) // 2

    for i, (hmap, (visited_map, _)) in enumerate(zip(heightmaps, floodfill_data)):
        if visited_map is None:
            continue
        
        ax = axes[i]
        
        # This corrects the visual representation to match the data model.
        flipped_visited_map = np.fliplr(visited_map)
        flipped_visited_map = np.flipud(flipped_visited_map)
        flipped_visited_map = np.rot90(flipped_visited_map)
        flipped_visited_map = np.fliplr(flipped_visited_map)
        
        building_img = np.zeros((env_xy, env_xy))
        building_img[offset_xy:offset_xy+design_xy, offset_xy:offset_xy+design_xy] = hmap.reshape((design_xy, design_xy))
        flipped_building_img = np.fliplr(building_img)

        # 1. Background
        ax.imshow(flipped_visited_map[:, :, 0] == 0, cmap='gray_r', alpha=0.1)

        # 2. Buildings
        masked_buildings = np.ma.masked_where(flipped_building_img == 0, flipped_building_img)
        ax.imshow(masked_buildings, cmap='viridis')

        # 3. Flood overlay
        visited_slice = flipped_visited_map[:, :, 0].astype(float)
        masked_visited = np.ma.masked_where(visited_slice == 0, visited_slice)
        ax.imshow(masked_visited, cmap='cool', alpha=0.4, vmin=0, vmax=1)
        
        ax.set_title(f"Sample {i}", fontsize=10)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    debug_plot_path = os.path.join(output_path, 'debug_floodfill_visuals.png')
    plt.savefig(debug_plot_path, dpi=150)
    plt.close(fig)
    print(f"DEBUG: Saved flood-fill visualizations to {debug_plot_path}")

def save_airflow_visualizations(heightmaps, airflow_data, output_path):
    """
    Saves a grid of heightmap plots with superimposed airflow vectors.
    """
    n_images = len(heightmaps)
    if n_images == 0 or not airflow_data:
        return

    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    # Set a background color for the figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), 
                             constrained_layout=True,
                             facecolor='#4F4F4F') # Dark grey background
    axes = axes.flatten()

    for i, (hmap, (uq, vq)) in enumerate(zip(heightmaps, airflow_data)):
        ax = axes[i]
        ax.set_facecolor('#4F4F4F')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                       left=False, right=False, labelbottom=False, labelleft=False)

        if uq is None or vq is None:
            # If there's no data, just show the heightmap on the dark background
            l = int(np.sqrt(hmap.size))
            heightmap = hmap.reshape((l, l))
            cmap = plt.cm.viridis
            cmap.set_under(alpha=0) # Make '0' values transparent
            ax.imshow(heightmap, cmap=cmap, norm=colors.Normalize(vmin=0.1, vmax=heightmap.max() or 1))
            ax.set_title(f"Sample {i}\n(No Airflow Data)", fontsize=10, color='white')
            continue
            
        # Get dimensions
        airflow_dim = uq.shape[0]
        heightmap_dim = int(np.sqrt(hmap.size))
        
        # Create a full-size canvas for the heightmap, initialized to 0
        full_heightmap = np.zeros((airflow_dim, airflow_dim))
        
        # Calculate padding to center the smaller heightmap
        pad = (airflow_dim - heightmap_dim) // 2
        
        # Embed the heightmap into the center of the canvas
        if pad >= 0:
            full_heightmap[pad:pad+heightmap_dim, pad:pad+heightmap_dim] = hmap.reshape((heightmap_dim, heightmap_dim))
        else: # Should not happen if heightmap is smaller or equal
            full_heightmap = hmap.reshape((heightmap_dim, heightmap_dim))

        # Display the building heightmap as the background
        # Use a colormap where values <= 0 are transparent
        cmap = plt.cm.viridis
        cmap.set_under(alpha=0) 
        ax.imshow(full_heightmap, cmap=cmap, norm=colors.Normalize(vmin=0.1, vmax=full_heightmap.max() or 1))
        
        # Create grid for the quiver plot based on airflow dimensions
        x = np.arange(0, airflow_dim, 1)
        y = np.arange(0, airflow_dim, 1)
        X, Y = np.meshgrid(x, y)
        
        # Downsample the vector field for clarity
        skip = max(1, airflow_dim // 15)  # Aim for ~15 arrows per side
        
        # Plot the vector field (quiver plot) with a white color
        # Scale arrows to be visible: scale parameter controls arrow length
        # Lower scale = longer arrows. With velocities ~0.5 m/s, scale=5 gives good visibility
        ax.quiver(
            X[::skip, ::skip], Y[::skip, ::skip],
            uq[::skip, ::skip], vq[::skip, ::skip],
            color='white',
            scale_units='xy',
            scale=0.01,  # Adjusted for visibility with m/s velocities
            headwidth=4,
            headlength=5,
            width=0.003
        )
        
        ax.set_title(f"Sample {i}", fontsize=10, color='white')

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    debug_plot_path = os.path.join(output_path, 'debug_airflow_visuals.png')
    plt.savefig(debug_plot_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"DEBUG: Saved airflow visualizations to {debug_plot_path}")

def save_velocity_field_visualizations(heightmaps, airflow_data, output_path, config_encoding=None):
    """
    Saves comprehensive KLAM-21 simulation visualization with inputs and outputs.
    Each row shows one KLAM run with 10 columns:
    1. Landuse (input)
    2. Building height (input)
    3. Terrain height (input)
    4. uq velocity at 2m height with streamlines (output)
    5. vq velocity at 2m height with streamlines (output)
    6. uz velocity column average (output)
    7. vz velocity column average (output)
    8. Hx cold airflow height (output)
    9. Ex cold air content (output)
    10. Statistics text box (wind speeds over ROI)
    """
    n_samples = len(heightmaps)
    if n_samples == 0 or not airflow_data:
        return

    # Pre-compute global ranges for consistent colorbars
    all_heightmaps = []
    all_uq = []
    all_vq = []
    all_uz = []
    all_vz = []
    all_hx = []
    all_ex = []
    
    # Collect metrics for correlation analysis
    metrics_data = []  # List of [wind_2m, wind_col, hx, ex, flux] for each sample
    
    for i, (hmap, data_tuple) in enumerate(zip(heightmaps, airflow_data)):
        uq, vq, uz, vz, hx, ex = data_tuple
        if uq is not None and vq is not None:
            airflow_dim = uq.shape[0]
            heightmap_dim = int(np.sqrt(hmap.size))
            full_heightmap = np.zeros((airflow_dim, airflow_dim))
            pad = (airflow_dim - heightmap_dim) // 2
            if pad >= 0:
                full_heightmap[pad:pad+heightmap_dim, pad:pad+heightmap_dim] = hmap.reshape((heightmap_dim, heightmap_dim))
            else:
                full_heightmap = hmap.reshape((heightmap_dim, heightmap_dim))
            
            all_heightmaps.append(full_heightmap)
            all_uq.append(uq)
            all_vq.append(vq)
            if uz is not None:
                all_uz.append(uz)
            if vz is not None:
                all_vz.append(vz)
            if hx is not None:
                all_hx.append(hx)
            if ex is not None:
                all_ex.append(ex)
    
    # Calculate global ranges
    if all_heightmaps:
        global_height_max = max(hm.max() for hm in all_heightmaps)
    else:
        global_height_max = 1
    
    # Calculate global range for Hx (cold airflow height)
    if all_hx:
        global_hx_max = max(h.max() for h in all_hx)
    else:
        global_hx_max = 10  # Default max height in meters
    
    # Calculate global range for Ex (cold air content in 100 J/m²)
    if all_ex:
        global_ex_max = max(e.max() for e in all_ex)
    else:
        global_ex_max = 100  # Default max
    
    if all_uq and all_vq:
        # Combine all velocity components to get shared velocity range for all 4 plots
        all_velocities = [np.array(all_uq).flatten(), np.array(all_vq).flatten()]
        if all_uz:
            all_velocities.append(np.array(all_uz).flatten())
        if all_vz:
            all_velocities.append(np.array(all_vz).flatten())
        
        all_velocities = np.concatenate(all_velocities)
        vel_min, vel_max = np.min(all_velocities), np.max(all_velocities)
        # Center at zero (white) with symmetric range
        vel_abs_max = max(abs(vel_min), abs(vel_max), 0.1)
        global_vel_min = -vel_abs_max
        global_vel_max = vel_abs_max
        
        # Calculate global range for wind speed magnitude (2m height)
        all_wind_speeds_2m = [np.sqrt(uq**2 + vq**2) for uq, vq in zip(all_uq, all_vq)]
        
        # Calculate global range for wind speed magnitude (column average)
        if all_uz and all_vz:
            all_wind_speeds_column = [np.sqrt(uz**2 + vz**2) for uz, vz in zip(all_uz, all_vz)]
            # Combine both for unified scale
            all_wind_speeds = all_wind_speeds_2m + all_wind_speeds_column
        else:
            all_wind_speeds = all_wind_speeds_2m
        
        global_speed_max = max(ws.max() for ws in all_wind_speeds)
    else:
        global_vel_min, global_vel_max = -1, 1
        global_speed_max = 2

    # Create figure: 9 columns per sample
    fig, axes = plt.subplots(n_samples, 10, figsize=(40, n_samples * 3.5), 
                             constrained_layout=True)
    
    # Ensure axes is always 2D
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (hmap, data_tuple) in enumerate(zip(heightmaps, airflow_data)):
        uq, vq, uz, vz, hx, ex = data_tuple
        
        # DEBUG: Print velocity field statistics
        if i == 0 and uq is not None:
            print(f"DEBUG: uq shape: {uq.shape}, min={np.min(uq):.3f}, max={np.max(uq):.3f}, unique values: {len(np.unique(uq))}")
            print(f"DEBUG: vq shape: {vq.shape}, min={np.min(vq):.3f}, max={np.max(vq):.3f}, unique values: {len(np.unique(vq))}")
            if uz is not None:
                print(f"DEBUG: uz shape: {uz.shape}, min={np.min(uz):.3f}, max={np.max(uz):.3f}, unique values: {len(np.unique(uz))}")
            if vz is not None:
                print(f"DEBUG: vz shape: {vz.shape}, min={np.min(vz):.3f}, max={np.max(vz):.3f}, unique values: {len(np.unique(vz))}")
            if hx is not None:
                print(f"DEBUG: hx shape: {hx.shape}, min={np.min(hx):.3f}, max={np.max(hx):.3f}, unique values: {len(np.unique(hx))}")
            print(f"DEBUG: Global velocity range: [{global_vel_min:.3f}, {global_vel_max:.3f}]")
            print(f"DEBUG: Global speed range: [0, {global_speed_max:.3f}]")
            print(f"DEBUG: Global Hx range: [0, {global_hx_max:.3f}]")
            print(f"DEBUG: Global Ex range: [0, {global_ex_max:.3f}]")
        
        if uq is None or vq is None:
            # No airflow data available
            for k in range(10):
                ax = axes[i, k]
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.axis('off')
            continue
        
        # Get dimensions (domain may be rectangular: rows=y, cols=x)
        airflow_dim_y, airflow_dim_x = uq.shape
        heightmap_dim = int(np.sqrt(hmap.size))
        
        # Calculate offsets matching evaluation_klam.py logic
        # Original base size would give equal padding
        # But domain is extended 100% to the left, so x-offset is larger
        original_offset = (airflow_dim_y - heightmap_dim) // 2  # y-offset (unchanged)
        left_extension = original_offset  # Same extension amount
        
        offset_cells_y = original_offset
        offset_cells_x = original_offset + left_extension  # Shifted right
        
        # Create full-size canvas for the heightmap (rectangular)
        full_heightmap = np.zeros((airflow_dim_y, airflow_dim_x))
        if offset_cells_y >= 0 and offset_cells_x >= 0:
            full_heightmap[offset_cells_y:offset_cells_y+heightmap_dim, 
                          offset_cells_x:offset_cells_x+heightmap_dim] = hmap.reshape((heightmap_dim, heightmap_dim))
        else:
            full_heightmap = hmap.reshape((heightmap_dim, heightmap_dim))
        
        # Calculate parcel center for landuse split
        design_cells = heightmap_dim
        parcel_center_col = offset_cells_x + design_cells // 2
        
        # Create landuse array (matching evaluation_klam.py)
        env_landuse = np.zeros((airflow_dim_y, airflow_dim_x), dtype=np.int8)
        env_landuse[:, :parcel_center_col] = 7  # Left = free space (matches evaluation_klam.py)
        env_landuse[:, parcel_center_col:] = 2  # Right = low-density buildings
        
        # Create terrain height array with 5° slope left of parcel (matching evaluation_klam.py)
        xy_scale = config_encoding.get('xy_scale', 1.0) if config_encoding else 1.0
        slope_angle_deg = 5.0
        slope_gradient = np.tan(np.radians(slope_angle_deg))  # ~0.01746 * slope_angle_deg
        
        env_terrain = np.zeros((airflow_dim_y, airflow_dim_x))
        # Hill starts at left edge, slopes down to parcel left edge
        slope_distance_cells = offset_cells_x  # cells from left edge to parcel start
        slope_distance_m = slope_distance_cells * xy_scale  # meters
        hill_height = slope_distance_m * slope_gradient  # height at left edge
        
        # For columns left of the parcel (col < offset_cells_x), create downward slope
        for col in range(offset_cells_x):
            distance_from_left_m = col * xy_scale
            elevation = hill_height - distance_from_left_m * slope_gradient
            env_terrain[:, col] = max(0, elevation)  # Terrain at parcel starts at 0
        
        # Get terrain height range for colorbar
        terrain_max = hill_height if hill_height > 0 else 1.0
        
        # Calculate wind speed statistics
        # Ground level (2m height) - uq/vq
        wind_speed_2m = np.sqrt(uq**2 + vq**2)
        
        # Air column average - uz/vz (if available)
        if uz is not None and vz is not None:
            wind_speed_column = np.sqrt(uz**2 + vz**2)
        else:
            wind_speed_column = wind_speed_2m  # Fallback if uz/vz not available
        
        # Define parcel region + 50% buffer, but extend to right edge of domain (downwind)
        parcel_start_y = offset_cells_y
        parcel_end_y = offset_cells_y + design_cells
        parcel_start_x = offset_cells_x
        parcel_end_x = offset_cells_x + design_cells
        buffer_size = int(design_cells * 0.5)
        
        roi_start_y = max(0, parcel_start_y - buffer_size)
        roi_end_y = min(airflow_dim_y, parcel_end_y + buffer_size)
        roi_start_x = max(0, parcel_start_x - buffer_size)
        roi_end_x = airflow_dim_x  # Extend to right edge (downwind direction)
        
        # Extract region of interest for statistics
        wind_speed_2m_roi = wind_speed_2m[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        wind_speed_column_roi = wind_speed_column[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        
        # Calculate mean over ROI only
        mean_wind_speed_2m = np.mean(wind_speed_2m_roi)
        mean_wind_speed_column = np.mean(wind_speed_column_roi)
        
        # Plot 1: Landuse (input)
        ax1 = axes[i, 0]
        cmap_landuse = plt.cm.get_cmap('tab10', 10)
        im1 = ax1.imshow(env_landuse, cmap=cmap_landuse, vmin=0, vmax=9)
        ax1.set_title(f"Sample {i}\nLanduse", fontsize=10, fontweight='bold')
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Category', fontsize=8)
        
        # Plot 2: Building height (input) - use global range
        ax2 = axes[i, 1]
        cmap_buildings = plt.cm.viridis
        cmap_buildings.set_under('lightgray')
        im2 = ax2.imshow(full_heightmap, cmap=cmap_buildings, 
                        norm=colors.Normalize(vmin=0.1, vmax=global_height_max))
        ax2.set_title(f"Sample {i}\nBuilding Height", fontsize=10, fontweight='bold')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Floors', fontsize=8)
        
        # Plot 3: Terrain height (input)
        ax3 = axes[i, 2]
        im3 = ax3.imshow(env_terrain, cmap='terrain', vmin=0, vmax=terrain_max)
        ax3.set_title(f"Sample {i}\nTerrain Height", fontsize=10, fontweight='bold')
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Meters', fontsize=8)
        
        # Plot 4: uq velocity at 2m with streamlines (output) - use global velocity range
        ax4 = axes[i, 3]
        im4 = ax4.imshow(uq, cmap='RdBu_r', vmin=global_vel_min, vmax=global_vel_max)
        
        # Add streamlines (flip vq sign to correct direction)
        y_grid, x_grid = np.mgrid[0:airflow_dim_y, 0:airflow_dim_x]
        ax4.streamplot(x_grid, y_grid, uq, -vq, color='black', linewidth=0.5, 
                      density=1.5, arrowsize=0.8, arrowstyle='->')
        
        ax4.set_title(f"Sample {i}\nuq at 2m + Streamlines", fontsize=10, fontweight='bold')
        ax4.axis('off')
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('m/s', fontsize=8)
        
        # Plot 5: vq velocity at 2m with streamlines (output) - use global velocity range
        ax5 = axes[i, 4]
        im5 = ax5.imshow(vq, cmap='RdBu_r', vmin=global_vel_min, vmax=global_vel_max)
        
        # Add streamlines (flip vq sign to correct direction)
        ax5.streamplot(x_grid, y_grid, uq, -vq, color='black', linewidth=0.5, 
                      density=1.5, arrowsize=0.8, arrowstyle='->')
        
        ax5.set_title(f"Sample {i}\nvq at 2m + Streamlines", fontsize=10, fontweight='bold')
        ax5.axis('off')
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label('m/s', fontsize=8)
        
        # Plot 6: uz velocity column average
        ax6 = axes[i, 5]
        if uz is not None:
            im6 = ax6.imshow(uz, cmap='RdBu_r', vmin=global_vel_min, vmax=global_vel_max)
        else:
            im6 = ax6.imshow(uq, cmap='RdBu_r', vmin=global_vel_min, vmax=global_vel_max, alpha=0.3)
        
        # Overlay ROI rectangle
        from matplotlib.patches import Rectangle
        roi_rect = Rectangle((roi_start_x, roi_start_y), roi_end_x - roi_start_x, roi_end_y - roi_start_y,
                            linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
        ax6.add_patch(roi_rect)
        
        ax6.set_title(f"Sample {i}\nuz Column Avg", fontsize=10, fontweight='bold')
        ax6.axis('off')
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        cbar6.set_label('m/s', fontsize=8)
        
        # Plot 7: vz velocity column average
        ax7 = axes[i, 6]
        if vz is not None:
            im7 = ax7.imshow(vz, cmap='RdBu_r', vmin=global_vel_min, vmax=global_vel_max)
        else:
            im7 = ax7.imshow(vq, cmap='RdBu_r', vmin=global_vel_min, vmax=global_vel_max, alpha=0.3)
        
        # Overlay ROI rectangle
        roi_rect2 = Rectangle((roi_start_x, roi_start_y), roi_end_x - roi_start_x, roi_end_y - roi_start_y,
                             linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
        ax7.add_patch(roi_rect2)
        
        ax7.set_title(f"Sample {i}\nvz Column Avg", fontsize=10, fontweight='bold')
        ax7.axis('off')
        cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        cbar7.set_label('m/s', fontsize=8)
        
        # Plot 8: Hx cold airflow height
        ax8 = axes[i, 7]
        if hx is not None:
            im8 = ax8.imshow(hx, cmap='viridis', vmin=0, vmax=global_hx_max)
        else:
            # Show placeholder if no Hx data
            im8 = ax8.imshow(np.zeros_like(uq), cmap='viridis', vmin=0, vmax=global_hx_max, alpha=0.3)
        
        # Overlay ROI rectangle
        roi_rect3 = Rectangle((roi_start_x, roi_start_y), roi_end_x - roi_start_x, roi_end_y - roi_start_y,
                             linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
        ax8.add_patch(roi_rect3)
        
        ax8.set_title(f"Sample {i}\nHx Cold Air Height", fontsize=10, fontweight='bold')
        ax8.axis('off')
        cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        cbar8.set_label('Meters', fontsize=8)
        
        # Plot 9: Ex cold air content (Kälteinhalt in 100 J/m²)
        ax9 = axes[i, 8]
        if ex is not None:
            im9 = ax9.imshow(ex, cmap='viridis', vmin=0, vmax=global_ex_max)
        else:
            # Show placeholder if no Ex data
            im9 = ax9.imshow(np.zeros_like(uq), cmap='viridis', vmin=0, vmax=global_ex_max, alpha=0.3)
        
        # Overlay ROI rectangle
        roi_rect4 = Rectangle((roi_start_x, roi_start_y), roi_end_x - roi_start_x, roi_end_y - roi_start_y,
                             linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
        ax9.add_patch(roi_rect4)
        
        ax9.set_title(f"Sample {i}\nEx Cold Air Content", fontsize=10, fontweight='bold')
        ax9.axis('off')
        cbar9 = plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
        cbar9.set_label('100 J/m²', fontsize=8)
        
        # Plot 10: Statistics text box
        ax10 = axes[i, 9]
        ax10.axis('off')
        
        # Calculate mean Hx over ROI if available
        if hx is not None:
            hx_roi = hx[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
            mean_hx = np.mean(hx_roi)
            hx_stats = f"Mean Hx: {mean_hx:.2f} m"
        else:
            mean_hx = 0
            hx_stats = "Hx: N/A"
        
        # Calculate mean Ex over ROI if available
        if ex is not None:
            ex_roi = ex[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
            mean_ex = np.mean(ex_roi)
            ex_stats = f"Mean Ex: {mean_ex:.1f}"
        else:
            mean_ex = 0
            ex_stats = "Ex: N/A"
        
        # Calculate Cold Air Content Flux: Ex * u_2m (cooling potential metric)
        cold_air_flux = mean_ex * mean_wind_speed_2m
        
        # Collect metrics for correlation analysis
        metrics_data.append([mean_wind_speed_2m, mean_wind_speed_column, mean_hx, mean_ex, cold_air_flux])
        
        stats_text = f"Sample {i}\n\n" \
                    f"Mean Values\n" \
                    f"(Parcel to Right Edge)\n" \
                    f"{'='*30}\n\n" \
                    f"Wind at 2m (uq/vq):\n" \
                    f"  Mean: {mean_wind_speed_2m:.3f} m/s\n\n" \
                    f"Wind Column (uz/vz):\n" \
                    f"  Mean: {mean_wind_speed_column:.3f} m/s\n\n" \
                    f"Cold Air Height:\n" \
                    f"  {hx_stats}\n\n" \
                    f"Cold Air Content:\n" \
                    f"  {ex_stats} (100 J/m²)\n\n" \
                    f"Cold Air Flux (Ex*u):\n" \
                    f"  {cold_air_flux:.1f}"
        ax10.text(0.5, 0.5, stats_text, 
                ha='center', va='center',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=1', 
                         facecolor='lightblue', 
                         edgecolor='black', 
                         linewidth=2))
    
    debug_plot_path = os.path.join(output_path, 'debug_klam_comprehensive.png')
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"DEBUG: Saved comprehensive KLAM visualization to {debug_plot_path}")
    
    # Generate correlation matrix heatmap
    if len(metrics_data) >= 2:
        metrics_array = np.array(metrics_data)
        metric_names = ['Wind 2m\n(m/s)', 'Wind Col\n(m/s)', 'Hx\n(m)', 'Ex\n(100 J/m²)', 'Flux\n(Ex×u)']
        
        # Compute Pearson correlation matrix
        corr_matrix = np.corrcoef(metrics_array.T)
        
        # Create correlation heatmap figure
        fig_corr, ax_corr = plt.subplots(figsize=(8, 7))
        
        # Plot heatmap
        im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson Correlation', fontsize=11)
        
        # Set ticks and labels
        ax_corr.set_xticks(np.arange(len(metric_names)))
        ax_corr.set_yticks(np.arange(len(metric_names)))
        ax_corr.set_xticklabels(metric_names, fontsize=10)
        ax_corr.set_yticklabels(metric_names, fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add correlation values as text annotations
        for i_row in range(len(metric_names)):
            for j_col in range(len(metric_names)):
                corr_val = corr_matrix[i_row, j_col]
                # Choose text color based on background
                text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                ax_corr.text(j_col, i_row, f'{corr_val:.2f}',
                           ha='center', va='center', color=text_color,
                           fontsize=11, fontweight='bold')
        
        ax_corr.set_title(f'Metric Correlations (n={len(metrics_data)} samples)', 
                         fontsize=14, fontweight='bold', pad=15)
        
        # Save correlation figure
        corr_plot_path = os.path.join(output_path, 'correlation_matrix.png')
        plt.savefig(corr_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig_corr)
        print(f"DEBUG: Saved correlation matrix to {corr_plot_path}")
    else:
        print(f"DEBUG: Not enough samples ({len(metrics_data)}) for correlation analysis (need >= 2)")

def save_debug_visualizations(heightmaps, fitnesses, output_path):
    """Saves a grid of heightmap plots with building height labels for debugging."""
    n_images = len(heightmaps)
    if n_images == 0:
        return
    
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # Z-scale: meters per floor (typically 3.0m)
    z_scale = 3.0  # meters per floor

    for i, (hmap, fit) in enumerate(zip(heightmaps, fitnesses)):
        l = int(np.sqrt(hmap.size))
        heightmap_2d = hmap.reshape((l, l))
        axes[i].imshow(heightmap_2d, cmap='viridis')
        axes[i].set_title(f"Sample {i}\nFitness: {fit:.3f}", fontsize=8)
        axes[i].axis('off')
        
        # Label each building with its height in METERS
        from scipy.ndimage import label as scipy_label, center_of_mass
        labeled_buildings, num_buildings = scipy_label(heightmap_2d > 0)
        
        if num_buildings > 0:
            for building_id in range(1, num_buildings + 1):
                building_mask = labeled_buildings == building_id
                building_height_floors = heightmap_2d[building_mask].max()
                building_height_meters = building_height_floors * z_scale
                cy, cx = center_of_mass(building_mask)
                
                # Add text label with building height in meters
                axes[i].text(cx, cy, f"{building_height_meters:.1f}m", 
                           color='white', fontsize=6, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    debug_plot_path = os.path.join(output_path, 'debug_initial_samples.png')
    plt.savefig(debug_plot_path)
    plt.close(fig)
    print(f"DEBUG: Saved initial sample visualizations to {debug_plot_path}")


def visualize_solutions(heightmaps, fitness_values):
    l = int(np.sqrt(heightmaps.shape[1]))
    # Visualize all heightmaps in a rectangular grid
    fig, axs = plt.subplots(l, l, figsize=(15, 15))
    for i, ax in enumerate(axs.flat):
        if i >= len(heightmaps):
            break
        ax.imshow(heightmaps[i].reshape((l, l)), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Fitness: {fitness_values[i]:.2f}')
    plt.tight_layout()
    plt.draw()
    plt.pause(2.0)

# Helper function to filter out samples that are too close
def filter_close_samples(X, y, tol=1e-3):
    distances = np.linalg.norm(X[:, None] - X, axis=2)
    np.fill_diagonal(distances, np.inf)
    close_indices = np.where(distances < tol)
    mask = np.ones(len(X), dtype=bool)
    mask[close_indices[0]] = False
    X = X[mask]
    y = y[mask]
    return X, y

def run_sail_optimization(config_environment, solution_template, progress_callback=None, result_path=None, lambda_ucb=None, config_optimization=None, config_encoding=None, run_parallel=True, debug=False, resume_archive=None, resume_stats=None, resume_generation=0, resume_gp_data=None):
    eval_method = config_environment.get('evaluation_method', 'flood_fill')
    print(f"Using evaluation method: {eval_method}")
    eval_module_name = f"domain_description.evaluation_{eval_method}" if eval_method == 'klam' else "domain_description.evaluation"
    eval_module = importlib.import_module(eval_module_name)
    eval_multiple = eval_module.eval_multiple

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(result_path, current_datetime)
    os.makedirs(output_path, exist_ok=True)
    
    # Save configs (if files exist)
    if os.path.exists('optimization/cfg.yml'):
        shutil.copyfile('optimization/cfg.yml', os.path.join(output_path, 'cfg_optimization.yml'))
    if os.path.exists('domain_description/cfg.yml'):
        shutil.copyfile('domain_description/cfg.yml', os.path.join(output_path, 'cfg_domain_description.yml'))
    if os.path.exists('encodings/parametric/cfg.yml'):
        shutil.copyfile('encodings/parametric/cfg.yml', os.path.join(output_path, 'cfg_encodings_parametric.yml'))
    
    if config_optimization is None:
        with open('optimization/cfg.yml') as f:
            config_optimization = yaml.safe_load(f)

    if debug:
        config_optimization['num_generations'] = 10 # Just a few generations
        config_optimization['surrogate_num_init_samples'] = 8 # A smaller batch for viz
        config_optimization['surrogate_update_frequency'] = 100 # Avoid updates
        # run_parallel = False # Avoid multiprocessing issues in debug    
    if lambda_ucb is None:
        lambda_ucb = config_optimization['lambda_ucb']

    # Use passed config_encoding or load from file
    if config_encoding is None:
        with open('encodings/parametric/cfg.yml') as f:
            config_encoding: Dict = yaml.safe_load(f)

    selected_features = config_environment['features']
    labels = config_environment['labels']
    labels = [labels[i] for i in selected_features]
    feat_ranges = np.array(config_environment['feat_ranges']).T
    feat_ranges = [feat_ranges[i] for i in selected_features]
    dims = [config_optimization["num_niches"]] * len(selected_features)
    
    l = solution_template.config['length_design']
    
    # Resume from existing archive if provided
    if resume_archive is not None:
        print(f"\n{'='*60}")
        print(f"RESUMING from existing archive")
        print(f"  Starting generation: {resume_generation:,}")
        print(f"  Archive has {len(resume_archive.data()['objective']):,} elites")
        print(f"{'='*60}\n")
        working_archive = resume_archive
    else:
        # Define the archives and emitters
        working_archive = GridArchive(
            solution_dim=solution_template.get_dimension(),
            dims=dims,
            ranges=feat_ranges,
            learning_rate=config_optimization['learning_rate'],
            threshold_min=0.0,
            extra_fields={'heightmaps': ((l*l,), np.float32)}
        )
    emitters = [
        GaussianEmitter(
            working_archive,
            x0=[0.0] * solution_template.get_dimension(),
            sigma=config_optimization['sigma'],
            bounds=None,
            batch_size=config_optimization['batch_size'],
        ) for _ in range(config_optimization['num_emitters'])
    ]
    
    scheduler = Scheduler(working_archive, emitters) # , result_archive=result_archive)
    if run_parallel:
        nb_cpus = psutil.cpu_count(logical=True)
        pool = multiprocessing.Pool(processes=nb_cpus)
    else:
        pool = None

    # Initialize surrogate training data via Sobol sampling if enabled
    # SKIP initialization if resuming from existing archive
    if config_optimization['surrogate_update_frequency'] > 0 and resume_archive is None:
        dim = solution_template.get_dimension()        
        num_sobol = config_optimization['surrogate_num_init_samples']
        
        print(f"\n[INIT] Generating {num_sobol} Sobol samples for surrogate initialization...", flush=True)
        sobol_engine = qmc.Sobol(d=dim, scramble=True)
        sobol_samples = sobol_engine.random(n=num_sobol)
        # Scale samples from [0,1] to [-1,1]
        sobol_samples = 2 * sobol_samples - 1
        
        print(f"[INIT] Evaluating {num_sobol} initial samples with KLAM_21...", flush=True)
        init_start = datetime.now()
        init_results, debug_data_list, _ = eval_multiple(sobol_samples, config_environment, config_encoding, solution_template, surrogate_model=None, pool=pool, debug=debug)
        init_eval_time = datetime.now() - init_start
        print(f"[INIT] Initial evaluation completed in {init_eval_time}", flush=True)
        
        init_fitness = init_results[:, 0]
        init_features = init_results[:, 1:len(selected_features)+1]
        init_heightmaps = init_results[:, len(selected_features)+1:]
        
        print(f"[INIT] Initial fitness range: [{init_fitness.min():.2f}, {init_fitness.max():.2f}]", flush=True)

        if debug and debug_data_list:
            print(f'num_sobol: {num_sobol}')
            print(f'Number of debug data entries: {len(debug_data_list)}')
            if config_environment['evaluation_method'] == 'klam':
                save_velocity_field_visualizations(init_heightmaps, debug_data_list, output_path, config_encoding)
            elif config_environment['evaluation_method'] == 'flood_fill':
                save_floodfill_visualizations(init_heightmaps, debug_data_list, config_environment, output_path)
            # Early exit in debug mode - return empty archive and labels
            if pool:
                pool.close()
                pool.join()
            return working_archive, labels, output_path
        
        gp_training_X = init_heightmaps
        gp_training_y = init_fitness
        
        # Filter out NaN fitness values (failed KLAM evaluations)
        valid_mask = ~np.isnan(gp_training_y)
        if not np.all(valid_mask):
            num_invalid = np.sum(~valid_mask)
            print(f"[INIT] WARNING: Filtering {num_invalid} samples with NaN fitness values", flush=True)
            gp_training_X = gp_training_X[valid_mask]
            gp_training_y = gp_training_y[valid_mask]
        
        if len(gp_training_y) < 10:
            raise ValueError(f"Too few valid samples for GP training: {len(gp_training_y)}. Most KLAM simulations failed.")
        
        gp_training_X, gp_training_y = filter_close_samples(gp_training_X, gp_training_y, tol=1e-3)

        print(f"[INIT] Training initial surrogate model with {len(gp_training_y)} valid samples...", flush=True)
        start_training = datetime.now()                
        surrogate_model = train_gp(gp_training_X, gp_training_y, training_iter=config_optimization['surrogate_num_epochs'])
        training_time = datetime.now() - start_training
        print(f"[INIT] Surrogate training completed in {training_time}", flush=True)
        
        init_results, debug_airflow, _ = eval_multiple(sobol_samples, config_environment, config_encoding, solution_template, surrogate_model=surrogate_model, lambda_ucb=lambda_ucb, pool=pool)
        init_fitness = init_results[:, 0]
        working_archive.clear()
        working_archive.add(sobol_samples, init_fitness, init_features, heightmaps=init_heightmaps)
        init_solution_in_archive = working_archive.data(fields='solution')
        print(f'[INIT] Archive initialized with {init_solution_in_archive.shape[0]} solutions', flush=True)

    elif config_optimization['surrogate_update_frequency'] > 0 and resume_archive is not None:
        # Resume case: load existing GP training data (actual KLAM evaluations)
        print(f"\n[RESUME] Loading GP training data (real KLAM evaluations)...", flush=True)
        
        if resume_gp_data is not None and len(resume_gp_data) == 2:
            # GP training data provided explicitly
            gp_training_X, gp_training_y = resume_gp_data
            print(f"[RESUME] Loaded {len(gp_training_y)} KLAM-evaluated samples from resume_gp_data", flush=True)
        else:
            raise ValueError(
                "Cannot resume SAIL without GP training data!\n"
                "The archive contains surrogate-predicted fitness, not real KLAM evaluations.\n"
                "You must provide resume_gp_data=(X, y) with actual KLAM-evaluated samples.\n"
                "These should be loaded from the saved *_gp_data.pkl file."
            )
        
        # Train surrogate model with actual KLAM evaluations
        print(f"[RESUME] Training surrogate model with {len(gp_training_y)} KLAM-evaluated samples...", flush=True)
        start_training = datetime.now()
        surrogate_model = train_gp(gp_training_X, gp_training_y, training_iter=config_optimization['surrogate_num_epochs'])
        training_time = datetime.now() - start_training
        print(f"[RESUME] Surrogate training completed in {training_time}", flush=True)
                    
    else:
        gp_training_X, gp_training_y = None, None
        surrogate_model = None
        likelihood = None

    # Helper function that runs the QD loop.
    # The lambda_val parameter controls the UCB exploration bonus.
    # When update_surrogate is True, the surrogate model is updated from simulation evaluations.
    def optimization_loop(lambda_val, num_iterations, phase_label, update_surrogate=True, start_iteration=0, resume_stats_list=None):
        stats = resume_stats_list if resume_stats_list is not None else []
        start_time = datetime.now()
        last_log_time = start_time
        nonlocal gp_training_X, gp_training_y, surrogate_model, likelihood
        
        # Print start message immediately
        if start_iteration == 0:
            print(f"\n{'='*60}", flush=True)
            print(f"Starting {phase_label} optimization: {num_iterations:,} generations", flush=True)
            print(f"{'='*60}", flush=True)
        else:
            print(f"\n{'='*60}", flush=True)
            print(f"RESUMING {phase_label} optimization from generation {start_iteration:,}", flush=True)
            print(f"Target: {num_iterations:,} generations (remaining: {num_iterations - start_iteration:,})", flush=True)
            print(f"{'='*60}", flush=True)
        
        for itr in range(start_iteration, num_iterations):
            # Calculate progress metrics
            elapsed = datetime.now() - start_time
            progress_pct = (itr + 1) / num_iterations * 100
            
            # Log progress: at start, every output_inv_frequency, or every 5 minutes
            time_since_log = (datetime.now() - last_log_time).total_seconds()
            should_log = (itr == 0 or 
                         itr % config_optimization['output_inv_frequency'] == 0 or 
                         time_since_log >= 300)  # Log at least every 5 minutes
            
            if should_log:
                last_log_time = datetime.now()
                # Estimate time remaining
                if itr > 0:
                    avg_time_per_iter = elapsed.total_seconds() / itr
                    remaining_iters = num_iterations - itr
                    eta_seconds = avg_time_per_iter * remaining_iters
                    eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = "calculating..."
                
                print(f"[{phase_label}] Gen {itr:,}/{num_iterations:,} ({progress_pct:.1f}%) | "
                      f"Elapsed: {elapsed} | ETA: {eta_str} | "
                      f"Coverage: {working_archive.stats.coverage:.2%} | "
                      f"QD: {working_archive.stats.qd_score:.2f}", flush=True)
                
            if update_surrogate and (itr % config_optimization['surrogate_update_frequency']) == 0:
                if itr != 0:
                    # Evaluate elite solutions to augment the GP training set
                    new_solutions_dict = working_archive.sample_elites(config_optimization['surrogate_num_samples'])                    
                    new_results, debug_airflow, _ = eval_multiple(new_solutions_dict['solution'], config_environment, config_encoding, solution_template, surrogate_model=None, pool=pool, debug=debug)
                    new_heightmaps = new_results[:, len(selected_features)+1:]
                    new_fitness = new_results[:, 0]
                    
                    # Filter out NaN fitness values before adding to training set
                    valid_mask = ~np.isnan(new_fitness)
                    if not np.all(valid_mask):
                        num_invalid = np.sum(~valid_mask)
                        print(f"  → WARNING: Filtering {num_invalid} samples with NaN fitness", flush=True)
                        new_heightmaps = new_heightmaps[valid_mask]
                        new_fitness = new_fitness[valid_mask]
                    
                    if len(new_fitness) > 0:
                        gp_training_X = np.concatenate((gp_training_X, new_heightmaps), axis=0)
                        gp_training_y = np.concatenate((gp_training_y, new_fitness), axis=0)
                    gp_training_X, gp_training_y = filter_close_samples(gp_training_X, gp_training_y, tol=1e-3)
                    
                    # surrogate_model, likelihood 
                    # Measure training time
                    print(f"  → Training surrogate model with {gp_training_X.shape[0]} samples...", flush=True)
                    surrogate_model = train_gp(gp_training_X, gp_training_y, training_iter=config_optimization['surrogate_num_epochs'])
                    print(f"  → Surrogate training complete. Reevaluating archive...", flush=True)
                    
                    # Reevaluate all archive solutions using the current surrogate model and lambda_val
                    current_solutions = working_archive.data(fields='solution')
                    updated_results, debug_airflow, _ = eval_multiple(np.asarray(current_solutions), config_environment, config_encoding, solution_template,
                                            surrogate_model=surrogate_model, lambda_ucb=lambda_val, pool=pool)
                    updated_fitness_values = updated_results[:, 0]
                    current_feature_values = updated_results[:, 1:len(selected_features)+1]
                    current_heightmaps = updated_results[:, len(selected_features)+1:]                    
                    working_archive.clear()
                    working_archive.add(current_solutions, updated_fitness_values, current_feature_values, heightmaps=current_heightmaps)
                    
            
            # Emit solutions and evaluate them
            emitted_solutions = scheduler.ask()
            emitted_results, debug_airflow, _ = eval_multiple(emitted_solutions, config_environment, config_encoding, solution_template,
                                    surrogate_model=surrogate_model, lambda_ucb=lambda_val, pool=pool, debug=debug)
            emitted_fitness_values = emitted_results[:, 0]
            emitted_feature_values = emitted_results[:, 1:len(selected_features)+1]
            emitted_heightmaps = emitted_results[:, len(selected_features)+1:]
            scheduler.tell(emitted_fitness_values, emitted_feature_values, heightmaps=emitted_heightmaps)

            # Update the stats and save the archive
            stats.append(working_archive.stats)
            if progress_callback:
                progress_callback(itr + 1, num_iterations)
            if itr % config_optimization['output_inv_frequency'] == 0 or itr == num_iterations - 1:
                # Append generation and stats to text file
                with open(f'{output_path}stats.txt', 'a') as f:
                    f.write(f'SAIL Generation: {itr}\n')
                    f.write(f'QD score: {working_archive.stats.qd_score}\n')
                    f.write(f'Coverage: {working_archive.stats.coverage}\n')
                    f.write(f'Time passed: {datetime.now() - start_time}\n')
                    f.write('\n')

                with open(f'{output_path}{phase_label}_archive.pkl', 'wb') as output:
                    pickle.dump(working_archive, output)
                with open(f'{output_path}{phase_label}_stats.pkl', 'wb') as output:
                    pickle.dump(stats, output)
                # Save GP training data (actual KLAM evaluations)
                if gp_training_X is not None and gp_training_y is not None:
                    with open(f'{output_path}{phase_label}_gp_data.pkl', 'wb') as output:
                        pickle.dump({'X': gp_training_X, 'y': gp_training_y}, output)
        
        # Print completion message
        total_time = datetime.now() - start_time
        print(f"\n{'='*60}", flush=True)
        print(f"Completed {phase_label}: {num_iterations:,} generations in {total_time}", flush=True)
        print(f"Final QD score: {working_archive.stats.qd_score:.2f}", flush=True)
        print(f"Final Coverage: {working_archive.stats.coverage:.2%}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # Phase 1: Run surrogate‐assisted illumination with the original lambda_ucb for the exploration phase and training the surrogate model
    # Determine where to start based on resume parameters
    sail_start_gen = resume_generation if resume_generation < config_optimization['num_generations'] else config_optimization['num_generations']
    sail_resume_stats = resume_stats if (resume_stats is not None and resume_generation < config_optimization['num_generations']) else None
    
    if sail_start_gen < config_optimization['num_generations']:
        optimization_loop(lambda_ucb, config_optimization['num_generations'], phase_label="SAIL", 
                         update_surrogate=True, start_iteration=sail_start_gen, resume_stats_list=sail_resume_stats)
    else:
        print(f"\n{'='*60}")
        print(f"SAIL phase already complete (generation {resume_generation:,} >= {config_optimization['num_generations']:,})")
        print(f"Skipping to FinalQD phase...")
        print(f"{'='*60}\n")
    
    # Phase 2: Run final QD using the surrogate model but with lambda_ucb = 0 (pure exploitation)
    # Reevaluate all archive solutions using the current surrogate model and lambda_val set to 0, because we are now exploiting!
    final_qd_start_gen = max(0, resume_generation - config_optimization['num_generations'])  # How far into FinalQD we are
    
    # If resuming into FinalQD phase, use the provided stats (which should be FinalQD stats)
    # Otherwise start fresh
    if resume_generation >= config_optimization['num_generations'] and resume_stats is not None:
        final_qd_resume_stats = resume_stats
        print(f"[FinalQD] Resuming with provided stats from generation {final_qd_start_gen}")
    else:
        final_qd_resume_stats = None
    
    if final_qd_start_gen == 0:
        # Starting FinalQD phase from beginning - reevaluate all solutions
        current_solutions = working_archive.data(fields='solution')
        updated_results, debug_airflow, _ = eval_multiple(np.asarray(current_solutions), config_environment, config_encoding, solution_template,
                                surrogate_model=surrogate_model, lambda_ucb=0, pool=pool, debug=debug)
        updated_fitness_values = updated_results[:, 0]
        current_feature_values = updated_results[:, 1:len(selected_features)+1]
        current_heightmaps = updated_results[:, len(selected_features)+1:]                    
        working_archive.clear()
        working_archive.add(current_solutions, updated_fitness_values, current_feature_values, heightmaps=current_heightmaps)
    
    optimization_loop(0, config_optimization['num_generations'], phase_label="FinalQD", 
                     update_surrogate=False, start_iteration=final_qd_start_gen, resume_stats_list=final_qd_resume_stats)

    return working_archive, labels, output_path
