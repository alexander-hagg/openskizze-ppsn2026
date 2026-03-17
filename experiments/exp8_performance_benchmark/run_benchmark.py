#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Experiment 8: Performance Benchmarking for QD Optimization

This script benchmarks individual components and proposed optimizations
to identify bottlenecks and measure improvements.

Usage:
    # Run all benchmarks
    python experiments/exp8_performance_benchmark/run_benchmark.py --all
    
    # Run specific benchmark
    python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark features
    python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark domain
    python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark inference
    python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark flux
    python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark full
"""

import argparse
import json
import logging
import time
import cProfile
import pstats
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
import gpytorch
from scipy.ndimage import label, center_of_mass
import numba
from numba import jit, prange

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)
from domain_description.evaluation_klam import calculate_planning_features
from scipy.special import erf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    batch_sizes: List[int] = None
    num_warmup: int = 3
    num_iterations: int = 10
    parcel_size: int = 60
    device: str = 'cuda'
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [64, 128, 256, 512, 1024, 2048]


# ============================================================================
# Fast norm2unif Implementation
# ============================================================================

def fast_norm2unif(x, min_val=0.0, max_val=1.0, mu=None, sd=None):
    """Fast replacement for scipy.stats-based norm2unif using erf.
    
    Converts N(mu, sd) to Uniform[min_val, max_val] without scipy.stats overhead.
    """
    if mu is None:
        mu = np.mean(x)
    if sd is None:
        sd = np.std(x, ddof=1)
    
    # Standardize to N(0, 1)
    z = (x - mu) / sd
    
    # CDF of standard normal: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    uniform_01 = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # Scale to [min_val, max_val]
    return min_val + uniform_01 * (max_val - min_val)


# ============================================================================
# Baseline Implementations (Current Code)
# ============================================================================

def baseline_compute_features_batch(genomes: np.ndarray, parcel_size: int) -> np.ndarray:
    """Current (slow) feature computation - one encoding per solution."""
    features = []
    length_design = parcel_size // 3
    
    for genome in genomes:
        config_encoding = {
            'length_design': length_design,
            'max_num_buildings': 10,
            'max_num_floors': 10,
            'xy_scale': 3.0,
            'z_scale': 3.0
        }
        encoding = ParametricEncoding(config=config_encoding)
        heightmap = encoding.express(genome, as_height_map=True)
        feat = calculate_planning_features(heightmap, config_encoding)
        features.append(feat)
    
    return np.array(features)


def baseline_construct_domain_grids(
    genomes: np.ndarray, 
    parcel_size: int,
    grid_h: int = 66,
    grid_w: int = 94
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Current (slow) domain grid construction - one loop per solution."""
    parcel_size_cells = parcel_size // 3
    
    terrain_list = []
    buildings_list = []
    landuse_list = []
    
    for genome in genomes:
        config_encoding = {
            'length_design': parcel_size_cells,
            'max_num_buildings': 10,
            'max_num_floors': 10,
            'xy_scale': 3.0,
            'z_scale': 3.0
        }
        encoding = ParametricEncoding(config=config_encoding)
        heightmap = encoding.express(genome, as_height_map=True)
        
        terrain = np.zeros((grid_h, grid_w), dtype=np.float32)
        buildings = np.zeros((grid_h, grid_w), dtype=np.float32)
        landuse = np.full((grid_h, grid_w), 7, dtype=np.float32)
        
        offset_h = (grid_h - parcel_size_cells) // 2
        offset_w = (grid_w - parcel_size_cells) // 2
        
        buildings[offset_h:offset_h+parcel_size_cells, 
                 offset_w:offset_w+parcel_size_cells] = heightmap * 3.0
        landuse[offset_h:offset_h+parcel_size_cells,
               offset_w:offset_w+parcel_size_cells] = np.where(heightmap > 0, 2, 7)
        
        terrain_list.append(terrain)
        buildings_list.append(buildings)
        landuse_list.append(landuse)
    
    return np.array(terrain_list), np.array(buildings_list), np.array(landuse_list)


def baseline_compute_flux(
    Ex: np.ndarray,
    uq: np.ndarray,
    vq: np.ndarray,
    roi_mask: np.ndarray
) -> np.ndarray:
    """Current (slow) cold air flux computation - Python loop."""
    uq_ms = uq / 100.0
    vq_ms = vq / 100.0
    wind_speed = np.sqrt(uq_ms**2 + vq_ms**2)
    
    flux = np.zeros(len(Ex))
    for i in range(len(Ex)):
        Ex_roi = Ex[i][roi_mask]
        wind_roi = wind_speed[i][roi_mask]
        flux[i] = np.mean(Ex_roi) * np.mean(wind_roi)
    
    return flux


# ============================================================================
# Optimized Implementations
# ============================================================================

class FastEncoding:
    """Encoding with fast norm2unif (no scipy.stats overhead)."""
    def __init__(self, parcel_size: int):
        self.parcel_size = parcel_size
        self.length_design = parcel_size // 3
        self.config = {
            'length_design': self.length_design,
            'max_num_buildings': 10,
            'max_num_floors': 10,
            'xy_scale': 3.0,
            'z_scale': 3.0
        }
    
    def express_batch(self, genomes: np.ndarray) -> np.ndarray:
        """Vectorized expression with fast norm2unif."""
        batch_size = len(genomes)
        heightmaps = np.zeros((batch_size, self.length_design, self.length_design), dtype=np.float32)
        
        # Apply fast_norm2unif to all genomes at once
        genomes_uniform = np.clip(fast_norm2unif(genomes), 0, 1)
        
        for i in range(batch_size):
            genome = genomes_uniform[i]
            phenotype = np.zeros((self.length_design, self.length_design))
            
            for j in range(10):  # max_num_buildings
                if genome[j*6 + 5] > 0.5:  # active_bit
                    # Match original ParametricEncoding logic exactly
                    x_origin = int(genome[j*6 + 3] * self.length_design)
                    y_origin = int(genome[j*6 + 4] * self.length_design)
                    width = int(genome[j*6 + 0] * self.length_design)
                    length = int(genome[j*6 + 1] * self.length_design)
                    # Height: np.floor(genome * (max_num_floors + 1))
                    num_floors = int(np.floor(genome[j*6 + 2] * (10 + 1)))
                    num_floors = min(num_floors, 10)
                    
                    # Calculate bounds (center ± 0.5 * dimension)
                    x_start = max(0, int(x_origin - 0.5 * width))
                    x_end = min(self.length_design, int(x_origin + 0.5 * width))
                    y_start = max(0, int(y_origin - 0.5 * length))
                    y_end = min(self.length_design, int(y_origin + 0.5 * length))
                    
                    phenotype[y_start:y_end, x_start:x_end] = np.maximum(
                        phenotype[y_start:y_end, x_start:x_end], 
                        num_floors
                    )
            
            heightmaps[i] = phenotype
        
        return heightmaps


class OptimizedEncoding:
    """Optimized encoding with reusable config and vectorized operations."""
    
    def __init__(self, parcel_size: int):
        self.length_design = parcel_size // 3
        self.config = {
            'length_design': self.length_design,
            'max_num_buildings': 10,
            'max_num_floors': 10,
            'xy_scale': 3.0,
            'z_scale': 3.0
        }
        # Create reusable encoding object
        self._encoding = ParametricEncoding(config=self.config)
    
    def express_batch(self, genomes: np.ndarray) -> np.ndarray:
        """Express multiple genomes to heightmaps (still sequential, but reuses encoding)."""
        heightmaps = np.zeros((len(genomes), self.length_design, self.length_design), 
                              dtype=np.float32)
        for i, genome in enumerate(genomes):
            heightmaps[i] = self._encoding.express(genome, as_height_map=True)
        return heightmaps


def optimized_compute_features_batch(
    genomes: np.ndarray, 
    parcel_size: int,
    encoding: Optional[OptimizedEncoding] = None,
    heightmaps: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized feature computation with shared encoding and heightmap caching.
    
    Returns (features, heightmaps) so heightmaps can be reused.
    """
    if encoding is None:
        encoding = OptimizedEncoding(parcel_size)
    
    if heightmaps is None:
        heightmaps = encoding.express_batch(genomes)
    
    # Compute features (still uses calculate_planning_features, but avoids encoding overhead)
    features = []
    for heightmap in heightmaps:
        feat = calculate_planning_features(heightmap, encoding.config)
        features.append(feat)
    
    return np.array(features), heightmaps


def optimized_construct_domain_grids(
    heightmaps: np.ndarray,
    parcel_size_cells: int,
    grid_h: int = 66,
    grid_w: int = 94
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized domain grid construction using vectorized NumPy operations.
    
    Uses pre-computed heightmaps (avoiding duplicate encoding work).
    """
    batch_size = len(heightmaps)
    
    # Pre-allocate all arrays at once
    terrain = np.zeros((batch_size, grid_h, grid_w), dtype=np.float32)
    buildings = np.zeros((batch_size, grid_h, grid_w), dtype=np.float32)
    landuse = np.full((batch_size, grid_h, grid_w), 7, dtype=np.float32)
    
    # Compute placement offsets
    offset_h = (grid_h - parcel_size_cells) // 2
    offset_w = (grid_w - parcel_size_cells) // 2
    
    # Place all heightmaps at once (vectorized)
    buildings[:, offset_h:offset_h+parcel_size_cells, 
             offset_w:offset_w+parcel_size_cells] = heightmaps * 3.0
    
    # Vectorized landuse assignment
    landuse[:, offset_h:offset_h+parcel_size_cells,
           offset_w:offset_w+parcel_size_cells] = np.where(
               heightmaps > 0, 2, 7
           )
    
    return terrain, buildings, landuse


def optimized_compute_flux(
    Ex: np.ndarray,
    uq: np.ndarray,
    vq: np.ndarray,
    roi_mask: np.ndarray
) -> np.ndarray:
    """Optimized cold air flux computation using vectorized operations."""
    # Convert cm/s to m/s
    uq_ms = uq / 100.0
    vq_ms = vq / 100.0
    
    # Compute wind speed (vectorized)
    wind_speed = np.sqrt(uq_ms**2 + vq_ms**2)
    
    # Apply ROI mask and compute means (vectorized)
    # Reshape for broadcasting: (N, H, W) with mask (H, W) -> (N, num_roi_pixels)
    Ex_roi = Ex[:, roi_mask]  # (N, M) where M = sum(roi_mask)
    wind_roi = wind_speed[:, roi_mask]
    
    # Compute mean across ROI pixels
    mean_Ex = np.mean(Ex_roi, axis=1)
    mean_wind = np.mean(wind_roi, axis=1)
    
    return mean_Ex * mean_wind


# ============================================================================
# Numba-Optimized Feature Computation
# ============================================================================

@jit(nopython=True, cache=True)
def numba_find(parent: np.ndarray, x: int) -> int:
    """Find with path compression for Union-Find (iterative to avoid stack overflow)."""
    root = x
    while parent[root] != root:
        root = parent[root]
    
    # Path compression
    while parent[x] != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    
    return root


@jit(nopython=True, cache=True)
def numba_union(parent: np.ndarray, x: int, y: int):
    """Union two sets (simplified version without rank)."""
    px = numba_find(parent, x)
    py = numba_find(parent, y)
    if px != py:
        parent[px] = py


@jit(nopython=True, cache=True)
def numba_connected_components(occupied: np.ndarray) -> tuple:
    """
    Fast connected components labeling using Union-Find algorithm.
    Numba-compatible replacement for scipy.ndimage.label.
    """
    h, w = occupied.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = np.arange(h * w, dtype=np.int32)
    
    # First pass: assign labels and union neighbors
    label_counter = 0
    for i in range(h):
        for j in range(w):
            if occupied[i, j]:
                label_counter += 1
                idx = i * w + j
                labels[i, j] = label_counter
                parent[idx] = idx
                
                # Check neighbors
                if i > 0 and occupied[i-1, j]:
                    neighbor_idx = (i-1) * w + j
                    numba_union(parent, idx, neighbor_idx)
                if j > 0 and occupied[i, j-1]:
                    neighbor_idx = i * w + (j-1)
                    numba_union(parent, idx, neighbor_idx)
    
    # Second pass: relabel with root labels
    # Use simple array instead of typed.Dict (avoids Numba typing issues)
    component_map = np.full(h * w, -1, dtype=np.int32)
    num_components = 0
    for i in range(h):
        for j in range(w):
            if occupied[i, j]:
                idx = i * w + j
                root = numba_find(parent, idx)
                if component_map[root] == -1:
                    num_components += 1
                    component_map[root] = num_components
                labels[i, j] = component_map[root]
    
    return labels, num_components


@jit(nopython=True, cache=True)
def numba_compute_centroids(labeled_array: np.ndarray, num_components: int) -> np.ndarray:
    """Compute centroids of labeled components."""
    h, w = labeled_array.shape
    centroids = np.zeros((num_components, 2), dtype=np.float64)
    counts = np.zeros(num_components, dtype=np.int32)
    
    for i in range(h):
        for j in range(w):
            label_id = labeled_array[i, j]
            if label_id > 0:
                centroids[label_id - 1, 0] += i
                centroids[label_id - 1, 1] += j
                counts[label_id - 1] += 1
    
    for k in range(num_components):
        if counts[k] > 0:
            centroids[k, 0] /= counts[k]
            centroids[k, 1] /= counts[k]
    
    return centroids


@jit(nopython=True, cache=True)
def numba_compute_pairwise_distances(centroids: np.ndarray) -> float:
    """Compute mean pairwise distance between centroids."""
    n = len(centroids)
    if n <= 1:
        return 0.0
    
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centroids[i, 0] - centroids[j, 0]
            dy = centroids[i, 1] - centroids[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            total_dist += dist
            count += 1
    
    return total_dist / count if count > 0 else 0.0


@jit(nopython=True, cache=True)
def numba_calculate_compactness(heightmap: np.ndarray, pixel_size: float) -> float:
    """Calculate Surface-to-Volume ratio (A/V) - Numba compatible."""
    volume = np.sum(heightmap) * (pixel_size ** 2)
    if volume == 0.0:
        return 0.0
    
    rows, cols = heightmap.shape
    
    # Roof area
    roof_area = 0.0
    for i in range(rows):
        for j in range(cols):
            if heightmap[i, j] > 0:
                roof_area += pixel_size ** 2
    
    # Wall area calculation
    total_height_diff = 0.0
    for i in range(rows):
        for j in range(cols):
            h = heightmap[i, j]
            # Check north neighbor
            if i > 0:
                total_height_diff += max(0.0, h - heightmap[i-1, j])
            else:
                total_height_diff += h  # Boundary
            # Check south neighbor
            if i < rows - 1:
                total_height_diff += max(0.0, h - heightmap[i+1, j])
            else:
                total_height_diff += h  # Boundary
            # Check west neighbor
            if j > 0:
                total_height_diff += max(0.0, h - heightmap[i, j-1])
            else:
                total_height_diff += h  # Boundary
            # Check east neighbor
            if j < cols - 1:
                total_height_diff += max(0.0, h - heightmap[i, j+1])
            else:
                total_height_diff += h  # Boundary
    
    wall_area = total_height_diff * pixel_size
    surface_area = roof_area + wall_area
    
    return surface_area / volume


@jit(nopython=True, cache=True)
def numba_distance_transform_edt(binary_mask: np.ndarray) -> np.ndarray:
    """
    Simplified distance transform for park factor calculation.
    Computes Euclidean distance to nearest zero (building) pixel.
    """
    rows, cols = binary_mask.shape
    dist = np.full((rows, cols), 1e10, dtype=np.float64)
    
    # Initialize distances for building pixels (where mask is False/0)
    for i in range(rows):
        for j in range(cols):
            if not binary_mask[i, j]:
                dist[i, j] = 0.0
    
    # Forward pass
    for i in range(rows):
        for j in range(cols):
            if dist[i, j] > 0:
                # Check north
                if i > 0:
                    dist[i, j] = min(dist[i, j], dist[i-1, j] + 1.0)
                # Check west
                if j > 0:
                    dist[i, j] = min(dist[i, j], dist[i, j-1] + 1.0)
    
    # Backward pass
    for i in range(rows-1, -1, -1):
        for j in range(cols-1, -1, -1):
            # Check south
            if i < rows - 1:
                dist[i, j] = min(dist[i, j], dist[i+1, j] + 1.0)
            # Check east
            if j < cols - 1:
                dist[i, j] = min(dist[i, j], dist[i, j+1] + 1.0)
    
    # Compute Euclidean distances (approximation using Chamfer distances)
    # For better accuracy, check diagonals
    for _ in range(2):  # Multiple passes for better approximation
        for i in range(rows):
            for j in range(cols):
                if dist[i, j] > 0:
                    # Check all 8 neighbors
                    min_dist = dist[i, j]
                    sqrt2 = 1.41421356
                    if i > 0:
                        min_dist = min(min_dist, dist[i-1, j] + 1.0)
                        if j > 0:
                            min_dist = min(min_dist, dist[i-1, j-1] + sqrt2)
                        if j < cols - 1:
                            min_dist = min(min_dist, dist[i-1, j+1] + sqrt2)
                    if i < rows - 1:
                        min_dist = min(min_dist, dist[i+1, j] + 1.0)
                        if j > 0:
                            min_dist = min(min_dist, dist[i+1, j-1] + sqrt2)
                        if j < cols - 1:
                            min_dist = min(min_dist, dist[i+1, j+1] + sqrt2)
                    if j > 0:
                        min_dist = min(min_dist, dist[i, j-1] + 1.0)
                    if j < cols - 1:
                        min_dist = min(min_dist, dist[i, j+1] + 1.0)
                    dist[i, j] = min_dist
    
    return dist


@jit(nopython=True, cache=True)
def numba_calculate_park_factor(heightmap: np.ndarray, pixel_size: float) -> float:
    """Calculate park factor (average distance to nearest building) - Numba compatible."""
    rows, cols = heightmap.shape
    
    # Check if there's any open space
    has_open_space = False
    for i in range(rows):
        for j in range(cols):
            if heightmap[i, j] == 0:
                has_open_space = True
                break
        if has_open_space:
            break
    
    if not has_open_space:
        return 0.0
    
    # Create open space mask (True where heightmap == 0)
    open_space = heightmap == 0
    
    # Distance transform: distance to nearest building (heightmap > 0)
    dist_map_pixels = numba_distance_transform_edt(open_space)
    
    # Average over open pixels only
    dist_sum = 0.0
    count = 0
    for i in range(rows):
        for j in range(cols):
            if open_space[i, j]:
                dist_sum += dist_map_pixels[i, j]
                count += 1
    
    if count == 0:
        return 0.0
    
    avg_dist_pixels = dist_sum / count
    return avg_dist_pixels * pixel_size


@jit(nopython=True, cache=True)
def numba_calculate_features(
    heightmap: np.ndarray,
    pixel_size: float
) -> np.ndarray:
    """
    Numba-accelerated feature calculation.
    
    Returns 8 features: [GRZ, GFZ, avg_height, height_std, avg_distance, 
                         num_buildings, compactness, park_factor]
    """
    grid_res_y, grid_res_x = heightmap.shape
    pixel_area = pixel_size * pixel_size
    buildable_area = grid_res_y * grid_res_x * pixel_area
    
    # Count occupied pixels and collect heights manually (Numba doesn't support boolean indexing on 2D)
    occupied_count = 0
    height_sum = 0.0
    height_sum_sq = 0.0
    total_floor_area = 0.0
    
    for i in range(grid_res_y):
        for j in range(grid_res_x):
            h = heightmap[i, j]
            if h > 0:
                occupied_count += 1
                height_sum += h
                height_sum_sq += h * h
            total_floor_area += h
    
    if occupied_count == 0:
        return np.zeros(8, dtype=np.float64)
    
    # [0] GRZ - Site Coverage
    built_area = occupied_count * pixel_area
    grz = built_area / buildable_area if buildable_area > 0 else 0.0
    grz = min(max(grz, 0.0), 1.0)
    
    # [1] GFZ - Floor Area Ratio
    gfz = (total_floor_area * pixel_area) / buildable_area if buildable_area > 0 else 0.0
    
    # [2] Average Height
    avg_height = height_sum / occupied_count
    
    # [3] Height Variability (std dev)
    variance = (height_sum_sq / occupied_count) - (avg_height * avg_height)
    height_std = np.sqrt(max(variance, 0.0))
    
    # [4] Average Building Distance & [5] Number of Buildings
    occupied = heightmap > 0
    labeled_array, num_buildings = numba_connected_components(occupied)
    
    if num_buildings > 1:
        centroids = numba_compute_centroids(labeled_array, num_buildings)
        avg_spacing_pixels = numba_compute_pairwise_distances(centroids)
        avg_spacing = avg_spacing_pixels * pixel_size
    else:
        avg_spacing = 0.0
    
    # [6] Compactness (proper A/V ratio with wall area)
    compactness = numba_calculate_compactness(heightmap, pixel_size)
    
    # [7] Park Factor (distance-based metric)
    park_factor = numba_calculate_park_factor(heightmap, pixel_size)
    
    return np.array([
        grz, gfz, avg_height, height_std,
        avg_spacing, float(num_buildings), compactness, park_factor
    ], dtype=np.float64)


def numba_compute_features_batch(
    genomes: np.ndarray,
    parcel_size: int,
    encoding: Optional[OptimizedEncoding] = None,
    heightmaps: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated feature computation.
    
    Returns (features, heightmaps) so heightmaps can be reused.
    """
    if encoding is None:
        encoding = OptimizedEncoding(parcel_size)
    
    if heightmaps is None:
        heightmaps = encoding.express_batch(genomes)
    
    pixel_size = encoding.config['xy_scale']
    
    # Compute features using Numba (parallelizable)
    features = np.zeros((len(heightmaps), 8), dtype=np.float64)
    for i in range(len(heightmaps)):
        features[i] = numba_calculate_features(heightmaps[i], pixel_size)
    
    return features, heightmaps


# ============================================================================
# Benchmarking Functions
# ============================================================================

def benchmark_function(
    func,
    args,
    num_warmup: int = 3,
    num_iterations: int = 10,
    name: str = "function"
) -> Dict:
    """Run a function multiple times and collect timing statistics."""
    
    # Warmup
    for _ in range(num_warmup):
        _ = func(*args)
    
    # Timed runs
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        'name': name,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'times_ms': times,
    }


def run_feature_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Benchmark feature computation (baseline vs optimized)."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Feature Computation")
    logger.info("=" * 60)
    
    results = []
    
    for batch_size in config.batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")
        
        # Generate random genomes
        genomes = np.random.randn(batch_size, 60).astype(np.float32)
        
        # Baseline
        baseline_result = benchmark_function(
            baseline_compute_features_batch,
            (genomes, config.parcel_size),
            config.num_warmup,
            config.num_iterations,
            f"baseline_features_N{batch_size}"
        )
        baseline_result['batch_size'] = batch_size
        baseline_result['version'] = 'baseline'
        results.append(baseline_result)
        logger.info(f"  Baseline: {baseline_result['mean_ms']:.2f} ± {baseline_result['std_ms']:.2f} ms")
        
        # Optimized
        encoding = OptimizedEncoding(config.parcel_size)
        optimized_result = benchmark_function(
            lambda g: optimized_compute_features_batch(g, config.parcel_size, encoding),
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"optimized_features_N{batch_size}"
        )
        optimized_result['batch_size'] = batch_size
        optimized_result['version'] = 'optimized'
        results.append(optimized_result)
        logger.info(f"  Optimized: {optimized_result['mean_ms']:.2f} ± {optimized_result['std_ms']:.2f} ms")
        
        speedup = baseline_result['mean_ms'] / optimized_result['mean_ms']
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Numba-optimized (JIT compilation happens on first call)
        logger.info(f"  Compiling Numba JIT (first call)...")
        # Trigger JIT compilation with small sample
        _ = numba_compute_features_batch(genomes[:1], config.parcel_size, encoding)
        
        numba_result = benchmark_function(
            lambda g: numba_compute_features_batch(g, config.parcel_size, encoding),
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"numba_features_N{batch_size}"
        )
        numba_result['batch_size'] = batch_size
        numba_result['version'] = 'numba'
        results.append(numba_result)
        logger.info(f"  Numba: {numba_result['mean_ms']:.2f} ± {numba_result['std_ms']:.2f} ms")
        
        speedup_numba = baseline_result['mean_ms'] / numba_result['mean_ms']
        logger.info(f"  Speedup (Numba): {speedup_numba:.2f}x")
        
        # Fast encoding (no scipy.stats overhead)
        fast_encoding = FastEncoding(config.parcel_size)
        # Trigger JIT compilation
        _ = numba_compute_features_batch(genomes[:1], config.parcel_size, fast_encoding)
        
        fast_result = benchmark_function(
            lambda g: numba_compute_features_batch(g, config.parcel_size, fast_encoding),
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"fast_features_N{batch_size}"
        )
        fast_result['batch_size'] = batch_size
        fast_result['version'] = 'fast'
        results.append(fast_result)
        logger.info(f"  Fast (Numba + fast encoding): {fast_result['mean_ms']:.2f} ± {fast_result['std_ms']:.2f} ms")
        
        speedup_fast = baseline_result['mean_ms'] / fast_result['mean_ms']
        logger.info(f"  Speedup (Fast): {speedup_fast:.2f}x")
    
    return results


def run_domain_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Benchmark domain grid construction (baseline vs optimized)."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Domain Grid Construction")
    logger.info("=" * 60)
    
    results = []
    parcel_size_cells = config.parcel_size // 3
    
    for batch_size in config.batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")
        
        # Generate random genomes
        genomes = np.random.randn(batch_size, 60).astype(np.float32)
        
        # Pre-compute heightmaps for optimized version
        encoding = OptimizedEncoding(config.parcel_size)
        heightmaps = encoding.express_batch(genomes)
        
        # Baseline
        baseline_result = benchmark_function(
            baseline_construct_domain_grids,
            (genomes, config.parcel_size),
            config.num_warmup,
            config.num_iterations,
            f"baseline_domain_N{batch_size}"
        )
        baseline_result['batch_size'] = batch_size
        baseline_result['version'] = 'baseline'
        results.append(baseline_result)
        logger.info(f"  Baseline: {baseline_result['mean_ms']:.2f} ± {baseline_result['std_ms']:.2f} ms")
        
        # Optimized
        optimized_result = benchmark_function(
            optimized_construct_domain_grids,
            (heightmaps, parcel_size_cells),
            config.num_warmup,
            config.num_iterations,
            f"optimized_domain_N{batch_size}"
        )
        optimized_result['batch_size'] = batch_size
        optimized_result['version'] = 'optimized'
        results.append(optimized_result)
        logger.info(f"  Optimized: {optimized_result['mean_ms']:.2f} ± {optimized_result['std_ms']:.2f} ms")
        
        speedup = baseline_result['mean_ms'] / optimized_result['mean_ms']
        logger.info(f"  Speedup: {speedup:.2f}x")
    
    return results


def run_flux_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Benchmark cold air flux computation (baseline vs optimized)."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Cold Air Flux Computation")
    logger.info("=" * 60)
    
    results = []
    grid_h, grid_w = 66, 94
    roi_mask = np.ones((grid_h, grid_w), dtype=bool)
    
    for batch_size in config.batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")
        
        # Generate random predictions
        Ex = np.random.rand(batch_size, grid_h, grid_w).astype(np.float32) * 100
        uq = np.random.rand(batch_size, grid_h, grid_w).astype(np.float32) * 200 - 100
        vq = np.random.rand(batch_size, grid_h, grid_w).astype(np.float32) * 200 - 100
        
        # Baseline
        baseline_result = benchmark_function(
            baseline_compute_flux,
            (Ex, uq, vq, roi_mask),
            config.num_warmup,
            config.num_iterations,
            f"baseline_flux_N{batch_size}"
        )
        baseline_result['batch_size'] = batch_size
        baseline_result['version'] = 'baseline'
        results.append(baseline_result)
        logger.info(f"  Baseline: {baseline_result['mean_ms']:.2f} ± {baseline_result['std_ms']:.2f} ms")
        
        # Optimized
        optimized_result = benchmark_function(
            optimized_compute_flux,
            (Ex, uq, vq, roi_mask),
            config.num_warmup,
            config.num_iterations,
            f"optimized_flux_N{batch_size}"
        )
        optimized_result['batch_size'] = batch_size
        optimized_result['version'] = 'optimized'
        results.append(optimized_result)
        logger.info(f"  Optimized: {optimized_result['mean_ms']:.2f} ± {optimized_result['std_ms']:.2f} ms")
        
        speedup = baseline_result['mean_ms'] / optimized_result['mean_ms']
        logger.info(f"  Speedup: {speedup:.2f}x")
    
    return results


def run_inference_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Benchmark model inference (SVGP and U-Net)."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Model Inference")
    logger.info("=" * 60)
    
    results = []
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Try to load models
    svgp_path = project_root / 'results/exp3_hpo/hyperparameterization/model_optimized_ind2500_kmeans_rep1.pth'
    unet_path = project_root / 'results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth'
    
    # SVGP Benchmark
    if svgp_path.exists():
        logger.info("\nSVGP Inference:")
        try:
            from experiments.exp3_hpo.train_gp_hpo import SVGPModel
            
            checkpoint = torch.load(svgp_path, map_location=device)
            inducing_points = checkpoint['model_state_dict']['variational_strategy.inducing_points']
            num_inducing = inducing_points.size(0)
            input_dim = inducing_points.size(1)
            
            model = SVGPModel(inducing_points.to(device), input_dim=input_dim).to(device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            model.eval()
            likelihood.eval()
            
            train_x_mean = checkpoint['train_x_mean'].to(device)
            train_x_std = checkpoint['train_x_std'].to(device)
            
            for batch_size in config.batch_sizes:
                genomes = np.random.randn(batch_size, 60).astype(np.float32)
                parcel_cols = np.full((batch_size, 2), config.parcel_size, dtype=np.float32)
                X = np.column_stack([genomes, parcel_cols])
                X_torch = torch.tensor(X, dtype=torch.float32).to(device)
                X_norm = (X_torch - train_x_mean) / (train_x_std + 1e-6)
                
                def svgp_inference():
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        pred = likelihood(model(X_norm))
                        return pred.mean.cpu().numpy()
                
                result = benchmark_function(
                    svgp_inference,
                    (),
                    config.num_warmup,
                    config.num_iterations,
                    f"svgp_inference_N{batch_size}"
                )
                result['batch_size'] = batch_size
                result['model'] = 'svgp'
                results.append(result)
                logger.info(f"  N={batch_size}: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
        
        except Exception as e:
            logger.warning(f"SVGP benchmark failed: {e}")
    else:
        logger.warning(f"SVGP model not found at {svgp_path}")
    
    # U-Net Benchmark
    if unet_path.exists():
        logger.info("\nU-Net Inference:")
        try:
            from models.unet import UNet, UNetConfig
            
            checkpoint = torch.load(unet_path, map_location=device)
            unet_config = UNetConfig(**checkpoint['config'])
            model = UNet(unet_config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            grid_h, grid_w = unet_config.input_height, unet_config.input_width
            
            for batch_size in config.batch_sizes:
                X = np.random.randn(batch_size, 3, grid_h, grid_w).astype(np.float32)
                X_torch = torch.tensor(X, dtype=torch.float32).to(device)
                
                def unet_inference():
                    with torch.no_grad():
                        pred = model(X_torch)
                        return pred.cpu().numpy()
                
                result = benchmark_function(
                    unet_inference,
                    (),
                    config.num_warmup,
                    config.num_iterations,
                    f"unet_inference_N{batch_size}"
                )
                result['batch_size'] = batch_size
                result['model'] = 'unet'
                results.append(result)
                logger.info(f"  N={batch_size}: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
        
        except Exception as e:
            logger.warning(f"U-Net benchmark failed: {e}")
    else:
        logger.warning(f"U-Net model not found at {unet_path}")
    
    return results


def run_full_pipeline_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Benchmark complete evaluation pipeline (baseline vs optimized)."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Full Pipeline (Features + Domain + Flux)")
    logger.info("=" * 60)
    
    results = []
    grid_h, grid_w = 66, 94
    roi_mask = np.ones((grid_h, grid_w), dtype=bool)
    parcel_size_cells = config.parcel_size // 3
    
    for batch_size in config.batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")
        
        # Generate random genomes
        genomes = np.random.randn(batch_size, 60).astype(np.float32)
        
        # Baseline pipeline
        def baseline_pipeline(genomes):
            features = baseline_compute_features_batch(genomes, config.parcel_size)
            terrain, buildings, landuse = baseline_construct_domain_grids(genomes, config.parcel_size)
            # Simulate U-Net output
            Ex = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 100
            uq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            vq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            flux = baseline_compute_flux(Ex, uq, vq, roi_mask)
            return features, flux
        
        baseline_result = benchmark_function(
            baseline_pipeline,
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"baseline_full_N{batch_size}"
        )
        baseline_result['batch_size'] = batch_size
        baseline_result['version'] = 'baseline'
        results.append(baseline_result)
        logger.info(f"  Baseline: {baseline_result['mean_ms']:.2f} ± {baseline_result['std_ms']:.2f} ms")
        
        # Optimized pipeline
        encoding = OptimizedEncoding(config.parcel_size)
        
        def optimized_pipeline(genomes):
            features, heightmaps = optimized_compute_features_batch(genomes, config.parcel_size, encoding)
            terrain, buildings, landuse = optimized_construct_domain_grids(heightmaps, parcel_size_cells)
            # Simulate U-Net output
            Ex = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 100
            uq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            vq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            flux = optimized_compute_flux(Ex, uq, vq, roi_mask)
            return features, flux
        
        optimized_result = benchmark_function(
            optimized_pipeline,
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"optimized_full_N{batch_size}"
        )
        optimized_result['batch_size'] = batch_size
        optimized_result['version'] = 'optimized'
        results.append(optimized_result)
        logger.info(f"  Optimized: {optimized_result['mean_ms']:.2f} ± {optimized_result['std_ms']:.2f} ms")
        
        speedup = baseline_result['mean_ms'] / optimized_result['mean_ms']
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Numba-optimized pipeline
        def numba_pipeline(genomes):
            features, heightmaps = numba_compute_features_batch(genomes, config.parcel_size, encoding)
            terrain, buildings, landuse = optimized_construct_domain_grids(heightmaps, parcel_size_cells)
            # Simulate U-Net output
            Ex = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 100
            uq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            vq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            flux = optimized_compute_flux(Ex, uq, vq, roi_mask)
            return features, flux
        
        numba_result = benchmark_function(
            numba_pipeline,
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"numba_full_N{batch_size}"
        )
        numba_result['batch_size'] = batch_size
        numba_result['version'] = 'numba'
        results.append(numba_result)
        logger.info(f"  Numba: {numba_result['mean_ms']:.2f} ± {numba_result['std_ms']:.2f} ms")
        
        numba_speedup = baseline_result['mean_ms'] / numba_result['mean_ms']
        logger.info(f"  Numba Speedup: {numba_speedup:.2f}x")
        
        # Fast pipeline (Numba + fast encoding, no scipy.stats)
        fast_encoding = FastEncoding(config.parcel_size)
        
        def fast_pipeline(genomes):
            features, heightmaps = numba_compute_features_batch(genomes, config.parcel_size, fast_encoding)
            terrain, buildings, landuse = optimized_construct_domain_grids(heightmaps, parcel_size_cells)
            # Simulate U-Net output
            Ex = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 100
            uq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            vq = np.random.rand(len(genomes), grid_h, grid_w).astype(np.float32) * 200 - 100
            flux = optimized_compute_flux(Ex, uq, vq, roi_mask)
            return features, flux
        
        fast_result = benchmark_function(
            fast_pipeline,
            (genomes,),
            config.num_warmup,
            config.num_iterations,
            f"fast_full_N{batch_size}"
        )
        fast_result['batch_size'] = batch_size
        fast_result['version'] = 'fast'
        results.append(fast_result)
        logger.info(f"  Fast (Numba + fast encoding): {fast_result['mean_ms']:.2f} ± {fast_result['std_ms']:.2f} ms")
        
        fast_speedup = baseline_result['mean_ms'] / fast_result['mean_ms']
        logger.info(f"  Fast Speedup: {fast_speedup:.2f}x")
    
    return results


# ============================================================================
# Deep Profiling
# ============================================================================

def profile_numba_pipeline(config: BenchmarkConfig, batch_size: int = 1024):
    """Deep profile the Numba-optimized pipeline to find remaining bottlenecks."""
    logger.info("=" * 80)
    logger.info("DEEP PROFILING: Numba Pipeline (Component Breakdown)")
    logger.info("=" * 80)
    
    genomes = np.random.randn(batch_size, 60).astype(np.float32)
    encoding = OptimizedEncoding(config.parcel_size)
    parcel_size_cells = config.parcel_size // 3
    grid_h, grid_w = 66, 94
    roi_mask = np.ones((grid_h, grid_w), dtype=bool)
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(3):
        heightmaps = encoding.express_batch(genomes)
        features, _ = numba_compute_features_batch(genomes, config.parcel_size, encoding)
        terrain, buildings, landuse = optimized_construct_domain_grids(heightmaps, parcel_size_cells)
    
    num_runs = 20
    
    # Component timing breakdown
    times = {
        'express_batch': [],
        'numba_features': [],
        'construct_domain': [],
        'total': []
    }
    
    logger.info(f"\nTiming {num_runs} runs with batch_size={batch_size}...")
    
    for i in range(num_runs):
        t_start = time.perf_counter()
        
        # 1. Express batch (genome → heightmap)
        t1 = time.perf_counter()
        heightmaps = encoding.express_batch(genomes)
        t2 = time.perf_counter()
        times['express_batch'].append((t2 - t1) * 1000)
        
        # 2. Numba features (pass heightmaps to avoid duplicate express_batch)
        t1 = time.perf_counter()
        features, _ = numba_compute_features_batch(genomes, config.parcel_size, encoding, heightmaps=heightmaps)
        t2 = time.perf_counter()
        times['numba_features'].append((t2 - t1) * 1000)
        
        # 3. Domain construction
        t1 = time.perf_counter()
        terrain, buildings, landuse = optimized_construct_domain_grids(heightmaps, parcel_size_cells)
        t2 = time.perf_counter()
        times['construct_domain'].append((t2 - t1) * 1000)
        
        t_end = time.perf_counter()
        times['total'].append((t_end - t_start) * 1000)
    
    # Print statistics
    logger.info(f"\n{'Component':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'% of Total':<12}")
    logger.info("-" * 80)
    
    total_mean = np.mean(times['total'])
    
    for component in ['express_batch', 'numba_features', 'construct_domain']:
        mean_time = np.mean(times[component])
        std_time = np.std(times[component])
        pct = (mean_time / total_mean) * 100
        logger.info(f"{component:<30} {mean_time:>10.2f}   {std_time:>10.2f}   {pct:>10.1f}%")
    
    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<30} {total_mean:>10.2f}   {np.std(times['total']):>10.2f}   {100.0:>10.1f}%")
    
    return times


def profile_express_batch_internals(config: BenchmarkConfig, batch_size: int = 1024):
    """Profile express_batch() to see where time is spent."""
    logger.info("\n" + "=" * 80)
    logger.info("DEEP PROFILING: express_batch() Internals")
    logger.info("=" * 80)
    
    genomes = np.random.randn(batch_size, 60).astype(np.float32)
    encoding = OptimizedEncoding(config.parcel_size)
    
    # Use cProfile to profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple times for better statistics
    for _ in range(10):
        heightmaps = encoding.express_batch(genomes)
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    logger.info("\nTop 30 functions by cumulative time:\n")
    logger.info(s.getvalue())
    
    return profiler


def profile_numba_features_internals(config: BenchmarkConfig, batch_size: int = 1024):
    """Profile Numba feature computation in detail."""
    logger.info("\n" + "=" * 80)
    logger.info("DEEP PROFILING: Numba Feature Computation Breakdown")
    logger.info("=" * 80)
    
    genomes = np.random.randn(batch_size, 60).astype(np.float32)
    encoding = OptimizedEncoding(config.parcel_size)
    
    # Warmup
    for _ in range(3):
        heightmaps = encoding.express_batch(genomes)
        for hm in heightmaps[:10]:  # Just a few for warmup
            _ = numba_calculate_features(hm.astype(np.float32), 3.0)
    
    num_runs = 20
    times_per_sample = []
    
    logger.info(f"\nTiming per-sample Numba feature computation ({num_runs} runs)...")
    
    heightmaps = encoding.express_batch(genomes)
    
    for _ in range(num_runs):
        t_start = time.perf_counter()
        for hm in heightmaps:
            _ = numba_calculate_features(hm.astype(np.float32), 3.0)
        t_end = time.perf_counter()
        times_per_sample.append((t_end - t_start) * 1000 / batch_size)
    
    mean_time = np.mean(times_per_sample)
    std_time = np.std(times_per_sample)
    
    logger.info(f"Per-sample Numba feature time: {mean_time:.4f} ± {std_time:.4f} ms")
    logger.info(f"Full batch ({batch_size} samples): {mean_time * batch_size:.2f} ms")
    
    # Compare to loop overhead
    logger.info("\nChecking loop overhead vs actual computation...")
    
    # Time just the loop
    t_start = time.perf_counter()
    for hm in heightmaps:
        pass
    t_end = time.perf_counter()
    loop_overhead = (t_end - t_start) * 1000
    
    logger.info(f"Loop overhead: {loop_overhead:.2f} ms ({loop_overhead / (mean_time * batch_size) * 100:.1f}% of total)")
    
    return times_per_sample


def run_deep_profiling(config: BenchmarkConfig):
    """Run comprehensive deep profiling."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING DEEP PROFILING")
    logger.info("=" * 80)
    
    # 1. Component breakdown
    component_times = profile_numba_pipeline(config, batch_size=1024)
    
    # 2. express_batch internals
    express_profiler = profile_express_batch_internals(config, batch_size=1024)
    
    # 3. Numba features internals
    numba_times = profile_numba_features_internals(config, batch_size=1024)
    
    return {
        'component_times': component_times,
        'express_profiler': express_profiler,
        'numba_times': numba_times
    }


# ============================================================================
# Results Summary
# ============================================================================

def print_summary(all_results: Dict[str, List[Dict]]):
    """Print a summary of all benchmark results."""
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)
    
    for benchmark_name, results in all_results.items():
        logger.info(f"\n{benchmark_name}:")
        
        # Group by batch size
        batch_sizes = sorted(set(r['batch_size'] for r in results))
        
        for bs in batch_sizes:
            bs_results = [r for r in results if r['batch_size'] == bs]
            baseline = next((r for r in bs_results if r.get('version') == 'baseline'), None)
            optimized = next((r for r in bs_results if r.get('version') == 'optimized'), None)
            numba = next((r for r in bs_results if r.get('version') == 'numba'), None)
            fast = next((r for r in bs_results if r.get('version') == 'fast'), None)
            
            if baseline and optimized:
                speedup = baseline['mean_ms'] / optimized['mean_ms']
                line = f"  N={bs:4d}: {baseline['mean_ms']:8.2f}ms -> {optimized['mean_ms']:8.2f}ms ({speedup:.2f}x)"
                
                if numba:
                    numba_speedup = baseline['mean_ms'] / numba['mean_ms']
                    line += f" -> {numba['mean_ms']:8.2f}ms ({numba_speedup:.2f}x Numba)"
                
                if fast:
                    fast_speedup = baseline['mean_ms'] / fast['mean_ms']
                    line += f" -> {fast['mean_ms']:8.2f}ms ({fast_speedup:.2f}x Fast)"
                
                logger.info(line)


def save_results(all_results: Dict[str, List[Dict]], output_dir: Path):
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Performance Benchmarking')
    parser.add_argument('--benchmark', type=str, default='all',
                       choices=['all', 'features', 'domain', 'inference', 'flux', 'full', 'profile'],
                       help='Which benchmark to run')
    parser.add_argument('--batch-sizes', type=int, nargs='+', 
                       default=[64, 128, 256, 512, 1024, 2048],
                       help='Batch sizes to test')
    parser.add_argument('--num-iterations', type=int, default=10,
                       help='Number of timing iterations')
    parser.add_argument('--parcel-size', type=int, default=60,
                       help='Parcel size in meters')
    parser.add_argument('--output-dir', type=str, 
                       default='results/exp8_performance_benchmark',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        num_iterations=args.num_iterations,
        parcel_size=args.parcel_size,
        device=args.device,
    )
    
    output_dir = Path(args.output_dir)
    all_results = {}
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT 8: PERFORMANCE BENCHMARKING")
    logger.info("=" * 80)
    logger.info(f"Batch sizes: {config.batch_sizes}")
    logger.info(f"Iterations: {config.num_iterations}")
    logger.info(f"Parcel size: {config.parcel_size}m")
    
    if args.benchmark == 'profile':
        # Deep profiling mode
        profiling_results = run_deep_profiling(config)
        logger.info("\nDeep profiling complete!")
        return
    
    if args.benchmark in ['all', 'features']:
        all_results['features'] = run_feature_benchmark(config)
    
    if args.benchmark in ['all', 'domain']:
        all_results['domain'] = run_domain_benchmark(config)
    
    if args.benchmark in ['all', 'flux']:
        all_results['flux'] = run_flux_benchmark(config)
    
    if args.benchmark in ['all', 'inference']:
        all_results['inference'] = run_inference_benchmark(config)
    
    if args.benchmark in ['all', 'full']:
        all_results['full_pipeline'] = run_full_pipeline_benchmark(config)
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    save_results(all_results, output_dir)
    
    logger.info("\nBenchmarking complete!")


if __name__ == '__main__':
    main()
