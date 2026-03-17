#!/usr/bin/env python3
"""
Benchmark U-Net Pipeline Optimizations

Evaluates performance improvements from:
1. Baseline (current implementation)
2. GPU domain grid construction
3. GPU grids + Numba JIT FastEncoding
4. GPU grids + Numba JIT + Pinned memory

Run with: python experiments/exp8_performance_benchmark/benchmark_unet_optimizations.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import erf
from numba import jit
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# BASELINE IMPLEMENTATION (Current)
# ============================================================================

def fast_norm2unif(x, min_val=0.0, max_val=1.0, mu=None, sd=None):
    """Fast replacement for scipy.stats-based norm2unif using erf."""
    if mu is None:
        mu = np.mean(x)
    if sd is None:
        sd = np.std(x)
    
    if sd < 1e-10:
        return np.full_like(x, (min_val + max_val) / 2.0)
    
    # Standardize
    z = (x - mu) / sd
    
    # CDF: Φ(z) = 0.5 * (1 + erf(z / √2))
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # Map to [min_val, max_val]
    return min_val + (max_val - min_val) * cdf


class BaselineFastEncoding:
    """Baseline FastEncoding (matches production implementation)."""
    
    def __init__(self, parcel_size: int):
        self.parcel_size = parcel_size
        self.length_design = parcel_size // 3  # e.g., 60m / 3m = 20 cells
        self.max_num_floors = 5
    
    def express_batch(self, genomes: np.ndarray) -> np.ndarray:
        """Express batch of genomes to heightmaps.
        
        IMPORTANT: Apply fast_norm2unif to ALL genomes at once (vectorized).
        This computes mean/std across the entire batch, giving proper 0-1 distribution.
        Applying per-element always returns 0.5 (single-element mean=value, std=0).
        """
        batch_size = len(genomes)
        heightmaps = np.zeros((batch_size, self.length_design, self.length_design), dtype=np.float64)
        
        # Vectorized conversion: apply to all genomes at once
        genomes_uniform = np.clip(fast_norm2unif(genomes), 0, 1)
        
        for i in range(batch_size):
            genome = genomes_uniform[i]
            phenotype = np.zeros((self.length_design, self.length_design), dtype=np.float64)
            
            # Process 10 buildings
            for b in range(10):
                idx = b * 6
                width_unif = genome[idx]
                length_unif = genome[idx + 1]
                height_unif = genome[idx + 2]
                x_unif = genome[idx + 3]
                y_unif = genome[idx + 4]
                active_gene = genome[idx + 5]
                
                # Check if building is active (threshold at 0.5 in uniform space)
                if active_gene > 0.5:
                    # Scale to design space
                    width = max(1, int(width_unif * self.length_design))
                    length = max(1, int(length_unif * self.length_design))
                    num_floors = max(1, int(height_unif * self.max_num_floors))
                    
                    # Center-based positioning
                    x_origin = x_unif * self.length_design
                    y_origin = y_unif * self.length_design
                    
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


def baseline_construct_domain_grids(
    heightmaps: np.ndarray,
    parcel_size_cells: int,
    grid_h: int = 66,
    grid_w: int = 94
) -> tuple:
    """Baseline domain grid construction (CPU, NumPy)."""
    batch_size = len(heightmaps)
    
    # Pre-allocate arrays
    terrain = np.zeros((batch_size, grid_h, grid_w), dtype=np.float32)
    buildings = np.zeros((batch_size, grid_h, grid_w), dtype=np.float32)
    landuse = np.full((batch_size, grid_h, grid_w), 7, dtype=np.float32)
    
    # Calculate offsets
    offset_h = (grid_h - parcel_size_cells) // 2
    offset_w = (grid_w - parcel_size_cells) // 2
    
    # Vectorized placement
    buildings[:, offset_h:offset_h+parcel_size_cells, 
              offset_w:offset_w+parcel_size_cells] = heightmaps * 3.0
    landuse[:, offset_h:offset_h+parcel_size_cells,
            offset_w:offset_w+parcel_size_cells] = np.where(heightmaps > 0, 2, 7)
    
    return terrain, buildings, landuse


# ============================================================================
# OPTIMIZATION 1: GPU Domain Grid Construction
# ============================================================================

def gpu_construct_domain_grids(
    heightmaps_torch: torch.Tensor,
    parcel_size_cells: int,
    grid_h: int = 66,
    grid_w: int = 94
) -> tuple:
    """GPU domain grid construction (PyTorch)."""
    batch_size = heightmaps_torch.shape[0]
    device = heightmaps_torch.device
    
    # Pre-allocate on GPU
    terrain = torch.zeros((batch_size, grid_h, grid_w), device=device, dtype=torch.float32)
    buildings = torch.zeros((batch_size, grid_h, grid_w), device=device, dtype=torch.float32)
    landuse = torch.full((batch_size, grid_h, grid_w), 7, device=device, dtype=torch.float32)
    
    # Calculate offsets
    offset_h = (grid_h - parcel_size_cells) // 2
    offset_w = (grid_w - parcel_size_cells) // 2
    
    # Vectorized GPU placement
    buildings[:, offset_h:offset_h+parcel_size_cells, 
              offset_w:offset_w+parcel_size_cells] = heightmaps_torch * 3.0
    landuse[:, offset_h:offset_h+parcel_size_cells,
            offset_w:offset_w+parcel_size_cells] = torch.where(heightmaps_torch > 0, 2, 7)
    
    return terrain, buildings, landuse


# ============================================================================
# OPTIMIZATION 2: Numba JIT FastEncoding
# ============================================================================

@jit(nopython=True, cache=True)
def numba_erf(x):
    """Numba-compatible erf approximation."""
    # Abramowitz and Stegun approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y


@jit(nopython=True, cache=True)
def numba_fast_norm2unif(x, mu, sd, min_val=0.0, max_val=1.0):
    """JIT-compiled norm2unif."""
    if sd < 1e-10:
        return (min_val + max_val) / 2.0
    
    z = (x - mu) / sd
    cdf = 0.5 * (1.0 + numba_erf(z / np.sqrt(2.0)))
    
    return min_val + (max_val - min_val) * cdf


@jit(nopython=True, cache=True)
def numba_express_batch(genomes_uniform: np.ndarray, length_design: int, max_floors: int) -> np.ndarray:
    """JIT-compiled express_batch.
    
    NOTE: Expects genomes_uniform to already be in [0,1] range (pre-converted).
    The fast_norm2unif conversion is done outside Numba for vectorization.
    """
    batch_size = genomes_uniform.shape[0]
    heightmaps = np.zeros((batch_size, length_design, length_design), dtype=np.float32)
    
    for i in range(batch_size):
        genome = genomes_uniform[i]
        phenotype = np.zeros((length_design, length_design), dtype=np.float32)
        
        for b in range(10):
            idx = b * 6
            width_unif = genome[idx]
            length_unif = genome[idx + 1]
            height_unif = genome[idx + 2]
            x_unif = genome[idx + 3]
            y_unif = genome[idx + 4]
            active_unif = genome[idx + 5]
            
            # Active if uniform value > 0.5
            if active_unif > 0.5:
                # Scale to design space
                width = max(1, int(width_unif * length_design))
                length_val = max(1, int(length_unif * length_design))
                num_floors = max(1, int(height_unif * max_floors))
                
                # Center-based positioning
                x_origin = x_unif * length_design
                y_origin = y_unif * length_design
                
                # Calculate bounds
                x_start = max(0, int(x_origin - 0.5 * width))
                x_end = min(length_design, int(x_origin + 0.5 * width))
                y_start = max(0, int(y_origin - 0.5 * length_val))
                y_end = min(length_design, int(y_origin + 0.5 * length_val))
                
                # Place building
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        phenotype[y, x] = max(phenotype[y, x], num_floors)
        
        heightmaps[i] = phenotype
    
    return heightmaps


class NumbaFastEncoding:
    """FastEncoding with Numba JIT optimization."""
    
    def __init__(self, parcel_size: int):
        self.parcel_size = parcel_size
        self.length_design = parcel_size // 3
        self.max_num_floors = 5
    
    def express_batch(self, genomes: np.ndarray) -> np.ndarray:
        # Vectorized conversion to uniform [0,1] (done outside Numba)
        genomes_uniform = np.clip(fast_norm2unif(genomes), 0, 1).astype(np.float32)
        return numba_express_batch(genomes_uniform, self.length_design, self.max_num_floors)


# ============================================================================
# Benchmark Functions
# ============================================================================

def generate_test_genomes(batch_size: int, seed: int = 42) -> np.ndarray:
    """Generate random genomes for testing."""
    np.random.seed(seed)
    return np.random.randn(batch_size, 60).astype(np.float32)


def benchmark_baseline(genomes: np.ndarray, device: torch.device, num_iterations: int = 10) -> dict:
    """Benchmark baseline implementation."""
    print("\n" + "="*80)
    print("BENCHMARK 1: BASELINE (Current Implementation)")
    print("="*80)
    
    encoding = BaselineFastEncoding(60)
    parcel_cells = 20
    
    times = []
    checksum = 0.0
    for i in range(num_iterations):
        start = time.time()
        
        # Step 1: Express genomes to heightmaps (CPU)
        heightmaps = encoding.express_batch(genomes)
        
        # Step 2: Construct domain grids (CPU)
        terrain, buildings, landuse = baseline_construct_domain_grids(heightmaps, parcel_cells)
        
        # Step 3: Transfer to GPU
        terrain_gpu = torch.tensor(terrain, device=device)
        buildings_gpu = torch.tensor(buildings, device=device)
        landuse_gpu = torch.tensor(landuse, device=device)
        
        # Force computation by accessing specific elements (prevents caching/optimization)
        checksum += terrain_gpu[0, 0, 0].item()
        checksum += buildings_gpu[0, 33, 47].item()  # Center of parcel
        checksum += landuse_gpu[512, 40, 60].item()  # Different sample
        checksum += torch.sum(terrain_gpu).item()
        checksum += torch.sum(buildings_gpu).item()
        checksum += torch.sum(landuse_gpu).item()
        
        # Ensure GPU sync
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if i == 0:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (warmup, checksum: {checksum:.2f})")
        elif i == num_iterations - 1:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (final checksum: {checksum:.2f})")
    
    # Remove warmup
    times = times[1:]
    
    result = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'checksum': checksum,
    }
    
    print(f"\n  Results (excluding warmup):")
    print(f"    Mean: {result['mean']:.2f} ± {result['std']:.2f} ms")
    print(f"    Min:  {result['min']:.2f} ms")
    print(f"    Max:  {result['max']:.2f} ms")
    
    return result


def benchmark_gpu_grids(genomes: np.ndarray, device: torch.device, num_iterations: int = 10) -> dict:
    """Benchmark with GPU domain grid construction."""
    print("\n" + "="*80)
    print("BENCHMARK 2: GPU Domain Grid Construction")
    print("="*80)
    
    encoding = BaselineFastEncoding(60)
    parcel_cells = 20
    
    times = []
    checksum = 0.0
    for i in range(num_iterations):
        start = time.time()
        
        # Step 1: Express genomes to heightmaps (CPU)
        heightmaps = encoding.express_batch(genomes)
        
        # Step 2: Transfer heightmaps to GPU (small transfer)
        heightmaps_gpu = torch.tensor(heightmaps, device=device, dtype=torch.float32)
        
        # Step 3: Construct domain grids on GPU
        terrain, buildings, landuse = gpu_construct_domain_grids(heightmaps_gpu, parcel_cells)
        
        # Force computation by accessing specific elements (prevents caching/optimization)
        checksum += terrain[0, 0, 0].item()
        checksum += buildings[0, 33, 47].item()  # Center of parcel
        checksum += landuse[512, 40, 60].item()  # Different sample
        checksum += torch.sum(terrain).item()
        checksum += torch.sum(buildings).item()
        checksum += torch.sum(landuse).item()
        
        # Ensure GPU sync
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if i == 0:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (warmup, checksum: {checksum:.2f})")
        elif i == num_iterations - 1:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (final checksum: {checksum:.2f})")
    
    times = times[1:]
    
    result = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'checksum': checksum,
    }
    
    print(f"\n  Results (excluding warmup):")
    print(f"    Mean: {result['mean']:.2f} ± {result['std']:.2f} ms")
    print(f"    Min:  {result['min']:.2f} ms")
    print(f"    Max:  {result['max']:.2f} ms")
    print(f"    Speedup vs baseline: {benchmark_baseline.result['mean'] / result['mean']:.2f}×")
    
    return result


def benchmark_gpu_grids_numba(genomes: np.ndarray, device: torch.device, num_iterations: int = 10) -> dict:
    """Benchmark with GPU grids + Numba JIT."""
    print("\n" + "="*80)
    print("BENCHMARK 3: GPU Grids + Numba JIT FastEncoding")
    print("="*80)
    
    encoding = NumbaFastEncoding(60)
    parcel_cells = 20
    
    # Warmup Numba
    _ = encoding.express_batch(genomes[:10])
    
    times = []
    checksum = 0.0
    for i in range(num_iterations):
        start = time.time()
        
        # Step 1: Express genomes to heightmaps (CPU, Numba JIT)
        heightmaps = encoding.express_batch(genomes)
        
        # Step 2: Transfer heightmaps to GPU
        heightmaps_gpu = torch.tensor(heightmaps, device=device, dtype=torch.float32)
        
        # Step 3: Construct domain grids on GPU
        terrain, buildings, landuse = gpu_construct_domain_grids(heightmaps_gpu, parcel_cells)
        
        # Force computation by accessing specific elements (prevents caching/optimization)
        checksum += terrain[0, 0, 0].item()
        checksum += buildings[0, 33, 47].item()  # Center of parcel
        checksum += landuse[512, 40, 60].item()  # Different sample
        checksum += torch.sum(terrain).item()
        checksum += torch.sum(buildings).item()
        checksum += torch.sum(landuse).item()
        
        # Ensure GPU sync
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if i == 0:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (warmup, checksum: {checksum:.2f})")
        elif i == num_iterations - 1:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (final checksum: {checksum:.2f})")
    
    times = times[1:]
    
    result = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'checksum': checksum,
    }
    
    print(f"\n  Results (excluding warmup):")
    print(f"    Mean: {result['mean']:.2f} ± {result['std']:.2f} ms")
    print(f"    Min:  {result['min']:.2f} ms")
    print(f"    Max:  {result['max']:.2f} ms")
    print(f"    Speedup vs baseline: {benchmark_baseline.result['mean'] / result['mean']:.2f}×")
    print(f"    Speedup vs GPU grids: {benchmark_gpu_grids.result['mean'] / result['mean']:.2f}×")
    
    return result


def benchmark_gpu_grids_numba_pinned(genomes: np.ndarray, device: torch.device, num_iterations: int = 10) -> dict:
    """Benchmark with GPU grids + Numba + Pinned memory."""
    print("\n" + "="*80)
    print("BENCHMARK 4: GPU Grids + Numba JIT + Pinned Memory")
    print("="*80)
    
    encoding = NumbaFastEncoding(60)
    parcel_cells = 20
    
    # Warmup Numba
    _ = encoding.express_batch(genomes[:10])
    
    times = []
    checksum = 0.0
    for i in range(num_iterations):
        start = time.time()
        
        # Step 1: Express genomes to heightmaps (CPU, Numba JIT)
        heightmaps = encoding.express_batch(genomes)
        
        # Step 2: Transfer heightmaps to GPU with pinned memory
        if device.type == 'cuda':
            # Create tensor, then pin it
            heightmaps_torch = torch.from_numpy(heightmaps).float()
            heightmaps_torch = heightmaps_torch.pin_memory()
            heightmaps_gpu = heightmaps_torch.to(device, non_blocking=True)
        else:
            heightmaps_gpu = torch.tensor(heightmaps, device=device, dtype=torch.float32)
        
        # Step 3: Construct domain grids on GPU
        terrain, buildings, landuse = gpu_construct_domain_grids(heightmaps_gpu, parcel_cells)
        
        # Force computation by accessing specific elements (prevents caching/optimization)
        checksum += terrain[0, 0, 0].item()
        checksum += buildings[0, 33, 47].item()  # Center of parcel
        checksum += landuse[512, 40, 60].item()  # Different sample
        checksum += torch.sum(terrain).item()
        checksum += torch.sum(buildings).item()
        checksum += torch.sum(landuse).item()
        
        # Ensure GPU sync
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if i == 0:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (warmup, checksum: {checksum:.2f})")
        elif i == num_iterations - 1:
            print(f"  Iteration {i+1}: {elapsed:.2f} ms (final checksum: {checksum:.2f})")
    
    times = times[1:]
    
    result = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'checksum': checksum,
    }
    
    print(f"\n  Results (excluding warmup):")
    print(f"    Mean: {result['mean']:.2f} ± {result['std']:.2f} ms")
    print(f"    Min:  {result['min']:.2f} ms")
    print(f"    Max:  {result['max']:.2f} ms")
    print(f"    Speedup vs baseline: {benchmark_baseline.result['mean'] / result['mean']:.2f}×")
    print(f"    Speedup vs GPU grids: {benchmark_gpu_grids.result['mean'] / result['mean']:.2f}×")
    print(f"    Speedup vs GPU+Numba: {benchmark_gpu_grids_numba.result['mean'] / result['mean']:.2f}×")
    
    return result


def validate_outputs(genomes: np.ndarray, device: torch.device) -> dict:
    """Validate that all methods produce identical outputs."""
    print("\n" + "="*80)
    print("VALIDATION: Checking Output Consistency")
    print("="*80)
    
    # Use small batch for validation
    test_genomes = genomes[:10]
    parcel_cells = 20
    
    # Method 1: Baseline
    encoding_baseline = BaselineFastEncoding(60)
    heightmaps_baseline = encoding_baseline.express_batch(test_genomes)
    terrain_base, buildings_base, landuse_base = baseline_construct_domain_grids(heightmaps_baseline, parcel_cells)
    
    # Method 2: GPU grids
    heightmaps_gpu = torch.tensor(heightmaps_baseline, device=device, dtype=torch.float32)
    terrain_gpu, buildings_gpu, landuse_gpu = gpu_construct_domain_grids(heightmaps_gpu, parcel_cells)
    terrain_gpu_np = terrain_gpu.cpu().numpy()
    buildings_gpu_np = buildings_gpu.cpu().numpy()
    landuse_gpu_np = landuse_gpu.cpu().numpy()
    
    # Method 3: Numba JIT
    encoding_numba = NumbaFastEncoding(60)
    heightmaps_numba = encoding_numba.express_batch(test_genomes)
    heightmaps_numba_gpu = torch.tensor(heightmaps_numba, device=device, dtype=torch.float32)
    terrain_numba, buildings_numba, landuse_numba = gpu_construct_domain_grids(heightmaps_numba_gpu, parcel_cells)
    terrain_numba_np = terrain_numba.cpu().numpy()
    buildings_numba_np = buildings_numba.cpu().numpy()
    landuse_numba_np = landuse_numba.cpu().numpy()
    
    # Method 4: Pinned memory (same as Method 3, just different transfer method)
    if device.type == 'cuda':
        heightmaps_pinned_torch = torch.from_numpy(heightmaps_numba).float()
        heightmaps_pinned_torch = heightmaps_pinned_torch.pin_memory()
        heightmaps_pinned_gpu = heightmaps_pinned_torch.to(device, non_blocking=True)
    else:
        heightmaps_pinned_gpu = torch.tensor(heightmaps_numba, device=device, dtype=torch.float32)
    terrain_pinned, buildings_pinned, landuse_pinned = gpu_construct_domain_grids(heightmaps_pinned_gpu, parcel_cells)
    terrain_pinned_np = terrain_pinned.cpu().numpy()
    buildings_pinned_np = buildings_pinned.cpu().numpy()
    landuse_pinned_np = landuse_pinned.cpu().numpy()
    
    # Compare outputs
    validation_results = {}
    
    # Heightmaps: Baseline vs Numba
    heightmap_diff = np.max(np.abs(heightmaps_baseline - heightmaps_numba))
    heightmap_match = np.allclose(heightmaps_baseline, heightmaps_numba, rtol=1e-5, atol=1e-5)
    validation_results['heightmaps_baseline_vs_numba'] = {
        'max_diff': float(heightmap_diff),
        'match': heightmap_match
    }
    print(f"\n  Heightmaps (Baseline vs Numba):")
    print(f"    Max difference: {heightmap_diff:.6e}")
    print(f"    Match (rtol=1e-5): {heightmap_match}")
    
    # Domain grids: Baseline vs GPU
    terrain_diff = np.max(np.abs(terrain_base - terrain_gpu_np))
    buildings_diff = np.max(np.abs(buildings_base - buildings_gpu_np))
    landuse_diff = np.max(np.abs(landuse_base - landuse_gpu_np))
    
    grids_match = (
        np.allclose(terrain_base, terrain_gpu_np, rtol=1e-5, atol=1e-5) and
        np.allclose(buildings_base, buildings_gpu_np, rtol=1e-5, atol=1e-5) and
        np.allclose(landuse_base, landuse_gpu_np, rtol=1e-5, atol=1e-5)
    )
    
    validation_results['grids_baseline_vs_gpu'] = {
        'terrain_max_diff': float(terrain_diff),
        'buildings_max_diff': float(buildings_diff),
        'landuse_max_diff': float(landuse_diff),
        'match': grids_match
    }
    
    print(f"\n  Domain Grids (Baseline vs GPU):")
    print(f"    Terrain max diff:   {terrain_diff:.6e}")
    print(f"    Buildings max diff: {buildings_diff:.6e}")
    print(f"    Landuse max diff:   {landuse_diff:.6e}")
    print(f"    Match (rtol=1e-5): {grids_match}")
    
    # Full pipeline: Baseline vs Numba+GPU
    terrain_diff_numba = np.max(np.abs(terrain_base - terrain_numba_np))
    buildings_diff_numba = np.max(np.abs(buildings_base - buildings_numba_np))
    landuse_diff_numba = np.max(np.abs(landuse_base - landuse_numba_np))
    
    pipeline_match = (
        np.allclose(terrain_base, terrain_numba_np, rtol=1e-4, atol=1e-4) and
        np.allclose(buildings_base, buildings_numba_np, rtol=1e-4, atol=1e-4) and
        np.allclose(landuse_base, landuse_numba_np, rtol=1e-4, atol=1e-4)
    )
    
    validation_results['pipeline_baseline_vs_numba'] = {
        'terrain_max_diff': float(terrain_diff_numba),
        'buildings_max_diff': float(buildings_diff_numba),
        'landuse_max_diff': float(landuse_diff_numba),
        'match': pipeline_match
    }
    
    print(f"\n  Full Pipeline (Baseline vs Numba+GPU):")
    print(f"    Terrain max diff:   {terrain_diff_numba:.6e}")
    print(f"    Buildings max diff: {buildings_diff_numba:.6e}")
    print(f"    Landuse max diff:   {landuse_diff_numba:.6e}")
    print(f"    Match (rtol=1e-4): {pipeline_match}")
    
    # Pinned vs Numba (should be identical - same computation, just different transfer)
    pinned_vs_numba_match = (
        np.allclose(terrain_pinned_np, terrain_numba_np, rtol=1e-6, atol=1e-6) and
        np.allclose(buildings_pinned_np, buildings_numba_np, rtol=1e-6, atol=1e-6) and
        np.allclose(landuse_pinned_np, landuse_numba_np, rtol=1e-6, atol=1e-6)
    )
    
    validation_results['pinned_vs_numba'] = {
        'match': pinned_vs_numba_match
    }
    
    print(f"\n  Pinned Memory vs Numba (should be identical):")
    print(f"    Match (rtol=1e-6): {pinned_vs_numba_match}")
    
    # Overall validation status
    all_valid = (
        validation_results['heightmaps_baseline_vs_numba']['match'] and
        validation_results['grids_baseline_vs_gpu']['match'] and
        validation_results['pipeline_baseline_vs_numba']['match'] and
        validation_results['pinned_vs_numba']['match']
    )
    
    validation_results['all_valid'] = all_valid
    
    print(f"\n  Overall Validation: {'✓ PASSED' if all_valid else '✗ FAILED'}")
    
    if not all_valid:
        print("\n  WARNING: Output validation failed! Results may not be comparable.")
    
    return validation_results


def main():
    """Run all benchmarks."""
    print("="*80)
    print("U-Net Pipeline Optimization Benchmark")
    print("="*80)
    
    # Configuration
    batch_size = 1024
    num_iterations = 20
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Generate test data
    genomes = generate_test_genomes(batch_size)
    print(f"  Test genomes: {genomes.shape}")
    
    # Validate output consistency
    validation_results = validate_outputs(genomes, device)
    
    # Run benchmarks
    results = {}
    
    results['baseline'] = benchmark_baseline(genomes, device, num_iterations)
    benchmark_baseline.result = results['baseline']  # Store for speedup calculations
    
    results['gpu_grids'] = benchmark_gpu_grids(genomes, device, num_iterations)
    benchmark_gpu_grids.result = results['gpu_grids']
    
    results['gpu_grids_numba'] = benchmark_gpu_grids_numba(genomes, device, num_iterations)
    benchmark_gpu_grids_numba.result = results['gpu_grids_numba']
    
    results['gpu_grids_numba_pinned'] = benchmark_gpu_grids_numba_pinned(genomes, device, num_iterations)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    baseline_time = results['baseline']['mean']
    baseline_checksum = results['baseline']['checksum']
    
    print(f"\n{'Configuration':<30} {'Time (ms)':<15} {'Speedup':<10} {'Checksum Match':<15}")
    print("-"*75)
    print(f"{'1. Baseline':<30} {results['baseline']['mean']:>6.2f} ± {results['baseline']['std']:>5.2f}   {'1.00×':<10} {'(reference)':<15}")
    
    for name, label in [
        ('gpu_grids', '2. + GPU grids'),
        ('gpu_grids_numba', '3. + Numba JIT'),
        ('gpu_grids_numba_pinned', '4. + Pinned memory')
    ]:
        checksum_diff = abs(results[name]['checksum'] - baseline_checksum)
        checksum_match = checksum_diff < 1.0  # Allow small floating point errors
        match_str = '✓' if checksum_match else f'✗ (Δ={checksum_diff:.2f})'
        print(f"{label:<30} {results[name]['mean']:>6.2f} ± {results[name]['std']:>5.2f}   {baseline_time/results[name]['mean']:>4.2f}×   {match_str:<15}")
    
    # Check if all checksums match
    all_match = all(
        abs(results[name]['checksum'] - baseline_checksum) < 1.0
        for name in ['gpu_grids', 'gpu_grids_numba', 'gpu_grids_numba_pinned']
    )
    
    print(f"\n  {'✓ All methods produce identical outputs' if all_match else '✗ WARNING: Output mismatch detected!'}")
    
    if not all_match:
        print("\n  Checksums:")
        for name in ['baseline', 'gpu_grids', 'gpu_grids_numba', 'gpu_grids_numba_pinned']:
            print(f"    {name}: {results[name]['checksum']:.2f}")
    
    # Projected QD times
    print("\n" + "="*80)
    print("PROJECTED QD OPTIMIZATION TIMES (10K generations)")
    print("="*80)
    
    print(f"\n{'Configuration':<30} {'Time/Gen (ms)':<15} {'10K Gens (min)':<15}")
    print("-"*60)
    
    for name, label in [
        ('baseline', '1. Baseline'),
        ('gpu_grids', '2. + GPU grids'),
        ('gpu_grids_numba', '3. + Numba JIT'),
        ('gpu_grids_numba_pinned', '4. + Pinned memory')
    ]:
        time_ms = results[name]['mean']
        time_10k_min = (time_ms * 10000) / 1000 / 60
        print(f"{label:<30} {time_ms:>6.2f}          {time_10k_min:>6.1f}")
    
    print("\n" + "="*80)
    print("Note: Add ~150ms for U-Net inference (not benchmarked here)")
    print("      SVGP baseline: ~75ms/gen = 12.5 min for 10K gens")
    print("="*80)
    
    # Save results
    output_dir = project_root / 'results' / 'exp8_performance_benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'unet_optimization_benchmark.json'
    
    save_data = {
        'config': {
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'device': str(device),
        },
        'validation': validation_results,
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
