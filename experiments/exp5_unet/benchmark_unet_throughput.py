#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Benchmark U-Net inference throughput on GPU.

Measures samples/second for different batch sizes to determine
optimal throughput for real-time QD optimization.

Usage:
    # On HPC with trained model
    python experiments/exp5_unet/benchmark_unet_throughput.py \
        --model-path results/unet_experiment/sail_mse_seed42/best_model.pth \
        --norm-path results/unet_experiment/sail_mse_seed42/normalization.json \
        --device cuda \
        --batch-sizes 1 8 16 32 64 128 256 \
        --num-warmup 50 \
        --num-samples 1000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.unet import UNet, UNetConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path, device: str) -> Tuple[UNet, UNetConfig]:
    """
    Load trained U-Net model from checkpoint.
    
    Args:
        model_path: Path to best_model.pth
        device: 'cuda' or 'cpu'
    
    Returns:
        Tuple of (model, config)
    """
    logger.info(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config
    config_dict = checkpoint['config']
    config = UNetConfig(**config_dict)
    
    # Create and load model
    model = UNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Input size: {config.input_height} × {config.input_width}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    
    return model, config


def load_normalization_stats(norm_path: Path) -> Dict:
    """Load normalization statistics from training."""
    logger.info(f"Loading normalization stats from {norm_path}")
    
    with open(norm_path, 'r') as f:
        stats = json.load(f)
    
    return stats


def generate_random_inputs(
    batch_size: int,
    config: UNetConfig,
    norm_stats: Dict,
    device: str
) -> torch.Tensor:
    """
    Generate random normalized inputs matching training distribution.
    
    Args:
        batch_size: Number of samples
        config: Model config with input dimensions
        norm_stats: Normalization statistics from training
        device: Target device
    
    Returns:
        Input tensor (B, 3, H, W)
    """
    H, W = config.input_height, config.input_width
    
    # Generate random inputs (terrain, buildings, landuse)
    terrain = np.random.randn(batch_size, H, W).astype(np.float32)
    buildings = np.random.randn(batch_size, H, W).astype(np.float32)
    landuse = np.random.randn(batch_size, H, W).astype(np.float32)
    
    # Normalize using training stats (nested structure: input -> variable -> mean/std)
    terrain = (terrain - norm_stats['input']['terrain']['mean']) / norm_stats['input']['terrain']['std']
    buildings = (buildings - norm_stats['input']['buildings']['mean']) / norm_stats['input']['buildings']['std']
    landuse = (landuse - norm_stats['input']['landuse']['mean']) / norm_stats['input']['landuse']['std']
    
    # Stack and convert to tensor
    inputs = np.stack([terrain, buildings, landuse], axis=1)  # (B, 3, H, W)
    inputs_tensor = torch.from_numpy(inputs).to(device)
    
    return inputs_tensor


def benchmark_throughput(
    model: UNet,
    config: UNetConfig,
    norm_stats: Dict,
    device: str,
    batch_size: int,
    num_warmup: int = 50,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """
    Benchmark inference throughput for given batch size.
    
    Args:
        model: Trained U-Net model
        config: Model config
        norm_stats: Normalization statistics
        device: 'cuda' or 'cpu'
        batch_size: Batch size to test
        num_warmup: Number of warmup iterations
        num_samples: Total samples to process
    
    Returns:
        Dict with timing statistics
    """
    logger.info(f"\nBenchmarking batch_size={batch_size}")
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    actual_samples = num_batches * batch_size
    
    # Warmup
    logger.info(f"  Warmup: {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            inputs = generate_random_inputs(batch_size, config, norm_stats, device)
            _ = model(inputs)
            
            # Synchronize GPU
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"  Benchmark: {num_batches} batches ({actual_samples} samples)...")
    
    batch_times = []
    
    with torch.no_grad():
        for i in range(num_batches):
            # Generate inputs
            inputs = generate_random_inputs(batch_size, config, norm_stats, device)
            
            # Time forward pass
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            outputs = model(inputs)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            batch_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                logger.info(f"    Processed {(i+1)*batch_size}/{actual_samples} samples")
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    
    total_time = np.sum(batch_times)
    mean_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    min_batch_time = np.min(batch_times)
    max_batch_time = np.max(batch_times)
    
    # Throughput metrics
    samples_per_second = actual_samples / total_time
    mean_sample_time_ms = (mean_batch_time / batch_size) * 1000
    
    results = {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_samples': actual_samples,
        'total_time_sec': float(total_time),
        'mean_batch_time_sec': float(mean_batch_time),
        'std_batch_time_sec': float(std_batch_time),
        'min_batch_time_sec': float(min_batch_time),
        'max_batch_time_sec': float(max_batch_time),
        'samples_per_second': float(samples_per_second),
        'mean_sample_time_ms': float(mean_sample_time_ms),
    }
    
    logger.info(f"  Results:")
    logger.info(f"    Throughput: {samples_per_second:.1f} samples/sec")
    logger.info(f"    Per-sample: {mean_sample_time_ms:.2f} ms/sample")
    logger.info(f"    Per-batch:  {mean_batch_time*1000:.2f} ± {std_batch_time*1000:.2f} ms")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark U-Net inference throughput',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to best_model.pth checkpoint'
    )
    parser.add_argument(
        '--norm-path',
        type=Path,
        required=True,
        help='Path to normalization.json file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 8, 16, 32, 64, 128, 256],
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=50,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Total samples to process per batch size'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output JSON file for results (default: <model_dir>/throughput_benchmark.json)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and normalization stats
    model, config = load_model(args.model_path, args.device)
    norm_stats = load_normalization_stats(args.norm_path)
    
    # Run benchmarks
    all_results = []
    
    for batch_size in args.batch_sizes:
        try:
            results = benchmark_throughput(
                model=model,
                config=config,
                norm_stats=norm_stats,
                device=args.device,
                batch_size=batch_size,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
            )
            all_results.append(results)
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error(f"  OOM error at batch_size={batch_size}, skipping larger batches")
                break
            else:
                raise
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Batch Size':<12} {'Throughput':<20} {'Per-Sample':<15} {'Per-Batch':<15}")
    logger.info(f"{'':12} {'(samples/sec)':<20} {'(ms)':<15} {'(ms)':<15}")
    logger.info("-" * 70)
    
    for res in all_results:
        logger.info(
            f"{res['batch_size']:<12} "
            f"{res['samples_per_second']:<20.1f} "
            f"{res['mean_sample_time_ms']:<15.2f} "
            f"{res['mean_batch_time_sec']*1000:<15.2f}"
        )
    
    # Find optimal batch size
    best_result = max(all_results, key=lambda x: x['samples_per_second'])
    logger.info("-" * 70)
    logger.info(f"OPTIMAL: batch_size={best_result['batch_size']} "
                f"→ {best_result['samples_per_second']:.1f} samples/sec")
    logger.info("=" * 70)
    
    # Calculate QD timing estimate
    logger.info("\nQD OPTIMIZATION TIME ESTIMATES (for 15-minute target):")
    logger.info("-" * 70)
    
    throughput = best_result['samples_per_second']
    
    # Phase 1: 3000 gens × 25 evals = 75,000 samples
    phase1_time = 75000 / throughput
    logger.info(f"Phase 1 (exploration): 75,000 evals @ {throughput:.0f}/s = {phase1_time:.1f} sec ({phase1_time/60:.1f} min)")
    
    # Phase 2: 1000 archive re-eval + 1000 gens × 25 evals
    phase2a_time = 1000 / throughput
    phase2b_time = 25000 / throughput
    phase2_total = phase2a_time + phase2b_time
    logger.info(f"Phase 2a (re-eval):    1,000 evals @ {throughput:.0f}/s = {phase2a_time:.1f} sec")
    logger.info(f"Phase 2b (SAIL):      25,000 evals @ {throughput:.0f}/s = {phase2b_time:.1f} sec ({phase2b_time/60:.1f} min)")
    
    # Phase 3: 2000 archive validation
    phase3_time = 2000 / throughput
    logger.info(f"Phase 3 (validation):  2,000 evals @ {throughput:.0f}/s = {phase3_time:.1f} sec")
    
    total_time = phase1_time + phase2_total + phase3_time
    logger.info("-" * 70)
    logger.info(f"TOTAL TIME: {total_time:.1f} sec = {total_time/60:.1f} min")
    
    if total_time <= 900:  # 15 minutes
        logger.info(f"✓ Target achieved! ({900-total_time:.0f} sec under budget)")
    else:
        logger.info(f"✗ Over budget by {total_time-900:.0f} sec ({(total_time-900)/60:.1f} min)")
        logger.info(f"  Suggestions:")
        
        # Calculate required throughput
        required_throughput = 102000 / (900 - phase3_time - phase2a_time)  # Adjust for fixed phases
        logger.info(f"  - Need {required_throughput:.0f} samples/s throughput")
        
        # Calculate generations for current throughput
        max_phase1_time = 900 - phase2_total - phase3_time
        max_phase1_samples = max_phase1_time * throughput
        max_gens = int(max_phase1_samples / 25)
        logger.info(f"  - Or reduce Phase 1 to {max_gens} generations")
        
        # Suggest 5D archive
        logger.info(f"  - Or reduce archive to 5D (3,125 cells vs 390,625)")
    
    logger.info("=" * 70)
    
    # Save results
    if args.output is None:
        args.output = args.model_path.parent / 'throughput_benchmark.json'
    
    output_data = {
        'device': args.device,
        'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        'model_path': str(args.model_path),
        'model_parameters': model.count_parameters(),
        'input_shape': [config.input_height, config.input_width],
        'batch_sizes_tested': args.batch_sizes,
        'num_samples_per_batch': args.num_samples,
        'results': all_results,
        'optimal_batch_size': best_result['batch_size'],
        'optimal_throughput': best_result['samples_per_second'],
        'qd_time_estimates': {
            'phase1_sec': float(phase1_time),
            'phase2a_sec': float(phase2a_time),
            'phase2b_sec': float(phase2b_time),
            'phase3_sec': float(phase3_time),
            'total_sec': float(total_time),
            'total_min': float(total_time / 60),
            'target_met': total_time <= 900,
        }
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
