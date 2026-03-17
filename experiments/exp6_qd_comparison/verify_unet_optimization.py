#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Verify U-Net Optimization Speedup

Quick benchmark to confirm torch.compile + FP16 optimization is working
in the actual evaluator pipeline.
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp6_qd_comparison.run_mapelites_offline import UNetEvaluator, ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Verify U-Net optimization speedup")
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--num-warmup', type=int, default=5,
                       help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=20,
                       help='Number of benchmark runs')
    parser.add_argument('--unet-model', type=str,
                       default='results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth',
                       help='Path to U-Net model')
    parser.add_argument('--parcel-size', type=int, default=120,
                       choices=[60, 120, 240],
                       help='Parcel size in meters')
    parser.add_argument('--output', type=str,
                       default='results/exp6_qd_comparison/verify_optimization.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VERIFY U-NET OPTIMIZATION SPEEDUP")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Batch size: {args.batch_size}")
    
    results = {}
    
    # Test unoptimized evaluator
    print("\n" + "=" * 80)
    print("1. UNOPTIMIZED (FP32, no compile)")
    print("=" * 80)
    
    evaluator_unopt = UNetEvaluator(
        Path(args.unet_model), 
        device, 
        use_compile=False, 
        use_fp16=False
    )
    
    genomes = np.random.randn(args.batch_size, 60).astype(np.float32)
    parcel_sizes = np.full(args.batch_size, args.parcel_size, dtype=np.float32)
    
    # Warmup
    print("Warmup...")
    for i in range(args.num_warmup):
        _ = evaluator_unopt.evaluate(genomes, parcel_sizes, None)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for i in range(args.num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = evaluator_unopt.evaluate(genomes, parcel_sizes, None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    print(f"Inference: {np.mean(times):.1f} ± {np.std(times):.1f} ms")
    results['unoptimized'] = {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times))}
    baseline = np.mean(times)
    
    # Clean up
    del evaluator_unopt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Test optimized evaluator
    print("\n" + "=" * 80)
    print("2. OPTIMIZED (torch.compile + FP16)")
    print("=" * 80)
    
    evaluator_opt = UNetEvaluator(
        Path(args.unet_model), 
        device, 
        use_compile=True, 
        use_fp16=True
    )
    
    # Extended warmup for compilation
    print("Warmup (includes JIT compilation)...")
    for i in range(args.num_warmup + 3):
        _ = evaluator_opt.evaluate(genomes, parcel_sizes, None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if i < 5:
            print(f"  Warmup {i+1}/{args.num_warmup + 3}")
    
    # Benchmark
    times = []
    for i in range(args.num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = evaluator_opt.evaluate(genomes, parcel_sizes, None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    print(f"Inference: {np.mean(times):.1f} ± {np.std(times):.1f} ms")
    results['optimized'] = {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times))}
    optimized = np.mean(times)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    speedup = baseline / optimized
    print(f"  Unoptimized: {baseline:.1f} ms")
    print(f"  Optimized:   {optimized:.1f} ms")
    print(f"  Speedup:     {speedup:.2f}x")
    
    # Time estimates for 10K generations
    time_10k_unopt = baseline * 10000 / 1000 / 60  # minutes
    time_10k_opt = optimized * 10000 / 1000 / 60  # minutes
    print(f"\n  For 10,000 generations:")
    print(f"    Unoptimized: {time_10k_unopt:.1f} minutes ({time_10k_unopt/60:.1f} hours)")
    print(f"    Optimized:   {time_10k_opt:.1f} minutes ({time_10k_opt/60:.1f} hours)")
    print(f"    Time saved:  {time_10k_unopt - time_10k_opt:.1f} minutes")
    
    results['speedup'] = float(speedup)
    results['time_10k_unopt_min'] = float(time_10k_unopt)
    results['time_10k_opt_min'] = float(time_10k_opt)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
