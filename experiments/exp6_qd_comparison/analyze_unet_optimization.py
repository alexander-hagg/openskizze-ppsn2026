#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
U-Net Optimization Analysis

Tests various optimization strategies for U-Net inference:
1. Current configuration (baseline)
2. FP16 mixed precision
3. torch.compile (PyTorch 2.0+)
4. Smaller model architecture
5. TensorRT (if available)

Run on HPC with GPU to get accurate benchmarks.
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unet import UNet, UNetConfig


def benchmark_model(model, x, num_warmup=5, num_runs=20, name="Model"):
    """Benchmark model inference time."""
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        'name': name,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'times_ms': times
    }


def main():
    parser = argparse.ArgumentParser(description="U-Net Optimization Analysis")
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for benchmarking')
    parser.add_argument('--num-warmup', type=int, default=5,
                       help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=20,
                       help='Number of benchmark runs')
    parser.add_argument('--output', type=str,
                       default='results/exp6_qd_comparison/unet_optimization_analysis.json',
                       help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("U-NET OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Runs: {args.num_runs}")
    
    results = {
        'config': {
            'batch_size': args.batch_size,
            'num_warmup': args.num_warmup,
            'num_runs': args.num_runs,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if device.type == 'cuda' else None,
        },
        'benchmarks': []
    }
    
    # Current configuration
    print("\n" + "=" * 80)
    print("1. CURRENT CONFIGURATION (BASELINE)")
    print("=" * 80)
    
    config = UNetConfig(
        input_channels=3,
        output_channels=6,
        base_channels=64,
        depth=4,
        dropout=0.1,
        input_height=66,
        input_width=94
    )
    
    model = UNet(config).to(device)
    model.eval()
    
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape: ({args.batch_size}, 3, 66, 94)")
    print(f"Padded to: ({args.batch_size}, 3, {config.padded_height}, {config.padded_width})")
    
    # Generate test input
    x = torch.randn(args.batch_size, 3, 66, 94, device=device)
    
    result = benchmark_model(model, x, args.num_warmup, args.num_runs, "FP32 Baseline")
    result['parameters'] = model.count_parameters()
    result['dtype'] = 'float32'
    results['benchmarks'].append(result)
    print(f"Inference: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
    
    baseline_time = result['mean_ms']
    
    # FP16 Mixed Precision
    print("\n" + "=" * 80)
    print("2. FP16 MIXED PRECISION")
    print("=" * 80)
    
    model_fp16 = model.half()
    x_fp16 = x.half()
    
    result = benchmark_model(model_fp16, x_fp16, args.num_warmup, args.num_runs, "FP16")
    result['parameters'] = model.count_parameters()
    result['dtype'] = 'float16'
    result['speedup'] = baseline_time / result['mean_ms']
    results['benchmarks'].append(result)
    print(f"Inference: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
    print(f"Speedup: {result['speedup']:.2f}x")
    
    # torch.compile (PyTorch 2.0+)
    print("\n" + "=" * 80)
    print("3. TORCH.COMPILE (PyTorch 2.0+)")
    print("=" * 80)
    
    try:
        # Reset to FP32
        model_compile = UNet(config).to(device)
        model_compile.eval()
        
        print("Compiling model (this may take a minute)...")
        model_compiled = torch.compile(model_compile, mode='reduce-overhead')
        
        # Warmup includes compilation
        print("Warmup (includes JIT compilation)...")
        with torch.no_grad():
            for i in range(args.num_warmup):
                _ = model_compiled(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                print(f"  Warmup {i+1}/{args.num_warmup}")
        
        result = benchmark_model(model_compiled, x, 0, args.num_runs, "torch.compile")
        result['parameters'] = model.count_parameters()
        result['dtype'] = 'float32'
        result['speedup'] = baseline_time / result['mean_ms']
        results['benchmarks'].append(result)
        print(f"Inference: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
        print(f"Speedup: {result['speedup']:.2f}x")
    except Exception as e:
        print(f"Error: {e}")
        results['benchmarks'].append({
            'name': 'torch.compile',
            'error': str(e)
        })
    
    # torch.compile + FP16
    print("\n" + "=" * 80)
    print("4. TORCH.COMPILE + FP16")
    print("=" * 80)
    
    try:
        model_compile_fp16 = UNet(config).to(device).half()
        model_compile_fp16.eval()
        
        print("Compiling FP16 model...")
        model_compiled_fp16 = torch.compile(model_compile_fp16, mode='reduce-overhead')
        
        print("Warmup...")
        with torch.no_grad():
            for i in range(args.num_warmup):
                _ = model_compiled_fp16(x_fp16)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                print(f"  Warmup {i+1}/{args.num_warmup}")
        
        result = benchmark_model(model_compiled_fp16, x_fp16, 0, args.num_runs, "torch.compile + FP16")
        result['parameters'] = model.count_parameters()
        result['dtype'] = 'float16'
        result['speedup'] = baseline_time / result['mean_ms']
        results['benchmarks'].append(result)
        print(f"Inference: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
        print(f"Speedup: {result['speedup']:.2f}x")
    except Exception as e:
        print(f"Error: {e}")
        results['benchmarks'].append({
            'name': 'torch.compile + FP16',
            'error': str(e)
        })
    
    # Smaller model (base_channels=32, depth=3)
    print("\n" + "=" * 80)
    print("5. SMALLER MODEL (base_channels=32, depth=3)")
    print("=" * 80)
    print("NOTE: Would require retraining to verify accuracy")
    
    config_small = UNetConfig(
        input_channels=3,
        output_channels=6,
        base_channels=32,
        depth=3,
        dropout=0.1,
        input_height=66,
        input_width=94
    )
    
    model_small = UNet(config_small).to(device)
    model_small.eval()
    
    print(f"Parameters: {model_small.count_parameters():,} ({model_small.count_parameters()/model.count_parameters()*100:.1f}% of original)")
    
    result = benchmark_model(model_small, x, args.num_warmup, args.num_runs, "Small Model (32ch, depth=3)")
    result['parameters'] = model_small.count_parameters()
    result['dtype'] = 'float32'
    result['speedup'] = baseline_time / result['mean_ms']
    result['requires_retraining'] = True
    results['benchmarks'].append(result)
    print(f"Inference: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
    print(f"Speedup: {result['speedup']:.2f}x")
    
    # Small model + FP16
    print("\n" + "=" * 80)
    print("6. SMALLER MODEL + FP16")
    print("=" * 80)
    
    model_small_fp16 = model_small.half()
    
    result = benchmark_model(model_small_fp16, x_fp16, args.num_warmup, args.num_runs, "Small Model + FP16")
    result['parameters'] = model_small.count_parameters()
    result['dtype'] = 'float16'
    result['speedup'] = baseline_time / result['mean_ms']
    result['requires_retraining'] = True
    results['benchmarks'].append(result)
    print(f"Inference: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
    print(f"Speedup: {result['speedup']:.2f}x")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Time (ms)':<15} {'Speedup':<10} {'Notes'}")
    print("-" * 80)
    
    for r in results['benchmarks']:
        if 'error' in r:
            print(f"{r['name']:<30} {'ERROR':<15} {'-':<10} {r['error'][:30]}")
        else:
            notes = ""
            if r.get('requires_retraining'):
                notes = "Requires retraining"
            print(f"{r['name']:<30} {r['mean_ms']:<15.1f} {r.get('speedup', 1.0):<10.2f}x {notes}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best options
    valid_results = [r for r in results['benchmarks'] if 'error' not in r and not r.get('requires_retraining')]
    if valid_results:
        best = min(valid_results, key=lambda x: x['mean_ms'])
        print(f"• Best option (no retraining): {best['name']}")
        print(f"  Time: {best['mean_ms']:.1f}ms, Speedup: {best.get('speedup', 1.0):.2f}x")
    
    retrain_results = [r for r in results['benchmarks'] if 'error' not in r and r.get('requires_retraining')]
    if retrain_results:
        best_retrain = min(retrain_results, key=lambda x: x['mean_ms'])
        print(f"• Best option (requires retraining): {best_retrain['name']}")
        print(f"  Time: {best_retrain['mean_ms']:.1f}ms, Speedup: {best_retrain.get('speedup', 1.0):.2f}x")
        print(f"  Parameters: {best_retrain['parameters']:,} vs {model.count_parameters():,}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
