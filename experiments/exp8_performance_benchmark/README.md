# Experiment 8: Performance Benchmarking

## Purpose

Measure and validate performance bottlenecks in QD optimization to achieve "coffee break" (<15 min) optimization runs.

## Current Problem

| Model | Time/1K gens | Projected 10K | Solutions/gen |
|-------|--------------|---------------|---------------|
| SVGP  | 572.5s       | ~96 min       | 1024          |
| U-Net | 1486.1s      | ~248 min      | 1024          |
| Hybrid| 1588.5s      | ~265 min      | 1024          |

**Target**: <15 min for 10K generations → <90ms per generation

## Benchmarks

### 1. Feature Computation
Tests baseline vs optimized `compute_features_batch()`:
- Baseline: Creates new `ParametricEncoding` per solution
- Optimized: Reuses encoding, shares heightmaps

### 2. Domain Grid Construction  
Tests baseline vs optimized U-Net input preparation:
- Baseline: Sequential Python loops per solution
- Optimized: Vectorized NumPy operations on pre-computed heightmaps

### 3. Cold Air Flux Computation
Tests baseline vs optimized flux calculation:
- Baseline: Python for-loop with per-sample masking
- Optimized: Fully vectorized NumPy operations

### 4. Model Inference
Benchmarks SVGP and U-Net inference across batch sizes.

### 5. Full Pipeline
End-to-end benchmark of complete evaluation cycle.

## Usage

```bash
# Run all benchmarks
python experiments/exp8_performance_benchmark/run_benchmark.py --all

# Run specific benchmark
python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark features
python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark domain
python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark inference
python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark flux
python experiments/exp8_performance_benchmark/run_benchmark.py --benchmark full

# Custom batch sizes
python experiments/exp8_performance_benchmark/run_benchmark.py --all --batch-sizes 256 512 1024

# HPC
sbatch hpc/exp8_performance_benchmark/submit_benchmark.sh
```

## Expected Results

| Component | Baseline (1024 samples) | Optimized | Speedup |
|-----------|-------------------------|-----------|---------|
| Feature Computation | ~400ms | ~100ms | 4× |
| Domain Construction | ~800ms | ~50ms | 16× |
| Flux Computation | ~100ms | ~10ms | 10× |
| Full Pipeline | ~1300ms | ~160ms | 8× |

## Output

Results saved to `results/exp8_performance_benchmark/benchmark_results.json`

```json
{
  "features": [...],
  "domain": [...],
  "flux": [...],
  "inference": [...],
  "full_pipeline": [...]
}
```

## Next Steps

After benchmarking:
1. Apply optimizations to `run_mapelites_offline.py`
2. Re-run Experiment 6 with optimized code
3. Validate "coffee break" target achieved

See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for detailed analysis.
