# U-Net Throughput Benchmarking

This directory contains scripts to measure U-Net inference throughput for determining feasibility of 15-minute QD runs.

## Quick Start (HPC)

```bash
# Navigate to project root
cd ~/openskizze-klam21-optimization

# Submit benchmark job
sbatch hpc/exp5_unet/benchmark_throughput.sh

# Check job status
squeue -u $USER

# View results when complete
cat results/unet_experiment/sail_mse_seed42/throughput_benchmark.json
```

## What It Does

The benchmark:
1. Loads your trained U-Net model (`best_model.pth`)
2. Tests multiple batch sizes (1, 4, 8, 16, 32, 64, 128, 256)
3. Measures throughput (samples/second) for each
4. Identifies optimal batch size
5. **Calculates QD timing estimates** for 15-minute target

## Output

### Console Output
```
BENCHMARK SUMMARY
======================================================================
Batch Size   Throughput          Per-Sample      Per-Batch      
             (samples/sec)       (ms)            (ms)           
----------------------------------------------------------------------
1            45.2                22.13           22.13          
8            180.5               5.54            44.32          
16           285.7               3.50            56.00          
32           350.4               2.85            91.32          
64           420.8               2.38            152.08         
128          465.3               2.15            275.12         
----------------------------------------------------------------------
OPTIMAL: batch_size=128 → 465.3 samples/sec
======================================================================

QD OPTIMIZATION TIME ESTIMATES (for 15-minute target):
----------------------------------------------------------------------
Phase 1 (exploration): 75,000 evals @ 465/s = 161.3 sec (2.7 min)
Phase 2a (re-eval):    1,000 evals @ 465/s = 2.2 sec
Phase 2b (SAIL):      25,000 evals @ 465/s = 53.8 sec (0.9 min)
Phase 3 (validation):  2,000 evals @ 465/s = 4.3 sec
----------------------------------------------------------------------
TOTAL TIME: 221.5 sec = 3.7 min
✓ Target achieved! (678 sec under budget)
======================================================================
```

### JSON Output
Saved to `results/unet_experiment/sail_mse_seed42/throughput_benchmark.json`:

```json
{
  "device": "cuda",
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "optimal_batch_size": 128,
  "optimal_throughput": 465.3,
  "qd_time_estimates": {
    "phase1_sec": 161.3,
    "phase2a_sec": 2.2,
    "phase2b_sec": 53.8,
    "phase3_sec": 4.3,
    "total_sec": 221.5,
    "total_min": 3.7,
    "target_met": true
  },
  "results": [...]
}
```

## Manual Run (Interactive)

If you want to run interactively on a GPU node:

```bash
# Request GPU node
srun --partition=gpu --gres=gpu:1 --mem=32G --pty bash

# Activate environment
conda activate openskizze_klam_qd

# Run benchmark
python experiments/exp5_unet/benchmark_unet_throughput.py \
    --model-path results/unet_experiment/sail_mse_seed42/best_model.pth \
    --norm-path results/unet_experiment/sail_mse_seed42/normalization.json \
    --device cuda \
    --batch-sizes 1 8 16 32 64 128 256 \
    --num-warmup 100 \
    --num-samples 2000
```

## Interpreting Results

### Throughput Metrics
- **samples/sec**: How many layouts can be evaluated per second
- **Per-sample time**: Latency for single evaluation (ms)
- **Optimal batch size**: Best balance of throughput vs memory

### QD Time Estimates
Based on dual U-Net strategy (see `DUAL_UNET_STRATEGY.md`):

- **Phase 1**: Fast exploration with small U-Net (75,000 evaluations)
- **Phase 2a**: Re-evaluate archive with large U-Net (1,000 elites)
- **Phase 2b**: Continue SAIL with large U-Net (25,000 evaluations)
- **Phase 3**: Final validation (2,000 elites)

**Target**: Total < 15 minutes (900 seconds)

### Decision Points

**If target met (< 900 sec)**:
✓ Proceed with implementation of dual U-Net strategy

**If over budget**:
- Reduce Phase 1 generations (e.g., 2000 instead of 3000)
- Use 5D archive instead of 8D (3,125 vs 390,625 cells)
- Accept longer runtime (20-25 min)
- Use flood-fill proxy for Phase 1 instead

## Troubleshooting

### Out of Memory (OOM)
If benchmark crashes with OOM at large batch sizes:
- This is expected and handled gracefully
- Script will report max feasible batch size
- Optimal is usually before OOM limit anyway

### Slow Throughput
If throughput is unexpectedly low:
- Check GPU utilization: `nvidia-smi dmon`
- Verify CUDA is being used (not CPU fallback)
- Check for other jobs on same GPU

### Model Not Found
Ensure model path points to trained model:
```bash
ls -lh results/unet_experiment/sail_mse_seed42/
# Should contain:
#   best_model.pth
#   normalization.json
#   history.json
#   results.json
```

## Next Steps

After benchmark completes:

1. **Check if 15-min target is feasible**
   - If yes: Proceed with dual U-Net implementation
   - If no: Adjust strategy (see `DUAL_UNET_STRATEGY.md`)

2. **Report results**
   - Share `throughput_benchmark.json` 
   - Confirm optimal batch size for production

3. **Validate ranking correlation**
   - Run correlation test (small U-Net vs KLAM)
   - Target: Spearman ρ > 0.80

4. **Begin implementation**
   - Create `evaluation_unet.py`
   - Implement two-phase QD runner
   - See `DUAL_UNET_STRATEGY.md` for full plan
