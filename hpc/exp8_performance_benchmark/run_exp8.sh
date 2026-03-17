#!/bin/bash

# Experiment 8: Performance Benchmarking
# Master script for running benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs
mkdir -p results/exp8_performance_benchmark

# Print usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    submit      Submit benchmark job to HPC
    local       Run benchmarks locally
    status      Check job status
    results     Show benchmark results

Examples:
    # Submit to HPC
    bash hpc/exp8_performance_benchmark/run_exp8.sh submit
    
    # Run locally (GPU)
    bash hpc/exp8_performance_benchmark/run_exp8.sh local
    
    # Run locally (CPU only)
    bash hpc/exp8_performance_benchmark/run_exp8.sh local --cpu
    
    # Check status
    bash hpc/exp8_performance_benchmark/run_exp8.sh status
    
    # View results
    bash hpc/exp8_performance_benchmark/run_exp8.sh results
EOF
}

# Parse arguments
COMMAND=${1:-help}
shift || true

case $COMMAND in
    submit)
        echo "Submitting benchmark job to HPC..."
        sbatch "$SCRIPT_DIR/submit_benchmark.sh"
        echo "Job submitted! Check status with: $0 status"
        ;;
    
    local)
        echo "Running benchmarks locally..."
        DEVICE="cuda"
        if [ "$1" == "--cpu" ]; then
            DEVICE="cpu"
        fi
        
        python experiments/exp8_performance_benchmark/run_benchmark.py \
            --benchmark all \
            --batch-sizes 8 16 32 64 128 256 512 1024 \
            --num-iterations 10 \
            --parcel-size 60 \
            --output-dir results/exp8_performance_benchmark \
            --device "$DEVICE"
        ;;
    
    status)
        echo "Checking benchmark job status..."
        squeue -u $USER --name=exp8_bench || echo "No jobs found"
        echo ""
        sacct --name=exp8_bench -S $(date -d '1 days ago' +%Y-%m-%d) \
            --format=JobID,JobName,State,Elapsed,MaxRSS 2>/dev/null | tail -10 || true
        ;;
    
    results)
        RESULTS_FILE="results/exp8_performance_benchmark/benchmark_results.json"
        if [ -f "$RESULTS_FILE" ]; then
            echo "Benchmark Results:"
            echo ""
            python -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)

for benchmark_name, results in data.items():
    print(f'\n{benchmark_name.upper()}:')
    print('-' * 60)
    
    # Group by batch size
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    
    for bs in batch_sizes:
        bs_results = [r for r in results if r['batch_size'] == bs]
        baseline = next((r for r in bs_results if r.get('version') == 'baseline'), None)
        optimized = next((r for r in bs_results if r.get('version') == 'optimized'), None)
        numba = next((r for r in bs_results if r.get('version') == 'numba'), None)
        fast = next((r for r in bs_results if r.get('version') == 'fast'), None)
        
        if baseline and optimized:
            line = f'  N={bs:4d}: {baseline[\"mean_ms\"]:8.2f}ms -> {optimized[\"mean_ms\"]:8.2f}ms ({baseline[\"mean_ms\"] / optimized[\"mean_ms\"]:.2f}x)'
            
            if numba:
                line += f' -> {numba[\"mean_ms\"]:8.2f}ms ({baseline[\"mean_ms\"] / numba[\"mean_ms\"]:.2f}x Numba)'
            
            if fast:
                line += f' -> {fast[\"mean_ms\"]:8.2f}ms ({baseline[\"mean_ms\"] / fast[\"mean_ms\"]:.2f}x Fast)'
            
            print(line)
        elif baseline:
            print(f'  N={bs:4d}: {baseline[\"mean_ms\"]:8.2f}ms (baseline only)')
        elif optimized:
            print(f'  N={bs:4d}: {optimized[\"mean_ms\"]:8.2f}ms (optimized only)')
"
        else
            echo "No results found at $RESULTS_FILE"
            echo "Run benchmarks first with: $0 submit  OR  $0 local"
        fi
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        usage
        exit 1
        ;;
esac
