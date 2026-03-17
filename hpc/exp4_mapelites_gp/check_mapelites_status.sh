#!/bin/bash
# Check status of MAP-Elites GP grid sweep

RESULTS_DIR="${1:-results/exp4_mapelites_gp/mapelites_gp}"

echo "=========================================="
echo "MAP-Elites GP Sweep Status"
echo "=========================================="
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "Results directory: $RESULTS_DIR"
echo ""

# Count completed runs
total_expected=27  # 3 emitters × 3 batch sizes × 3 replicates
completed=0
failed=0
running=0

echo "Configuration Status:"
echo "--------------------"

for emit in 8 64 256; do
    for batch in 4 16 64; do
        for rep in 1 2 3; do
            run_dir="${RESULTS_DIR}/emit${emit}_batch${batch}_rep${rep}"
            
            if [ -d "$run_dir" ]; then
                if [ -f "$run_dir/archive_final.pkl" ]; then
                    status="✓ COMPLETE"
                    ((completed++))
                elif [ -f "$run_dir/history.pkl" ]; then
                    status="⚠ RUNNING"
                    ((running++))
                else
                    status="✗ FAILED"
                    ((failed++))
                fi
            else
                status="○ NOT STARTED"
            fi
            
            printf "E=%3d B=%2d Rep=%d: %s\n" $emit $batch $rep "$status"
        done
    done
done

echo ""
echo "Summary:"
echo "--------"
echo "Total expected: $total_expected"
echo "Completed:      $completed / $total_expected"
echo "Running:        $running / $total_expected"
echo "Failed:         $failed / $total_expected"
echo "Not started:    $((total_expected - completed - running - failed)) / $total_expected"

# Check if ready for analysis
if [ $completed -eq $total_expected ]; then
    echo ""
    echo "✓ All runs complete! Ready for analysis:"
    echo "  python experiments/exp4_mapelites_gp/analyze_mapelites_gp.py --results-dir $RESULTS_DIR"
elif [ $completed -gt 0 ]; then
    echo ""
    echo "⚠ Partial results available. Progress: $(awk "BEGIN {printf \"%.1f\", ($completed/$total_expected)*100}")%"
fi

echo ""
