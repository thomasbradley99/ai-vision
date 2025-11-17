#!/bin/bash
# Monitor GAA Batch Processing Progress

echo "========================================="
echo "GAA Batch Processing Monitor"
echo "========================================="
echo ""

# Check if process is running
if ps aux | grep -q "[g]aa_identify"; then
    echo "✓ Status: RUNNING"
    echo ""
    
    # Show current process
    echo "Current Process:"
    ps aux | grep "[g]aa_identify" | awk '{print "  " $11 " " $12 " " $13 " " $14}'
    echo ""
    
    # GPU Usage
    echo "GPU Status:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk '{printf "  Memory: %s/%s MB (%.1f%% util)\n", $1, $2, $3}'
    echo ""
else
    echo "✗ Status: NOT RUNNING"
    echo ""
fi

# Show completed outputs
echo "Completed Outputs:"
for duration in "10s" "30s" "1min" "2min"; do
    output="/home/ubuntu/clann/ai-vision/outputs/gaa_${duration}_reid_full/player_overlay.mp4"
    if [ -f "$output" ]; then
        size=$(du -h "$output" | cut -f1)
        echo "  ✓ ${duration}: ${size}"
    else
        echo "  ⏳ ${duration}: pending"
    fi
done
echo ""

# Show recent log entries
echo "Recent Log (last 15 lines):"
echo "----------------------------------------"
tail -15 /home/ubuntu/clann/ai-vision/outputs/gaa_batch_nohup.log 2>/dev/null || echo "  No log file yet"
echo "========================================="

