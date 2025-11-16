#!/bin/bash
echo "=== GAA Processing Status ==="
echo
echo "Process Status:"
ps aux | grep gaa_identify | grep -v grep | awk '{print $2, $3"% CPU", $4"% MEM", $10}'
echo
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | awk -F', ' '{print "Memory: "$1"/"$2" | GPU: "$3" | Temp: "$4}'
echo
echo "Chunks Created:"
ls -1 /home/ubuntu/clann/ai-vision/outputs/gaa_full_chunked/temp_chunks/videos/ 2>/dev/null | wc -l
echo
echo "Chunks Processed:"
find /home/ubuntu/clann/ai-vision/outputs/gaa_full_chunked/temp_chunks/output_* -name "player_overlay.mp4" 2>/dev/null | wc -l
echo
echo "Output Sizes:"
du -sh /home/ubuntu/clann/ai-vision/outputs/gaa_full_chunked/temp_chunks/output_* 2>/dev/null
echo
echo "Runtime:"
ps -p $(pgrep -f "gaa_identify.py --video" | head -1) -o etime= 2>/dev/null | xargs echo "Elapsed:"

