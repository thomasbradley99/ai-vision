#!/bin/bash
# Process all 13 GAA clips with Script 4 (Multi-anchor SAM2 + Clann Green)

cd /home/ubuntu/clann/ai-vision/gaa-vision-ai

echo "========================================="
echo "Processing all 13 clips with Script 4"
echo "Multi-anchor SAM2 + Clann Green (#016F32)"
echo "========================================="
echo ""

start_time=$(date +%s)

for i in {1..13}; do
    echo "[$i/13] Processing gaa${i}.mp4..."
    echo "Started at: $(date +%H:%M:%S)"
    
    clip_start=$(date +%s)
    
    conda run -n hooper-ai python scripts/4_hybrid_teams_bb_ball_sam2.py \
        --video inputs2/gaa${i}.mp4 \
        --output outputs/hybrid_clips/gaa${i}_hybrid.mp4 \
        --model /home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt
    
    clip_end=$(date +%s)
    clip_duration=$((clip_end - clip_start))
    
    echo "Finished gaa${i} at: $(date +%H:%M:%S)"
    echo "Time: ${clip_duration}s ($(($clip_duration / 60))m $(($clip_duration % 60))s)"
    echo ""
done

end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo "========================================="
echo "ALL CLIPS COMPLETE!"
echo "Total time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
echo "Finished at: $(date +%H:%M:%S)"
echo "========================================="

