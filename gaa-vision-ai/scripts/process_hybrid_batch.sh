#!/bin/bash
# Process multiple clips with hybrid approach

INPUT_DIR="/home/ubuntu/clann/ai-vision/inputs2"
OUTPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hybrid_clips"
SCRIPT="/home/ubuntu/clann/ai-vision/gaa-vision-ai/scripts/4_hybrid_teams_bb_ball_sam2.py"
MODEL="/home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt"

mkdir -p "$OUTPUT_DIR"

# Process selected clips
for clip in gaa2.mp4 gaa4.mp4 gaa5.mp4; do
    echo "=========================================="
    echo "Processing: $clip"
    echo "=========================================="
    
    conda run -n hooper-ai python "$SCRIPT" \
        --video "$INPUT_DIR/$clip" \
        --output "$OUTPUT_DIR/${clip%.mp4}_hybrid.mp4" \
        --model "$MODEL"
    
    echo "✓ Completed: $clip"
    echo ""
done

echo "=========================================="
echo "All clips processed!"
echo "=========================================="

