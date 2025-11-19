#!/bin/bash
# Process ALL 13 clips with fixed hybrid approach

INPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs2"
OUTPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hybrid_clips"
SCRIPT="/home/ubuntu/clann/ai-vision/gaa-vision-ai/scripts/4_hybrid_teams_bb_ball_sam2.py"
MODEL="/home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Processing ALL 13 GAA clips with hybrid approach"
echo "Players: Team colored bounding boxes"
echo "Ball: Yellow SAM2 mask"
echo "=========================================="
echo ""

# Process all clips
for clip in gaa1.mp4 gaa2.mp4 gaa3.mp4 gaa4.mp4 gaa5.mp4 gaa6.mp4 gaa7.mp4 gaa8.mp4 gaa9.mp4 gaa10.mp4 gaa11.mp4 gaa12.mp4 gaa13.mp4; do
    OUTPUT_FILE="$OUTPUT_DIR/${clip%.mp4}_hybrid.mp4"
    
    echo "=========================================="
    echo "Processing: $clip"
    echo "=========================================="
    
    conda run -n hooper-ai python "$SCRIPT" \
        --video "$INPUT_DIR/$clip" \
        --output "$OUTPUT_FILE" \
        --model "$MODEL"
    
    echo "✓ Completed: $clip"
    echo ""
done

echo "=========================================="
echo "ALL CLIPS PROCESSED!"
echo "=========================================="

