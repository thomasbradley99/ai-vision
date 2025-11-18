#!/bin/bash
# Process remaining clips with hybrid approach

INPUT_DIR="/home/ubuntu/clann/ai-vision/inputs2"
OUTPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hybrid_clips"
SCRIPT="/home/ubuntu/clann/ai-vision/gaa-vision-ai/scripts/4_hybrid_teams_bb_ball_sam2.py"
MODEL="/home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt"

mkdir -p "$OUTPUT_DIR"

# Process remaining clips
for clip in gaa3.mp4 gaa6.mp4 gaa7.mp4 gaa8.mp4 gaa11.mp4 gaa12.mp4 gaa13.mp4; do
    OUTPUT_FILE="$OUTPUT_DIR/${clip%.mp4}_hybrid.mp4"
    
    # Skip if already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Skipping $clip (already exists)"
        continue
    fi
    
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
echo "All clips processed!"
echo "=========================================="

