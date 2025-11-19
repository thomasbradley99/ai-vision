#!/bin/bash
# Process all GAA clips with team tracking and bounding boxes

set -e

INPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs2"
OUTPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hero_clips"
MODEL_PATH="/home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt"
SCRIPT_PATH="/home/ubuntu/clann/ai-vision/gaa-vision-ai/scripts/1_ball_filter_jersey_color_bb.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each clip
for VIDEO in "$INPUT_DIR"/gaa*.mp4; do
    BASENAME=$(basename "$VIDEO" .mp4)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_tracked.mp4"
    
    echo "=========================================="
    echo "Processing: $BASENAME"
    echo "=========================================="
    
    conda run -n hooper-ai python "$SCRIPT_PATH" \
        --video "$VIDEO" \
        --output "$OUTPUT_FILE" \
        --model "$MODEL_PATH"
    
    echo "✓ Completed: $BASENAME"
    echo ""
done

echo "=========================================="
echo "All clips processed successfully!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

