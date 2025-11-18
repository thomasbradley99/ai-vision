#!/bin/bash
# Process multiple clips with SAM2 ball tracking

INPUT_DIR="/home/ubuntu/clann/ai-vision/inputs2"
OUTPUT_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/ball_sam2_clips"
SCRIPT="/home/ubuntu/clann/ai-vision/gaa-vision-ai/scripts/3_ball_sam2_green.py"
MODEL="/home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt"

mkdir -p "$OUTPUT_DIR"

# Process selected clips
for clip in gaa1.mp4 gaa2.mp4 gaa4.mp4 gaa5.mp4; do
    echo "=========================================="
    echo "Processing: $clip"
    echo "=========================================="
    
    conda run -n hooper-ai python "$SCRIPT" \
        --video "$INPUT_DIR/$clip" \
        --output "$OUTPUT_DIR/${clip%.mp4}_ball_sam2.mp4" \
        --model "$MODEL" \
        --frame-stride 2
    
    echo "✓ Completed: $clip"
    echo ""
done

echo "=========================================="
echo "All clips processed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

