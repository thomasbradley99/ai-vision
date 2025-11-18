#!/bin/bash
# Combine hybrid clips into hero video

CLIPS_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hybrid_clips"
OUTPUT_FILE="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/gaa_hero_hybrid_$(date +%Y%m%d_%H%M%S).mp4"
CONCAT_FILE="/tmp/hybrid_concat_list.txt"

echo "=========================================="
echo "Creating hero video from hybrid clips"
echo "=========================================="

# Create concat list (sorted order)
> "$CONCAT_FILE"
for VIDEO in "$CLIPS_DIR"/gaa1_hybrid.mp4 \
             "$CLIPS_DIR"/gaa2_hybrid.mp4 \
             "$CLIPS_DIR"/gaa4_hybrid.mp4 \
             "$CLIPS_DIR"/gaa5_hybrid.mp4; do
    if [ -f "$VIDEO" ]; then
        echo "file '$VIDEO'" >> "$CONCAT_FILE"
        echo "  Adding: $(basename $VIDEO)"
    fi
done

# Check if we have clips
if [ ! -s "$CONCAT_FILE" ]; then
    echo "Error: No clips found!"
    exit 1
fi

echo ""
echo "Combining $(wc -l < "$CONCAT_FILE") clips..."

# Concatenate with re-encode to fix timing issues
ffmpeg -f concat -safe 0 -i "$CONCAT_FILE" \
    -c:v libx264 -preset fast -crf 23 \
    -pix_fmt yuv420p -r 30 \
    "$OUTPUT_FILE" -y -loglevel error

echo ""
echo "=========================================="
echo "✓ Hero video created!"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Show details
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")
SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')

echo "Duration: $(printf "%.1f" $DURATION)s"
echo "File size: $SIZE"

# Clean up
rm "$CONCAT_FILE"

