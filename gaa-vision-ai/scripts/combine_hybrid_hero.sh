#!/bin/bash
# Combine hybrid clips into hero video

CLIPS_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hybrid_clips"
OUTPUT_FILE="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/gaa_hero_hybrid_$(date +%Y%m%d_%H%M%S).mp4"
CONCAT_FILE="/tmp/hybrid_concat.txt"

echo "=========================================="
echo "Creating hero video from hybrid clips"
echo "=========================================="

# Create concat list (sorted by filename)
> "$CONCAT_FILE"
for VIDEO in $(ls "$CLIPS_DIR"/*_hybrid.mp4 | sort -V); do
    if [ -f "$VIDEO" ]; then
        echo "file '$VIDEO'" >> "$CONCAT_FILE"
        echo "Adding: $(basename $VIDEO)"
    fi
done

# Check if we have files
if [ ! -s "$CONCAT_FILE" ]; then
    echo "Error: No clips found in $CLIPS_DIR"
    exit 1
fi

echo ""
echo "Combining $(wc -l < "$CONCAT_FILE") clips..."

# Concatenate with re-encoding to ensure compatibility
ffmpeg -f concat -safe 0 -i "$CONCAT_FILE" \
    -c:v libx264 -preset fast -crf 23 \
    -pix_fmt yuv420p -r 30 \
    "$OUTPUT_FILE" -y -loglevel error

echo ""
echo "=========================================="
echo "✓ Hero video created!"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Show file info
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")
SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')

echo "Duration: $(printf "%.1f" $DURATION)s"
echo "File size: $SIZE"

# Clean up
rm "$CONCAT_FILE"

