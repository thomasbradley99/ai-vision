#!/bin/bash
# Combine all processed clips into one hero video

set -e

CLIPS_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hero_clips"
OUTPUT_FILE="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/gaa_hero_video_$(date +%Y%m%d_%H%M%S).mp4"
CONCAT_FILE="/tmp/concat_list.txt"

echo "=========================================="
echo "Creating hero video from processed clips"
echo "=========================================="

# Create concat list file
> "$CONCAT_FILE"
for VIDEO in $(ls "$CLIPS_DIR"/gaa*_tracked.mp4 | sort -V); do
    if [ -f "$VIDEO" ]; then
        echo "file '$VIDEO'" >> "$CONCAT_FILE"
    fi
done

# Check if we have any files
if [ ! -s "$CONCAT_FILE" ]; then
    echo "Error: No processed clips found in $CLIPS_DIR"
    exit 1
fi

echo "Combining $(wc -l < "$CONCAT_FILE") clips..."

# Concatenate all videos
ffmpeg -f concat -safe 0 -i "$CONCAT_FILE" -c copy "$OUTPUT_FILE" -y

echo ""
echo "=========================================="
echo "✓ Hero video created successfully!"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Show file size
ls -lh "$OUTPUT_FILE"

# Clean up
rm "$CONCAT_FILE"

