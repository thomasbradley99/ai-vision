#!/bin/bash
# Create short hero video - 2 seconds from each clip

set -e

CLIPS_DIR="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/hero_clips"
OUTPUT_FILE="/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/gaa_hero_video_short_$(date +%Y%m%d_%H%M%S).mp4"
TEMP_DIR="/tmp/hero_short_clips"
CONCAT_FILE="/tmp/concat_short_list.txt"
SECONDS_PER_CLIP=2

echo "=========================================="
echo "Creating SHORT hero video (${SECONDS_PER_CLIP}s per clip)"
echo "=========================================="

# Clean and create temp directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Process each clip - take first N seconds
counter=0
for VIDEO in $(ls "$CLIPS_DIR"/gaa*_tracked.mp4 | sort -V); do
    if [ -f "$VIDEO" ]; then
        counter=$((counter + 1))
        TEMP_FILE="$TEMP_DIR/clip_$(printf "%02d" $counter).mp4"
        
        echo "Processing $(basename $VIDEO) -> ${SECONDS_PER_CLIP}s..."
        
        # Extract first N seconds with re-encoding to ensure consistent format
        ffmpeg -i "$VIDEO" -t $SECONDS_PER_CLIP \
            -c:v libx264 -preset fast -crf 23 \
            -pix_fmt yuv420p -r 30 \
            "$TEMP_FILE" -y -loglevel error
    fi
done

# Create concat list
> "$CONCAT_FILE"
for VIDEO in "$TEMP_DIR"/clip_*.mp4 | sort -V; do
    if [ -f "$VIDEO" ]; then
        echo "file '$VIDEO'" >> "$CONCAT_FILE"
    fi
done

echo ""
echo "Combining $counter clips..."

# Concatenate all videos
ffmpeg -f concat -safe 0 -i "$CONCAT_FILE" -c copy "$OUTPUT_FILE" -y -loglevel error

echo ""
echo "=========================================="
echo "✓ Short hero video created!"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Show file info
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")
SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')

echo "Duration: $(printf "%.1f" $DURATION)s"
echo "File size: $SIZE"

# Clean up
rm -rf "$TEMP_DIR"
rm "$CONCAT_FILE"

