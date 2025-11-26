# Track Yourself in a 30-Minute GAA Session

This guide explains how to track yourself throughout a full 30-minute GAA video using person re-identification.

## Quick Start

```bash
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai
conda activate hooper-ai

python scripts/track_specific_person.py \
    --video /path/to/your_30min_session.mp4 \
    --reference-image /path/to/your_photo.jpg \
    --output outputs/tracked_you.mp4 \
    --similarity-threshold 0.7
```

## How It Works

1. **Reference Image**: You provide a photo/crop of yourself
2. **Re-ID Embedding**: The system extracts a feature vector from your photo
3. **Video Processing**: Every person detected in the video is compared against your embedding
4. **Tracking**: Matches are tracked across frames, even when you leave/re-enter the frame
5. **Output**: Video with you highlighted in green boxes

## Requirements

### Reference Image
- Clear photo of yourself (face + upper body visible)
- Can be a crop from the video itself
- Best if wearing the same clothes as in the video
- Recommended: 200x300 pixels or larger

### Video
- Any format (will be converted to MP4)
- 30 minutes is fine - the system handles long videos
- Frame stride of 2 means it processes every other frame (faster)

## Parameters

### Key Parameters

- `--similarity-threshold` (default: 0.7)
  - Lower = more matches (but more false positives)
  - Higher = fewer matches (but more accurate)
  - Start with 0.7, adjust if needed

- `--frame-stride` (default: 2)
  - 1 = process every frame (slowest, most accurate)
  - 2 = process every other frame (balanced)
  - 4 = process every 4th frame (fastest, less accurate)

- `--max-track-gap` (default: 60)
  - How many frames you can disappear before track resets
  - For 30fps video: 60 frames = 2 seconds
  - Increase if you disappear for longer periods

### Performance Tuning

For a 30-minute video at 30fps:
- **Total frames**: ~54,000 frames
- **With stride=2**: Process ~27,000 frames
- **Expected time**: 2-4 hours on GPU, 8-12 hours on CPU

To speed up:
```bash
# Process every 4th frame (4x faster)
--frame-stride 4

# Larger batch size (if you have GPU memory)
--batch-size 32
```

## Example Workflow

### Step 1: Extract Reference Image from Video

If you don't have a separate photo, extract a frame from the video:

```bash
# Extract frame at 10 seconds
ffmpeg -i your_video.mp4 -ss 00:00:10 -vframes 1 reference_frame.jpg

# Or use a video player to find a good frame, then extract it
```

### Step 2: Run Tracking

```bash
python scripts/track_specific_person.py \
    --video /path/to/your_30min_session.mp4 \
    --reference-image reference_frame.jpg \
    --output outputs/tracked_you.mp4 \
    --similarity-threshold 0.7 \
    --frame-stride 2
```

### Step 3: Check Results

The output video will have:
- Green bounding boxes around you
- Label showing "You (0.XX)" with similarity score
- Only frames where you're detected

## Troubleshooting

### No tracks found

**Problem**: Script completes but no detections

**Solutions**:
1. Lower similarity threshold: `--similarity-threshold 0.6`
2. Use a better reference image (clearer, better lighting)
3. Check that you appear in the video
4. Lower detection threshold: `--min-person-score 0.5`

### Tracking breaks frequently

**Problem**: Track resets when you briefly disappear

**Solutions**:
1. Increase max track gap: `--max-track-gap 120` (4 seconds at 30fps)
2. Lower similarity threshold slightly: `--similarity-threshold 0.65`
3. Process more frames: `--frame-stride 1`

### Too many false positives

**Problem**: Other people are being tracked as you

**Solutions**:
1. Increase similarity threshold: `--similarity-threshold 0.75`
2. Use a more distinctive reference image
3. Increase detection threshold: `--min-person-score 0.7`

### Processing too slow

**Solutions**:
1. Increase frame stride: `--frame-stride 4`
2. Use GPU: `--device cuda` (if available)
3. Increase batch size: `--batch-size 32` (if GPU memory allows)

## Advanced: Multiple Reference Images

For better accuracy, you can modify the script to use multiple reference images and average their embeddings. This helps handle different poses/lighting.

## Output Statistics

The script prints:
- Number of tracks found
- Total detections
- Average similarity score

A good result:
- 1-3 tracks (you might be split into multiple tracks if you disappear for long)
- Average similarity > 0.75
- Detections in most frames where you're visible

## Integration with Other Scripts

After tracking yourself, you can:
1. Extract clips where you appear: Use frame numbers from tracks
2. Create highlight reel: Combine tracked segments
3. Analyze your movements: Use bounding box positions over time

## Notes

- The system uses **Solider Re-ID** model (state-of-the-art person re-identification)
- Works best when you're wearing the same clothes as in reference
- Handles occlusions, pose changes, and lighting variations
- For 30-minute videos, expect 2-4 hour processing time on GPU

