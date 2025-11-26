# Input Videos Directory

Put your video files here!

## Supported Formats
- `.mp4` (recommended)
- `.mov`
- `.avi`
- `.mkv`

## Example
```bash
# Copy your video here
cp /path/to/your/video.mp4 /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/

# Then run tracking
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai
python scripts/track_specific_person.py \
    --video inputs/your_video.mp4 \
    --reference-image inputs/your_photo.jpg \
    --output outputs/tracked_output.mp4
```

