# Where to Put Your Video

## Quick Answer

Put your video **anywhere you want** - the script accepts any path!

But for organization, I recommend:

```
/home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/your_video.mp4
```

## Directory Structure

I've created these folders for you:

```
ai-vision/gaa-vision-ai/
├── inputs/          ← PUT YOUR VIDEO HERE
│   └── your_30min_session.mp4
│
├── outputs/         ← Results will go here
│   └── tracked_you.mp4
│
└── scripts/
    └── track_specific_person.py
```

## Example Commands

### Option 1: Video in inputs folder (recommended)
```bash
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai

# Put your video here:
# inputs/your_30min_session.mp4

# Run tracking:
python scripts/track_specific_person.py \
    --video inputs/your_30min_session.mp4 \
    --reference-image inputs/your_photo.jpg \
    --output outputs/tracked_you.mp4
```

### Option 2: Video anywhere on your system
```bash
# Video can be anywhere:
python scripts/track_specific_person.py \
    --video /path/to/anywhere/your_video.mp4 \
    --reference-image /path/to/your_photo.jpg \
    --output outputs/tracked_you.mp4
```

### Option 3: Use absolute path
```bash
python scripts/track_specific_person.py \
    --video /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/your_video.mp4 \
    --reference-image /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/your_photo.jpg \
    --output /home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/tracked_you.mp4
```

## Copy Your Video

If your video is somewhere else, copy it:

```bash
# Copy video to inputs folder
cp /path/to/your/video.mp4 /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/

# Or use a symbolic link (saves space)
ln -s /path/to/your/video.mp4 /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/your_video.mp4
```

## Reference Image

Put your reference photo in the same place:

```bash
# Copy your photo
cp /path/to/your/photo.jpg /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/your_photo.jpg
```

Or extract a frame from your video:

```bash
# Extract frame at 10 seconds as reference
ffmpeg -i inputs/your_30min_session.mp4 -ss 00:00:10 -vframes 1 inputs/reference_frame.jpg
```

## Supported Video Formats

The script accepts:
- `.mp4` (recommended)
- `.mov`
- `.avi`
- `.mkv`
- Any format that OpenCV can read

If your video is in a different format, convert it:

```bash
ffmpeg -i your_video.mov inputs/your_video.mp4
```

## Full Example

```bash
# 1. Copy your video
cp ~/Downloads/my_gaa_session.mp4 /home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/

# 2. Extract reference frame (or use a separate photo)
ffmpeg -i inputs/my_gaa_session.mp4 -ss 00:00:10 -vframes 1 inputs/reference.jpg

# 3. Run tracking
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai
conda activate hooper-ai

python scripts/track_specific_person.py \
    --video inputs/my_gaa_session.mp4 \
    --reference-image inputs/reference.jpg \
    --output outputs/tracked_me.mp4 \
    --similarity-threshold 0.7
```

## Summary

**You can put the video anywhere!** The script accepts any path.

But for organization:
- **Inputs**: `/home/ubuntu/clann/ai-vision/gaa-vision-ai/inputs/`
- **Outputs**: `/home/ubuntu/clann/ai-vision/gaa-vision-ai/outputs/`

