# Quick Setup for Matteo's VM

## What's Done
✅ Re-ID model weights (`swin_base_msmt17.pth`) are now in the repo with Git LFS  
✅ Just pull the repo and the weights will download automatically

## Setup Steps (Super Simple Now!)

```bash
# 1. Pull latest code (includes weights via LFS)
cd /home/ubuntu/ai-vision
git pull

# 2. Make sure LFS files are downloaded
git lfs pull

# 3. Verify weights exist
ls -lh hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth
# Should show ~336MB file

# 4. Activate environment (you already have this)
conda activate hooper-ai

# 5. Run the script!
cd gaa-vision-ai
python scripts/track_individual_improved.py \
    --video inputs/your_video.mp4 \
    --reference-image inputs/reference.jpg \
    --output outputs/all_people.mp4 \
    --match-output outputs/just_you.mp4 \
    --device cuda \
    --batch-size 32 \
    --frame-stride 2
```

That's it! No more copying files manually. The weights are in the repo with LFS.

## If LFS files don't download automatically

```bash
# Install Git LFS if not already installed
git lfs install

# Pull LFS files manually
git lfs pull
```

## Verify Setup

```bash
# Check weights file exists and is correct size
ls -lh hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth
# Should be ~336MB

# Test the script
python scripts/track_individual_improved.py --help
# Should work without errors
```

Done! Much simpler than before.

