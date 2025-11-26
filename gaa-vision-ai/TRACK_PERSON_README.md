# Track Yourself in GAA Videos - Complete Guide

This system tracks a specific person (you) throughout long GAA videos using person re-identification (Re-ID).

## 🎯 What It Does

1. **Detects ALL people** in the video
2. **Tracks each person** with unique IDs using Re-ID
3. **Counts unique people** in the video
4. **Matches your reference image** to one of the tracks
5. **Creates videos** with you highlighted

## 📁 Files Created

- `scripts/track_all_people_then_match.py` - Main script (detects all, then matches you)
- `scripts/track_specific_person.py` - Alternative (only tracks matching person)
- `TRACK_PERSON_GUIDE.md` - Detailed usage guide
- `REID_EXPLAINED_SIMPLE.md` - How Re-ID works
- `WHERE_TO_PUT_VIDEO.md` - File organization

## 🚀 Quick Start

### Prerequisites

1. **Conda environment**: `hooper-ai`
2. **Model weights**: Re-ID model (`swin_base_msmt17.pth`)
3. **Video file**: Your GAA session video
4. **Reference image**: Photo/crop of yourself

### Step 1: Setup

```bash
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai
conda activate hooper-ai
```

### Step 2: Put Your Video

```bash
# Copy your video to inputs folder
cp /path/to/your/video.mp4 inputs/your_video.mp4
```

### Step 3: Extract Reference Frame

```bash
# Extract a frame where you're visible (adjust time as needed)
ffmpeg -i inputs/your_video.mp4 -ss 00:00:10 -vframes 1 inputs/reference.jpg
```

### Step 4: Run Tracking

```bash
python scripts/track_all_people_then_match.py \
    --video inputs/your_video.mp4 \
    --reference-image inputs/reference.jpg \
    --output outputs/all_people.mp4 \
    --match-output outputs/just_you.mp4 \
    --frame-stride 4 \
    --device cpu
```

## 📊 What You Get

### Output Files

1. **`outputs/all_people.mp4`**
   - All people tracked with different colored boxes
   - Each person has a unique ID
   - Shows how many unique people were detected

2. **`outputs/just_you.mp4`**
   - Only you highlighted (green box)
   - Tracked throughout the entire video

3. **`outputs/track_data/tracks/`**
   - Tag images showing each person detected
   - Named `track_01_tag.jpg`, `track_02_tag.jpg`, etc.
   - Use these to verify which track is you

4. **`outputs/track_data/player_tracks.json`**
   - Metadata about all tracks
   - Frame numbers, embeddings, etc.

## 🔧 Parameters Explained

### Key Parameters

- `--frame-stride` (default: 2)
  - Process every Nth frame
  - `2` = every other frame (faster)
  - `1` = every frame (slowest, most accurate)
  - `4` = every 4th frame (fastest)

- `--similarity-threshold` (default: 0.7)
  - How similar must someone be to count as "you"
  - Lower = more matches (but more false positives)
  - Higher = fewer matches (but more accurate)
  - Range: 0.0 - 1.0

- `--device` (`cuda` or `cpu`)
  - Use GPU if available (`cuda`)
  - Use CPU if no GPU (`cpu`)
  - CPU is slower but works everywhere

- `--max-track-gap` (default: 60)
  - How many frames you can disappear before track resets
  - 60 frames at 30fps = 2 seconds
  - Increase if you disappear for longer periods

### Performance Tuning

**For 30-minute video at 30fps:**
- Total frames: ~54,000
- With `--frame-stride 4`: Process ~13,500 frames
- Expected time: 4-8 hours on CPU, 1-2 hours on GPU

**To speed up:**
```bash
--frame-stride 4        # Process every 4th frame
--batch-size 32        # Larger batches (if GPU memory allows)
--device cuda          # Use GPU if available
```

## 🏗️ System Requirements

### Required Models

1. **Detectron2 Keypoint Model**
   - Auto-downloads on first run
   - If SSL issues: Download manually from:
     `https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl`
   - Save to: `~/.cache/detectron2/`

2. **Re-ID Model** (`swin_base_msmt17.pth`)
   - Location: `/home/ubuntu/clann/ai-vision/hooper-glean/checkpoints/PERSON-Tracking/swin_base_msmt17.pth`
   - Symlink points to: `/home/ubuntu/hooper-ai/ai/checkpoints/PERSON-Tracking/`
   - If missing: Need to download or copy from another system

### Dependencies

All in `hooper-ai` conda environment:
- PyTorch
- Detectron2
- OpenCV
- NumPy
- tqdm
- Solider Re-ID model

## 🐛 Troubleshooting

### "Re-ID weights not found"

**Problem**: Missing `swin_base_msmt17.pth`

**Solutions**:
1. Check if weights exist elsewhere:
   ```bash
   find /home/ubuntu -name "swin_base_msmt17.pth"
   ```
2. Copy from another system if available
3. Download from model repository (if you have access)
4. Use alternative script without Re-ID (YOLO-based only)

### "No NVIDIA driver found"

**Problem**: Script trying to use GPU but none available

**Solution**: Add `--device cpu` to command

### "SSL certificate verify failed"

**Problem**: Can't download Detectron2 model

**Solutions**:
1. Download manually and place in `~/.cache/detectron2/`
2. Fix SSL certificates:
   ```bash
   conda install certifi
   ```
3. Use pre-downloaded model from another system

### "No tracks found"

**Problem**: No people detected

**Solutions**:
1. Lower detection threshold: `--min-person-score 0.5`
2. Process more frames: `--frame-stride 1`
3. Check video quality/format

### "No match found" (for your reference)

**Problem**: Can't match your reference image to any track

**Solutions**:
1. Lower similarity threshold: `--similarity-threshold 0.6`
2. Use a better reference image (clearer, better lighting)
3. Extract reference from the same video
4. Check tag images in `track_data/tracks/` to find yourself manually

## 📝 Running on Different VM

### What to Copy

1. **Code**:
   ```bash
   # Already in git repo
   cd /home/ubuntu/clann/ai-vision
   git push  # Push your changes
   ```

2. **Model Weights**:
   ```bash
   # Copy Re-ID weights
   scp /path/to/swin_base_msmt17.pth user@new-vm:/path/to/checkpoints/
   
   # Or use rsync
   rsync -av /home/ubuntu/clann/ai-vision/hooper-glean/checkpoints/ \
       user@new-vm:/path/to/checkpoints/
   ```

3. **Conda Environment**:
   ```bash
   # Export environment
   conda env export -n hooper-ai > hooper-ai-env.yml
   
   # On new VM
   conda env create -f hooper-ai-env.yml
   ```

### Setup on New VM

```bash
# 1. Clone repo
git clone <your-repo> ai-vision
cd ai-vision

# 2. Create conda environment
conda env create -f hooper-ai-env.yml
conda activate hooper-ai

# 3. Install any missing packages
pip install -r requirements.txt  # If exists

# 4. Copy model weights
mkdir -p hooper-glean/checkpoints/PERSON-Tracking
cp /path/to/swin_base_msmt17.pth hooper-glean/checkpoints/PERSON-Tracking/

# 5. Test
python gaa-vision-ai/scripts/track_all_people_then_match.py --help
```

## 🔄 Alternative: Use Existing Script

If Re-ID weights are missing, use the existing `gaa_identify.py`:

```bash
cd /home/ubuntu/clann/ai-vision/hooper-glean/scripts
conda activate hooper-ai

python gaa_identify.py \
    --video /path/to/video.mp4 \
    --out-dir /path/to/output \
    --frame-stride 4 \
    --device cpu
```

This will:
- Detect and track all people
- Create tag images for each person
- You manually identify which track is you from the tag images

## 📚 Additional Documentation

- `TRACK_PERSON_GUIDE.md` - Detailed usage guide
- `REID_EXPLAINED_SIMPLE.md` - How Re-ID works (simple explanation)
- `WHERE_TO_PUT_VIDEO.md` - File organization

## 🎬 Example Workflow

```bash
# 1. Setup
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai
conda activate hooper-ai

# 2. Add video
cp ~/Downloads/my_session.mp4 inputs/

# 3. Extract reference
ffmpeg -i inputs/my_session.mp4 -ss 00:00:10 -vframes 1 inputs/me.jpg

# 4. Run tracking
python scripts/track_all_people_then_match.py \
    --video inputs/my_session.mp4 \
    --reference-image inputs/me.jpg \
    --output outputs/all_people.mp4 \
    --match-output outputs/just_me.mp4 \
    --frame-stride 4 \
    --device cpu \
    --similarity-threshold 0.7

# 5. Check results
ls -lh outputs/
# - all_people.mp4 (everyone tracked)
# - just_me.mp4 (only you)
# - track_data/tracks/ (tag images)
```

## ✅ Checklist for New VM

- [ ] Code cloned from git
- [ ] Conda environment created (`hooper-ai`)
- [ ] Re-ID weights copied (`swin_base_msmt17.pth`)
- [ ] Detectron2 model downloaded (or cached)
- [ ] Test run with `--help` works
- [ ] Video file accessible
- [ ] Reference image extracted

## 💡 Tips

1. **Start with small test**: Use a 30-second clip first to verify everything works
2. **Check tag images**: Always look at `track_data/tracks/` to verify which track is you
3. **Adjust similarity**: If no match, lower threshold to 0.6 or 0.65
4. **Frame stride**: Use 4 for speed, 2 for balance, 1 for accuracy
5. **GPU vs CPU**: GPU is 4-8x faster, but CPU works fine for testing

## 🆘 Need Help?

1. Check tag images in `track_data/tracks/` - these show each detected person
2. Lower similarity threshold if no match found
3. Use `--device cpu` if GPU issues
4. Check video format (MP4 works best)
5. Verify model weights are in correct location

---

**Last Updated**: November 2024  
**Location**: `/home/ubuntu/clann/ai-vision/gaa-vision-ai/`

