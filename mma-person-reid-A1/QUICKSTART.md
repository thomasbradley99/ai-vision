# Quick Start Guide - JP's Person ReID Trial

**Goal**: Count unique people in MMA videos using SAM2 + embedding vectors

## 30-Second Overview

```bash
# 1. Setup
cd /home/ubuntu/clann/clann-jujisu/mma-person-reid-jp
conda activate hooper-ai
./setup_test_videos.sh

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install SAM2 (one-time)
cd scripts
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../../..

# 4. Run pipeline
./run_pipeline.sh videos/test-video-1 2 0.7 2

# 5. Check results
cat videos/test-video-1/outputs/person_count.json
```

## Detailed Setup

### 1. Environment Setup
```bash
# Activate conda environment
conda activate hooper-ai

# Navigate to project
cd /home/ubuntu/clann/clann-jujisu/mma-person-reid-jp

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install SAM2
```bash
# Clone SAM2 repository
cd scripts
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# Install SAM2
pip install -e .

# Download model checkpoints (~2GB)
cd checkpoints
./download_ckpts.sh

# Return to project root
cd ../../..
```

### 3. Setup Test Videos
```bash
# Create symbolic links to existing BJJ videos
./setup_test_videos.sh
```

This creates:
- `videos/test-video-1/` → ryan-thomas (2 people, 6 min)
- `videos/test-video-2/` → columba (2 people, 1 min)
- `videos/test-video-3/` → gio-thomas (2 people, 3 min)

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)
```bash
# Run all 5 stages automatically
./run_pipeline.sh videos/test-video-1 2 0.7 2

# Arguments:
#   videos/test-video-1  = video directory
#   2                    = FPS to extract
#   0.7                  = similarity threshold
#   2                    = ground truth count (for validation)
```

### Option 2: Stage by Stage
```bash
# Stage 1: Extract frames (2 FPS)
python scripts/1_extract_frames.py videos/test-video-1 --fps 2

# Stage 2: Segment persons with SAM2
python scripts/2_segment_persons.py videos/test-video-1

# Stage 3: Extract embeddings
python scripts/3_extract_embeddings.py videos/test-video-1

# Stage 4: Reidentify persons
python scripts/4_reidentify.py videos/test-video-1 --threshold 0.7

# Stage 5: Generate person count
python scripts/5_count_people.py videos/test-video-1 --ground-truth 2
```

## Understanding the Output

### Output Files
```
videos/test-video-1/outputs/
├── frames/                      # Extracted frames
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── segmentations/              # SAM2 masks
│   ├── frame_0000_masks.npz
│   ├── frame_0000_boxes.json
│   └── ...
├── visualizations/             # Visual outputs
│   ├── frame_0000_vis.jpg     # Segmentation overlay
│   └── ...
├── embeddings.pkl              # Person embeddings
├── person_tracks.json          # Person IDs across frames
├── person_count.json           # Final report
└── frames_metadata.json        # Video metadata
```

### Key Output: person_count.json
```json
{
  "unique_people_count": 2,
  "people": [
    {
      "person_id": 1,
      "total_appearances": 698,
      "first_frame": 0,
      "last_frame": 715,
      "confidence": 0.95
    },
    {
      "person_id": 2,
      "total_appearances": 682,
      "first_frame": 5,
      "last_frame": 710,
      "confidence": 0.92
    }
  ]
}
```

## Validation

### Check Results
```bash
# View person count
cat videos/test-video-1/outputs/person_count.json | grep "unique_people_count"

# Expected: "unique_people_count": 2
```

### Visualize Segmentation
```bash
# Open segmentation visualizations
ls videos/test-video-1/outputs/visualizations/

# View a sample frame
# (Use your preferred image viewer or copy to local machine)
```

### Validate All Test Videos
```bash
# Test video 1 (ryan-thomas)
./run_pipeline.sh videos/test-video-1 2 0.7 2

# Test video 2 (columba)
./run_pipeline.sh videos/test-video-2 2 0.7 2

# Test video 3 (gio-thomas)
./run_pipeline.sh videos/test-video-3 2 0.7 2
```

**Expected**: All 3 should detect 2 people

## Troubleshooting

### Issue: SAM2 not found
```bash
# Solution: Install SAM2
cd scripts/segment-anything-2
pip install -e .
cd checkpoints
./download_ckpts.sh
```

### Issue: CUDA out of memory
```bash
# Solution: Use CPU (slower) or reduce FPS
python scripts/1_extract_frames.py videos/test-video-1 --fps 1

# Force CPU for SAM2
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Wrong person count
```bash
# Solution: Tune similarity threshold
python scripts/4_reidentify.py videos/test-video-1 --threshold 0.6  # Lower = merge more
python scripts/4_reidentify.py videos/test-video-1 --threshold 0.8  # Higher = split more
python scripts/5_count_people.py videos/test-video-1 --ground-truth 2
```

### Issue: Very slow processing
```bash
# Solutions:
# 1. Lower FPS
python scripts/1_extract_frames.py videos/test-video-1 --fps 1

# 2. Use GPU (if available)
# Check GPU: nvidia-smi

# 3. Process shorter clips
# (Use ffmpeg to trim video first)
```

## Next Steps

### Week 1 Goals
1. ✓ Get pipeline running
2. ✓ Process first video
3. Validate accuracy on all 3 test videos
4. Debug any issues

### Week 2 Goals
1. Improve accuracy to 95%+
2. Optimize processing speed
3. Handle edge cases

### Week 3 Goals
1. Production-ready code
2. Integration with existing BJJ pipeline
3. Final demo

## Documentation

- **README.md**: Full project overview
- **METHODOLOGY.md**: Technical details
- **EVALUATION.md**: How to measure accuracy
- **PROGRESS.md**: Weekly progress tracker
- **QUICKSTART.md**: This file

## Getting Help

If you're stuck:
1. Check error messages carefully
2. Review relevant documentation
3. Check existing BJJ pipeline for reference:
   - `/home/ubuntu/clann/clann-jujisu/bjj-ai-testing/`
4. Ask team for help

## Useful Commands

```bash
# Check conda environment
conda env list

# Activate environment
conda activate hooper-ai

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# Count frames
ls videos/test-video-1/outputs/frames/ | wc -l

# View JSON files nicely
cat videos/test-video-1/outputs/person_count.json | python -m json.tool

# Quick test (just Stage 1)
python scripts/1_extract_frames.py videos/test-video-1 --fps 1
ls videos/test-video-1/outputs/frames/
```

---

**Ready to start? Run:**
```bash
./setup_test_videos.sh
./run_pipeline.sh videos/test-video-1 2 0.7 2
```

Good luck! 🚀

