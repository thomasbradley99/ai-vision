# MMA Person Reidentification - JP Trial Project

**3-week trial project: Count unique people in MMA/BJJ videos using SAM2 and embedding vectors**

## Project Goal

Develop a system to accurately count and track unique individuals in MMA/BJJ training videos using:
- **SAM2** (Segment Anything Model 2) for person segmentation
- **Embedding vectors** for person reidentification across frames
- **Person tracking** to maintain identity throughout the video

## Use Cases

1. **Gym attendance tracking** - Automatically count unique people in training footage
2. **Multi-fighter analysis** - Track individual fighters across long sparring sessions
3. **Class size monitoring** - Analyze training density and participation
4. **Fighter identification** - Maintain consistent IDs for analysis pipelines

## Project Structure

```
mma-person-reid-jp/
├── videos/                    # Test videos
│   ├── test-video-1/
│   │   ├── input/
│   │   │   └── video.mp4
│   │   └── outputs/
│   │       ├── person_count.json
│   │       ├── segmentation_masks/
│   │       └── embeddings/
│   └── [more test videos]
│
├── scripts/
│   ├── 1_extract_frames.py       # Extract frames from video
│   ├── 2_segment_persons.py      # SAM2 person segmentation
│   ├── 3_extract_embeddings.py   # Generate person embeddings
│   ├── 4_reidentify.py           # Match embeddings across frames
│   └── 5_count_people.py         # Final person count
│
├── docs/
│   ├── METHODOLOGY.md            # Technical approach
│   ├── EVALUATION.md             # How to measure accuracy
│   └── PROGRESS.md               # Weekly progress tracker
│
├── outputs/                      # Shared outputs/visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Technical Approach

### Phase 1: Segmentation (Week 1)
1. Extract frames from MMA videos at 2-5 fps
2. Use SAM2 to segment all people in each frame
3. Save segmentation masks and bounding boxes

### Phase 2: Embedding & Reidentification (Week 2)
1. Extract person crops from segmentation masks
2. Generate embedding vectors using:
   - Option A: ResNet50 + ArcFace (person ReID)
   - Option B: CLIP embeddings
   - Option C: Custom MMA-tuned model
3. Compare embeddings across frames (cosine similarity)
4. Cluster similar embeddings → unique person IDs

### Phase 3: Tracking & Validation (Week 3)
1. Implement temporal tracking (connect IDs across frames)
2. Handle occlusions, pose changes, lighting variations
3. Validate on ground truth data (manual counts)
4. Optimize for speed and accuracy

## Setup

### 1. Environment Setup
```bash
# Use existing hooper-ai conda environment
conda activate hooper-ai

# Install project dependencies
cd /home/ubuntu/clann/clann-jujisu/mma-person-reid-jp
pip install -r requirements.txt
```

### 2. Get SAM2 Model
```bash
# Clone SAM2 repository
cd scripts
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download pretrained weights
cd checkpoints
./download_ckpts.sh
```

### 3. Test Videos
Use existing BJJ videos as test data:
```bash
# Copy test videos from bjj-ai-testing
ln -s /home/ubuntu/clann/clann-jujisu/bjj-ai-testing/videos/ryan-thomas/input/video.mov \
      videos/test-video-1/input/video.mov

ln -s /home/ubuntu/clann/clann-jujisu/bjj-ai-testing/videos/columba/input/video.mov \
      videos/test-video-2/input/video.mov

ln -s /home/ubuntu/clann/clann-jujisu/bjj-ai-testing/videos/gio-thomas/input/video.mov \
      videos/test-video-3/input/video.mov
```

## Usage

### Quick Start - Run Full Pipeline
```bash
cd /home/ubuntu/clann/clann-jujisu/mma-person-reid-jp

# Process a single video
python scripts/1_extract_frames.py videos/test-video-1
python scripts/2_segment_persons.py videos/test-video-1
python scripts/3_extract_embeddings.py videos/test-video-1
python scripts/4_reidentify.py videos/test-video-1
python scripts/5_count_people.py videos/test-video-1

# Output: videos/test-video-1/outputs/person_count.json
```

### Individual Stages

**Stage 1: Extract Frames**
```bash
python scripts/1_extract_frames.py videos/test-video-1 --fps 2
# Output: videos/test-video-1/outputs/frames/
```

**Stage 2: Segment Persons**
```bash
python scripts/2_segment_persons.py videos/test-video-1
# Output: videos/test-video-1/outputs/segmentation_masks/
```

**Stage 3: Extract Embeddings**
```bash
python scripts/3_extract_embeddings.py videos/test-video-1
# Output: videos/test-video-1/outputs/embeddings.pkl
```

**Stage 4: Reidentify**
```bash
python scripts/4_reidentify.py videos/test-video-1 --threshold 0.7
# Output: videos/test-video-1/outputs/person_tracks.json
```

**Stage 5: Count People**
```bash
python scripts/5_count_people.py videos/test-video-1
# Output: videos/test-video-1/outputs/person_count.json
```

## Expected Outputs

### person_count.json
```json
{
  "video_name": "test-video-1",
  "duration_seconds": 360,
  "frames_analyzed": 720,
  "unique_people_count": 2,
  "people": [
    {
      "person_id": 1,
      "first_seen_frame": 0,
      "last_seen_frame": 715,
      "total_appearances": 698,
      "confidence": 0.95
    },
    {
      "person_id": 2,
      "first_seen_frame": 5,
      "last_seen_frame": 710,
      "total_appearances": 682,
      "confidence": 0.92
    }
  ]
}
```

## Evaluation Metrics

1. **Person Count Accuracy**: Does it match manual count?
2. **Identity Consistency**: Same person gets same ID across video?
3. **False Positives**: Detects non-existent people?
4. **False Negatives**: Misses actual people?
5. **Processing Speed**: Frames per second

## Ground Truth

For BJJ test videos:
- **ryan-thomas**: 2 people (Ryan, Thomas)
- **columba**: 2 people (Columba, opponent)
- **gio-thomas**: 2 people (Gio, Thomas)

## Week-by-Week Goals

### Week 1: Segmentation Working
- [ ] SAM2 installed and running
- [ ] Can segment people in MMA frames
- [ ] Visualize segmentation masks
- [ ] Process all 3 test videos

### Week 2: Reidentification Working  
- [ ] Person embeddings extracted
- [ ] Similarity matching implemented
- [ ] Person IDs assigned
- [ ] Track people across frames

### Week 3: Production-Ready
- [ ] Accurate person counts on all 3 videos
- [ ] Fast processing (< 5 min per video)
- [ ] Clean code and documentation
- [ ] Integration plan with existing BJJ pipeline

## Resources

### SAM2
- GitHub: https://github.com/facebookresearch/segment-anything-2
- Paper: https://ai.meta.com/sam2/

### Person ReID
- Torchreid library: https://github.com/KaiyangZhou/deep-person-reid
- ArcFace: https://github.com/deepinsight/insightface

### Related Work
- Existing person detection: `/home/ubuntu/clann/CLANNAI/video-editor/detection/person_detect_api.py`
- Existing tracking: `/home/ubuntu/clann/CLANNAI/video-editor/tracking/`

## Notes

- Use existing `.env` file at `/home/ubuntu/clann/clann-jujisu/.env` for API keys
- Test videos already available in `bjj-ai-testing/videos/`
- Consider GPU usage for SAM2 (if available on EC2)
- Start simple, iterate quickly

## Contact

**Assigned to**: JP  
**Duration**: 3 weeks  
**Start Date**: [To be filled in]  
**Demo Date**: [To be filled in]

---

*Good luck! Focus on getting something working end-to-end in Week 1, then optimize in Weeks 2-3.*

