# Evaluation Guide

## How to Measure Accuracy

This guide explains how to evaluate the person reidentification system.

## Metrics

### 1. Person Count Accuracy
**Question**: Does the system count the correct number of unique people?

**Calculation**:
```
Accuracy = (Correct counts / Total videos) × 100%
```

**Example**:
- Video 1: Ground truth = 2, Detected = 2 ✓
- Video 2: Ground truth = 2, Detected = 3 ✗
- Video 3: Ground truth = 2, Detected = 2 ✓
- **Accuracy = 2/3 = 66.7%**

**Target**: 95%+ accuracy

---

### 2. Identity Consistency
**Question**: Does the same person get the same ID across frames?

**Measurement**:
1. Manually verify 10 random frames
2. Check if person IDs are consistent
3. Count ID switches

**Calculation**:
```
Consistency = (Frames with correct IDs / Total frames checked) × 100%
```

**Target**: 85%+ consistency

---

### 3. False Positives
**Question**: Does the system detect people that don't exist?

**Examples**:
- Detecting equipment/bags as people
- Splitting one person into multiple IDs
- Detecting referees in background

**Measurement**:
```
False Positive Rate = (False positives / Total detections) × 100%
```

**Target**: <5% false positive rate

---

### 4. False Negatives
**Question**: Does the system miss actual people?

**Examples**:
- Person partially occluded
- Person in background
- Person enters/exits frame

**Measurement**:
```
False Negative Rate = (Missed people / Actual people) × 100%
```

**Target**: <10% false negative rate

---

### 5. Processing Speed
**Question**: How fast does the system process videos?

**Measurement**:
- Time each stage of pipeline
- Calculate frames per second
- Measure total pipeline time

**Benchmarks**:
| Video Length | Target Time | Max Time |
|--------------|-------------|----------|
| 1 minute | 3 min | 5 min |
| 5 minutes | 10 min | 20 min |
| 10 minutes | 15 min | 30 min |

---

## Evaluation Protocol

### Step 1: Prepare Test Set
Use 3 BJJ videos with known ground truth:

```bash
./setup_test_videos.sh
```

**Test videos**:
1. `test-video-1` (ryan-thomas): 2 people, ~6 min
2. `test-video-2` (columba): 2 people, ~1 min
3. `test-video-3` (gio-thomas): 2 people, ~3 min

---

### Step 2: Run Pipeline
Process each video:

```bash
./run_pipeline.sh videos/test-video-1 2 0.7 2
./run_pipeline.sh videos/test-video-2 2 0.7 2
./run_pipeline.sh videos/test-video-3 2 0.7 2
```

---

### Step 3: Automated Validation
Check person count accuracy:

```bash
python scripts/5_count_people.py videos/test-video-1 --ground-truth 2
```

**Expected output**:
```
VALIDATION:
  Ground truth: 2 people
  Detected: 2 people
  ✓ CORRECT!
```

---

### Step 4: Manual Validation
Manually inspect results:

1. **Check visualizations**:
   ```bash
   ls videos/test-video-1/outputs/visualizations/
   # Open random frames and verify:
   # - Are all people detected?
   # - Are person IDs consistent?
   # - Any false detections?
   ```

2. **Check person tracks**:
   ```bash
   cat videos/test-video-1/outputs/person_tracks.json
   # Verify:
   # - Correct number of unique person IDs
   # - Each person appears in most frames
   # - No suspicious short tracks (<5 frames)
   ```

3. **Check embeddings**:
   ```bash
   python scripts/analyze_embeddings.py videos/test-video-1
   # (Create this script to visualize embedding similarities)
   ```

---

### Step 5: Error Analysis
If accuracy is low, diagnose the issue:

#### Problem: Wrong person count
**Debug steps**:
1. Check segmentation visualizations
   - Are people correctly segmented?
   - Are there false object detections?
2. Check similarity threshold
   - Too low → merges different people
   - Too high → splits same person
3. Adjust threshold and re-run Stage 4-5

#### Problem: Inconsistent IDs
**Debug steps**:
1. Check embedding similarities
   - Are same-person embeddings similar?
   - Are different-person embeddings dissimilar?
2. Check for appearance changes
   - Lighting changes
   - Pose changes
   - Occlusions
3. Consider temporal smoothing

#### Problem: Slow processing
**Debug steps**:
1. Profile each stage
   ```bash
   time python scripts/1_extract_frames.py videos/test-video-1
   time python scripts/2_segment_persons.py videos/test-video-1
   # etc.
   ```
2. Optimize slowest stage:
   - Stage 2 (SAM2): Use GPU, reduce frame rate
   - Stage 3 (Embeddings): Batch processing
   - All stages: Use smaller input frames

---

## Evaluation Checklist

### Week 1 (MVP)
- [ ] Pipeline runs end-to-end without errors
- [ ] Processes at least 1 video successfully
- [ ] Outputs person count (even if incorrect)
- [ ] Visualizations generated

### Week 2 (Accuracy)
- [ ] Person count correct on all 3 test videos (95%+)
- [ ] Identity consistency >80%
- [ ] False positive rate <10%
- [ ] False negative rate <15%

### Week 3 (Production)
- [ ] Person count correct on all 3 test videos (95%+)
- [ ] Identity consistency >85%
- [ ] False positive rate <5%
- [ ] False negative rate <10%
- [ ] Processing time <20 min per video
- [ ] Clean, documented code

---

## Comparison with Baselines

### Baseline 1: Simple YOLO Person Detection
**Method**: Use YOLO to detect people, count unique boxes

**Expected performance**:
- Person count: 60-70% (often overcounts due to tracking errors)
- Speed: Fast (2-3 min)

### Baseline 2: Google Video Intelligence API
**Method**: Use existing person detection API

**Expected performance**:
- Person count: 80-90%
- Speed: Slow (API calls)
- Cost: Paid API

### Our System: SAM2 + Embeddings + ReID
**Target performance**:
- Person count: 95%+
- Speed: Medium (10-20 min)
- Cost: Free (local processing)

---

## Tips for Improving Accuracy

### 1. Tune Similarity Threshold
Try different thresholds:
```bash
for threshold in 0.5 0.6 0.7 0.8 0.9; do
    python scripts/4_reidentify.py videos/test-video-1 --threshold $threshold
    python scripts/5_count_people.py videos/test-video-1 --ground-truth 2
done
```

### 2. Adjust Frame Rate
Lower FPS = fewer frames = less chance of errors:
```bash
python scripts/1_extract_frames.py videos/test-video-1 --fps 1
```

### 3. Better Embedding Model
Try different models:
- CLIP embeddings (more robust)
- Torchreid models (person-specific)
- Fine-tuned ResNet on MMA data

### 4. Temporal Smoothing
Use position tracking to smooth ID assignments:
- People don't teleport
- Track position + appearance
- Use Kalman filtering

### 5. Post-processing
Clean up tracks:
- Merge very short tracks (<5 frames)
- Remove tracks far from main action
- Use majority voting for IDs

---

## Reporting Results

### Format for Weekly Reports

```markdown
## Week X Evaluation

### Test Results
| Video | Ground Truth | Detected | Status |
|-------|--------------|----------|--------|
| test-video-1 | 2 | 2 | ✓ |
| test-video-2 | 2 | 2 | ✓ |
| test-video-3 | 2 | 3 | ✗ |

### Metrics
- Person count accuracy: 66.7%
- Identity consistency: 85%
- False positive rate: 5%
- Processing time: 15 min/video

### Issues Found
- test-video-3: Detected background person as fighter

### Improvements Made
- Tuned similarity threshold to 0.75
- Added size filtering for segmentation

### Next Steps
- Implement temporal smoothing
- Test on longer videos
```

---

## Questions?

If you encounter evaluation issues:
1. Check `METHODOLOGY.md` for technical details
2. Review `PROGRESS.md` for weekly goals
3. Ask team for help

**Good luck with testing!**

