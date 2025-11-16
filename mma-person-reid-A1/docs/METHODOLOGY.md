# Person Reidentification Methodology

## Overview

This project counts unique people in MMA/BJJ videos using a 5-stage pipeline:

```
Video → Frames → Segmentation → Embeddings → ReID → Count
```

## Technical Pipeline

### Stage 1: Frame Extraction
**Goal**: Sample frames from video at fixed FPS

**Method**:
- Extract frames at 2 FPS (configurable)
- Save as JPEG images
- Lower FPS = faster processing, but might miss people who appear briefly

**Trade-offs**:
- 1 FPS: Fast, but might miss people
- 2 FPS: Good balance for most videos  
- 5 FPS: More accurate, slower processing

**Output**: `frame_0000.jpg`, `frame_0001.jpg`, ...

---

### Stage 2: Person Segmentation
**Goal**: Detect and segment all people in each frame

**Method**: SAM2 (Segment Anything Model 2)
- **Input**: RGB frame
- **Process**: 
  - SAM2 generates masks for all objects
  - Filter masks by size (people are usually > 5% of frame)
  - Extract bounding boxes
- **Output**: Binary masks + bounding boxes

**Why SAM2?**
- State-of-the-art segmentation accuracy
- No training data needed (zero-shot)
- Works well with challenging poses (grappling, submissions)
- Better than YOLO for overlapping people

**Limitations**:
- Slow (GPU recommended)
- May segment non-person objects (can filter by size/shape)

**Alternative approaches**:
- YOLOv8 person detection (faster, less accurate)
- Detectron2 instance segmentation
- Google Video Intelligence API (existing in codebase)

---

### Stage 3: Embedding Extraction
**Goal**: Convert person image to fixed-length vector for comparison

**Method**: ResNet50 pretrained on ImageNet
- **Input**: Person crop from segmentation mask
- **Process**:
  - Resize to 224×224
  - Pass through ResNet50 (without classification layer)
  - Extract 2048-dimensional embedding
  - L2 normalize
- **Output**: Embedding vector (2048,)

**Why embeddings?**
- Converts images to vectors we can compare mathematically
- Similar people → similar vectors
- Invariant to small changes (lighting, pose)

**Embedding models compared**:

| Model | Pros | Cons |
|-------|------|------|
| **ResNet50-ImageNet** | Fast, pre-trained, no setup | Not trained for person ReID |
| **CLIP** | Robust to appearance changes | Slower, larger model |
| **ArcFace** | Best for face ReID | Only works for faces (not full body) |
| **Torchreid** | Purpose-built for person ReID | Requires installation, more setup |

**Current choice**: ResNet50 for simplicity. Can upgrade to Torchreid if accuracy insufficient.

---

### Stage 4: Reidentification
**Goal**: Match embeddings across frames to track unique people

**Method**: Cosine similarity clustering
- **Input**: All embeddings from all frames
- **Process**:
  1. For each new detection:
     - Compare embedding to all existing person tracks
     - Use cosine similarity: `sim = dot(emb1, emb2)`
     - If similarity > threshold (default 0.7), assign to existing person
     - Otherwise, create new person ID
  2. Use average of last 5 appearances for robust matching

**Similarity threshold tuning**:
- **0.5-0.6**: Aggressive matching (may merge different people)
- **0.7**: Balanced (recommended default)
- **0.8-0.9**: Conservative (may split same person)

**Challenges**:
- Appearance changes (lighting, camera angle)
- Occlusions (grappling, submissions)
- Similar-looking people (same gi color)

**Improvements**:
- Temporal smoothing (people don't teleport between frames)
- Kalman filtering for position tracking
- Multi-modal features (position + appearance + size)

---

### Stage 5: Person Counting
**Goal**: Generate final unique person count

**Method**: Count unique person IDs
- **Process**:
  - Count number of person tracks
  - Calculate confidence based on appearance frequency
  - Generate summary report

**Confidence calculation**:
```
confidence = min(appearances / total_frames * 2, 1.0)
```

- Person appears in 50%+ frames → confidence = 1.0
- Person appears briefly → lower confidence

**Validation**:
- Compare against manual ground truth counts
- Check if person IDs are consistent across frames

---

## Key Challenges in MMA/BJJ

### 1. Overlapping People
MMA involves close contact, making segmentation hard.

**Solution**: SAM2 handles overlapping objects better than bounding boxes

### 2. Pose Variations
Fighters change poses rapidly (standing, ground, submissions)

**Solution**: ResNet50 embeddings somewhat robust to pose. Consider pose-invariant ReID models.

### 3. Occlusions
People partially hidden by opponent or camera angle

**Solution**: Track through occlusions using temporal smoothing

### 4. Similar Appearance
Fighters wear similar gi/gear

**Solution**: Use position tracking + appearance. People don't teleport.

### 5. Lighting Changes
Video quality varies, lighting changes

**Solution**: Normalized embeddings help. Data augmentation during embedding extraction.

---

## Performance Benchmarks

### Expected Performance (on EC2 with GPU)

| Stage | Time (per video) | Notes |
|-------|------------------|-------|
| 1. Extract frames | 10-30s | Fast, CPU-bound |
| 2. Segment (SAM2) | 5-15 min | GPU helps significantly |
| 3. Embeddings | 1-3 min | GPU helps |
| 4. ReID | 1-10s | CPU, depends on # detections |
| 5. Count | <1s | CPU |
| **Total** | **7-20 min** | Depends on video length, FPS |

### Accuracy Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Person count accuracy | 95%+ | Correct unique count |
| Identity consistency | 85%+ | Same person = same ID |
| False positives | <5% | Non-existent people |
| False negatives | <10% | Missed people |

---

## Evaluation Strategy

### Test Videos
Use existing BJJ videos with known ground truth:

| Video | People | Duration | Notes |
|-------|--------|----------|-------|
| ryan-thomas | 2 | 6 min | Clear footage, 2 fighters |
| columba | 2 | 1 min | Shorter, good test case |
| gio-thomas | 2 | 3 min | Medium length |

### Validation Checklist
- [ ] Correct person count (2 people in all test videos)
- [ ] Consistent person IDs across frames
- [ ] No false person detections
- [ ] Both people tracked throughout video
- [ ] Processing time < 20 min per video

### Error Analysis
When errors occur, check:
1. **Segmentation quality**: Are people correctly segmented?
2. **Embedding similarity**: What's the similarity between same/different people?
3. **Threshold tuning**: Would different threshold help?
4. **Temporal gaps**: Are there frames where people disappear?

---

## Future Improvements

### Week 1 (MVP)
- [x] Basic pipeline working
- [x] SAM2 segmentation
- [x] ResNet50 embeddings
- [x] Simple cosine similarity matching

### Week 2 (Accuracy)
- [ ] Tune similarity threshold per video
- [ ] Add temporal smoothing
- [ ] Better embedding model (Torchreid)
- [ ] Handle occlusions

### Week 3 (Production)
- [ ] Optimize speed (batching, GPU)
- [ ] Integration with existing BJJ pipeline
- [ ] Web UI for visualization
- [ ] Deploy to Lambda/Docker

---

## References

- **SAM2**: https://github.com/facebookresearch/segment-anything-2
- **Person ReID**: https://github.com/KaiyangZhou/deep-person-reid
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity
- **Existing person detection**: `/home/ubuntu/clann/CLANNAI/video-editor/detection/person_detect_api.py`

