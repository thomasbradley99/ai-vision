# AI Vision

A collection of AI vision projects for sports analytics, focusing on object detection, tracking, segmentation, and re-identification for various sports including GAA, BJJ, MMA, and football.

## Projects

### hooper-glean
Sports analytics pipeline adapted from the Hooper BJJ system. Includes:
- Object detection and tracking for athletes
- Keypoint detection for pose analysis
- SAM2-powered segmentation
- Person re-identification
- Scripts for GAA player detection and analysis

**Key Scripts:**
- `scripts/gaa_identify.py` - GAA player detection and tracking with crowd filtering
- `scripts/bjj_identify.py` - BJJ athlete tracking and analysis
- `scripts/gaa_sam.py` - SAM2-powered segmentation for GAA

### player-counting-sam2
Player counting and segmentation system using SAM2 (Segment Anything Model 2).
- Frame extraction from video
- Person detection and segmentation
- Player tracking across frames
- Counting unique players

### mma-person-reid-A1
MMA person re-identification pipeline for tracking fighters across different camera angles and time periods.
- Frame extraction
- Person segmentation
- Feature embedding extraction
- Re-identification across video segments
- People counting

### Football-Object-Detection
Football/soccer object detection system (external project from [Mostafa-Nafie](https://github.com/Mostafa-Nafie/Football-Object-Detection)).

## Setup

Each project has its own dependencies and setup requirements. Refer to individual project READMEs:
- `hooper-glean/README.md`
- `player-counting-sam2/QUICKSTART.md`
- `mma-person-reid-A1/QUICKSTART.md`
- `Football-Object-Detection/README.md`

## General Requirements

Common dependencies across projects:
- Python 3.8+
- PyTorch with CUDA support
- OpenCV
- Detectron2
- SAM2 (Segment Anything Model 2)

## Usage

### GAA Player Detection Example

```bash
cd hooper-glean
python scripts/gaa_identify.py \
    --video path/to/gaa_video.mp4 \
    --output-dir outputs/gaa_analysis \
    --batch-size 16 \
    --min-track-length 3
```

### BJJ Analysis Example

```bash
cd hooper-glean
python scripts/bjj_identify.py \
    --video path/to/bjj_video.mp4 \
    --output-dir outputs/bjj_analysis
```

## Features

- Real-time object detection and tracking
- GPU-accelerated inference with batching
- SAM2 segmentation for high-quality masks
- Crowd filtering and athlete classification
- Person re-identification embeddings
- Multi-sport support (GAA, BJJ, MMA, Football)

## Repository Structure

```
ai-vision/
├── hooper-glean/          # Main sports analytics pipeline
├── player-counting-sam2/  # SAM2-based player counting
├── mma-person-reid-A1/    # MMA re-identification
├── Football-Object-Detection/  # Football detection (submodule)
├── outputs/               # Generated outputs (excluded from git)
└── README.md
```

## Notes

- Large video files, model checkpoints, and outputs are excluded from version control
- The `Football-Object-Detection` project is included as a git submodule
- GPU with adequate VRAM (8GB+) recommended for optimal performance

## License

Each project may have its own license. Please refer to individual project directories for details.



