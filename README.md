# AI Vision - GAA Player Detection

Repository for GAA (Gaelic Football) player detection and tracking experiments.

## Structure

```
ai-vision/
├── inputs/                   # Test videos
│   ├── gaa_full.mp4         # Full 3.5min source (146MB)
│   ├── gaa_2s.mp4           # 2 second test clip (seconds 10-12)
│   ├── gaa_10s.mp4          # 10 second test clip
│   ├── gaa_30s.mp4          # 30 second test clip
│   ├── gaa_1min.mp4         # 1 minute test clip
│   └── gaa_2min.mp4         # 2 minute test clip
│
├── outputs/                  # Processed results
│   └── gaa_10s_example/     # Example successful output
│
├── hooper-glean/            # BJJ/GAA detection pipeline (Detectron2 + SAM2 + Re-ID)
├── Football-Object-Detection/ # Simple YOLO-based approach
├── player-counting-sam2/    # SAM2 player counting experiments
└── mma-person-reid-A1/      # Person re-identification experiments
```

## Test Videos

All clips from "The Magic of Gaelic Football" YouTube compilation:
- **Full video:** 212 seconds, 1920x1080, 60fps
- **Challenge:** Multiple matches with different teams/colors
- Test clips range from 10s to 2min for different approaches

## Approaches Tested

1. **hooper-glean pipeline** (scripts/gaa_identify.py)
   - Detectron2 for person detection
   - Solider for re-identification 
   - SAM2 for segmentation (optional)
   - Works: 10s clips (19MB output)
   - Fails: 1min+ clips (OOM errors)

2. **Football-Object-Detection**
   - Simple YOLO detection
   - Color-based team classification
   - Problem: Assumes same teams throughout video

3. **Others:** Various SAM2 and re-ID experiments

## Current Goal

Create a hero video for website with clean player detection. Need to find an approach that:
- Handles scene transitions between different matches
- Processes efficiently without OOM errors
- Produces clean visual output
