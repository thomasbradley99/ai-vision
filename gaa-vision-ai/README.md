# GAA Vision AI

Clean, production-ready pipeline for GAA player detection, tracking, and team separation.

## What It Does

1. **YOLO Detection** - Detects players using football-trained model
2. **ByteTrack** - Tracks each player across frames with stable IDs
3. **Smart Team Clustering** - Uses K-Means on jersey colors, assigned PER TRACK (no flickering)
4. **Future: SAM2 Masks** - Will add pixel-perfect overlays instead of boxes

## Key Innovation

**Problem:** Simple per-frame color matching causes flickering (player switches teams frame-to-frame)

**Solution:** 
- **Pass 1:** Random sampling of N frames throughout video (default 50)
- Extract jersey colors from all players in sampled frames
- Run K-Means once to get 2 team centroids
- **Pass 2:** Assign each TRACK (not frame) to a team based on jersey color
- Lock in assignment = stable team colors throughout video

**Why random sampling?** 
- Better coverage of all players and lighting conditions
- More robust than just using first 30 frames
- Captures different camera angles and play situations

## Usage

```bash
cd /home/ubuntu/clann/ai-vision/gaa-vision-ai

conda activate hooper-ai

python scripts/gaa_team_tracker.py \
  --video /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --model /path/to/yolo_model.pt \
  --sample-frames 50
```

## Example

```bash
python scripts/gaa_team_tracker.py \
  --video ../inputs2/gaa12.mp4 \
  --output outputs/gaa12_tracked.mp4 \
  --model /home/ubuntu/clann/ai-vision/Football-Object-Detection/weights/best.pt \
  --sample-frames 50
```

## Output

- Team 0: Blue boxes
- Team 1: Red boxes
- Each player labeled with Team (e.g., "T0")
- Two-pass approach: random sampling first, then full video processing

## Next Steps

1. Test on single video
2. Fine-tune sample count if needed (increase for more robust clustering)
3. Add SAM2 segmentation masks
4. Batch process all clips
5. Stitch into hero video

