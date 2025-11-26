#!/bin/bash
# Run improved individual re-ID on 5-minute clip with GPU optimizations
# Expected runtime: ~15-20 minutes (safe for 30 min window)

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate hooper-ai

cd /home/ubuntu/ai-vision/gaa-vision-ai

# Update these paths to your video and reference image
VIDEO="inputs/your_video_5min.mp4"
REFERENCE="inputs/reference_frame.jpg"
OUTPUT_ALL="outputs/5min_all_people.mp4"
OUTPUT_MATCH="outputs/5min_just_you.mp4"

python scripts/track_individual_improved.py \
    --video "$VIDEO" \
    --reference-image "$REFERENCE" \
    --output "$OUTPUT_ALL" \
    --match-output "$OUTPUT_MATCH" \
    --device cuda \
    --batch-size 32 \
    --frame-stride 2 \
    --similarity-threshold 0.6 \
    --adaptive-threshold \
    2>&1 | tee outputs/5min_gpu_run.log

echo "Done! Check outputs/ directory for results"


