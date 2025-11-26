#!/bin/bash
# Run improved individual re-ID on 2-second clip with GPU optimizations

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate hooper-ai

cd /home/ubuntu/ai-vision/gaa-vision-ai

python scripts/track_individual_improved.py \
    --video inputs/test_2s.mp4 \
    --reference-image inputs/reference_frame.jpg \
    --output outputs/test_2s_all_people.mp4 \
    --match-output outputs/test_2s_just_you.mp4 \
    --device cuda \
    --batch-size 32 \
    --frame-stride 1 \
    --similarity-threshold 0.6 \
    --adaptive-threshold \
    2>&1 | tee outputs/test_2s_gpu_run.log

echo "Done! Check outputs/ directory for results"


