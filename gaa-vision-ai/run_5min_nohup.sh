#!/bin/bash
cd /home/ubuntu/ai-vision/gaa-vision-ai
nohup bash -c "source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate hooper-ai && cd /home/ubuntu/ai-vision/gaa-vision-ai && python scripts/track_individual_improved.py --video inputs/video_5min.mp4 --reference-image inputs/reference_frame_5min.jpg --output outputs/5min_all_people.mp4 --match-output outputs/5min_just_you.mp4 --device cuda --batch-size 32 --frame-stride 2 --similarity-threshold 0.6 --adaptive-threshold" > outputs/nohup_5min.log 2>&1 &
echo "Job started with PID: $!"
echo "Monitor: tail -f outputs/nohup_5min.log"

