#!/bin/bash
# FULL NOHUP COMMAND FOR 5-MINUTE VIDEO
# Copy-paste this entire command

# First, create 5-minute clip from your source video (adjust path):
# ffmpeg -i /path/to/your/source_video.mp4 -t 300 -c copy inputs/video_5min.mp4

# Then run this nohup command:
nohup bash -c "source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate hooper-ai && cd /home/ubuntu/ai-vision/gaa-vision-ai && python scripts/track_individual_improved.py --video inputs/video_5min.mp4 --reference-image inputs/reference_frame.jpg --output outputs/5min_all_people.mp4 --match-output outputs/5min_just_you.mp4 --device cuda --batch-size 32 --frame-stride 2 --similarity-threshold 0.6 --adaptive-threshold" > outputs/nohup_5min.log 2>&1 &

echo "Job started! Process ID: $!"
echo "Monitor with: tail -f outputs/nohup_5min.log"
echo "Or check GPU: watch -n 2 nvidia-smi"


