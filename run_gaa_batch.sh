#!/bin/bash
# GAA Player Detection - Batch Processing with Full SAM2 Segmentation
# Runs 10s, 30s, 1min, and 2min clips overnight

set -e

# Activate conda environment
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate hooper-ai

# Base paths
BASE_DIR="/home/ubuntu/clann/ai-vision"
INPUT_DIR="${BASE_DIR}/inputs"
OUTPUT_DIR="${BASE_DIR}/outputs"
SCRIPT_DIR="${BASE_DIR}/hooper-glean/scripts"
LOG_FILE="${OUTPUT_DIR}/gaa_batch_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "${LOG_FILE}"
echo "GAA Batch Processing Started" | tee -a "${LOG_FILE}"
echo "Start Time: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Array of clips to process
declare -a clips=("gaa_10s" "gaa_30s" "gaa_1min" "gaa_2min")
declare -a durations=("10s" "30s" "1min" "2min")

# Process each clip
for i in "${!clips[@]}"; do
    clip="${clips[$i]}"
    duration="${durations[$i]}"
    
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Processing: ${clip} (${duration})" | tee -a "${LOG_FILE}"
    echo "Start: $(date)" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    start_time=$(date +%s)
    
    # Run the detection pipeline with re-ID (no SAM2 to avoid OOM)
    cd "${SCRIPT_DIR}" && \
    python gaa_identify.py \
        --video "${INPUT_DIR}/${clip}.mp4" \
        --out-dir "${OUTPUT_DIR}/${clip}_reid_full" \
        --batch-size 16 \
        --min-track-length 3 \
        --keep-top-k 15 \
        2>&1 | tee -a "${LOG_FILE}"
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    echo "" | tee -a "${LOG_FILE}"
    echo "✓ Completed: ${clip}" | tee -a "${LOG_FILE}"
    echo "Duration: ${elapsed}s ($(date -u -d @${elapsed} +%H:%M:%S))" | tee -a "${LOG_FILE}"
    echo "End: $(date)" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "All Processing Complete!" | tee -a "${LOG_FILE}"
echo "End Time: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Output videos:" | tee -a "${LOG_FILE}"
for clip in "${clips[@]}"; do
    output_path="${OUTPUT_DIR}/${clip}_reid_full/player_overlay.mp4"
    if [ -f "${output_path}" ]; then
        size=$(du -h "${output_path}" | cut -f1)
        echo "  - ${clip}: ${size}" | tee -a "${LOG_FILE}"
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

