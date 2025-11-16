#!/bin/bash
# Run full person reidentification pipeline on a video

if [ $# -eq 0 ]; then
    echo "Usage: ./run_pipeline.sh <video_dir> [fps] [similarity_threshold] [ground_truth_count]"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline.sh videos/test-video-1"
    echo "  ./run_pipeline.sh videos/test-video-1 2 0.7 2"
    echo ""
    echo "Arguments:"
    echo "  video_dir:            Video directory (e.g., videos/test-video-1)"
    echo "  fps:                  Frames per second to extract (default: 2)"
    echo "  similarity_threshold: ReID similarity threshold (default: 0.7)"
    echo "  ground_truth_count:   Expected person count for validation (optional)"
    exit 1
fi

VIDEO_DIR=$1
FPS=${2:-2}
THRESHOLD=${3:-0.7}
GROUND_TRUTH=${4:-""}

echo "========================================"
echo "Person ReID Pipeline"
echo "========================================"
echo "Video: $VIDEO_DIR"
echo "FPS: $FPS"
echo "Similarity threshold: $THRESHOLD"
if [ -n "$GROUND_TRUTH" ]; then
    echo "Ground truth: $GROUND_TRUTH people"
fi
echo "========================================"
echo ""

# Check if video exists
if [ ! -d "$VIDEO_DIR/input" ]; then
    echo "ERROR: Video directory not found: $VIDEO_DIR"
    echo "Run ./setup_test_videos.sh first!"
    exit 1
fi

# Stage 1: Extract frames
echo "Stage 1/5: Extracting frames..."
python scripts/1_extract_frames.py "$VIDEO_DIR" --fps "$FPS"
if [ $? -ne 0 ]; then
    echo "ERROR: Frame extraction failed"
    exit 1
fi
echo ""

# Stage 2: Segment persons
echo "Stage 2/5: Segmenting persons with SAM2..."
python scripts/2_segment_persons.py "$VIDEO_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Segmentation failed"
    exit 1
fi
echo ""

# Stage 3: Extract embeddings
echo "Stage 3/5: Extracting embeddings..."
python scripts/3_extract_embeddings.py "$VIDEO_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Embedding extraction failed"
    exit 1
fi
echo ""

# Stage 4: Reidentify persons
echo "Stage 4/5: Reidentifying persons..."
python scripts/4_reidentify.py "$VIDEO_DIR" --threshold "$THRESHOLD"
if [ $? -ne 0 ]; then
    echo "ERROR: Reidentification failed"
    exit 1
fi
echo ""

# Stage 5: Count people
echo "Stage 5/5: Generating person count report..."
if [ -n "$GROUND_TRUTH" ]; then
    python scripts/5_count_people.py "$VIDEO_DIR" --ground-truth "$GROUND_TRUTH"
else
    python scripts/5_count_people.py "$VIDEO_DIR"
fi
if [ $? -ne 0 ]; then
    echo "ERROR: Report generation failed"
    exit 1
fi
echo ""

echo "========================================"
echo "✓ Pipeline complete!"
echo "========================================"
echo "Results:"
echo "  Person count: $VIDEO_DIR/outputs/person_count.json"
echo "  Person tracks: $VIDEO_DIR/outputs/person_tracks.json"
echo "  Visualizations: $VIDEO_DIR/outputs/visualizations/"
echo ""

