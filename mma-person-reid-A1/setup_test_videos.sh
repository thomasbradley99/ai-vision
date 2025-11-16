#!/bin/bash
# Setup test videos from existing BJJ test data

echo "Setting up test videos for person reidentification..."

# Create video directories
mkdir -p videos/test-video-1/input
mkdir -p videos/test-video-2/input  
mkdir -p videos/test-video-3/input

# Link existing BJJ videos
echo "Linking ryan-thomas video..."
ln -sf /home/ubuntu/clann/clann-jujisu/bjj-ai-testing/videos/ryan-thomas/input/video.mov \
       videos/test-video-1/input/video.mov

echo "Linking columba video..."
ln -sf /home/ubuntu/clann/clann-jujisu/bjj-ai-testing/videos/columba/input/video.mov \
       videos/test-video-2/input/video.mov

echo "Linking gio-thomas video..."
ln -sf /home/ubuntu/clann/clann-jujisu/bjj-ai-testing/videos/gio-thomas/input/video.mov \
       videos/test-video-3/input/video.mov

# Create ground truth files
echo "Creating ground truth files..."

cat > videos/test-video-1/input/ground_truth.json << 'EOF'
{
  "video_name": "ryan-thomas",
  "people_count": 2,
  "people": [
    {
      "id": 1,
      "name": "Ryan",
      "description": "Black rashguard"
    },
    {
      "id": 2,
      "name": "Thomas",
      "description": "Green gi"
    }
  ]
}
EOF

cat > videos/test-video-2/input/ground_truth.json << 'EOF'
{
  "video_name": "columba",
  "people_count": 2,
  "people": [
    {
      "id": 1,
      "name": "Columba",
      "description": "Fighter 1"
    },
    {
      "id": 2,
      "name": "Opponent",
      "description": "Fighter 2"
    }
  ]
}
EOF

cat > videos/test-video-3/input/ground_truth.json << 'EOF'
{
  "video_name": "gio-thomas",
  "people_count": 2,
  "people": [
    {
      "id": 1,
      "name": "Gio",
      "description": "Fighter 1"
    },
    {
      "id": 2,
      "name": "Thomas",
      "description": "Fighter 2"
    }
  ]
}
EOF

echo ""
echo "✓ Test videos setup complete!"
echo ""
echo "Videos available:"
echo "  1. videos/test-video-1/ (ryan-thomas, 2 people)"
echo "  2. videos/test-video-2/ (columba, 2 people)"
echo "  3. videos/test-video-3/ (gio-thomas, 2 people)"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Install SAM2: see README.md"
echo "  3. Run pipeline: python scripts/1_extract_frames.py videos/test-video-1"
echo ""

