#!/usr/bin/env python3
"""
Stage 2: Segment persons using SAM2

Usage:
    python 2_segment_persons.py videos/test-video-1
    
Requires:
    - SAM2 installed: pip install git+https://github.com/facebookresearch/segment-anything-2.git
    - Frames extracted from Stage 1
    
Output:
    videos/test-video-1/outputs/segmentations/frame_0000_masks.npz
    videos/test-video-1/outputs/segmentations/frame_0000_boxes.json
    videos/test-video-1/outputs/visualizations/frame_0000_vis.jpg
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm
import torch

# Try to import SAM2 - provide helpful error if not installed
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("ERROR: SAM2 not installed!")
    print("\nInstall SAM2 with:")
    print("  cd scripts")
    print("  git clone https://github.com/facebookresearch/segment-anything-2.git")
    print("  cd segment-anything-2")
    print("  pip install -e .")
    print("\nThen download checkpoints:")
    print("  cd checkpoints")
    print("  ./download_ckpts.sh")
    sys.exit(1)


def detect_persons_yolo(frame):
    """
    Detect person bounding boxes using YOLO
    This provides initial prompts for SAM2
    
    Returns:
        List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    # TODO: Implement YOLO person detection
    # For now, return empty list (SAM2 can work without prompts)
    # In production, use YOLOv8 or similar to get person boxes
    
    # Placeholder: Detect using simple method or SAM2's automatic mode
    return []


def segment_frame_sam2(predictor, frame, person_boxes=None):
    """
    Segment persons in a frame using SAM2
    
    Args:
        predictor: SAM2ImagePredictor instance
        frame: Input frame (numpy array)
        person_boxes: Optional list of person bounding boxes for prompts
        
    Returns:
        masks: List of segmentation masks (H, W) bool arrays
        boxes: List of bounding boxes
        scores: Confidence scores
    """
    # Set image in predictor
    predictor.set_image(frame)
    
    # If no boxes provided, use automatic mask generation
    if not person_boxes:
        # Use SAM2's automatic mask generation
        # This generates masks for all objects in the image
        masks, scores, boxes = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )
        
        # Filter for person-like masks (large, roughly vertical)
        person_masks = []
        person_boxes = []
        person_scores = []
        
        for mask, score, box in zip(masks, scores, boxes):
            # Simple heuristic: person masks are usually large and vertical
            area = mask.sum()
            h, w = frame.shape[:2]
            
            # Filter by size (persons are usually > 5% of frame)
            if area > (h * w * 0.05):
                person_masks.append(mask)
                person_boxes.append(box)
                person_scores.append(score)
        
        return person_masks, person_boxes, person_scores
    
    else:
        # Use provided boxes as prompts
        all_masks = []
        all_boxes = []
        all_scores = []
        
        for box in person_boxes:
            masks, scores, _ = predictor.predict(
                box=np.array(box),
                multimask_output=False,
            )
            all_masks.extend(masks)
            all_boxes.append(box)
            all_scores.extend(scores)
        
        return all_masks, all_boxes, all_scores


def visualize_segmentation(frame, masks, boxes):
    """
    Visualize segmentation masks on frame
    """
    vis = frame.copy()
    
    # Generate random colors for each mask
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
    
    for mask, box, color in zip(masks, boxes, colors):
        # Draw mask overlay
        mask_overlay = np.zeros_like(vis)
        mask_overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color.tolist(), 2)
    
    return vis


def segment_all_frames(frames_dir, output_dir, checkpoint_path=None):
    """
    Segment persons in all frames
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    seg_dir = output_dir / "segmentations"
    vis_dir = output_dir / "visualizations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SAM2
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = "scripts/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: SAM2 checkpoint not found: {checkpoint_path}")
        print("\nDownload checkpoints with:")
        print("  cd scripts/segment-anything-2/checkpoints")
        print("  ./download_ckpts.sh")
        sys.exit(1)
    
    # Build SAM2 model
    sam2_model = build_sam2("sam2_hiera_l.yaml", checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    
    if not frame_files:
        print(f"ERROR: No frames found in {frames_dir}")
        print("Run 1_extract_frames.py first!")
        sys.exit(1)
    
    print(f"Processing {len(frame_files)} frames...")
    
    for frame_path in tqdm(frame_files, desc="Segmenting frames"):
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read {frame_path}")
            continue
        
        # Segment persons
        masks, boxes, scores = segment_frame_sam2(predictor, frame)
        
        # Save segmentation data
        frame_name = frame_path.stem
        
        # Save masks as compressed numpy array
        mask_path = seg_dir / f"{frame_name}_masks.npz"
        np.savez_compressed(mask_path, masks=np.array(masks))
        
        # Save boxes and scores as JSON
        boxes_path = seg_dir / f"{frame_name}_boxes.json"
        with open(boxes_path, 'w') as f:
            json.dump({
                "boxes": [box.tolist() for box in boxes],
                "scores": [float(s) for s in scores],
                "num_persons": len(boxes)
            }, f, indent=2)
        
        # Save visualization
        vis = visualize_segmentation(frame, masks, boxes)
        vis_path = vis_dir / f"{frame_name}_vis.jpg"
        cv2.imwrite(str(vis_path), vis)
    
    print(f"✓ Segmented {len(frame_files)} frames")
    print(f"  Masks: {seg_dir}")
    print(f"  Visualizations: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Segment persons using SAM2")
    parser.add_argument("video_dir", help="Video directory (e.g., videos/test-video-1)")
    parser.add_argument("--checkpoint", help="SAM2 checkpoint path", default=None)
    
    args = parser.parse_args()
    
    # Construct paths
    video_dir = Path(args.video_dir)
    frames_dir = video_dir / "outputs" / "frames"
    output_dir = video_dir / "outputs"
    
    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        print("Run 1_extract_frames.py first!")
        sys.exit(1)
    
    # Segment frames
    try:
        segment_all_frames(frames_dir, output_dir, checkpoint_path=args.checkpoint)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

