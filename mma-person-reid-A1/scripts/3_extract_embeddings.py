#!/usr/bin/env python3
"""
Stage 3: Extract person embeddings for reidentification

Usage:
    python 3_extract_embeddings.py videos/test-video-1
    
Requires:
    - Segmentation masks from Stage 2
    
Output:
    videos/test-video-1/outputs/embeddings.pkl
    Contains: {frame_id: [(embedding_vector, bbox, mask_id), ...]}
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


class PersonEmbeddingExtractor:
    """
    Extract person embeddings using ResNet50 pretrained on ImageNet
    For better ReID, consider using torchreid or ArcFace models
    """
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading embedding model on {self.device}...")
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        
        # Remove classification layer to get embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = weights.transforms()
        
        print("✓ Model loaded")
    
    def extract_embedding(self, person_crop):
        """
        Extract embedding vector from person crop
        
        Args:
            person_crop: (H, W, 3) BGR image
            
        Returns:
            embedding: (2048,) numpy array
        """
        # Convert BGR to RGB
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 if needed
        if person_crop.shape[0] < 32 or person_crop.shape[1] < 32:
            # Skip very small crops
            return None
        
        # Apply transforms
        img_tensor = self.transform(person_rgb).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        # Convert to numpy and flatten
        embedding = embedding.cpu().numpy().flatten()
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding


def extract_person_crop(frame, mask, box, padding=0.1):
    """
    Extract person crop from frame using mask and box
    
    Args:
        frame: Full frame image
        mask: Binary mask (H, W)
        box: Bounding box [x1, y1, x2, y2]
        padding: Padding ratio around box
        
    Returns:
        person_crop: Cropped person image
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # Add padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    # Crop frame
    crop = frame[y1:y2, x1:x2].copy()
    
    # Optionally apply mask (set background to black)
    # mask_crop = mask[y1:y2, x1:x2]
    # crop[~mask_crop] = 0
    
    return crop


def extract_all_embeddings(frames_dir, seg_dir, output_path):
    """
    Extract embeddings for all detected persons
    """
    frames_dir = Path(frames_dir)
    seg_dir = Path(seg_dir)
    
    # Initialize embedding extractor
    extractor = PersonEmbeddingExtractor()
    
    # Get all frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    
    if not frame_files:
        print(f"ERROR: No frames found in {frames_dir}")
        sys.exit(1)
    
    print(f"Extracting embeddings from {len(frame_files)} frames...")
    
    all_embeddings = {}
    
    for frame_path in tqdm(frame_files, desc="Extracting embeddings"):
        frame_name = frame_path.stem
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Load segmentation data
        mask_path = seg_dir / f"{frame_name}_masks.npz"
        boxes_path = seg_dir / f"{frame_name}_boxes.json"
        
        if not mask_path.exists() or not boxes_path.exists():
            print(f"Warning: Segmentation data missing for {frame_name}")
            continue
        
        # Load masks and boxes
        masks_data = np.load(mask_path)
        masks = masks_data['masks']
        
        with open(boxes_path, 'r') as f:
            boxes_data = json.load(f)
        boxes = boxes_data['boxes']
        
        # Extract embeddings for each person
        frame_embeddings = []
        
        for mask_id, (mask, box) in enumerate(zip(masks, boxes)):
            # Extract person crop
            person_crop = extract_person_crop(frame, mask, box)
            
            # Extract embedding
            embedding = extractor.extract_embedding(person_crop)
            
            if embedding is not None:
                frame_embeddings.append({
                    'embedding': embedding,
                    'box': box,
                    'mask_id': mask_id
                })
        
        all_embeddings[frame_name] = frame_embeddings
    
    # Save embeddings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_embeddings, f)
    
    # Calculate statistics
    total_detections = sum(len(embs) for embs in all_embeddings.values())
    avg_per_frame = total_detections / len(all_embeddings) if all_embeddings else 0
    
    print(f"✓ Extracted {total_detections} embeddings from {len(all_embeddings)} frames")
    print(f"  Average: {avg_per_frame:.1f} persons per frame")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract person embeddings")
    parser.add_argument("video_dir", help="Video directory (e.g., videos/test-video-1)")
    
    args = parser.parse_args()
    
    # Construct paths
    video_dir = Path(args.video_dir)
    frames_dir = video_dir / "outputs" / "frames"
    seg_dir = video_dir / "outputs" / "segmentations"
    output_path = video_dir / "outputs" / "embeddings.pkl"
    
    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        print("Run 1_extract_frames.py first!")
        sys.exit(1)
    
    if not seg_dir.exists():
        print(f"ERROR: Segmentations directory not found: {seg_dir}")
        print("Run 2_segment_persons.py first!")
        sys.exit(1)
    
    # Extract embeddings
    try:
        extract_all_embeddings(frames_dir, seg_dir, output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

