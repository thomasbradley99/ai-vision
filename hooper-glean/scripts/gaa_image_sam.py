#!/usr/bin/env python3
"""
GAA Image Segmentation - SAM2 Single Image Mode

Creates awesome segmentation overlays for a single GAA image.
Much simpler than video - uses SAM2 image predictor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hooper.detectron_utils import get_detectron2_skeleton_model

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None

SAM2_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "SAM2-InstanceSegmentation" / "sam2.1_hiera_large.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAA image segmentation with SAM2")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-person-score", type=float, default=0.5, help="Min detection confidence")
    parser.add_argument("--overlay-opacity", type=float, default=0.4, help="Overlay opacity 0-1")
    parser.add_argument("--sam2-config", type=str, default=SAM2_DEFAULT_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=SAM2_DEFAULT_CHECKPOINT)
    return parser.parse_args()


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent color for a track ID."""
    rng = np.random.default_rng(seed=track_id)
    return tuple(rng.integers(low=64, high=256, size=3, dtype=np.int32))


def main():
    args = parse_args()
    
    print("🎨 GAA Image Segmentation")
    print("=" * 50)
    
    # Load image
    print(f"\n📸 Loading image: {args.image}")
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]
    print(f"   Size: {width}x{height}")
    
    # Load Detectron2 for person detection
    print("\n👥 Detecting players...")
    predictor, _ = get_detectron2_skeleton_model(device=args.device, batch_mode=False)
    predictions = predictor(image_bgr)
    
    instances = predictions["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    keypoints = instances.pred_keypoints.numpy() if instances.has("pred_keypoints") else None
    
    # Filter by score
    keep = scores >= args.min_person_score
    boxes = boxes[keep]
    scores = scores[keep]
    if keypoints is not None:
        keypoints = keypoints[keep]
    
    print(f"   Found {len(boxes)} players")
    
    if len(boxes) == 0:
        print("❌ No players detected!")
        return
    
    # Load SAM2 image predictor
    if build_sam2 is None or SAM2ImagePredictor is None:
        raise RuntimeError("SAM2 not installed! Install segment-anything-2")
    
    if not args.sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
    
    print("\n🎯 Loading SAM2 (image mode - way faster!)...")
    sam2_model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=args.device)
    sam_predictor = SAM2ImagePredictor(sam2_model)
    sam_predictor.set_image(image_rgb)
    print("✅ SAM2 ready!")
    
    # Segment each player with SAM2
    print(f"\n✨ Segmenting {len(boxes)} players with SAM2...")
    all_masks = []
    all_colors = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        box_array = np.array([x1, y1, x2, y2])
        
        # Use bounding box as prompt
        masks, scores_sam, _ = sam_predictor.predict(
            box=box_array,
            multimask_output=False,
        )
        
        # Take best mask
        best_mask = masks[0]
        all_masks.append(best_mask)
        all_colors.append(id_to_color(i + 1))
        
        print(f"   Player {i+1}: segmented ✓")
    
    # Render overlay
    print("\n🎬 Rendering overlay...")
    overlay = np.zeros_like(image_bgr)
    
    for mask, color in zip(all_masks, all_colors):
        mask_bool = mask.astype(bool)
        color_bgr = np.array([color[2], color[1], color[0]], dtype=np.uint8)  # RGB to BGR
        overlay[mask_bool] = color_bgr
    
    # Blend overlay with original
    result = cv2.addWeighted(
        image_bgr, 1.0 - args.overlay_opacity,
        overlay, args.overlay_opacity,
        0
    )
    
    # Draw labels
    for i, (box, color) in enumerate(zip(boxes, all_colors)):
        x1, y1, x2, y2 = box.astype(int)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # RGB to BGR
        
        label = f"Player {i+1}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Background for text
        cv2.rectangle(
            result,
            (x1, y1 - text_height - 12),
            (x1 + text_width + 6, y1),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            result, label,
            (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            color_bgr, 2,
            lineType=cv2.LINE_AA
        )
    
    # Save result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result)
    
    print("\n" + "=" * 50)
    print("🎉 DONE!")
    print(f"   📸 Output: {args.output}")
    print(f"   👥 Players: {len(boxes)}")
    print("=" * 50)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

