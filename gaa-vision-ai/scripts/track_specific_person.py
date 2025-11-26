#!/usr/bin/env python3
"""
Track a specific person throughout a long video (e.g., 30-minute GAA session).

This script uses person re-identification (Re-ID) to track one individual
across the entire video, even when they leave and re-enter the frame.

Usage:
    python track_specific_person.py \
        --video /path/to/30min_session.mp4 \
        --reference-image /path/to/your_photo.jpg \
        --output /path/to/tracked_output.mp4 \
        --similarity-threshold 0.7
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add hooper-glean to path for Re-ID model
REPO_ROOT = Path(__file__).resolve().parents[2]
HOOPER_PATH = REPO_ROOT / "hooper-glean"
if str(HOOPER_PATH) not in sys.path:
    sys.path.insert(0, str(HOOPER_PATH))

from hooper.detectron_utils import get_detectron2_skeleton_model
from hooper.solider_utils import get_solider_feature_extractor


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    # Normalize embeddings
    emb1_norm = emb1 / (torch.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (torch.norm(emb2) + 1e-8)
    similarity = torch.dot(emb1_norm, emb2_norm).item()
    # Convert from [-1, 1] to [0, 1]
    return (similarity + 1) / 2


def load_reference_embedding(reference_image_path: Path, extractor) -> torch.Tensor:
    """Load reference image and extract embedding."""
    print(f"[Reference] Loading reference image: {reference_image_path}")
    
    # Read image
    img = cv2.imread(str(reference_image_path))
    if img is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract embedding
    with torch.no_grad():
        embedding = extractor(img_rgb)
        embedding = torch.nn.functional.normalize(embedding.mean(0), dim=0)
    
    print(f"[Reference] Extracted embedding vector (dim={embedding.shape[0]})")
    return embedding


def process_batch_with_reid(
    frames: List[np.ndarray],
    frame_indices: List[int],
    predictor,
    extractor,
    reference_embedding: torch.Tensor,
    similarity_threshold: float,
    tracks: List[Dict],
    next_track_id: List[int],
    score_threshold: float,
    iou_threshold: float,
    max_age: int,
    device: str,
):
    """Process batch of frames and match detections to reference person."""
    predictions = predictor(frames)
    
    for frame, frame_idx, pred in zip(frames, frame_indices, predictions):
        instances = pred["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        
        if len(boxes) == 0:
            continue
        
        # Filter by score
        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        if len(boxes) == 0:
            continue
        
        # Sort by confidence
        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]
        
        # Extract crops and compute embeddings
        height, width = frame.shape[:2]
        crops = []
        valid_boxes = []
        valid_scores = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, x1 + 1, width))
            y2 = int(np.clip(y2, y1 + 1, height))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            valid_boxes.append([x1, y1, x2, y2])
            valid_scores.append(scores[len(valid_boxes) - 1])
        
        if not crops:
            continue
        
        # Compute embeddings for all crops
        with torch.no_grad():
            crop_embeddings = extractor(crops)
            crop_embeddings = torch.nn.functional.normalize(crop_embeddings, dim=1)
        
        # Match against reference
        similarities = []
        for emb in crop_embeddings:
            sim = cosine_similarity(emb, reference_embedding)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        matches = similarities >= similarity_threshold
        
        # Track assignment for matched detections
        active_track_indices = [
            idx for idx, tr in enumerate(tracks)
            if (frame_idx - tr["last_frame"]) <= max_age
        ]
        matched_tracks = set()
        assignments: List[int] = [-1] * len(valid_boxes)
        
        for det_idx, (box, is_match) in enumerate(zip(valid_boxes, matches)):
            if not is_match:
                continue  # Skip non-matching detections
            
            best_iou = 0.0
            best_track_idx = -1
            for track_idx in active_track_indices:
                if track_idx in matched_tracks:
                    continue
                iou_val = compute_iou(np.array(box), tracks[track_idx]["last_box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track_idx = track_idx
            
            if best_track_idx >= 0 and best_iou >= iou_threshold:
                assignments[det_idx] = best_track_idx
                matched_tracks.add(best_track_idx)
        
        # Update or create tracks
        for det_idx, (box, score, similarity, is_match) in enumerate(
            zip(valid_boxes, valid_scores, similarities, matches)
        ):
            if not is_match:
                continue
            
            track_idx = assignments[det_idx]
            if track_idx == -1:
                # Create new track
                track_idx = len(tracks)
                tracks.append({
                    "id": next_track_id[0],
                    "boxes": [],
                    "frames": [],
                    "scores": [],
                    "similarities": [],
                    "last_box": None,
                    "last_frame": -1,
                })
                next_track_id[0] += 1
            
            # Update track
            track = tracks[track_idx]
            track["last_box"] = np.array(box, dtype=np.float32)
            track["last_frame"] = frame_idx
            track["boxes"].append(box)
            track["frames"].append(frame_idx)
            track["scores"].append(float(score))
            track["similarities"].append(float(similarity))


def detect_and_track_person(
    video_path: Path,
    predictor,
    extractor,
    reference_embedding: torch.Tensor,
    similarity_threshold: float,
    frame_stride: int,
    batch_size: int,
    score_threshold: float,
    iou_threshold: float,
    max_age: int,
    device: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Detect and track the reference person throughout the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracks: List[Dict] = []
    next_track_id = [0]
    frames: List[np.ndarray] = []
    indices: List[int] = []
    frame_idx = 0
    
    print(f"[Tracking] Processing {total_frames} frames at {fps} FPS")
    print(f"[Tracking] Frame stride: {frame_stride}, Batch size: {batch_size}")
    print(f"[Tracking] Similarity threshold: {similarity_threshold}")
    
    with tqdm(total=total_frames, desc="Tracking person") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_stride == 0:
                frames.append(frame)
                indices.append(frame_idx)
                
                if len(frames) == batch_size:
                    process_batch_with_reid(
                        frames,
                        indices,
                        predictor,
                        extractor,
                        reference_embedding,
                        similarity_threshold,
                        tracks,
                        next_track_id,
                        score_threshold,
                        iou_threshold,
                        max_age,
                        device,
                    )
                    frames.clear()
                    indices.clear()
            
            frame_idx += 1
            pbar.update(1)
    
    # Process remaining frames
    if frames:
        process_batch_with_reid(
            frames,
            indices,
            predictor,
            extractor,
            reference_embedding,
            similarity_threshold,
            tracks,
            next_track_id,
            score_threshold,
            iou_threshold,
            max_age,
            device,
        )
    
    cap.release()
    
    metadata = {"fps": fps, "width": width, "height": height, "total_frames": total_frames}
    return tracks, metadata


def render_tracked_video(
    video_path: Path,
    tracks: List[Dict],
    output_path: Path,
    metadata: Dict,
    highlight_color: Tuple[int, int, int] = (0, 255, 0),  # Green
):
    """Render video with tracked person highlighted."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build frame map
    frame_map: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    for track in tracks:
        frames = track.get("frames", [])
        boxes = track.get("boxes", [])
        similarities = track.get("similarities", [])
        
        for frame_idx, box, sim in zip(frames, boxes, similarities):
            frame_map[int(frame_idx)].append((np.array(box), sim))
    
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata["fps"],
        (int(metadata["width"]), int(metadata["height"])),
    )
    
    frame_idx = 0
    total_frames = int(metadata["total_frames"])
    
    with tqdm(total=total_frames, desc="Rendering video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw boxes for this frame
            if frame_idx in frame_map:
                for box, similarity in frame_map[frame_idx]:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), highlight_color, 3)
                    
                    # Draw label with similarity score
                    label = f"You ({similarity:.2f})"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        highlight_color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )
            
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    writer.release()
    
    print(f"[Render] Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Track a specific person throughout a video using Re-ID"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        required=True,
        help="Path to reference image of the person to track",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output video",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for matching (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Process every Nth frame (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for detection (default: 16)",
    )
    parser.add_argument(
        "--min-person-score",
        type=float,
        default=0.6,
        help="Minimum confidence for person detection (default: 0.6)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for track association (default: 0.3)",
    )
    parser.add_argument(
        "--max-track-gap",
        type=int,
        default=60,
        help="Max frames before track is considered stale (default: 60)",
    )
    parser.add_argument(
        "--reid-weights",
        type=Path,
        default=None,
        help="Path to Re-ID model weights (auto-detected if not provided)",
    )
    
    args = parser.parse_args()
    
    # Auto-detect Re-ID weights if not provided
    if args.reid_weights is None:
        checkpoints_dir = HOOPER_PATH / "checkpoints"
        reid_weights = checkpoints_dir / "PERSON-Tracking" / "swin_base_msmt17.pth"
        if not reid_weights.exists():
            print(f"ERROR: Re-ID weights not found at {reid_weights}")
            print("Please provide --reid-weights or ensure weights are in checkpoints/")
            sys.exit(1)
        args.reid_weights = reid_weights
    
    print("=" * 60)
    print("Person Tracking with Re-ID")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Reference: {args.reference_image}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load models
    print("\n[Loading] Loading detection model...")
    predictor, _ = get_detectron2_skeleton_model(device=args.device, batch_mode=True)
    
    print("[Loading] Loading Re-ID model...")
    extractor = get_solider_feature_extractor(str(args.reid_weights), device=args.device)
    
    # Load reference embedding
    reference_embedding = load_reference_embedding(args.reference_image, extractor)
    
    # Detect and track
    print("\n[Tracking] Starting person detection and tracking...")
    tracks, metadata = detect_and_track_person(
        args.video,
        predictor,
        extractor,
        reference_embedding,
        args.similarity_threshold,
        args.frame_stride,
        args.batch_size,
        args.min_person_score,
        args.iou_threshold,
        args.max_track_gap,
        args.device,
    )
    
    if not tracks:
        print("\n❌ No tracks found! Try:")
        print("  - Lower --similarity-threshold (e.g., 0.6)")
        print("  - Use a clearer reference image")
        print("  - Check that the person appears in the video")
        sys.exit(1)
    
    # Print statistics
    print(f"\n[Stats] Found {len(tracks)} track(s)")
    total_detections = sum(len(t["frames"]) for t in tracks)
    avg_similarity = np.mean([np.mean(t["similarities"]) for t in tracks])
    print(f"[Stats] Total detections: {total_detections}")
    print(f"[Stats] Average similarity: {avg_similarity:.3f}")
    
    # Render video
    print("\n[Rendering] Creating output video...")
    render_tracked_video(args.video, tracks, args.output, metadata)
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print(f"   Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

