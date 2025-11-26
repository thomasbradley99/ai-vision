#!/usr/bin/env python3
"""
Improved individual person re-identification with temporal smoothing.

This script fixes the problems with track_specific_person.py by:
1. Tracking ALL people first (no early filtering)
2. Using temporal smoothing (averaging embeddings across frames)
3. Adaptive thresholds based on track history
4. Better matching using track-level embeddings

Why track_specific_person.py doesn't work:
- Filters by similarity BEFORE tracking → breaks tracks when similarity temporarily drops
- No temporal smoothing → single-frame embeddings are noisy
- Fixed threshold → doesn't adapt to person's appearance variations

This improved version:
- Tracks everyone, then matches → more robust
- Uses averaged embeddings from multiple frames → more stable
- Adaptive thresholding → handles appearance variations better
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add hooper-glean to path
REPO_ROOT = Path(__file__).resolve().parents[2]
HOOPER_PATH = REPO_ROOT / "hooper-glean"
if str(HOOPER_PATH) not in sys.path:
    sys.path.insert(0, str(HOOPER_PATH))

from hooper.detectron_utils import get_detectron2_skeleton_model
from hooper.solider_utils import get_solider_feature_extractor

# Import functions from gaa_identify script
sys.path.insert(0, str(HOOPER_PATH / "scripts"))
from gaa_identify import (
    detect_and_track,
    filter_tracks,
    compute_embeddings,
    render_tracks,
)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    # Ensure both tensors are on the same device
    if emb1.device != emb2.device:
        emb2 = emb2.to(emb1.device)
    
    emb1_norm = emb1 / (torch.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (torch.norm(emb2) + 1e-8)
    similarity = torch.dot(emb1_norm, emb2_norm).item()
    return (similarity + 1) / 2  # Convert from [-1, 1] to [0, 1]


def load_reference_embedding(reference_image_path: Path, extractor) -> torch.Tensor:
    """Load reference image and extract embedding."""
    print(f"[Reference] Loading reference image: {reference_image_path}")
    
    img = cv2.imread(str(reference_image_path))
    if img is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        embedding = extractor(img_rgb)
        embedding = torch.nn.functional.normalize(embedding.mean(0), dim=0)
    
    print(f"[Reference] Extracted embedding vector (dim={embedding.shape[0]})")
    return embedding


def find_matching_tracks_with_adaptive_threshold(
    reference_embedding: torch.Tensor, 
    track_summaries: List[Dict], 
    base_threshold: float = 0.7,
    adaptive_factor: float = 0.15,  # Allow 15% lower threshold for long tracks
) -> Tuple[Optional[int], Dict]:
    """
    Find matching track with adaptive thresholding.
    
    Long tracks (>100 frames) get slightly lower threshold since they're more reliable.
    Also returns similarity scores for all tracks for analysis.
    """
    print(f"\n[Matching] Comparing reference to {len(track_summaries)} tracks...")
    
    track_similarities = {}
    best_match_id = None
    best_similarity = 0.0
    
    for track_summary in track_summaries:
        track_id = track_summary["track_id"]
        track_embedding = torch.tensor(track_summary["embedding"], device=reference_embedding.device)
        num_observations = track_summary.get("num_observations", 1)
        
        similarity = cosine_similarity(reference_embedding, track_embedding)
        track_similarities[track_id] = {
            "similarity": similarity,
            "num_observations": num_observations,
        }
        
        # Adaptive threshold: longer tracks get slightly lower threshold
        # This helps because longer tracks have more reliable averaged embeddings
        threshold = base_threshold
        if num_observations > 100:
            threshold = base_threshold - adaptive_factor
        elif num_observations > 50:
            threshold = base_threshold - (adaptive_factor * 0.5)
        
        print(f"  Track {track_id:2d}: similarity = {similarity:.3f} "
              f"(threshold: {threshold:.3f}, frames: {num_observations})")
        
        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match_id = track_id
    
    if best_match_id is not None:
        print(f"\n[Matching] ✓ Found match! Track {best_match_id} (similarity: {best_similarity:.3f})")
        return best_match_id, track_similarities
    else:
        print(f"\n[Matching] ✗ No match found above threshold {base_threshold}")
        if track_similarities:
            best_id = max(track_similarities.items(), key=lambda x: x[1]["similarity"])[0]
            best_sim = track_similarities[best_id]["similarity"]
            print(f"  Best match: Track {best_id} (similarity: {best_sim:.3f})")
            print(f"  Try lowering --similarity-threshold to {best_sim:.2f} or lower")
        return None, track_similarities


def render_highlighted_person(
    video_path: Path,
    tracks: List[Dict],
    matching_track_id: int,
    output_path: Path,
    metadata: Dict,
    highlight_color: Tuple[int, int, int] = (0, 255, 0),  # Green
):
    """Render video highlighting only the matching person."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find the matching track
    matching_track = None
    for track in tracks:
        if track["id"] == matching_track_id:
            matching_track = track
            break
    
    if matching_track is None:
        print(f"[Error] Track {matching_track_id} not found!")
        return
    
    # Build frame map for just this track
    frame_map: Dict[int, List[np.ndarray]] = defaultdict(list)
    frames = matching_track.get("sam_frames") or matching_track.get("frames", [])
    boxes = matching_track.get("sam_boxes") or matching_track.get("boxes", [])
    
    for frame_idx, box in zip(frames, boxes):
        frame_map[int(frame_idx)].append(np.array(box))
    
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata["fps"],
        (int(metadata["width"]), int(metadata["height"])),
    )
    
    frame_idx = 0
    total_frames = int(metadata.get("total_frames", 0))
    
    with tqdm(total=total_frames, desc="Rendering highlighted video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw box for matching person
            if frame_idx in frame_map:
                for box in frame_map[frame_idx]:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), highlight_color, 3)
                    
                    # Draw label
                    label = f"You (Track {matching_track_id})"
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
    print(f"[Render] Highlighted video saved to: {output_path}")


def save_matching_report(
    output_dir: Path,
    matching_track_id: Optional[int],
    track_similarities: Dict,
    reference_image_path: Path,
    video_path: Path,
):
    """Save a JSON report with matching results and all track similarities."""
    report = {
        "reference_image": str(reference_image_path),
        "video": str(video_path),
        "matching_track_id": matching_track_id,
        "track_similarities": track_similarities,
        "matched": matching_track_id is not None,
    }
    
    report_path = output_dir / "matching_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Matching report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Improved individual person tracking with temporal smoothing"
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
        help="Path to reference image of yourself",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output video with all people tracked",
    )
    parser.add_argument(
        "--match-output",
        type=Path,
        default=None,
        help="Path to output video with just you highlighted (optional)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Base similarity threshold for matching (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help="Use adaptive thresholding (lower threshold for longer tracks)",
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
        "--min-track-length",
        type=int,
        default=10,
        help="Minimum frames for a track to be kept (default: 10)",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=None,
        help="Keep only top K longest tracks (default: None = keep all)",
    )
    parser.add_argument(
        "--reid-weights",
        type=Path,
        default=None,
        help="Path to Re-ID model weights (auto-detected if not provided)",
    )
    
    args = parser.parse_args()
    
    # Auto-detect Re-ID weights
    if args.reid_weights is None:
        checkpoints_dir = HOOPER_PATH / "checkpoints"
        reid_weights = checkpoints_dir / "PERSON-Tracking" / "swin_base_msmt17.pth"
        if not reid_weights.exists():
            print(f"ERROR: Re-ID weights not found at {reid_weights}")
            sys.exit(1)
        args.reid_weights = reid_weights
    
    print("=" * 70)
    print("Improved Individual Person Re-ID (with Temporal Smoothing)")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Reference: {args.reference_image}")
    print(f"Output (all people): {args.output}")
    if args.match_output:
        print(f"Output (just you): {args.match_output}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    if args.adaptive_threshold:
        print(f"Adaptive thresholding: ENABLED")
    print("=" * 70)
    
    # Load models
    print("\n[Loading] Loading detection model...")
    predictor, _ = get_detectron2_skeleton_model(device=args.device, batch_mode=True)
    
    print("[Loading] Loading Re-ID model...")
    extractor = get_solider_feature_extractor(str(args.reid_weights), device=args.device)
    
    # Step 1: Detect and track ALL people (no filtering yet!)
    print("\n[Step 1] Detecting and tracking ALL people in video...")
    print("  (This is key: we track everyone first, then match)")
    tracks, metadata = detect_and_track(
        args.video,
        predictor,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        score_threshold=args.min_person_score,
        iou_threshold=args.iou_threshold,
        max_age=args.max_track_gap,
        max_crops=60,
    )
    
    # Filter tracks
    frame_height = int(metadata.get('height', 1080))
    for track in tracks:
        track['frame_height'] = frame_height
    
    tracks = filter_tracks(tracks, args.min_track_length, args.keep_top_k)
    
    print(f"[Step 1] ✓ Found {len(tracks)} unique people")
    
    if not tracks:
        print("❌ No people detected!")
        sys.exit(1)
    
    # Step 2: Compute embeddings for all tracks (with temporal averaging!)
    print("\n[Step 2] Computing embeddings for all tracks...")
    print("  (Averaging embeddings across multiple frames for stability)")
    out_dir = args.output.parent / "track_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    track_summaries = compute_embeddings(
        tracks,
        extractor,
        args.device,
        out_dir,
        max_crops=60,  # Average up to 60 frames per track
    )
    
    print(f"[Step 2] ✓ Computed averaged embeddings for {len(track_summaries)} tracks")
    print(f"  Tag images saved to: {out_dir}/tracks/")
    print(f"  (Check these to see which track is you!)")
    
    # Step 3: Match reference to one of the tracks
    print("\n[Step 3] Matching your reference image to tracks...")
    reference_embedding = load_reference_embedding(args.reference_image, extractor)
    
    if args.adaptive_threshold:
        matching_track_id, track_similarities = find_matching_tracks_with_adaptive_threshold(
            reference_embedding, track_summaries, args.similarity_threshold
        )
    else:
        # Use simple fixed threshold matching
        matching_track_id = None
        track_similarities = {}
        best_similarity = 0.0
        
        print(f"\n[Matching] Comparing reference to {len(track_summaries)} tracks...")
        for track_summary in track_summaries:
            track_id = track_summary["track_id"]
            track_embedding = torch.tensor(track_summary["embedding"], device=reference_embedding.device)
            
            similarity = cosine_similarity(reference_embedding, track_embedding)
            track_similarities[track_id] = {
                "similarity": similarity,
                "num_observations": track_summary.get("num_observations", 1),
            }
            
            print(f"  Track {track_id:2d}: similarity = {similarity:.3f} "
                  f"(frames: {track_summary.get('num_observations', 1)})")
            
            if similarity >= args.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                matching_track_id = track_id
        
        if matching_track_id is not None:
            print(f"\n[Matching] ✓ Found match! Track {matching_track_id} (similarity: {best_similarity:.3f})")
        else:
            print(f"\n[Matching] ✗ No match found above threshold {args.similarity_threshold}")
    
    # Save matching report
    save_matching_report(
        out_dir, matching_track_id, track_similarities, args.reference_image, args.video
    )
    
    # Step 4: Render videos
    print("\n[Step 4] Rendering videos...")
    
    # Render all people (with different colors)
    print("  Rendering video with all people tracked...")
    render_tracks(args.video, tracks, args.output, frame_masks=None)
    print(f"  ✓ Saved to: {args.output}")
    
    # Render just you (if match found)
    if matching_track_id is not None and args.match_output:
        print(f"  Rendering video with just you highlighted (Track {matching_track_id})...")
        render_highlighted_person(
            args.video, tracks, matching_track_id, args.match_output, metadata
        )
        print(f"  ✓ Saved to: {args.match_output}")
    elif matching_track_id is None:
        print("  ⚠ No match found - skipping 'just you' video")
        print("  Check tag images in track_data/tracks/ to find yourself manually")
        print(f"  Check matching_report.json for similarity scores")
    
    print("\n" + "=" * 70)
    print("✅ Complete!")
    print(f"   All people tracked: {args.output}")
    if matching_track_id is not None:
        print(f"   You are: Track {matching_track_id}")
        if args.match_output:
            print(f"   Just you: {args.match_output}")
    else:
        print(f"   ⚠ No match found - check track_data/matching_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

