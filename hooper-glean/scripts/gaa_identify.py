#!/usr/bin/env python3
"""
GAA player identification pipeline - based on Hooper/BJJ system.

This script does the following:
  1. Loads the Detectron2 keypoint model to obtain per-frame person boxes.
  2. Tracks those detections across frames with a simple IoU-based tracker.
  3. Crops each tracked person and embeds the crops with the Solider re-ID model.
  4. Renders an overlay video with tracked players and optional SAM2 segmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hooper.detectron_utils import get_detectron2_skeleton_model
from hooper.solider_utils import get_solider_feature_extractor
from hooper.ml_utils import (
    add_person_to_tracking_prompt,
    init_inference_state,
    track_objects,
)

try:
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    build_sam2_video_predictor = None

SAM2_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

SAM2_DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "SAM2-InstanceSegmentation" / "sam2.1_hiera_large.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAA player identification pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the source video",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to store outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device for inference (default: cuda)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Sample every Nth frame for detection/tracking",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Detectron2 keypoint inference",
    )
    parser.add_argument(
        "--min-person-score",
        type=float,
        default=0.6,
        help="Minimum confidence to keep person detections",
    )
    parser.add_argument(
        "--max-track-gap",
        type=int,
        default=30,
        help="Maximum frame gap before a track is considered stale",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU to associate detections to existing tracks",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=60,
        help="Maximum number of crops to keep per track for embedding",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=10,
        help="Minimum number of frames for a track to be kept",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=None,
        help="If set, keep only the top-K longest tracks",
    )
    parser.add_argument(
        "--render-path",
        type=Path,
        default=None,
        help="Optional explicit path for rendered overlay video (mp4). Defaults to out-dir/player_overlay.mp4",
    )
    parser.add_argument(
        "--use-sam2",
        action="store_true",
        help="If set, refine tracks with SAM2 masks for better separation.",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default=SAM2_DEFAULT_CONFIG,
        help="Path to the SAM2 config yaml.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=SAM2_DEFAULT_CHECKPOINT,
        help="Path to the SAM2 checkpoint.",
    )
    parser.add_argument(
        "--sam-prompt-min-score",
        type=float,
        default=0.2,
        help="Minimum keypoint score retained when building SAM prompts.",
    )
    return parser.parse_args()


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
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


def process_batch(
    frames: List[np.ndarray],
    frame_indices: List[int],
    predictor,
    tracks: List[Dict],
    next_track_id: List[int],
    score_threshold: float,
    iou_threshold: float,
    max_age: int,
    max_crops: int,
):
    """Run Detectron2 on a batch of frames and update tracks in place."""
    predictions = predictor(frames)
    for frame, frame_idx, pred in zip(frames, frame_indices, predictions):
        instances = pred["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        keypoints = None
        if instances.has("pred_keypoints"):
            keypoints = instances.pred_keypoints.numpy()

        if len(boxes) == 0:
            continue

        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if len(boxes) == 0:
            continue

        # Sort detections by confidence (high to low)
        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]
        if keypoints is not None:
            keypoints = keypoints[order]

        # Track assignment
        active_track_indices = [
            idx
            for idx, tr in enumerate(tracks)
            if (frame_idx - tr["last_frame"]) <= max_age
        ]
        matched_tracks = set()
        assignments: List[int] = [-1] * len(boxes)

        for det_idx, box in enumerate(boxes):
            best_iou = 0.0
            best_track_idx = -1
            for track_idx in active_track_indices:
                if track_idx in matched_tracks:
                    continue
                iou_val = compute_iou(box, tracks[track_idx]["last_box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track_idx = track_idx
            if best_track_idx >= 0 and best_iou >= iou_threshold:
                assignments[det_idx] = best_track_idx
                matched_tracks.add(best_track_idx)

        height, width = frame.shape[:2]
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, x1 + 1, width))
            y2 = int(np.clip(y2, y1 + 1, height))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            track_idx = assignments[det_idx]
            if track_idx == -1:
                track_idx = len(tracks)
                tracks.append(
                    {
                        "id": next_track_id[0],
                        "boxes": [],
                        "frames": [],
                        "crops": [],
                        "keypoints": [],
                        "scores": [],
                        "last_box": None,
                        "last_frame": -1,
                    }
                )
                next_track_id[0] += 1

            track = tracks[track_idx]
            track["last_box"] = np.array([x1, y1, x2, y2], dtype=np.float32)
            track["last_frame"] = frame_idx
            track["boxes"].append(track["last_box"].tolist())
            track["frames"].append(frame_idx)
            track["scores"].append(float(scores[det_idx]))
            if len(track["crops"]) < max_crops:
                track["crops"].append(crop_rgb)
            if keypoints is not None:
                track["keypoints"].append(keypoints[det_idx].tolist())
            else:
                track["keypoints"].append(None)


def detect_and_track(
    video_path: Path,
    predictor,
    frame_stride: int,
    batch_size: int,
    score_threshold: float,
    iou_threshold: float,
    max_age: int,
    max_crops: int,
) -> Tuple[List[Dict], Dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    tracks: List[Dict] = []
    next_track_id = [0]
    frames: List[np.ndarray] = []
    indices: List[int] = []
    frame_idx = 0

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Detect+Track") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                frames.append(frame)
                indices.append(frame_idx)
                if len(frames) == batch_size:
                    process_batch(
                        frames,
                        indices,
                        predictor,
                        tracks,
                        next_track_id,
                        score_threshold,
                        iou_threshold,
                        max_age,
                        max_crops,
                    )
                    frames.clear()
                    indices.clear()
            frame_idx += 1
            pbar.update(1)

    if frames:
        process_batch(
            frames,
            indices,
            predictor,
            tracks,
            next_track_id,
            score_threshold,
            iou_threshold,
            max_age,
            max_crops,
        )
        frames.clear()
        indices.clear()

    cap.release()

    metadata = {"fps": fps, "width": width, "height": height}
    return tracks, metadata


def filter_tracks(
    tracks: List[Dict], min_track_length: int, keep_top_k: int | None
) -> List[Dict]:
    # First pass: basic filtering
    filtered = [
        trk
        for trk in tracks
        if len(trk["frames"]) >= min_track_length and len(trk["crops"]) >= 3
    ]
    
    if not filtered:
        return []
    
    # Calculate average box area for each track
    track_areas = []
    for trk in filtered:
        boxes = trk.get("boxes", [])
        if not boxes:
            track_areas.append((trk, 0, 0))
            continue
        
        avg_height = sum(box[3] - box[1] for box in boxes) / len(boxes)
        avg_width = sum(box[2] - box[0] for box in boxes) / len(boxes)
        avg_area = avg_height * avg_width
        avg_y = sum((box[1] + box[3]) / 2 for box in boxes) / len(boxes)
        
        track_areas.append((trk, avg_area, avg_y))
    
    # Find the largest box (primary detection - likely a player)
    max_area = max(area for _, area, _ in track_areas)
    
    # Filter out tracks that are too small relative to primary detection
    # Also filter tracks in top of frame (crowd in stands)
    field_filtered = []
    frame_height = filtered[0].get('frame_height', 1080)
    
    for trk, avg_area, avg_y in track_areas:
        # Classify as athlete or crowd based on:
        # 1. Size relative to biggest detection
        # 2. Position in frame
        size_ratio = avg_area / max_area if max_area > 0 else 0
        is_big_enough = size_ratio >= 0.3
        is_on_field = (avg_y / frame_height) > 0.4
        
        # Determine type
        if is_big_enough and is_on_field:
            trk['type'] = 'athlete'
            field_filtered.append(trk)
        elif size_ratio >= 0.15 and not is_on_field:
            # Keep some crowd for context, but label them
            trk['type'] = 'crowd'
            # Optionally add crowd - comment out if you don't want them
            # field_filtered.append(trk)
    
    # Sort by track length and keep top K
    field_filtered.sort(key=lambda trk: len(trk["frames"]), reverse=True)
    if keep_top_k is not None and keep_top_k > 0:
        field_filtered = field_filtered[:keep_top_k]

    for new_id, track in enumerate(field_filtered, start=1):
        track["id"] = new_id

    return field_filtered


def compute_embeddings(
    tracks: List[Dict],
    extractor,
    device: str,
    out_dir: Path,
    max_crops: int,
) -> List[Dict]:
    track_dir = out_dir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    for track in tracks:
        crops = track["crops"]
        if len(crops) == 0:
            continue

        if len(crops) > max_crops:
            sample_idx = np.linspace(0, len(crops) - 1, max_crops).astype(int)
            crops = [crops[int(i)] for i in sample_idx]

        features = extractor(crops).cpu()
        embedding = torch.nn.functional.normalize(features.mean(0), dim=0)
        embedding_vector = embedding.tolist()

        tag_image_path = track_dir / f"track_{track['id']:02d}_tag.jpg"
        tag_image_rgb = crops[0]
        cv2.imwrite(
            str(tag_image_path),
            cv2.cvtColor(tag_image_rgb, cv2.COLOR_RGB2BGR),
        )

        summaries.append(
            {
                "track_id": int(track["id"]),
                "num_observations": len(track.get("sam_frames", track["frames"])),
                "frames": track.get("sam_frames", track["frames"]),
                "embedding": embedding_vector,
                "tag_image": str(tag_image_path.relative_to(out_dir)),
            }
        )

    return summaries


def ensure_mp4_video(video_path: Path, workspace: Path) -> Path:
    if video_path.suffix.lower() == ".mp4":
        return video_path

    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / f"{video_path.stem}.mp4"
    if target.exists():
        return target

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        str(target),
    ]
    subprocess.run(cmd, check=True)
    return target


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    mask_arr = np.squeeze(np.asarray(mask))
    if mask_arr.ndim > 2:
        mask_arr = mask_arr.max(axis=0)
    if mask_arr.ndim != 2:
        return None

    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def build_sam_prompts_from_tracks(
    tracks: List[Dict], min_score: float
) -> List[Tuple[int, int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    prompts: List[
        Tuple[int, int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    ] = []
    for track in tracks:
        frames = track.get("frames", [])
        keypoints_list = track.get("keypoints", [])
        if not keypoints_list:
            continue
        best_idx = None
        best_valid = -1
        for idx, kp in enumerate(keypoints_list):
            if kp is None:
                continue
            kp_arr = np.asarray(kp, dtype=np.float32)
            valid = int((kp_arr[:, 2] >= min_score).sum())
            if valid == 0:
                continue
            if valid > best_valid:
                best_valid = valid
                best_idx = idx
        if best_idx is None:
            continue
        kp_arr = np.asarray(keypoints_list[best_idx], dtype=np.float32)
        frame_idx = int(frames[best_idx])
        prompts = add_person_to_tracking_prompt(
            prompts,
            frame_idx=frame_idx,
            obj_id=int(track["id"]),
            keypoints=kp_arr,
            min_score=min_score,
        )
    return prompts


def attach_sam_segments_to_tracks(
    tracks: List[Dict], segments: Dict[int, Dict[int, np.ndarray]]
) -> None:
    id_to_track = {int(track["id"]): track for track in tracks}
    for track in tracks:
        track["sam_frames"] = []
        track["sam_boxes"] = []
        track["sam_masks"] = {}

    for frame_idx, obj_masks in segments.items():
        for obj_id, mask in obj_masks.items():
            track = id_to_track.get(int(obj_id))
            if track is None:
                continue
            mask_bool = np.squeeze(mask.astype(bool))
            if mask_bool.ndim > 2:
                mask_bool = mask_bool.max(axis=0)
            if mask_bool.ndim != 2:
                continue
            bbox = mask_to_bbox(mask_bool)
            if bbox is None:
                continue
            track["sam_frames"].append(int(frame_idx))
            track["sam_boxes"].append([float(x) for x in bbox])
            track["sam_masks"][int(frame_idx)] = mask_bool

    for track in tracks:
        if not track["sam_frames"]:
            continue
        combined = sorted(zip(track["sam_frames"], track["sam_boxes"]), key=lambda x: x[0])
        track["sam_frames"] = [frame for frame, _ in combined]
        track["sam_boxes"] = [box for _, box in combined]


def collate_frame_masks(
    tracks: List[Dict],
) -> Dict[int, Dict[int, np.ndarray]]:
    frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
    for track in tracks:
        for frame_idx, mask in track.get("sam_masks", {}).items():
            frame_entry = frame_masks.setdefault(int(frame_idx), {})
            frame_entry[int(track["id"])] = mask
    return frame_masks


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id)
    color = rng.integers(low=64, high=256, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def render_tracks(
    video_path: Path,
    tracks: List[Dict],
    out_path: Path,
    frame_masks: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_map: Dict[int, List[Tuple[int, np.ndarray, str]]] = defaultdict(list)
    for track in tracks:
        track_id = int(track["id"])
        track_type = track.get("type", "athlete")  # Default to athlete
        frames = track.get("sam_frames") or track.get("frames", [])
        boxes = track.get("sam_boxes") or track.get("boxes", [])
        for frame_idx, box in zip(frames, boxes):
            frame_map[int(frame_idx)].append(
                (track_id, np.asarray(box, dtype=np.float32), track_type)
            )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video writer at {out_path}")

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Render overlay") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            masks_for_frame = (
                frame_masks.get(frame_idx, {}) if frame_masks is not None else {}
            )
            if masks_for_frame:
                overlay = np.zeros_like(frame)
                for track_id, mask in masks_for_frame.items():
                    mask_bool = mask.astype(bool)
                    color = np.array(id_to_color(track_id), dtype=np.uint8)
                    overlay[mask_bool] = color
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            for track_id, box, track_type in frame_map.get(frame_idx, []):
                x1, y1, x2, y2 = box.astype(int)
                x1 = int(np.clip(x1, 0, width - 1))
                y1 = int(np.clip(y1, 0, height - 1))
                x2 = int(np.clip(x2, x1 + 1, width))
                y2 = int(np.clip(y2, y1 + 1, height))

                # Different colors/styles for athletes vs crowd (no labels)
                if track_type == 'athlete':
                    color = id_to_color(track_id)
                    thickness = 2
                else:  # crowd
                    color = (128, 128, 128)  # Gray for crowd
                    thickness = 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    total_start = time.time()
    timings = {}

    # Load detectors - GPU optimized
    print("Loading models...")
    load_start = time.time()
    predictor, _ = get_detectron2_skeleton_model(
        device=args.device, batch_mode=True
    )
    checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    reid_weights = checkpoints_dir / "PERSON-Tracking" / "swin_base_msmt17.pth"
    extractor = get_solider_feature_extractor(str(reid_weights), device=args.device)
    timings['model_loading'] = time.time() - load_start
    print(f"✓ Models loaded ({timings['model_loading']:.1f}s)")

    # Detect + track + embed
    print("\nDetecting and tracking players...")
    detect_start = time.time()
    tracks, metadata = detect_and_track(
        args.video,
        predictor,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        score_threshold=args.min_person_score,
        iou_threshold=args.iou_threshold,
        max_age=args.max_track_gap,
        max_crops=args.max_crops,
    )
    # Store frame dimensions in tracks for crowd filtering
    frame_height = int(metadata.get('height', 1080))
    for track in tracks:
        track['frame_height'] = frame_height
    
    pre_filter_count = len([t for t in tracks if len(t["frames"]) >= args.min_track_length])
    tracks = filter_tracks(tracks, args.min_track_length, args.keep_top_k)
    
    # Count athletes vs crowd
    athlete_count = len([t for t in tracks if t.get('type') == 'athlete'])
    crowd_count = len([t for t in tracks if t.get('type') == 'crowd'])
    
    timings['detection_tracking'] = time.time() - detect_start
    print(f"✓ Detection and tracking complete ({timings['detection_tracking']:.1f}s)")
    print(f"  - Total detections: {pre_filter_count}")
    print(f"  - Athletes: {athlete_count}")
    print(f"  - Crowd: {crowd_count}")
    print(f"  - Filtered: {len(tracks)} total")

    if not tracks:
        print("No tracks satisfied the filtering criteria.")
        return

    sam_frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
    if args.use_sam2:
        print("\nRunning SAM2 segmentation...")
        sam_start = time.time()
        if build_sam2_video_predictor is None:
            raise RuntimeError(
                "SAM2 is not installed. Install segment-anything-2 to enable SAM2 refinement."
            )
        if not args.sam2_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {args.sam2_checkpoint}"
            )
        sam_model = build_sam2_video_predictor(
            args.sam2_config, str(args.sam2_checkpoint), device=args.device
        )
        video_for_sam = ensure_mp4_video(args.video, args.out_dir / "sam2_cache")
        prompts = build_sam_prompts_from_tracks(tracks, args.sam_prompt_min_score)
        if not prompts:
            print("SAM2: No suitable prompts were generated; skipping refinement.")
        else:
            inference_state = init_inference_state(sam_model, str(video_for_sam))
            sam_segments = track_objects(sam_model, prompts, inference_state)
            sam_model.reset_state(inference_state)
            attach_sam_segments_to_tracks(tracks, sam_segments)
            sam_frame_masks = collate_frame_masks(tracks)
            timings['sam2_segmentation'] = time.time() - sam_start
            print(f"✓ SAM2 segmentation complete ({timings['sam2_segmentation']:.1f}s)")
    else:
        timings['sam2_segmentation'] = 0

    print("\nComputing embeddings...")
    embed_start = time.time()
    summaries = compute_embeddings(
        tracks,
        extractor,
        args.device,
        args.out_dir,
        max_crops=args.max_crops,
    )
    timings['embeddings'] = time.time() - embed_start
    print(f"✓ Embeddings computed ({timings['embeddings']:.1f}s)")

    # Always render video
    print("\nRendering overlay video...")
    render_start = time.time()
    # Use output directory name as video filename
    render_path = args.render_path or (args.out_dir / f"{args.out_dir.name}.mp4")
    render_tracks(
        args.video,
        tracks,
        render_path,
        sam_frame_masks if sam_frame_masks else None,
    )

    output = {
        "video": str(args.video),
        "metadata": metadata,
        "tracks": summaries,
    }
    summary_path = args.out_dir / "player_tracks.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    timings['rendering'] = time.time() - render_start
    timings['total'] = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Model Loading:       {timings['model_loading']:>7.1f}s")
    print(f"Detection/Tracking:  {timings['detection_tracking']:>7.1f}s")
    if timings['sam2_segmentation'] > 0:
        print(f"SAM2 Segmentation:   {timings['sam2_segmentation']:>7.1f}s")
    print(f"Embeddings:          {timings['embeddings']:>7.1f}s")
    print(f"Rendering:           {timings['rendering']:>7.1f}s")
    print(f"{'-'*60}")
    print(f"TOTAL TIME:          {timings['total']:>7.1f}s")
    
    # Calculate speed metrics
    # Calculate actual video duration from metadata
    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or metadata.get('fps', 30)
    cap.release()
    video_duration = total_frames / fps if fps > 0 else 0
    
    if video_duration > 0:
        speed_factor = timings['total'] / video_duration
        print(f"Video Duration:      {video_duration:>7.1f}s")
        print(f"Processing Speed:    {speed_factor:>7.1f}x real-time")
    print(f"{'='*60}")
    
    print(f"\n✓ Wrote embeddings for {len(summaries)} tracks to {summary_path}")
    print(f"✓ Rendered overlay video to {render_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

