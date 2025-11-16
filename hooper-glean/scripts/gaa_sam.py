#!/usr/bin/env python3
"""
GAA Player Segmentation Demo - SAM2 Powered

Creates awesome segmentation overlays for Gaelic Football/Hurling videos.
Uses SAM2 for pixel-perfect player segmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
import gc
import time
from collections import defaultdict
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
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2_video_predictor = None
    build_sam2 = None
    SAM2ImagePredictor = None

SAM2_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "SAM2-InstanceSegmentation" / "sam2.1_hiera_large.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAA player segmentation with SAM2")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--frame-stride", type=int, default=10, help="Process every Nth frame")
    parser.add_argument("--batch-size", type=int, default=2, help="Detection batch size")
    parser.add_argument("--min-person-score", type=float, default=0.5)
    parser.add_argument("--max-track-gap", type=int, default=30)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-crops", type=int, default=20)
    parser.add_argument("--min-track-length", type=int, default=5)
    parser.add_argument("--keep-top-k", type=int, default=1, help="Top K players to track (keep low for memory)")
    parser.add_argument("--overlay-opacity", type=float, default=0.35, help="Overlay opacity 0-1")
    parser.add_argument("--sam-prompt-min-score", type=float, default=0.2)
    parser.add_argument("--sam2-config", type=str, default=SAM2_DEFAULT_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=SAM2_DEFAULT_CHECKPOINT)
    parser.add_argument("--use-image-predictor", action="store_true", help="Use fast image predictor instead of video predictor (much faster, slightly less accurate)")
    parser.add_argument("--sam-frame-stride", type=int, default=1, help="Process every Nth frame with SAM2 (only for image predictor)")
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
    return inter / union if union > 0 else 0.0


def process_batch(frames, frame_indices, predictor, tracks, next_track_id, 
                  score_threshold, iou_threshold, max_age, max_crops):
    predictions = predictor(frames)
    for frame, frame_idx, pred in zip(frames, frame_indices, predictions):
        instances = pred["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        keypoints = instances.pred_keypoints.numpy() if instances.has("pred_keypoints") else None

        if len(boxes) == 0:
            continue

        keep = scores >= score_threshold
        boxes, scores = boxes[keep], scores[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if len(boxes) == 0:
            continue

        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        if keypoints is not None:
            keypoints = keypoints[order]

        active_track_indices = [
            idx for idx, tr in enumerate(tracks)
            if (frame_idx - tr["last_frame"]) <= max_age
        ]
        matched_tracks = set()
        assignments = [-1] * len(boxes)

        for det_idx, box in enumerate(boxes):
            best_iou, best_track_idx = 0.0, -1
            for track_idx in active_track_indices:
                if track_idx in matched_tracks:
                    continue
                iou_val = compute_iou(box, tracks[track_idx]["last_box"])
                if iou_val > best_iou:
                    best_iou, best_track_idx = iou_val, track_idx
            if best_track_idx >= 0 and best_iou >= iou_threshold:
                assignments[det_idx] = best_track_idx
                matched_tracks.add(best_track_idx)

        height, width = frame.shape[:2]
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [
                np.clip(box[0], 0, width - 1),
                np.clip(box[1], 0, height - 1),
                np.clip(box[2], box[0] + 1, width),
                np.clip(box[3], box[1] + 1, height)
            ])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            track_idx = assignments[det_idx]
            if track_idx == -1:
                track_idx = len(tracks)
                tracks.append({
                    "id": next_track_id[0],
                    "boxes": [], "frames": [], "crops": [],
                    "keypoints": [], "scores": [],
                    "last_box": None, "last_frame": -1,
                })
                next_track_id[0] += 1

            track = tracks[track_idx]
            track["last_box"] = np.array([x1, y1, x2, y2], dtype=np.float32)
            track["last_frame"] = frame_idx
            track["boxes"].append(track["last_box"].tolist())
            track["frames"].append(frame_idx)
            track["scores"].append(float(scores[det_idx]))
            if len(track["crops"]) < max_crops:
                track["crops"].append(crop_rgb)
            track["keypoints"].append(keypoints[det_idx].tolist() if keypoints is not None else None)


def detect_and_track(video_path, predictor, frame_stride, batch_size,
                     score_threshold, iou_threshold, max_age, max_crops):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    tracks, next_track_id = [], [0]
    frames, indices = [], []
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Detecting players") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                frames.append(frame)
                indices.append(frame_idx)
                if len(frames) == batch_size:
                    process_batch(frames, indices, predictor, tracks, next_track_id,
                                score_threshold, iou_threshold, max_age, max_crops)
                    frames.clear()
                    indices.clear()
            frame_idx += 1
            pbar.update(1)

    if frames:
        process_batch(frames, indices, predictor, tracks, next_track_id,
                     score_threshold, iou_threshold, max_age, max_crops)

    cap.release()
    return tracks, {"fps": fps, "width": width, "height": height}


def filter_tracks(tracks, min_track_length, keep_top_k):
    filtered = [t for t in tracks if len(t["frames"]) >= min_track_length and len(t["crops"]) >= 3]
    filtered.sort(key=lambda t: len(t["frames"]), reverse=True)
    if keep_top_k:
        filtered = filtered[:keep_top_k]
    for i, track in enumerate(filtered, 1):
        track["id"] = i
    return filtered


def build_sam_prompts_from_tracks(tracks, min_score):
    prompts = []
    for track in tracks:
        frames = track.get("frames", [])
        keypoints_list = track.get("keypoints", [])
        if not keypoints_list:
            continue
        
        best_idx, best_valid = None, -1
        for idx, kp in enumerate(keypoints_list):
            if kp is None:
                continue
            kp_arr = np.asarray(kp, dtype=np.float32)
            valid = int((kp_arr[:, 2] >= min_score).sum())
            if valid > best_valid:
                best_valid, best_idx = valid, idx
        
        if best_idx is None:
            continue
        
        kp_arr = np.asarray(keypoints_list[best_idx], dtype=np.float32)
        frame_idx = int(frames[best_idx])
        prompts = add_person_to_tracking_prompt(
            prompts, frame_idx=frame_idx, obj_id=int(track["id"]),
            keypoints=kp_arr, min_score=min_score
        )
    return prompts


def attach_sam_segments_to_tracks(tracks, segments, scale_factor=None):
    """Attach SAM2 segments to tracks, optionally scaling masks if video was downsampled."""
    id_to_track = {int(t["id"]): t for t in tracks}
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
            
            # Scale mask if video was downsampled
            if scale_factor is not None and scale_factor != 1.0:
                h, w = mask_bool.shape
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            coords = np.argwhere(mask_bool > 0)
            if coords.size == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            
            track["sam_frames"].append(int(frame_idx))
            track["sam_boxes"].append([float(x) for x in bbox])
            track["sam_masks"][int(frame_idx)] = mask_bool

    for track in tracks:
        if track["sam_frames"]:
            combined = sorted(zip(track["sam_frames"], track["sam_boxes"]), key=lambda x: x[0])
            track["sam_frames"] = [f for f, _ in combined]
            track["sam_boxes"] = [b for _, b in combined]


def collate_frame_masks(tracks):
    frame_masks = {}
    for track in tracks:
        for frame_idx, mask in track.get("sam_masks", {}).items():
            frame_entry = frame_masks.setdefault(int(frame_idx), {})
            frame_entry[int(track["id"])] = mask
    return frame_masks


def segment_with_image_predictor(video_path, tracks, sam_predictor, frame_stride=1, min_score=0.2):
    """Fast SAM2 segmentation using image predictor (per-frame, no temporal propagation)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Build frame-to-track mapping
    frame_to_tracks = defaultdict(list)
    for track in tracks:
        track_id = int(track["id"])
        frames = track.get("frames", [])
        boxes = track.get("boxes", [])
        keypoints_list = track.get("keypoints", [])
        for frame_idx, box, kp in zip(frames, boxes, keypoints_list):
            if kp is not None:
                kp_arr = np.asarray(kp, dtype=np.float32)
                valid_kp = (kp_arr[:, 2] >= min_score).sum()
                if valid_kp >= 5:  # Need at least 5 valid keypoints
                    frame_to_tracks[int(frame_idx)].append((track_id, box, kp_arr))
    
    sam_frame_masks = {}
    frame_idx = 0
    
    frames_to_process = [f for f in frame_to_tracks.keys() if f % frame_stride == 0]
    print(f"   Processing {len(frames_to_process)} frames...")
    
    with tqdm(total=total_frames, desc="SAM2 image predictor") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_to_tracks and frame_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sam_predictor.set_image(frame_rgb)
                
                for track_id, box, keypoints in frame_to_tracks[frame_idx]:
                    # Build prompt from keypoints
                    prompts = add_person_to_tracking_prompt([], frame_idx, track_id, keypoints, min_score)
                    if prompts:
                        points = prompts[0][2]
                        labels = prompts[0][3]
                        masks, scores, _ = sam_predictor.predict(
                            point_coords=points,
                            point_labels=labels,
                            multimask_output=False,
                        )
                        if masks.any() and len(masks) > 0:
                            mask_bool = masks[0].astype(bool)
                            if track_id not in sam_frame_masks:
                                sam_frame_masks[track_id] = {}
                            sam_frame_masks[track_id][frame_idx] = mask_bool
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    # Convert to frame-based format
    result = defaultdict(dict)
    for track_id, frames_dict in sam_frame_masks.items():
        for frame_idx, mask in frames_dict.items():
            result[frame_idx][track_id] = mask
    
    return result


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id)
    return tuple(rng.integers(low=64, high=256, size=3, dtype=np.int32))


def render_overlay(video_path, tracks, out_path, frame_masks=None, opacity=0.35):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_map = defaultdict(list)
    for track in tracks:
        track_id = int(track["id"])
        frames = track.get("sam_frames") or track.get("frames", [])
        boxes = track.get("sam_boxes") or track.get("boxes", [])
        for frame_idx, box in zip(frames, boxes):
            frame_map[int(frame_idx)].append((track_id, np.asarray(box, dtype=np.float32)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {out_path}")

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Rendering overlay") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply SAM2 masks
            masks_for_frame = frame_masks.get(frame_idx, {}) if frame_masks else {}
            if masks_for_frame:
                overlay = np.zeros_like(frame)
                for track_id, mask in masks_for_frame.items():
                    mask_bool = mask.astype(bool)
                    color_rgb = id_to_color(track_id)
                    # Convert RGB to BGR for OpenCV
                    color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.uint8)
                    overlay[mask_bool] = color_bgr
                frame = cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0)

            # Draw labels
            for track_id, box in frame_map.get(frame_idx, []):
                x1, y1, x2, y2 = map(int, [
                    np.clip(box[0], 0, width - 1),
                    np.clip(box[1], 0, height - 1),
                    np.clip(box[2], box[0] + 1, width),
                    np.clip(box[3], box[1] + 1, height)
                ])

                color_rgb = id_to_color(track_id)
                # Convert RGB to BGR for OpenCV
                color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
                
                label = f"Player {track_id}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame, (x1, y1 - text_height - 10),
                    (x1 + text_width + 4, y1), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2,
                    lineType=cv2.LINE_AA
                )

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 GAA Player Segmentation Demo")
    print("=" * 50)
    
    # Load detection models
    print("\n📦 Loading detection models...")
    predictor, _ = get_detectron2_skeleton_model(device=args.device, batch_mode=True)
    checkpoints_dir = REPO_ROOT / "checkpoints"
    reid_weights = checkpoints_dir / "PERSON-Tracking" / "swin_base_msmt17.pth"
    extractor = get_solider_feature_extractor(str(reid_weights), device=args.device)

    # Detect and track
    print("\n👥 Detecting and tracking players...")
    tracks, metadata = detect_and_track(
        args.video, predictor, args.frame_stride, args.batch_size,
        args.min_person_score, args.iou_threshold,
        args.max_track_gap, args.max_crops
    )
    tracks = filter_tracks(tracks, args.min_track_length, args.keep_top_k)

    if not tracks:
        print("❌ No players detected!")
        return

    print(f"✅ Found {len(tracks)} players")
    
    # Skip SAM2 - just draw boxes
    sam_frame_masks = None

    # Render final video
    print("\n🎬 Rendering overlay video...")
    render_path = args.out_dir / "player_segmentation_overlay.mp4"
    render_overlay(
        args.video, tracks, render_path,
        sam_frame_masks if sam_frame_masks else None,
        args.overlay_opacity
    )

    # Save metadata
    output = {
        "video": str(args.video),
        "metadata": metadata,
        "num_players": len(tracks),
        "tracks": [
            {
                "track_id": int(t["id"]),
                "num_frames": len(t.get("sam_frames", t["frames"])),
                "frames": t.get("sam_frames", t["frames"])[:10]  # Sample
            }
            for t in tracks
        ],
    }
    summary_path = args.out_dir / "player_tracks.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 50)
    print("🎉 DONE!")
    print(f"   📹 Video: {render_path}")
    print(f"   📊 Data: {summary_path}")
    print(f"   👥 Players: {len(tracks)}")
    print("=" * 50)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
