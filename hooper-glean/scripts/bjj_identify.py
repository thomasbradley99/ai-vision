#!/usr/bin/env python3
"""
Lightweight person-identification pipeline for the Ryan/BJJ video.

This script sidesteps the basketball-specific logic in `infer.py` and instead
does the following:
  1. Loads the Detectron2 keypoint model to obtain per-frame person boxes.
  2. Tracks those detections across frames with a simple IoU-based tracker.
  3. Crops each tracked person and embeds the crops with the Solider re-ID model.
  4. Aggregates crops per track, writes a representative tag image, and saves one
     embedding vector per person.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
import subprocess
import tempfile
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
    repo_root = Path(__file__).resolve().parents[1]
    default_video = repo_root / "test_videos" / "ryan-thomas" / "input" / "video.mov"
    default_out = (
        repo_root / "test_videos" / "ryan-thomas" / "outputs" / "person-id"
    )

    parser = argparse.ArgumentParser(description="Simple BJJ person ID pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help="Path to the source video",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Directory to store embeddings, crops, and summary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device for inference",
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
        default=8,
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
        "--render-video",
        action="store_true",
        help="If set, render an overlay video with tracked fighters.",
    )
    parser.add_argument(
        "--render-path",
        type=Path,
        default=None,
        help="Optional explicit path for rendered overlay video (mp4).",
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
    parser.add_argument(
        "--pair-min-overlap",
        type=int,
        default=90,
        help="Minimum overlapping frames required to form a fighter pair.",
    )
    parser.add_argument(
        "--zoom-margin",
        type=float,
        default=0.2,
        help="Extra margin ratio to apply around pair crops.",
    )
    parser.add_argument(
        "--zoom-smoothing",
        type=float,
        default=0.3,
        help="Exponential smoothing factor (0-1) for pair crop panning.",
    )
    parser.add_argument(
        "--min-roll-seconds",
        type=float,
        default=6.0,
        help="Minimum duration in seconds for a roll segment to be kept.",
    )
    parser.add_argument(
        "--roll-gap-seconds",
        type=float,
        default=2.5,
        help="Maximum pause in seconds between frames before splitting rolls.",
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

    # Drop tracks with little evidence
    metadata = {"fps": fps, "width": width, "height": height}
    return tracks, metadata


def filter_tracks(
    tracks: List[Dict], min_track_length: int, keep_top_k: int | None
) -> List[Dict]:
    filtered = [
        trk
        for trk in tracks
        if len(trk["frames"]) >= min_track_length and len(trk["crops"]) >= 3
    ]
    filtered.sort(key=lambda trk: len(trk["frames"]), reverse=True)
    if keep_top_k is not None and keep_top_k > 0:
        filtered = filtered[:keep_top_k]

    for new_id, track in enumerate(filtered, start=1):
        track["id"] = new_id

    return filtered


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


def gather_track_boxes(tracks: List[Dict]) -> Dict[int, Dict[int, np.ndarray]]:
    track_boxes: Dict[int, Dict[int, np.ndarray]] = {}
    for track in tracks:
        track_id = int(track["id"])
        frame_to_box: Dict[int, np.ndarray] = {}
        frames = track.get("frames", [])
        boxes = track.get("boxes", [])
        for frame_idx, box in zip(frames, boxes):
            frame_to_box.setdefault(int(frame_idx), np.asarray(box, dtype=np.float32))
        sam_frames = track.get("sam_frames") or []
        sam_boxes = track.get("sam_boxes") or []
        for frame_idx, box in zip(sam_frames, sam_boxes):
            frame_to_box[int(frame_idx)] = np.asarray(box, dtype=np.float32)
        track_boxes[track_id] = frame_to_box
    return track_boxes


def compute_pairs(
    tracks: List[Dict], min_overlap: int
) -> List[Dict]:
    if len(tracks) < 2:
        return []

    from itertools import combinations

    frame_sets = {
        int(track["id"]): set(track.get("sam_frames", track["frames"]))
        for track in tracks
    }

    pairs: List[Dict] = []
    pair_id = 1
    for left, right in combinations(tracks, 2):
        l_id = int(left["id"])
        r_id = int(right["id"])
        overlap = sorted(frame_sets[l_id] & frame_sets[r_id])
        if len(overlap) < min_overlap:
            continue
        pairs.append(
            {
                "pair_id": pair_id,
                "fighters": [l_id, r_id],
                "num_overlap_frames": len(overlap),
                "start_frame": overlap[0],
                "end_frame": overlap[-1],
                "overlap_frames": overlap,
            }
        )
        pair_id += 1
    return pairs


def segment_rolls(
    pairs: List[Dict],
    fps: float,
    gap_seconds: float,
    min_duration_seconds: float,
) -> List[Dict]:
    if not pairs:
        return []

    fps = max(float(fps), 1e-6)
    gap_frames = max(1, int(round(gap_seconds * fps)))
    min_duration_frames = max(1, int(round(min_duration_seconds * fps)))

    roll_segments: List[Dict] = []
    per_pair_counts: Dict[int, int] = defaultdict(int)

    for pair in pairs:
        frames = pair.get("overlap_frames") or []
        if not frames:
            continue
        start = frames[0]
        prev = frames[0]

        def maybe_add_segment(seg_start: int, seg_end: int):
            duration_frames = seg_end - seg_start + 1
            if duration_frames < min_duration_frames:
                return
            pair_id = int(pair["pair_id"])
            per_pair_counts[pair_id] += 1
            roll_index = per_pair_counts[pair_id]
            start_time = seg_start / fps
            # end time aligns with the end of the frame, so add one frame
            end_time = (seg_end + 1) / fps
            duration_seconds = duration_frames / fps
            roll_segments.append(
                {
                    "pair_id": pair_id,
                    "fighters": list(pair["fighters"]),
                    "roll_index": roll_index,
                    "start_frame": int(seg_start),
                    "end_frame": int(seg_end),
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "duration_seconds": round(duration_seconds, 2),
                    "num_frames": int(duration_frames),
                }
            )

        for frame in frames[1:]:
            if frame - prev > gap_frames:
                maybe_add_segment(start, prev)
                start = frame
            prev = frame

        maybe_add_segment(start, prev)

    return roll_segments


def expand_box(
    box: np.ndarray, frame_width: int, frame_height: int, margin: float
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    width = x2 - x1
    height = y2 - y1
    pad_w = max(2.0, width * margin)
    pad_h = max(2.0, height * margin)

    x1 = max(0.0, x1 - pad_w)
    y1 = max(0.0, y1 - pad_h)
    x2 = min(float(frame_width), x2 + pad_w)
    y2 = min(float(frame_height), y2 + pad_h)

    if x2 <= x1 or y2 <= y1:
        return np.array([0, 0, min(frame_width, 1), min(frame_height, 1)], dtype=np.int32)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def adjust_box_to_size(
    box: np.ndarray,
    target_width: int,
    target_height: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    width = x2 - x1
    height = y2 - y1

    if width < target_width:
        deficit = target_width - width
        x1 -= deficit / 2.0
        x2 += deficit / 2.0
    if height < target_height:
        deficit = target_height - height
        y1 -= deficit / 2.0
        y2 += deficit / 2.0

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(frame_width), x2)
    y2 = min(float(frame_height), y2)

    if x2 - x1 < target_width:
        shift = target_width - (x2 - x1)
        x1 = max(0.0, x1 - shift)
        x2 = min(float(frame_width), x1 + target_width)
        x1 = max(0.0, x2 - target_width)
    if y2 - y1 < target_height:
        shift = target_height - (y2 - y1)
        y1 = max(0.0, y1 - shift)
        y2 = min(float(frame_height), y1 + target_height)
        y1 = max(0.0, y2 - target_height)

    return np.array(
        [
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        ],
        dtype=np.int32,
    )


def extract_roll_clips(
    video_path: Path,
    out_dir: Path,
    rolls: List[Dict],
    tracks: List[Dict],
    frame_masks: Optional[Dict[int, Dict[int, np.ndarray]]],
    metadata: Dict[str, float],
    margin: float,
    smoothing: float,
) -> List[Dict]:
    if not rolls:
        return []

    frame_width = int(metadata.get("width", 0) or 0)
    frame_height = int(metadata.get("height", 0) or 0)
    fps = float(metadata.get("fps", 30.0) or 30.0)
    if frame_width <= 0 or frame_height <= 0:
        cap_probe = cv2.VideoCapture(str(video_path))
        frame_width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or fps
        cap_probe.release()

    track_boxes = gather_track_boxes(tracks)
    roll_dir = out_dir / "roll_clips"
    roll_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Dict] = []
    for roll in rolls:
        fighters = roll["fighters"]
        start_frame = int(roll["start_frame"])
        end_frame = int(roll["end_frame"])
        frame_range = range(start_frame, end_frame + 1)

        per_frame_boxes: Dict[int, np.ndarray] = {}
        for frame_idx in frame_range:
            union_box = None
            if frame_masks and frame_idx in frame_masks:
                masks = [
                    frame_masks[frame_idx].get(fid)
                    for fid in fighters
                    if frame_masks[frame_idx].get(fid) is not None
                ]
                if masks:
                    stacked = np.stack(masks, axis=0)
                    merged = np.any(stacked, axis=0)
                    union_box = mask_to_bbox(merged)
            if union_box is None:
                boxes = [
                    track_boxes.get(int(fid), {}).get(frame_idx)
                    for fid in fighters
                    if track_boxes.get(int(fid), {}).get(frame_idx) is not None
                ]
                if boxes:
                    stacked = np.stack(boxes, axis=0)
                    union_box = np.array(
                        [
                            float(stacked[:, 0].min()),
                            float(stacked[:, 1].min()),
                            float(stacked[:, 2].max()),
                            float(stacked[:, 3].max()),
                        ],
                        dtype=np.float32,
                    )
            if union_box is None:
                continue
            expanded = expand_box(union_box, frame_width, frame_height, margin)
            per_frame_boxes[frame_idx] = expanded

        if not per_frame_boxes:
            continue

        if smoothing > 0.0:
            alpha = float(np.clip(smoothing, 0.0, 0.99))
            ordered_frames = sorted(per_frame_boxes.keys())
            prev_box = None
            for frame_idx in ordered_frames:
                box = per_frame_boxes[frame_idx].astype(np.float32)
                if prev_box is None:
                    prev_box = box
                else:
                    prev_box = alpha * prev_box + (1.0 - alpha) * box
                per_frame_boxes[frame_idx] = prev_box.copy()

            prev_box = None
            for frame_idx in reversed(ordered_frames):
                box = per_frame_boxes[frame_idx]
                if prev_box is None:
                    prev_box = box
                else:
                    prev_box = alpha * prev_box + (1.0 - alpha) * box
                    per_frame_boxes[frame_idx] = (
                        per_frame_boxes[frame_idx] + prev_box
                    ) / 2.0

        for frame_idx, box in per_frame_boxes.items():
            box = box.astype(np.float32)
            box[0] = np.clip(box[0], 0.0, float(frame_width))
            box[1] = np.clip(box[1], 0.0, float(frame_height))
            box[2] = np.clip(box[2], 0.0, float(frame_width))
            box[3] = np.clip(box[3], 0.0, float(frame_height))
            per_frame_boxes[frame_idx] = box

        max_width = 0
        max_height = 0
        for box in per_frame_boxes.values():
            width = int(box[2] - box[0])
            height = int(box[3] - box[1])
            max_width = max(max_width, width)
            max_height = max(max_height, height)

        max_width = max(32, max_width)
        max_height = max(32, max_height)

        out_path = roll_dir / f"roll_{roll['pair_id']:02d}_{roll['roll_index']:02d}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (max_width, max_height),
        )
        if not writer.isOpened():
            writer.release()
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            writer.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        last_box: Optional[np.ndarray] = None
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            box = per_frame_boxes.get(current_frame)
            if box is None:
                box = last_box
            if box is None:
                current_frame += 1
                continue

            adjusted = adjust_box_to_size(
                box, max_width, max_height, frame_width, frame_height
            )
            x1, y1, x2, y2 = adjusted
            x2 = max(x1 + 1, min(frame_width, x2))
            y2 = max(y1 + 1, min(frame_height, y2))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                current_frame += 1
                continue
            if crop.shape[1] != max_width or crop.shape[0] != max_height:
                crop = cv2.resize(crop, (max_width, max_height))

            writer.write(crop)
            last_box = adjusted
            current_frame += 1

        cap.release()
        writer.release()

        roll_output = dict(roll)
        roll_output["clip"] = str(out_path.relative_to(out_dir))
        outputs.append(roll_output)

    return outputs


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

    frame_map: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for track in tracks:
        track_id = int(track["id"])
        frames = track.get("sam_frames") or track.get("frames", [])
        boxes = track.get("sam_boxes") or track.get("boxes", [])
        for frame_idx, box in zip(frames, boxes):
            frame_map[int(frame_idx)].append(
                (track_id, np.asarray(box, dtype=np.float32))
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

            for track_id, box in frame_map.get(frame_idx, []):
                x1, y1, x2, y2 = box.astype(int)
                x1 = int(np.clip(x1, 0, width - 1))
                y1 = int(np.clip(y1, 0, height - 1))
                x2 = int(np.clip(x2, x1 + 1, width))
                y2 = int(np.clip(y2, y1 + 1, height))

                color = id_to_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Fighter {track_id}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load detectors
    predictor, _ = get_detectron2_skeleton_model(
        device=args.device, batch_mode=True
    )
    checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    reid_weights = checkpoints_dir / "PERSON-Tracking" / "swin_base_msmt17.pth"
    extractor = get_solider_feature_extractor(str(reid_weights), device=args.device)

    # Detect + track + embed
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
    tracks = filter_tracks(tracks, args.min_track_length, args.keep_top_k)

    if not tracks:
        print("No tracks satisfied the filtering criteria.")
        return

    sam_frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
    if args.use_sam2:
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

    pairs = compute_pairs(tracks, args.pair_min_overlap)
    fps_val = float(metadata.get("fps") or 30.0)
    if fps_val <= 0:
        fps_val = 30.0
    for pair in pairs:
        pair["overlap_seconds"] = round(
            pair["num_overlap_frames"] / max(fps_val, 1e-6), 2
        )

    rolls = segment_rolls(
        pairs,
        fps=fps_val,
        gap_seconds=args.roll_gap_seconds,
        min_duration_seconds=args.min_roll_seconds,
    )

    roll_clips = extract_roll_clips(
        args.video,
        args.out_dir,
        rolls,
        tracks,
        sam_frame_masks if sam_frame_masks else None,
        metadata,
        args.zoom_margin,
        args.zoom_smoothing,
    )

    pair_summaries = []
    for pair in pairs:
        pair_copy = dict(pair)
        pair_copy.pop("overlap_frames", None)
        pair_summaries.append(pair_copy)

    summaries = compute_embeddings(
        tracks,
        extractor,
        args.device,
        args.out_dir,
        max_crops=args.max_crops,
    )

    if args.render_video:
        render_path = args.render_path or (args.out_dir / "person_id_overlay.mp4")
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
        "pairs": pair_summaries,
        "rolls": roll_clips,
    }
    summary_path = args.out_dir / "person_tracks.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote embeddings for {len(summaries)} tracks to {summary_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

