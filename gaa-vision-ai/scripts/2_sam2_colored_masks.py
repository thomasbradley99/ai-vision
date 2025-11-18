#!/usr/bin/env python3
"""
GAA Team Tracker with SAM2 Segmentation Masks

This script extends the bounding box tracker with SAM2 pixel-perfect masks:
1. YOLO detection + ByteTrack for player tracking
2. K-Means clustering for team assignment (same as script 1)
3. SAM2 segmentation for pixel-perfect masks
4. Render colored masks by team (not boxes)

SAM2 workflow:
- Extract all frames to temp directory
- Build prompts from YOLO boxes at first appearance
- SAM2 propagates masks throughout video
- Color masks by team assignment
"""

import argparse
import sys
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
import random

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm

# SAM2 imports
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

SAM2_CHECKPOINT_DIR = Path("/home/ubuntu/clann/ai-vision/hooper-glean/checkpoints/SAM2-InstanceSegmentation")

# SAM2 model configurations
SAM2_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

SAM2_CHECKPOINTS = {
    "tiny": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_tiny.pt",
    "small": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_small.pt",
    "base": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_base_plus.pt",
    "large": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_large.pt",
}


class TeamColorAssigner:
    """Intelligently assigns team colors to tracked players."""
    
    def __init__(self):
        self.team_centroids = None
        self.track_teams = {}
        self.grass_hsv = None
        
    def extract_grass_color(self, frame):
        """Extract grass color from frame using green color range in HSV."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        grass_color = cv2.mean(frame, mask=mask)[:3]
        self.grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
        return self.grass_hsv
    
    def extract_jersey_color(self, player_img):
        """Extract jersey color from player crop."""
        if self.grass_hsv is None:
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([80, 255, 255])
        else:
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([self.grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([self.grass_hsv[0, 0, 0] + 10, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0]//2, :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        return kit_color
    
    def initialize_teams(self, all_jersey_colors):
        """Run K-Means on collected jersey colors to find 2 team centroids."""
        if len(all_jersey_colors) < 2:
            return False
        
        colors_array = np.array(all_jersey_colors)
        self.team_centroids = KMeans(n_clusters=2, random_state=42)
        self.team_centroids.fit(colors_array)
        print(f"[TeamAssigner] Initialized teams with {len(all_jersey_colors)} color samples")
        print(f"  Team 0 centroid (BGR): {self.team_centroids.cluster_centers_[0]}")
        print(f"  Team 1 centroid (BGR): {self.team_centroids.cluster_centers_[1]}")
        return True
    
    def get_team_colors(self):
        """Return team centroids as BGR tuples for rendering."""
        if self.team_centroids is None:
            return None
        return {
            0: tuple(map(int, self.team_centroids.cluster_centers_[0])),
            1: tuple(map(int, self.team_centroids.cluster_centers_[1]))
        }
    
    def assign_team(self, track_id, jersey_color):
        """Assign a track to a team based on its jersey color."""
        if track_id in self.track_teams:
            return self.track_teams[track_id]
        
        if self.team_centroids is None:
            return None
        
        team = self.team_centroids.predict([jersey_color])[0]
        self.track_teams[track_id] = team
        return team


class GAATeamTrackerSAM2:
    """Main tracking and SAM2 segmentation pipeline."""
    
    def __init__(self, video_path, output_path, model_path, sam2_model, frame_stride, sample_frames=50, max_duration=None, offload_to_cpu=False):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        self.sam2_model = sam2_model
        self.sam2_config = SAM2_CONFIGS[sam2_model]
        self.sam2_checkpoint = SAM2_CHECKPOINTS[sam2_model]
        self.frame_stride = frame_stride
        self.sample_frames = sample_frames
        self.max_duration = max_duration
        self.offload_to_cpu = offload_to_cpu
        
        print(f"[Config] SAM2 model: {sam2_model}")
        print(f"[Config] Frame stride: {frame_stride}")
        if offload_to_cpu:
            print(f"[Config] SAM2 offload to CPU: ENABLED (slower but uses ~80% less GPU memory)")
        if max_duration:
            print(f"[Config] Max duration: {max_duration}s (MEMORY TEST MODE)")
        
        # Load YOLO model
        print(f"[Tracker] Loading YOLO model from {model_path}")
        self.model = YOLO(str(model_path))
        self.log_gpu_memory("After YOLO model load")
        
        # Initialize team assigner
        self.team_assigner = TeamColorAssigner()
        
        # Colors for visualization
        self.team_colors = None
        
        # Track data
        self.track_first_frames = {}  # track_id -> first frame it appeared
        self.track_boxes = {}  # track_id -> {frame_idx: box}
        self.ball_track_ids = set()  # Track IDs that are balls (not players)
    
    def log_gpu_memory(self, stage):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"[GPU Memory @ {stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
        
    def extract_frames(self, frame_dir):
        """Extract frames from video to directory (with stride)."""
        print(f"[SAM2] Extracting frames (stride {self.frame_stride}) to {frame_dir}...")
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        max_frames = None
        
        if self.max_duration:
            max_frames = int(self.max_duration * fps)
            print(f"[SAM2] Limiting to first {max_frames} frames ({self.max_duration}s @ {fps:.1f}fps)")
        
        frame_idx = 0
        extracted_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check duration limit
            if max_frames and frame_idx >= max_frames:
                print(f"[SAM2] Reached duration limit at frame {frame_idx}")
                break
            
            # Only extract every Nth frame
            if frame_idx % self.frame_stride == 0:
                frame_path = frame_dir / f"{extracted_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_idx += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"[SAM2] Extracted {extracted_idx} frames from {frame_idx} total")
        self.log_gpu_memory("After frame extraction")
        return frame_idx, extracted_idx
    
    def random_sample_colors(self):
        """PASS 1: Random sampling to collect jersey colors for K-means initialization."""
        print(f"[Tracker] PASS 1: Random sampling {self.sample_frames} frames...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = sorted(random.sample(range(total_frames), min(self.sample_frames, total_frames)))
        print(f"[Tracker] Sampling frames: {sample_indices[:5]}...{sample_indices[-5:]} (showing first/last 5)")
        
        all_jersey_colors = []
        
        # Extract grass color from first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret:
            self.team_assigner.extract_grass_color(first_frame)
        
        # Sample frames and collect colors
        for idx, frame_num in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            results = self.model(frame, conf=0.3, verbose=False)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            for box in results[0].boxes:
                cls = int(box.cls.cpu().numpy()[0])
                if cls != 0:  # Only players
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                player_crop = frame[y1:y2, x1:x2]
                
                if player_crop.size == 0:
                    continue
                
                jersey_color = self.team_assigner.extract_jersey_color(player_crop)
                all_jersey_colors.append(jersey_color)
            
            if (idx + 1) % 10 == 0:
                print(f"[Tracker]   Sampled {idx + 1}/{len(sample_indices)} frames, collected {len(all_jersey_colors)} colors")
        
        cap.release()
        print(f"[Tracker] PASS 1 complete: Collected {len(all_jersey_colors)} jersey color samples")
        return all_jersey_colors
    
    def collect_tracks_and_teams(self):
        """PASS 2: Run YOLO+ByteTrack on video to collect track info and assign teams."""
        print(f"[Tracker] PASS 2: Collecting track data and assigning teams...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.max_duration:
            max_frames = int(self.max_duration * fps)
            total_frames = min(total_frames, max_frames)
        
        frame_idx = 0
        ball_squareness = {}  # track_id -> list of squareness scores
        
        with tqdm(total=total_frames, desc="Analyzing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration limit
                if self.max_duration and frame_idx >= total_frames:
                    break
                
                results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
                
                # First pass: collect player sizes for ball filtering
                player_areas = []
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        if cls == 0:  # Player
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            area = (x2 - x1) * (y2 - y1)
                            player_areas.append(area)
                
                median_player_area = np.median(player_areas) if len(player_areas) > 0 else float('inf')
                
                # Second pass: collect tracks and score balls
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                        
                        if track_id is None:  # Need track ID
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        box_coords = [x1, y1, x2, y2]
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = height / width if width > 0 else 0
                        
                        # Track both players (cls=0) and balls (cls!=0)
                        is_ball = (cls != 0)
                        if is_ball:
                            # Filter: Ball must be < 40% of median player size and roughly square
                            if area < 0.4 * median_player_area and 0.5 <= aspect_ratio <= 2.0:
                                # Score by squareness (closer to 1.0 = more square)
                                squareness = 1.0 - abs(1.0 - aspect_ratio)
                                if track_id not in ball_squareness:
                                    ball_squareness[track_id] = []
                                ball_squareness[track_id].append(squareness)
                                self.ball_track_ids.add(track_id)
                            else:
                                continue  # Skip invalid ball candidates
                        
                        # Record first frame
                        if track_id not in self.track_first_frames:
                            self.track_first_frames[track_id] = frame_idx
                        
                        # Store box
                        if track_id not in self.track_boxes:
                            self.track_boxes[track_id] = {}
                        self.track_boxes[track_id][frame_idx] = box_coords
                        
                        # Assign team (only for players, not balls)
                        if not is_ball:
                            player_crop = frame[y1:y2, x1:x2]
                            if player_crop.size > 0:
                                jersey_color = self.team_assigner.extract_jersey_color(player_crop)
                                self.team_assigner.assign_team(track_id, jersey_color)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Filter to only the most square ball (only one ball in the game!)
        if len(ball_squareness) > 1:
            avg_squareness = {tid: np.mean(scores) for tid, scores in ball_squareness.items()}
            best_ball_id = max(avg_squareness, key=avg_squareness.get)
            print(f"[Tracker] Found {len(ball_squareness)} ball candidates, keeping most square one (ID {best_ball_id})")
            # Remove all other balls
            for ball_id in list(self.ball_track_ids):
                if ball_id != best_ball_id:
                    self.ball_track_ids.remove(ball_id)
                    if ball_id in self.track_first_frames:
                        del self.track_first_frames[ball_id]
                    if ball_id in self.track_boxes:
                        del self.track_boxes[ball_id]
        
        print(f"[Tracker] PASS 2 complete: Found {len(self.track_first_frames)} unique tracks")
        print(f"  Team 0: {list(self.team_assigner.track_teams.values()).count(0)} players")
        print(f"  Team 1: {list(self.team_assigner.track_teams.values()).count(1)} players")
        print(f"  Balls: {len(self.ball_track_ids)}")
    
    def build_sam2_prompts(self):
        """Build SAM2 prompts from YOLO tracks (mapped to extracted frame indices)."""
        print(f"[SAM2] Building prompts...")
        prompts = []
        
        for track_id, first_frame in self.track_first_frames.items():
            # Map original frame index to extracted frame index
            extracted_frame_idx = first_frame // self.frame_stride
            
            # Find closest frame we actually have a box for
            closest_frame = min(self.track_boxes[track_id].keys(), 
                              key=lambda x: abs(x - first_frame))
            box = self.track_boxes[track_id][closest_frame]
            
            # SAM2 prompt format: (obj_id, frame_idx, points, labels, box)
            prompts.append((track_id, extracted_frame_idx, None, None, box))
        
        print(f"[SAM2] Built {len(prompts)} prompts")
        return prompts
    
    def run_sam2(self, frame_dir, prompts):
        """Run SAM2 segmentation."""
        if build_sam2_video_predictor is None:
            raise RuntimeError("SAM2 is not installed. Install segment-anything-2.")
        
        if not self.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found at {self.sam2_checkpoint}")
        
        print(f"[SAM2] Loading {self.sam2_model} model from {self.sam2_checkpoint}...")
        sam_model = build_sam2_video_predictor(self.sam2_config, str(self.sam2_checkpoint))
        self.log_gpu_memory("After SAM2 model load")
        
        print(f"[SAM2] Initializing inference state...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = sam_model.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=self.offload_to_cpu,
                offload_state_to_cpu=self.offload_to_cpu,
                async_loading_frames=self.offload_to_cpu,
            )
        self.log_gpu_memory("After init_state (frames loaded)")
        
        print(f"[SAM2] Adding prompts and propagating...")
        video_segments = {}
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Add all prompts
            for obj_id, frame_idx, points, labels, box in tqdm(prompts, desc="Adding prompts"):
                sam_model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    box=box,
                )
            self.log_gpu_memory("After adding prompts")
            
            # Propagate backward
            print("[SAM2] Propagating backward...")
            for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(
                inference_state, reverse=True
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[j] > 0.0).cpu().numpy()
                    for j, out_obj_id in enumerate(out_obj_ids)
                }
            self.log_gpu_memory("After backward propagation")
            
            # Propagate forward
            print("[SAM2] Propagating forward...")
            for out_frame_idx, out_obj_ids, out_mask_logits in sam_model.propagate_in_video(
                inference_state, reverse=False
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[j] > 0.0).cpu().numpy()
                    for j, out_obj_id in enumerate(out_obj_ids)
                }
            self.log_gpu_memory("After forward propagation")
        
        sam_model.reset_state(inference_state)
        print(f"[SAM2] Segmentation complete!")
        return video_segments
    
    def render_with_masks(self, frame_dir, video_segments, total_original_frames):
        """PASS 3: Render video with colored SAM2 masks."""
        print(f"[Tracker] PASS 3: Rendering video with colored masks...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        # Read original video frame by frame
        cap = cv2.VideoCapture(str(self.video_path))
        
        for original_frame_idx in tqdm(range(total_original_frames), desc="Rendering frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Map to extracted frame index
            extracted_frame_idx = original_frame_idx // self.frame_stride
            
            if extracted_frame_idx in video_segments:
                # Create colored overlay
                overlay = frame.copy()
                
                for track_id, mask in video_segments[extracted_frame_idx].items():
                    # Check if this is a ball
                    is_ball = track_id in self.ball_track_ids
                    
                    if is_ball:
                        # Ball gets dark kelly green mask (#016F32)
                        color = (50, 111, 1)  # BGR format
                        label = "BALL"
                    else:
                        # Players get team colors
                        team = self.team_assigner.track_teams.get(track_id)
                        if team is not None:
                            color = self.team_colors[team]
                            label = f"T{team}"
                        else:
                            continue
                    
                    # Apply colored mask
                    mask_bool = mask[0] > 0
                    overlay[mask_bool] = color
                    
                    # Add text label (find top-left corner of mask)
                    mask_coords = np.argwhere(mask_bool)
                    if len(mask_coords) > 0:
                        y_min = mask_coords[:, 0].min()
                        x_min = mask_coords[:, 1].min()
                        cv2.putText(frame, label, (x_min, y_min - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Blend overlay with original frame (more transparent)
                alpha = 0.4
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"\n[Tracker] Complete! Output saved to: {self.output_path}")
    
    def run(self):
        """Execute full pipeline: sampling + tracking + SAM2 + rendering."""
        # PASS 1: Collect colors via random sampling
        jersey_colors = self.random_sample_colors()
        
        if len(jersey_colors) < 2:
            print("[ERROR] Not enough jersey colors collected. Aborting.")
            return
        
        # Initialize K-means
        self.team_assigner.initialize_teams(jersey_colors)
        self.team_colors = self.team_assigner.get_team_colors()
        print(f"[Tracker] Using actual jersey colors for masks:")
        print(f"  Team 0: {self.team_colors[0]} (BGR)")
        print(f"  Team 1: {self.team_colors[1]} (BGR)")
        
        # PASS 2: Collect track data and assign teams
        self.collect_tracks_and_teams()
        
        # Create temp directory for frames
        temp_dir = Path(tempfile.mkdtemp(prefix="gaa_frames_"))
        print(f"[SAM2] Using temp directory: {temp_dir}")
        
        try:
            # Extract frames
            total_original_frames, total_extracted_frames = self.extract_frames(temp_dir)
            
            # Build SAM2 prompts
            prompts = self.build_sam2_prompts()
            
            # Run SAM2
            video_segments = self.run_sam2(temp_dir, prompts)
            
            # Render with colored masks
            self.render_with_masks(temp_dir, video_segments, total_original_frames)
            
        finally:
            # Cleanup temp directory
            print(f"[SAM2] Cleaning up temp directory...")
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="GAA Team Tracker with SAM2 colored masks")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path")
    parser.add_argument("--sam2-model", type=str, default="tiny", 
                       choices=["tiny", "small", "base", "large"],
                       help="SAM2 model size (default: tiny for speed)")
    parser.add_argument("--frame-stride", type=int, default=2,
                       help="Process every Nth frame for SAM2 (default: 2 for speed)")
    parser.add_argument("--sample-frames", type=int, default=50, 
                       help="Number of random frames to sample for K-means initialization (default: 50)")
    parser.add_argument("--max-duration", type=float, default=None,
                       help="Limit processing to first N seconds (for memory testing)")
    parser.add_argument("--offload-to-cpu", action="store_true",
                       help="Offload SAM2 video frames and state to CPU (slower but uses ~80%% less GPU memory)")
    args = parser.parse_args()
    
    tracker = GAATeamTrackerSAM2(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        sam2_model=args.sam2_model,
        frame_stride=args.frame_stride,
        sample_frames=args.sample_frames,
        max_duration=args.max_duration,
        offload_to_cpu=args.offload_to_cpu
    )
    
    tracker.run()


if __name__ == "__main__":
    main()

