#!/usr/bin/env python3
"""
Hybrid Approach: Team Bounding Boxes + Ball SAM2 Mask
- Players: Colored bounding boxes with team tracking (fast, reliable)
- Ball: SAM2 yellow segmentation mask (impressive, accurate)
"""

import cv2
import torch
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
from sam2.build_sam import build_sam2_video_predictor


class TeamColorAssigner:
    """Assign players to teams based on jersey colors."""
    
    def __init__(self, warmup_frames=50):
        self.warmup_frames = warmup_frames
        self.team_centroids = None
        self.track_teams = {}
        self.track_colors = {}
    
    def extract_jersey_color(self, player_crop):
        """Extract dominant jersey color from player crop (middle region)."""
        if player_crop.size == 0:
            return None
        
        h, w = player_crop.shape[:2]
        torso = player_crop[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]
        
        if torso.size == 0:
            return None
        
        pixels = torso.reshape(-1, 3)
        mean_color = np.median(pixels, axis=0)
        return mean_color
    
    def initialize_teams(self, all_jersey_colors):
        """Initialize K-means clustering with collected colors."""
        if len(all_jersey_colors) < 2:
            return False
        
        colors_array = np.array(all_jersey_colors)
        self.team_centroids = KMeans(n_clusters=2, random_state=42)
        self.team_centroids.fit(colors_array)
        return True
    
    def get_team_colors(self):
        """Return team centroids as BGR tuples."""
        if self.team_centroids is None:
            return None
        return {
            0: tuple(map(int, self.team_centroids.cluster_centers_[0])),
            1: tuple(map(int, self.team_centroids.cluster_centers_[1]))
        }
    
    def assign_team(self, track_id, jersey_color):
        """Assign track to team (cached)."""
        if track_id in self.track_teams:
            return self.track_teams[track_id]
        
        if self.team_centroids is None:
            return None
        
        team = self.team_centroids.predict([jersey_color])[0]
        self.track_teams[track_id] = team
        return team


class HybridTeamBallTracker:
    """Team tracking with bounding boxes + Ball SAM2 mask."""
    
    def __init__(self, video_path, output_path, model_path, sam2_checkpoint, 
                 sample_frames=50, frame_stride=2):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        self.sam2_checkpoint = Path(sam2_checkpoint)
        self.sample_frames = sample_frames
        self.frame_stride = frame_stride
        
        # Load YOLO
        print(f"[Tracker] Loading YOLO model from {model_path}")
        self.model = YOLO(str(model_path))
        
        # Load SAM2
        print(f"[SAM2] Loading SAM2 model from {sam2_checkpoint}")
        sam_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_model = build_sam2_video_predictor(sam_config, str(sam2_checkpoint), device=device)
        
        # Team assigner
        self.team_assigner = TeamColorAssigner(warmup_frames=sample_frames)
        self.team_colors = None
        
        # Ball tracking
        self.ball_track_id = None
        self.ball_first_frame = None
        self.ball_box = None
    
    def random_sample_colors(self):
        """PASS 1: Sample jersey colors for K-means."""
        print(f"[Tracker] PASS 1: Random sampling {self.sample_frames} frames...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sample_indices = sorted(random.sample(range(total_frames), min(self.sample_frames, total_frames)))
        all_jersey_colors = []
        
        for idx in tqdm(sample_indices, desc="Sampling"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
            if results[0].boxes is None:
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
        
        cap.release()
        
        # Initialize teams
        self.team_assigner.initialize_teams(all_jersey_colors)
        self.team_colors = self.team_assigner.get_team_colors()
        
        print(f"[Tracker] Collected {len(all_jersey_colors)} jersey color samples")
        print(f"  Team 0 color: {self.team_colors[0]}")
        print(f"  Team 1 color: {self.team_colors[1]}")
        return all_jersey_colors
    
    def track_teams_and_find_ball(self):
        """PASS 2: Track teams + find best ball."""
        print(f"[Tracker] PASS 2: Tracking teams and finding ball...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ball_squareness = {}
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Tracking") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
                
                if results[0].boxes is None:
                    frame_idx += 1
                    pbar.update(1)
                    continue
                
                # Get player sizes
                player_areas = []
                for box in results[0].boxes:
                    cls = int(box.cls.cpu().numpy()[0])
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        player_areas.append((x2 - x1) * (y2 - y1))
                
                median_player_area = np.median(player_areas) if player_areas else float('inf')
                
                # Track players and balls
                for box in results[0].boxes:
                    cls = int(box.cls.cpu().numpy()[0])
                    track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                    
                    if track_id is None:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Track ball
                    if cls != 0:
                        if area < 0.4 * median_player_area and 0.5 <= aspect_ratio <= 2.0:
                            squareness = 1.0 - abs(1.0 - aspect_ratio)
                            if track_id not in ball_squareness:
                                ball_squareness[track_id] = []
                            ball_squareness[track_id].append((squareness, frame_idx, [x1, y1, x2, y2]))
                    # Assign teams
                    else:
                        player_crop = frame[y1:y2, x1:x2]
                        if player_crop.size > 0:
                            jersey_color = self.team_assigner.extract_jersey_color(player_crop)
                            self.team_assigner.assign_team(track_id, jersey_color)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Find best ball
        if ball_squareness:
            avg_squareness = {tid: np.mean([s[0] for s in scores]) 
                             for tid, scores in ball_squareness.items()}
            best_ball_id = max(avg_squareness, key=avg_squareness.get)
            ball_data = sorted(ball_squareness[best_ball_id], key=lambda x: x[1])
            middle_idx = len(ball_data) // 2
            
            self.ball_track_id = best_ball_id
            self.ball_first_frame = ball_data[middle_idx][1]
            self.ball_box = ball_data[middle_idx][2]
            
            print(f"[Tracker] Found ball ID {best_ball_id} (squareness: {avg_squareness[best_ball_id]:.3f})")
            print(f"  Prompt frame: {self.ball_first_frame}, Box: {self.ball_box}")
        else:
            print("[WARNING] No ball found!")
        
        print(f"  Team 0: {list(self.team_assigner.track_teams.values()).count(0)} players")
        print(f"  Team 1: {list(self.team_assigner.track_teams.values()).count(1)} players")
    
    def extract_frames(self, frame_dir):
        """PASS 3: Extract frames for SAM2."""
        print(f"[SAM2] Extracting frames (stride={self.frame_stride})...")
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        extracted = 0
        
        with tqdm(total=total_frames, desc="Extracting") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % self.frame_stride == 0:
                    cv2.imwrite(str(frame_dir / f"{extracted:06d}.jpg"), frame)
                    extracted += 1
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        print(f"[SAM2] Extracted {extracted} frames")
        return extracted
    
    def run_sam2_on_ball(self, frame_dir):
        """PASS 4: Run SAM2 on ball only."""
        if self.ball_track_id is None:
            print("[WARNING] No ball to segment!")
            return {}
        
        print(f"[SAM2] Running segmentation on ball...")
        sam_frame_idx = self.ball_first_frame // self.frame_stride
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.sam2_model.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            
            # Add prompt
            self.sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=sam_frame_idx,
                obj_id=1,
                box=np.array(self.ball_box, dtype=np.float32),
            )
            
            # Propagate forward
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(
                inference_state, start_frame_idx=sam_frame_idx, reverse=False
            ):
                video_segments[out_frame_idx] = {
                    1: (out_mask_logits[0] > 0.0).cpu().numpy()
                }
            
            # Propagate backward
            self.sam2_model.reset_state(inference_state)
            self.sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=sam_frame_idx,
                obj_id=1,
                box=np.array(self.ball_box, dtype=np.float32),
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(
                inference_state, start_frame_idx=sam_frame_idx, reverse=True
            ):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {
                        1: (out_mask_logits[0] > 0.0).cpu().numpy()
                    }
        
        print(f"[SAM2] Segmented ball in {len(video_segments)} frames")
        return video_segments
    
    def render_hybrid(self, ball_segments):
        """PASS 5: Render players with boxes + ball with mask."""
        print(f"[Renderer] Rendering hybrid video...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Rendering") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
                
                if results[0].boxes is not None:
                    # Draw player bounding boxes
                    for box in results[0].boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                        
                        if track_id is None or cls != 0:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        team = self.team_assigner.assign_team(track_id, None)
                        
                        if team is not None:
                            color = self.team_colors[team]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, f"T{team}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Overlay ball SAM2 mask
                sam_frame_idx = frame_idx // self.frame_stride
                if sam_frame_idx in ball_segments and 1 in ball_segments[sam_frame_idx]:
                    mask = ball_segments[sam_frame_idx][1]
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    # Yellow overlay
                    overlay = frame.copy()
                    overlay[mask] = (0, 255, 255)  # Yellow
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                    
                    # Label
                    mask_coords = np.where(mask)
                    if len(mask_coords[0]) > 0:
                        y_min = mask_coords[0].min()
                        x_min = mask_coords[1].min()
                        cv2.putText(frame, "BALL", (x_min, max(y_min - 10, 20)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        print(f"[Renderer] Output saved to {self.output_path}")
    
    def run(self):
        """Main pipeline."""
        print(f"=== Hybrid Team BB + Ball SAM2 Pipeline ===")
        
        # PASS 1: Sample colors
        self.random_sample_colors()
        
        # PASS 2: Track teams + find ball
        self.track_teams_and_find_ball()
        
        # PASS 3: Extract frames
        frame_dir = Path("/tmp/hybrid_frames")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        self.extract_frames(frame_dir)
        
        # PASS 4: SAM2 on ball
        ball_segments = self.run_sam2_on_ball(frame_dir)
        
        # PASS 5: Render
        self.render_hybrid(ball_segments)
        
        # Cleanup
        shutil.rmtree(frame_dir)
        print(f"=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Hybrid: Team Boxes + Ball SAM2")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--output", required=True, help="Output video")
    parser.add_argument("--model", required=True, help="YOLO model")
    parser.add_argument("--sam2-checkpoint", 
                       default="../hooper-glean/checkpoints/SAM2-InstanceSegmentation/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sample-frames", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=2)
    
    args = parser.parse_args()
    
    tracker = HybridTeamBallTracker(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        sam2_checkpoint=args.sam2_checkpoint,
        sample_frames=args.sample_frames,
        frame_stride=args.frame_stride,
    )
    
    tracker.run()


if __name__ == "__main__":
    main()

