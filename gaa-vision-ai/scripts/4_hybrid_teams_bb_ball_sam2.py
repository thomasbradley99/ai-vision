#!/usr/bin/env python3
"""
Hybrid Approach: Team Bounding Boxes + Ball SAM2 Mask
- Players: Colored bounding boxes with team tracking (EXACT script 1 logic)
- Ball: SAM2 yellow segmentation mask (bidirectional tracking)
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
from collections import defaultdict
from sam2.build_sam import build_sam2_video_predictor


class TeamColorAssigner:
    """
    Intelligently assigns team colors to tracked players.
    IDENTICAL to script 1 - includes grass filtering.
    """
    
    def __init__(self, warmup_frames=50):
        self.warmup_frames = warmup_frames
        self.team_centroids = None
        self.track_colors = defaultdict(list)
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
        """
        Extract jersey color from player crop.
        Removes grass background and focuses on upper body.
        IDENTICAL to script 1.
        """
        if self.grass_hsv is None:
            # Fallback if grass color not set
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([80, 255, 255])
        else:
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([self.grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([self.grass_hsv[0, 0, 0] + 10, 255, 255])
        
        # Mask out grass
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        
        # Focus on upper half (jersey, not shorts)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0]//2, :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        
        # Get average color
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        return kit_color
    
    def initialize_teams(self, all_jersey_colors):
        """Initialize K-means clustering with collected colors."""
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
    
    def collect_color(self, track_id, jersey_color):
        """Collect jersey color observation for a track."""
        self.track_colors[track_id].append(jersey_color)


class HybridTeamBallTracker:
    """
    Script 1 logic for team tracking + SAM2 ball mask at the end.
    Uses EXACT script 1 approach, then adds SAM2 bidirectional ball tracking.
    """
    
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
        
        # Team assigner (with grass filtering like script 1)
        self.team_assigner = TeamColorAssigner(warmup_frames=sample_frames)
        self.team_colors = None
        
        # Ball tracking data
        self.ball_detections = []  # Store all ball detections for SAM2
        self.ball_frames = set()  # Frames where YOLO detected valid ball
        
        # Store tracking results from Pass 2 for replay in Pass 5
        self.tracking_results = []  # List of (frame_idx, boxes_data) tuples
        
        # SAM2 (lazy load)
        self.sam2_model = None
    
    def random_sample_colors(self):
        """
        PASS 1: Random sampling to collect jersey colors for K-means initialization.
        IDENTICAL to script 1.
        """
        print(f"[Tracker] PASS 1: Random sampling {self.sample_frames} frames...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generate random frame indices
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
            
            # Run YOLO detection (without tracking in this pass)
            results = self.model(frame, conf=0.3, verbose=False)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            # Extract jersey colors from all detected players
            for box in results[0].boxes:
                cls = int(box.cls.cpu().numpy()[0])
                
                # Filter: only process players (class 0), skip ball/referee/etc
                if cls != 0:
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
    
    def track_teams_and_find_ball(self):
        """
        PASS 2: Track teams + find best ball.
        Uses script 1's EXACT team assignment logic with grass filtering.
        """
        print(f"[Tracker] PASS 2: Tracking teams and finding ball (script 1 logic)...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ball_squareness = {}
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Run YOLO detection with tracking (EXACT script 1 approach)
            results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                self.tracking_results.append((frame_idx - 1, []))
                continue
            
            boxes = results[0].boxes
            
            # Store boxes for replay in Pass 5 (ensure temporal consistency!)
            boxes_data = []
            for box in boxes:
                box_dict = {
                    'xyxy': box.xyxy[0].cpu().numpy(),
                    'cls': int(box.cls.cpu().numpy()[0]),
                    'track_id': int(box.id.cpu().numpy()[0]) if box.id is not None else None,
                    'conf': float(box.conf.cpu().numpy()[0])
                }
                boxes_data.append(box_dict)
            self.tracking_results.append((frame_idx - 1, boxes_data))
            
            # First pass: collect player sizes for ball filtering (EXACT script 1)
            player_areas = []
            for box in boxes:
                cls = int(box.cls.cpu().numpy()[0])
                if cls == 0:  # Player
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)
                    player_areas.append(area)
            
            median_player_area = np.median(player_areas) if len(player_areas) > 0 else float('inf')
            
            # Second pass: find ball candidates (EXACT script 1 filtering)
            ball_candidates = []
            for box in boxes:
                cls = int(box.cls.cpu().numpy()[0])
                track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                
                if cls != 0 and track_id is not None:  # Not a player (likely ball)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Filter 1: Ball must be < 40% of median player size
                    # Filter 2: Ball must be roughly square (aspect ratio 0.5-2.0)
                    if area < 0.4 * median_player_area and 0.5 <= aspect_ratio <= 2.0:
                        # Score by how close aspect ratio is to 1.0 (perfect square)
                        squareness_score = 1.0 - abs(1.0 - aspect_ratio)
                        ball_candidates.append((squareness_score, x1, y1, x2, y2, track_id))
                        
                        # Store for SAM2
                        if track_id not in ball_squareness:
                            ball_squareness[track_id] = []
                        ball_squareness[track_id].append((squareness_score, frame_idx-1, [x1, y1, x2, y2]))
            
            # Track which frame has valid ball detection (most square one)
            if ball_candidates:
                best_ball_in_frame = max(ball_candidates, key=lambda x: x[0])
                self.ball_frames.add(frame_idx - 1)  # Store frame index where ball was detected
            
            # Third pass: assign teams to players (EXACT script 1 logic)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls.cpu().numpy()[0])
                track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                
                if track_id is None or cls != 0:
                    continue
                
                # Extract player crop and jersey color (with grass filtering!)
                player_crop = frame[y1:y2, x1:x2]
                if player_crop.size == 0:
                    continue
                
                jersey_color = self.team_assigner.extract_jersey_color(player_crop)
                self.team_assigner.assign_team(track_id, jersey_color)
            
            # Progress
            if frame_idx % 30 == 0:
                print(f"[Tracker] Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
        
        cap.release()
        
        # Collect ALL ball detections from ALL track IDs for multi-anchor SAM2
        if ball_squareness:
            # Merge all detections from all track IDs
            all_ball_detections = []
            for track_id, detections in ball_squareness.items():
                all_ball_detections.extend(detections)
            
            # Sort by frame index
            all_ball_detections = sorted(all_ball_detections, key=lambda x: x[1])
            
            # Find best track ID (for logging)
            avg_squareness = {tid: np.mean([s[0] for s in scores]) 
                             for tid, scores in ball_squareness.items()}
            best_ball_id = max(avg_squareness, key=avg_squareness.get)
            
            # Store ALL ball detections for multi-anchor SAM2
            self.ball_track_id = best_ball_id
            self.ball_detections = all_ball_detections  # ALL detections from ALL track IDs
            
            print(f"[Tracker] Found {len(ball_squareness)} ball track IDs")
            print(f"  Best ball ID: {best_ball_id} (squareness: {avg_squareness[best_ball_id]:.3f})")
            print(f"  Total ball detections for SAM2 anchoring: {len(all_ball_detections)}")
        else:
            print("[WARNING] No ball found!")
            self.ball_track_id = None
            self.ball_detections = []
        
        print(f"[Tracker] Tracked {len(self.team_assigner.track_teams)} unique players")
        print(f"  Team 0: {list(self.team_assigner.track_teams.values()).count(0)} players")
        print(f"  Team 1: {list(self.team_assigner.track_teams.values()).count(1)} players")
        print(f"  Ball detected in {len(self.ball_frames)}/{total_frames} frames ({100*len(self.ball_frames)/total_frames:.1f}%)")
    
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
    
    def load_sam2(self):
        """Lazy load SAM2 model."""
        if self.sam2_model is None:
            print(f"[SAM2] Loading SAM2 model from {self.sam2_checkpoint}")
            sam_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sam2_model = build_sam2_video_predictor(sam_config, str(self.sam2_checkpoint), device=device)
    
    def run_sam2_on_ball(self, frame_dir):
        """
        PASS 4: Run SAM2 with TRUE BIDIRECTIONAL tracking.
        - Forward pass: Track from earliest anchor to end
        - Backward pass: Track from latest anchor to start
        - Merge: Prefer masks closer to their anchor points (less drift)
        """
        if self.ball_track_id is None or len(self.ball_detections) == 0:
            print("[WARNING] No ball to segment!")
            return {}
        
        self.load_sam2()
        
        print(f"[SAM2] Bidirectional tracking with {len(self.ball_detections)} YOLO ball detections...")
        
        # Select anchor points
        anchor_interval = max(1, len(self.ball_detections) // 10)  # ~10 anchors
        anchor_detections = self.ball_detections[::anchor_interval]
        
        # Get first and last anchors for bidirectional tracking
        first_anchor = anchor_detections[0]  # (squareness, frame_idx, box)
        last_anchor = anchor_detections[-1]
        
        print(f"[SAM2] First anchor: frame {first_anchor[1]}")
        print(f"[SAM2] Last anchor: frame {last_anchor[1]}")
        print(f"[SAM2] Using {len(anchor_detections)} total anchors")
        
        forward_segments = {}
        backward_segments = {}
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # FORWARD PASS: Track from first anchor onwards
            print(f"[SAM2] === FORWARD PASS ===")
            inference_state_fwd = self.sam2_model.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            
            # Add all anchors for forward tracking
            for idx, (squareness, frame_idx, box) in enumerate(anchor_detections):
                sam_frame_idx = frame_idx // self.frame_stride
                self.sam2_model.add_new_points_or_box(
                    inference_state=inference_state_fwd,
                    frame_idx=sam_frame_idx,
                    obj_id=1,
                    box=np.array(box, dtype=np.float32),
                )
            
            print(f"[SAM2] Propagating forward from frame {first_anchor[1]}...")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state_fwd):
                forward_segments[out_frame_idx] = {
                    'mask': (out_mask_logits[0] > 0.0).cpu().numpy(),
                    'confidence': float(out_mask_logits[0].max().cpu().numpy())
                }
            
            print(f"[SAM2] Forward pass: {len(forward_segments)} frames")
            
            # BACKWARD PASS: Track from last anchor backwards
            print(f"[SAM2] === BACKWARD PASS ===")
            inference_state_bwd = self.sam2_model.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            
            # Add all anchors for backward tracking (reversed order to prioritize latest)
            for idx, (squareness, frame_idx, box) in enumerate(reversed(anchor_detections)):
                sam_frame_idx = frame_idx // self.frame_stride
                self.sam2_model.add_new_points_or_box(
                    inference_state=inference_state_bwd,
                    frame_idx=sam_frame_idx,
                    obj_id=1,
                    box=np.array(box, dtype=np.float32),
                )
            
            print(f"[SAM2] Propagating backward from frame {last_anchor[1]}...")
            last_sam_frame_idx = last_anchor[1] // self.frame_stride
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(
                inference_state_bwd, 
                start_frame_idx=last_sam_frame_idx,
                reverse=True
            ):
                backward_segments[out_frame_idx] = {
                    'mask': (out_mask_logits[0] > 0.0).cpu().numpy(),
                    'confidence': float(out_mask_logits[0].max().cpu().numpy())
                }
            
            print(f"[SAM2] Backward pass: {len(backward_segments)} frames")
        
        # MERGE: Combine forward and backward, preferring masks closer to anchors
        print(f"[SAM2] Merging bidirectional results...")
        video_segments = {}
        
        # Get anchor frame indices
        anchor_frames = {det[1] // self.frame_stride for det in anchor_detections}
        
        all_frames = set(forward_segments.keys()) | set(backward_segments.keys())
        for frame_idx in all_frames:
            fwd = forward_segments.get(frame_idx)
            bwd = backward_segments.get(frame_idx)
            
            # If only one exists, use it
            if fwd and not bwd:
                video_segments[frame_idx] = {1: fwd['mask']}
            elif bwd and not fwd:
                video_segments[frame_idx] = {1: bwd['mask']}
            elif fwd and bwd:
                # Both exist - prefer the one with higher confidence
                if fwd['confidence'] >= bwd['confidence']:
                    video_segments[frame_idx] = {1: fwd['mask']}
                else:
                    video_segments[frame_idx] = {1: bwd['mask']}
        
        print(f"[SAM2] Final merged: {len(video_segments)} frames (bidirectional)")
        return video_segments
    
    def render_hybrid(self, ball_segments):
        """
        PASS 5: Render with script 1's EXACT player logic + SAM2 ball overlay.
        Now uses STORED tracking results from Pass 2 for temporal consistency!
        """
        print(f"[Renderer] Rendering hybrid video using stored tracking results...")
        print(f"[Renderer] Stored {len(self.tracking_results)} frames of tracking data")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Renderer] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        for stored_frame_idx, boxes_data in self.tracking_results:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use stored tracking results (ensures temporal consistency!)
            if len(boxes_data) > 0:
                
                # First pass: collect player sizes for ball filtering (EXACT script 1)
                player_areas = []
                for box_dict in boxes_data:
                    cls = box_dict['cls']
                    if cls == 0:  # Player
                        x1, y1, x2, y2 = map(int, box_dict['xyxy'])
                        area = (x2 - x1) * (y2 - y1)
                        player_areas.append(area)
                
                median_player_area = np.median(player_areas) if len(player_areas) > 0 else float('inf')
                
                # Second pass: find the best ball candidate (most square) - EXACT script 1
                ball_candidates = []
                for box_dict in boxes_data:
                    cls = box_dict['cls']
                    track_id = box_dict['track_id']
                    
                    if cls != 0 and track_id is not None:  # Not a player (likely ball)
                        x1, y1, x2, y2 = map(int, box_dict['xyxy'])
                        width_b = x2 - x1
                        height_b = y2 - y1
                        area = width_b * height_b
                        aspect_ratio = height_b / width_b if width_b > 0 else 0
                        
                        # Filter 1: Ball must be < 40% of median player size
                        # Filter 2: Ball must be roughly square (aspect ratio 0.5-2.0)
                        if area < 0.4 * median_player_area and 0.5 <= aspect_ratio <= 2.0:
                            # Score by how close aspect ratio is to 1.0 (perfect square)
                            squareness_score = 1.0 - abs(1.0 - aspect_ratio)
                            ball_candidates.append((squareness_score, x1, y1, x2, y2, track_id))
                
                # Pick the most square ball (only one ball in the game!)
                best_ball = None
                if ball_candidates:
                    best_ball = max(ball_candidates, key=lambda x: x[0])
                
                # Third pass: render ALL detections with bounding boxes (EXACT script 1)
                for box_dict in boxes_data:
                    x1, y1, x2, y2 = map(int, box_dict['xyxy'])
                    cls = box_dict['cls']
                    track_id = box_dict['track_id']
                    
                    if track_id is None:
                        continue
                    
                    # Handle ball - render white bounding box if it's the best ball (EXACT script 1)
                    if cls != 0:  # Not a player (likely ball)
                        if best_ball and track_id == best_ball[5]:
                            _, bx1, by1, bx2, by2, _ = best_ball
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 3)  # White box
                            cv2.putText(frame, "BALL", (bx1, by1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        continue
                    
                    # Handle players - extract player crop and jersey color (with grass filtering!)
                    player_crop = frame[y1:y2, x1:x2]
                    if player_crop.size == 0:
                        continue
                    
                    jersey_color = self.team_assigner.extract_jersey_color(player_crop)
                    
                    # Assign team (will be cached after first assignment)
                    team = self.team_assigner.assign_team(track_id, jersey_color)
                    
                    if team is not None:
                        color = self.team_colors[team]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, f"T{team}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        # Fallback if team not assigned yet (shouldn't happen)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Overlay SAM2 ball mask (continuous tracking, fills gaps where YOLO misses)
            # Validate mask to ensure it's tracking the ball (circular) not hair (elongated)
            sam_frame_idx = frame_idx // self.frame_stride
            if sam_frame_idx in ball_segments and 1 in ball_segments[sam_frame_idx]:
                mask = ball_segments[sam_frame_idx][1]
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                # Validate mask shape - ball should be roughly circular
                mask_coords = np.where(mask)
                if len(mask_coords[0]) > 0:
                    y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
                    x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
                    
                    mask_width = x_max - x_min
                    mask_height = y_max - y_min
                    mask_area = np.sum(mask)
                    
                    # Check if mask is valid (circular ball, not elongated hair)
                    if mask_width > 0 and mask_height > 0:
                        aspect_ratio = mask_height / mask_width
                        bbox_area = mask_width * mask_height
                        
                        # Ball should be:
                        # 1. Roughly square (aspect ratio 0.5-2.0, like YOLO filter)
                        # 2. Small enough (< 15000 pixels - roughly 120x120)
                        # 3. Compact (mask fills >40% of bounding box = circular)
                        fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0
                        
                        if (0.5 <= aspect_ratio <= 2.0 and 
                            mask_area < 15000 and 
                            fill_ratio > 0.4):
                            # Valid ball mask - overlay Clann green (#016F32)
                            overlay = frame.copy()
                            overlay[mask] = (50, 111, 1)  # Clann Dark Kelly Green (BGR: #016F32)
                            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # 40% mask, 60% frame (more transparent)
                            
                            # Add white bounding box and "BALL" text (same as YOLO detection)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 3)
                            cv2.putText(frame, "BALL", (x_min, y_min-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            
            # Progress
            if frame_idx % 30 == 0:
                print(f"[Renderer] Rendered {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"\n[Renderer] Complete! Output saved to: {self.output_path}")
    
    def run(self):
        """
        Main pipeline: Script 1 logic + SAM2 ball at end.
        PASS 1: Random sample colors (script 1)
        PASS 2: Track teams + find ball (script 1 logic)
        PASS 3: Extract frames for SAM2
        PASS 4: SAM2 bidirectional ball tracking
        PASS 5: Render with script 1 player logic + SAM2 ball
        """
        print(f"=== Hybrid: Script 1 Team Tracking + SAM2 Ball ===\n")
        
        # PASS 1: Collect colors via random sampling (EXACT script 1)
        jersey_colors = self.random_sample_colors()
        
        if len(jersey_colors) < 2:
            print("[ERROR] Not enough jersey colors collected. Aborting.")
            return
        
        # Initialize K-means with collected colors
        self.team_assigner.initialize_teams(jersey_colors)
        
        # Set team colors from actual jersey centroids
        self.team_colors = self.team_assigner.get_team_colors()
        print(f"[Tracker] Using actual jersey colors for bounding boxes:")
        print(f"  Team 0: {self.team_colors[0]} (BGR)")
        print(f"  Team 1: {self.team_colors[1]} (BGR)\n")
        
        # PASS 2: Track teams + find ball (script 1 logic with ball collection)
        self.track_teams_and_find_ball()
        
        # If no ball found, skip this clip
        if self.ball_track_id is None or len(self.ball_detections) == 0:
            print("[WARNING] No ball found! Skipping this clip.")
            return
        
        # PASS 3: Extract frames for SAM2
        frame_dir = Path("/tmp/hybrid_frames")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        self.extract_frames(frame_dir)
        
        # PASS 4: SAM2 bidirectional on ball
        ball_segments = self.run_sam2_on_ball(frame_dir)
        
        # PASS 5: Render with script 1 player logic + SAM2 ball
        self.render_hybrid(ball_segments)
        
        # Cleanup
        shutil.rmtree(frame_dir)
        print(f"\n=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Hybrid: Team Boxes + Ball SAM2")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--output", required=True, help="Output video")
    parser.add_argument("--model", required=True, help="YOLO model")
    parser.add_argument("--sam2-checkpoint", 
                       default="/home/ubuntu/clann/ai-vision/hooper-glean/checkpoints/SAM2-InstanceSegmentation/sam2.1_hiera_tiny.pt")
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

