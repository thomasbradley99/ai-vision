#!/usr/bin/env python3
"""
GAA Team Tracker - Intelligent Team Detection & Tracking

This script combines:
1. YOLO detection (football-specific model)
2. ByteTrack for stable player tracking
3. Smart K-Means clustering for team assignment
4. (Future) SAM2 segmentation masks

The key innovation: Assign team colors PER TRACK (not per frame) to avoid flickering.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import random

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans


class TeamColorAssigner:
    """
    Intelligently assigns team colors to tracked players.
    Collects jersey colors over multiple frames before making a decision.
    """
    
    def __init__(self, warmup_frames=30):
        self.warmup_frames = warmup_frames
        self.team_centroids = None  # K-Means centroids for 2 teams
        self.track_colors = defaultdict(list)  # track_id -> list of observed colors
        self.track_teams = {}  # track_id -> team assignment (0 or 1)
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
        """
        Run K-Means on collected jersey colors to find 2 team centroids.
        Call this after collecting colors from warmup frames.
        """
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
        """
        Assign a track to a team based on its jersey color.
        If already assigned, return cached result.
        """
        if track_id in self.track_teams:
            return self.track_teams[track_id]
        
        if self.team_centroids is None:
            return None  # Teams not initialized yet
        
        # Assign based on closest centroid
        team = self.team_centroids.predict([jersey_color])[0]
        self.track_teams[track_id] = team
        return team
    
    def collect_color(self, track_id, jersey_color):
        """Collect jersey color observation for a track."""
        self.track_colors[track_id].append(jersey_color)


class GAATeamTracker:
    """Main tracking and rendering pipeline."""
    
    def __init__(self, video_path, output_path, model_path, sample_frames=50):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        self.sample_frames = sample_frames
        
        # Load YOLO model
        print(f"[Tracker] Loading YOLO model from {model_path}")
        self.model = YOLO(str(model_path))
        
        # Initialize team assigner
        self.team_assigner = TeamColorAssigner(warmup_frames=sample_frames)
        
        # Colors for visualization (will be set from K-means centroids)
        self.team_colors = None
    
    def random_sample_colors(self):
        """
        PASS 1: Random sampling to collect jersey colors for K-means initialization.
        Returns list of jersey colors from randomly sampled frames.
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
    
    def process_video(self):
        """
        PASS 2: Main processing with team-colored tracking.
        Assumes teams have already been initialized via random sampling.
        """
        print(f"[Tracker] PASS 2: Full video processing with team tracking...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Tracker] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Run YOLO detection with tracking
            results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                out.write(frame)
                continue
            
            # Process detections
            boxes = results[0].boxes
            
            # First pass: collect player sizes for ball filtering
            player_areas = []
            for box in boxes:
                cls = int(box.cls.cpu().numpy()[0])
                if cls == 0:  # Player
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)
                    player_areas.append(area)
            
            median_player_area = np.median(player_areas) if len(player_areas) > 0 else float('inf')
            
            # Second pass: render with ball filtering
            for box in boxes:
                # Get detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls.cpu().numpy()[0])
                
                # Get track ID if available
                track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                
                if track_id is None:
                    continue
                
                # Handle ball separately with filtering
                if cls != 0:  # Not a player (likely ball)
                    # Ball validation filters
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Filter 1: Ball must be < 40% of median player size
                    # Filter 2: Ball must be roughly square (aspect ratio 0.5-2.0)
                    if area < 0.4 * median_player_area and 0.5 <= aspect_ratio <= 2.0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)  # White box
                        cv2.putText(frame, "BALL", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Otherwise skip (likely false positive)
                    continue
                
                # Extract player crop and jersey color
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
            
            # Write frame
            out.write(frame)
            
            # Progress
            if frame_idx % 30 == 0:
                print(f"[Tracker] Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"\n[Tracker] Complete! Output saved to: {self.output_path}")
        print(f"[Tracker] Tracked {len(self.team_assigner.track_teams)} unique players")
        print(f"  Team 0: {list(self.team_assigner.track_teams.values()).count(0)} players")
        print(f"  Team 1: {list(self.team_assigner.track_teams.values()).count(1)} players")
    
    def run(self):
        """Execute full pipeline: random sampling + processing."""
        # PASS 1: Collect colors via random sampling
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
        print(f"  Team 1: {self.team_colors[1]} (BGR)")
        
        # PASS 2: Process full video with team tracking
        self.process_video()


def main():
    parser = argparse.ArgumentParser(description="GAA Team Tracker with intelligent clustering via random sampling")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path")
    parser.add_argument("--sample-frames", type=int, default=50, 
                       help="Number of random frames to sample for K-means initialization (default: 50)")
    args = parser.parse_args()
    
    tracker = GAATeamTracker(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        sample_frames=args.sample_frames
    )
    
    tracker.run()


if __name__ == "__main__":
    main()

