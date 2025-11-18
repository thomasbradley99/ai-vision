#!/usr/bin/env python3
"""
SAM2 Ball Segmentation with Green Mask
Tracks only the ball using SAM2 for pixel-perfect segmentation.
Uses frame extraction approach (Hooper method) to avoid GPU memory issues.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import shutil
from sam2.build_sam import build_sam2_video_predictor


class BallSAM2Tracker:
    """Track ball with YOLO + SAM2 segmentation."""
    
    def __init__(self, video_path, output_path, model_path, sam2_checkpoint, frame_stride=2):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        self.sam2_checkpoint = Path(sam2_checkpoint)
        self.frame_stride = frame_stride
        
        # Load YOLO model
        print(f"[Tracker] Loading YOLO model from {model_path}")
        self.yolo_model = YOLO(str(model_path))
        
        # Load SAM2 model
        print(f"[SAM2] Loading SAM2 model from {sam2_checkpoint}")
        sam_config = "configs/sam2.1/sam2.1_hiera_t.yaml"  # tiny model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_model = build_sam2_video_predictor(sam_config, str(sam2_checkpoint), device=device)
        
        self.ball_track_id = None
        self.ball_first_frame = None
        self.ball_box = None
    
    def find_best_ball(self):
        """PASS 1: Find the most square ball across the video."""
        print(f"[Tracker] PASS 1: Finding best ball candidate...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ball_squareness = {}  # track_id -> list of (squareness, frame_idx, box)
        
        frame_idx = 0
        with tqdm(total=total_frames, desc="Finding ball") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.yolo_model.track(frame, persist=True, conf=0.3, verbose=False)
                
                # Collect player sizes for filtering
                player_areas = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        if cls == 0:  # Player
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            area = (x2 - x1) * (y2 - y1)
                            player_areas.append(area)
                
                median_player_area = np.median(player_areas) if len(player_areas) > 0 else float('inf')
                
                # Find ball candidates
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else None
                        
                        if cls != 0 and track_id is not None:  # Ball
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            aspect_ratio = height / width if width > 0 else 0
                            
                            # Filter: size and aspect ratio
                            if area < 0.4 * median_player_area and 0.5 <= aspect_ratio <= 2.0:
                                squareness = 1.0 - abs(1.0 - aspect_ratio)
                                if track_id not in ball_squareness:
                                    ball_squareness[track_id] = []
                                ball_squareness[track_id].append((squareness, frame_idx, [x1, y1, x2, y2]))
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Find most square ball
        if not ball_squareness:
            print("[ERROR] No ball candidates found!")
            return False
        
        # Average squareness per track
        avg_squareness = {tid: np.mean([s[0] for s in scores]) 
                         for tid, scores in ball_squareness.items()}
        best_ball_id = max(avg_squareness, key=avg_squareness.get)
        
        # Get MIDDLE frame where ball appears (for better forward+backward propagation)
        ball_data = sorted(ball_squareness[best_ball_id], key=lambda x: x[1])
        middle_idx = len(ball_data) // 2
        self.ball_track_id = best_ball_id
        self.ball_first_frame = ball_data[middle_idx][1]
        self.ball_box = ball_data[middle_idx][2]
        
        print(f"[Tracker] Selected ball ID {best_ball_id} (avg squareness: {avg_squareness[best_ball_id]:.3f})")
        print(f"  Using middle frame: {self.ball_first_frame} (out of {len(ball_data)} frames)")
        print(f"  Box: {self.ball_box}")
        return True
    
    def extract_frames(self, frame_dir):
        """PASS 2: Extract frames to directory (every Nth frame)."""
        print(f"[Extractor] Extracting frames (stride={self.frame_stride})...")
        
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        extracted = 0
        
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract every Nth frame
                if frame_idx % self.frame_stride == 0:
                    frame_path = frame_dir / f"{extracted:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted += 1
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        print(f"[Extractor] Extracted {extracted} frames to {frame_dir}")
        return extracted
    
    def run_sam2(self, frame_dir):
        """PASS 3: Run SAM2 on extracted frames."""
        print(f"[SAM2] Running segmentation on ball...")
        
        # Map original frame index to extracted frame index
        sam_frame_idx = self.ball_first_frame // self.frame_stride
        
        print(f"[SAM2] Ball first appears at frame {self.ball_first_frame} -> SAM frame {sam_frame_idx}")
        print(f"[SAM2] Using box prompt: {self.ball_box}")
        
        # Initialize SAM2 inference state (pass frame DIRECTORY, not video file!)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.sam2_model.init_state(
                video_path=str(frame_dir),  # Directory of frames, not video!
                offload_video_to_cpu=True,  # Keep frames in CPU RAM
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            
            print(f"[SAM2] Adding box prompt at frame {sam_frame_idx}")
            # Add box prompt
            _, out_obj_ids, out_mask_logits = self.sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=sam_frame_idx,
                obj_id=1,  # Ball ID
                box=np.array(self.ball_box, dtype=np.float32),
            )
            
            print(f"[SAM2] Propagating FORWARD from frame {sam_frame_idx}...")
            # Propagate forward
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(
                inference_state, 
                start_frame_idx=sam_frame_idx,
                reverse=False
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            print(f"[SAM2] Propagating BACKWARD from frame {sam_frame_idx}...")
            # Reset state and propagate backward
            self.sam2_model.reset_state(inference_state)
            _, out_obj_ids, out_mask_logits = self.sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=sam_frame_idx,
                obj_id=1,
                box=np.array(self.ball_box, dtype=np.float32),
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(
                inference_state,
                start_frame_idx=sam_frame_idx,
                reverse=True
            ):
                if out_frame_idx not in video_segments:  # Don't overwrite forward propagation
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
        
        print(f"[SAM2] Segmented ball in {len(video_segments)} frames")
        print(f"  Frame range: {min(video_segments.keys())} to {max(video_segments.keys())}")
        return video_segments
    
    def render_video(self, frame_dir, video_segments):
        """PASS 4: Render video with yellow ball mask."""
        print(f"[Renderer] Creating output video with yellow ball mask...")
        
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
                
                # Map to extracted frame index
                sam_frame_idx = frame_idx // self.frame_stride
                
                # Overlay ball mask if available
                if sam_frame_idx in video_segments and 1 in video_segments[sam_frame_idx]:
                    mask = video_segments[sam_frame_idx][1]
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    # Create yellow overlay
                    overlay = frame.copy()
                    yellow_color = (0, 255, 255)  # BGR yellow (cyan + red = yellow)
                    overlay[mask] = yellow_color
                    
                    # Blend with transparency
                    alpha = 0.5
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Add BALL label
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
        print(f"=== Ball SAM2 Tracking Pipeline ===")
        print(f"Input: {self.video_path}")
        print(f"Output: {self.output_path}")
        print(f"Frame stride: {self.frame_stride}")
        
        # PASS 1: Find best ball
        if not self.find_best_ball():
            return
        
        # PASS 2: Extract frames
        frame_dir = Path("/tmp/ball_sam2_frames")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        self.extract_frames(frame_dir)
        
        # PASS 3: Run SAM2
        video_segments = self.run_sam2(frame_dir)
        
        # PASS 4: Render output
        self.render_video(frame_dir, video_segments)
        
        # Cleanup
        print(f"[Cleanup] Removing temporary frames...")
        shutil.rmtree(frame_dir)
        
        print(f"=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description="SAM2 Ball Tracking with Green Mask")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--sam2-checkpoint", default="../hooper-glean/checkpoints/SAM2-InstanceSegmentation/sam2.1_hiera_tiny.pt",
                       help="SAM2 checkpoint path")
    parser.add_argument("--frame-stride", type=int, default=2, 
                       help="Extract every Nth frame (lower = more accurate but slower)")
    
    args = parser.parse_args()
    
    tracker = BallSAM2Tracker(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        sam2_checkpoint=args.sam2_checkpoint,
        frame_stride=args.frame_stride,
    )
    
    tracker.run()


if __name__ == "__main__":
    main()

