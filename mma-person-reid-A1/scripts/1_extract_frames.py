#!/usr/bin/env python3
"""
Stage 1: Extract frames from MMA video

Usage:
    python 1_extract_frames.py videos/test-video-1 --fps 2
    
Output:
    videos/test-video-1/outputs/frames/frame_0000.jpg
    videos/test-video-1/outputs/frames/frame_0001.jpg
    ...
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, fps=2):
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 2)
    
    Returns:
        Number of frames extracted
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video: {video_path}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Extracting at {fps} fps...")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=int(duration * fps), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % frame_interval == 0:
                output_path = output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    
    print(f"✓ Extracted {saved_count} frames to {output_dir}")
    
    # Save metadata
    metadata = {
        "video_path": str(video_path),
        "video_fps": video_fps,
        "extract_fps": fps,
        "total_frames": saved_count,
        "duration_seconds": duration
    }
    
    import json
    metadata_path = output_dir.parent / "frames_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from MMA video")
    parser.add_argument("video_dir", help="Video directory (e.g., videos/test-video-1)")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second to extract")
    
    args = parser.parse_args()
    
    # Construct paths
    video_dir = Path(args.video_dir)
    video_path = None
    
    # Find video file in input directory
    input_dir = video_dir / "input"
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Look for video files
    for ext in ['.mp4', '.mov', '.avi', '.mkv']:
        candidates = list(input_dir.glob(f"*{ext}"))
        if candidates:
            video_path = candidates[0]
            break
    
    if not video_path:
        print(f"Error: No video file found in {input_dir}")
        print("Supported formats: .mp4, .mov, .avi, .mkv")
        sys.exit(1)
    
    # Create output directory
    output_dir = video_dir / "outputs" / "frames"
    
    # Extract frames
    try:
        extract_frames(video_path, output_dir, fps=args.fps)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

