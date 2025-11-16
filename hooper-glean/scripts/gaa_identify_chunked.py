#!/usr/bin/env python3
"""
Chunked GAA player identification - processes video in segments to avoid memory issues.
"""

import subprocess
import argparse
from pathlib import Path
import json
import cv2
import shutil


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0


def split_video_into_chunks(video_path, output_dir, chunk_duration=30):
    """Split video into chunks using ffmpeg."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.1f}s")
    print(f"Splitting into {chunk_duration}s chunks...")
    
    # Use ffmpeg to split without re-encoding
    chunk_pattern = output_dir / "chunk_%03d.mp4"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-c", "copy",  # Copy codec (no re-encoding)
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        str(chunk_pattern)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Get list of created chunks
    chunks = sorted(output_dir.glob("chunk_*.mp4"))
    print(f"Created {len(chunks)} chunks")
    return chunks


def process_chunk(chunk_path, output_dir, args):
    """Process a single chunk."""
    print(f"\n{'='*60}")
    print(f"Processing: {chunk_path.name}")
    print(f"{'='*60}")
    
    cmd = [
        "conda", "run", "-n", "hooper-ai", "python",
        str(Path(__file__).parent / "gaa_identify.py"),
        "--video", str(chunk_path),
        "--out-dir", str(output_dir),
        "--batch-size", str(args.batch_size),
        "--min-track-length", str(args.min_track_length),
        "--keep-top-k", str(args.keep_top_k),
    ]
    
    if args.use_sam2:
        cmd.append("--use-sam2")
    
    subprocess.run(cmd, check=True)


def combine_chunk_outputs(chunk_dirs, final_output_dir):
    """Combine outputs from all chunks."""
    final_output_dir = Path(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Combining chunk outputs...")
    print(f"{'='*60}")
    
    # Collect all overlay videos
    overlay_videos = []
    for chunk_dir in chunk_dirs:
        overlay = chunk_dir / "player_overlay.mp4"
        if overlay.exists():
            overlay_videos.append(str(overlay))
    
    if not overlay_videos:
        print("No overlay videos found!")
        return
    
    # Create concat file for ffmpeg
    concat_file = final_output_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for video in overlay_videos:
            f.write(f"file '{video}'\n")
    
    # Concatenate videos
    final_video = final_output_dir / "player_overlay.mp4"
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(final_video)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Created final video: {final_video}")
    
    # Combine track JSONs
    all_tracks = []
    track_offset = 0
    
    for chunk_dir in chunk_dirs:
        json_path = chunk_dir / "player_tracks.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                for track in data.get("tracks", []):
                    track["id"] = track["id"] + track_offset
                    all_tracks.append(track)
                track_offset += len(data.get("tracks", []))
    
    final_json = {
        "video": "combined",
        "total_chunks": len(chunk_dirs),
        "tracks": all_tracks
    }
    
    final_json_path = final_output_dir / "player_tracks_combined.json"
    with open(final_json_path, "w") as f:
        json.dump(final_json, f, indent=2)
    
    print(f"✓ Combined {len(all_tracks)} tracks into {final_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Chunked GAA player identification")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-duration", type=int, default=30, 
                       help="Duration of each chunk in seconds")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--keep-top-k", type=int, default=20)
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--keep-chunks", action="store_true",
                       help="Keep intermediate chunk files")
    
    args = parser.parse_args()
    
    # Create directories
    temp_dir = args.out_dir / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Split video
    chunks = split_video_into_chunks(args.video, temp_dir / "videos", args.chunk_duration)
    
    # Process each chunk
    chunk_output_dirs = []
    for i, chunk in enumerate(chunks):
        chunk_output_dir = temp_dir / f"output_{i:03d}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_chunk(chunk, chunk_output_dir, args)
            chunk_output_dirs.append(chunk_output_dir)
        except Exception as e:
            print(f"Error processing {chunk.name}: {e}")
            continue
    
    # Combine outputs
    combine_chunk_outputs(chunk_output_dirs, args.out_dir)
    
    # Cleanup
    if not args.keep_chunks:
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✓ Done")


if __name__ == "__main__":
    main()

