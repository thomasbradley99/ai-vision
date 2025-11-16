"""
Example: Count players in a sports video

Following Roboflow's pattern for sports analytics
"""

import argparse
from pathlib import Path
import json

from src.player_counter import PlayerCounter


def main():
    parser = argparse.ArgumentParser(
        description="Count unique players in sports videos"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save annotated video (optional)"
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--use-sam2",
        action="store_true",
        help="Use SAM2 for segmentation (requires SAM2 installed)"
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="models/checkpoints/sam2_hiera_large.pt",
        help="Path to SAM2 checkpoint"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="resnet50",
        choices=["resnet50", "clip"],
        help="Embedding model for re-ID (default: resnet50)"
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.7,
        help="Re-ID similarity threshold (default: 0.7)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame sampling rate (default: 2.0)"
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Path to save results as JSON"
    )
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = PlayerCounter(
        detection_model=args.detection_model,
        detection_conf=args.detection_conf,
        use_sam2=args.use_sam2,
        sam2_checkpoint=args.sam2_checkpoint if args.use_sam2 else None,
        embedding_model=args.embedding_model,
        reid_threshold=args.reid_threshold,
        device=args.device,
        fps=args.fps
    )
    
    # Count players
    results = counter.count(
        video_path=args.video_path,
        output_path=args.output,
        show_progress=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Unique players: {results['player_count']}")
    print(f"Frames processed: {results['frames_processed']}")
    print("\nPlayer details:")
    for stats in results['stats']:
        print(f"  Player {stats['player_id']}: "
              f"{stats['total_appearances']} appearances "
              f"(frames {stats['first_seen_frame']}-{stats['last_seen_frame']})")
    print("="*60)
    
    # Save JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()

