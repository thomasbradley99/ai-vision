#!/usr/bin/env python3
"""
Stage 5: Final person count and summary report

Usage:
    python 5_count_people.py videos/test-video-1
    
Requires:
    - Person tracks from Stage 4
    
Output:
    videos/test-video-1/outputs/person_count.json
    Final report with unique person count and detailed stats
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime


def generate_person_count_report(tracks_path, output_path, video_name=None):
    """
    Generate final person count report
    """
    # Load tracking data
    print(f"Loading tracking data from {tracks_path}...")
    with open(tracks_path, 'r') as f:
        tracks_data = json.load(f)
    
    num_persons = tracks_data['num_unique_persons']
    person_stats = tracks_data['persons']
    total_frames = tracks_data['total_frames']
    
    print(f"\n{'='*60}")
    print(f"PERSON COUNT REPORT")
    print(f"{'='*60}")
    print(f"Video: {video_name or 'Unknown'}")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Unique persons detected: {num_persons}")
    print(f"{'='*60}\n")
    
    # Detailed person info
    people_list = []
    
    for stats in person_stats:
        person_id = stats['person_id']
        appearances = stats['total_appearances']
        first_frame = stats['first_frame']
        last_frame = stats['last_frame']
        
        # Calculate confidence (based on number of appearances)
        confidence = min(appearances / total_frames * 2, 1.0)  # Cap at 1.0
        
        person_info = {
            'person_id': person_id,
            'total_appearances': appearances,
            'first_seen_frame': first_frame,
            'last_seen_frame': last_frame,
            'confidence': round(confidence, 3),
            'appears_in_percent': round(appearances / total_frames * 100, 1)
        }
        
        people_list.append(person_info)
        
        print(f"Person {person_id}:")
        print(f"  Appearances: {appearances}/{total_frames} frames ({person_info['appears_in_percent']}%)")
        print(f"  First seen: Frame {first_frame}")
        print(f"  Last seen: Frame {last_frame}")
        print(f"  Confidence: {person_info['confidence']:.3f}")
        print()
    
    # Load frame metadata if available
    video_dir = tracks_path.parent.parent
    metadata_path = video_dir / "outputs" / "frames_metadata.json"
    
    duration_seconds = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            duration_seconds = metadata.get('duration_seconds')
    
    # Create final report
    report = {
        'video_name': video_name or str(video_dir.name),
        'analysis_timestamp': datetime.now().isoformat(),
        'unique_people_count': num_persons,
        'frames_analyzed': total_frames,
        'duration_seconds': duration_seconds,
        'people': people_list,
        'method': {
            'segmentation': 'SAM2',
            'embedding': 'ResNet50-ImageNet',
            'similarity_threshold': tracks_data.get('similarity_threshold', 0.7)
        }
    }
    
    # Save report
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"{'='*60}")
    print(f"✓ Report saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return report


def validate_against_ground_truth(report, ground_truth_count):
    """
    Compare results against ground truth
    """
    detected = report['unique_people_count']
    
    print(f"VALIDATION:")
    print(f"  Ground truth: {ground_truth_count} people")
    print(f"  Detected: {detected} people")
    
    if detected == ground_truth_count:
        print(f"  ✓ CORRECT!")
    elif detected > ground_truth_count:
        print(f"  ⚠ Over-counted by {detected - ground_truth_count}")
    else:
        print(f"  ⚠ Under-counted by {ground_truth_count - detected}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate person count report")
    parser.add_argument("video_dir", help="Video directory (e.g., videos/test-video-1)")
    parser.add_argument("--ground-truth", type=int, help="Expected person count for validation")
    
    args = parser.parse_args()
    
    # Construct paths
    video_dir = Path(args.video_dir)
    tracks_path = video_dir / "outputs" / "person_tracks.json"
    output_path = video_dir / "outputs" / "person_count.json"
    
    if not tracks_path.exists():
        print(f"ERROR: Person tracks file not found: {tracks_path}")
        print("Run 4_reidentify.py first!")
        sys.exit(1)
    
    # Generate report
    try:
        report = generate_person_count_report(
            tracks_path, 
            output_path,
            video_name=video_dir.name
        )
        
        # Validate if ground truth provided
        if args.ground_truth is not None:
            validate_against_ground_truth(report, args.ground_truth)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

