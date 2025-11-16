#!/usr/bin/env python3
"""
Stage 4: Reidentify persons across frames using embedding similarity

Usage:
    python 4_reidentify.py videos/test-video-1 --threshold 0.7
    
Requires:
    - Embeddings from Stage 3
    
Output:
    videos/test-video-1/outputs/person_tracks.json
    Contains: Person IDs and their appearances across frames
"""

import os
import sys
import argparse
from pathlib import Path
import pickle
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def cosine_similarity(emb1, emb2):
    """
    Calculate cosine similarity between two embeddings
    
    Returns:
        Similarity score in [0, 1] (1 = identical, 0 = orthogonal)
    """
    # Embeddings should already be L2 normalized
    similarity = np.dot(emb1, emb2)
    # Clip to [0, 1] range
    similarity = (similarity + 1) / 2
    return similarity


class PersonTracker:
    """
    Track persons across frames using embedding similarity
    """
    
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.person_tracks = []  # List of person tracks
        self.next_person_id = 1
    
    def find_matching_person(self, embedding):
        """
        Find existing person track that matches this embedding
        
        Returns:
            person_id if match found, None otherwise
        """
        best_match_id = None
        best_similarity = 0
        
        for track in self.person_tracks:
            # Compare with all embeddings in this track
            track_embeddings = [e['embedding'] for e in track['appearances']]
            
            # Use average embedding or most recent
            # For robustness, use average of last N appearances
            recent_embeddings = track_embeddings[-5:]  # Last 5 appearances
            avg_embedding = np.mean(recent_embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            # Calculate similarity
            similarity = cosine_similarity(embedding, avg_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track['person_id']
        
        # Return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id
        else:
            return None
    
    def add_detection(self, frame_name, embedding, box, mask_id):
        """
        Add a person detection and assign to existing track or create new
        """
        # Try to match with existing person
        person_id = self.find_matching_person(embedding)
        
        if person_id is None:
            # Create new person track
            person_id = self.next_person_id
            self.next_person_id += 1
            
            self.person_tracks.append({
                'person_id': person_id,
                'appearances': []
            })
        
        # Add appearance to track
        for track in self.person_tracks:
            if track['person_id'] == person_id:
                track['appearances'].append({
                    'frame': frame_name,
                    'embedding': embedding,
                    'box': box,
                    'mask_id': mask_id
                })
                break
        
        return person_id
    
    def get_tracks(self):
        """
        Get all person tracks
        """
        return self.person_tracks


def reidentify_persons(embeddings_path, output_path, similarity_threshold=0.7):
    """
    Reidentify persons across all frames
    """
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        all_embeddings = pickle.load(f)
    
    print(f"Loaded embeddings for {len(all_embeddings)} frames")
    
    # Initialize tracker
    tracker = PersonTracker(similarity_threshold=similarity_threshold)
    
    # Process frames in order
    frame_names = sorted(all_embeddings.keys())
    
    print(f"Reidentifying persons (threshold={similarity_threshold})...")
    
    frame_assignments = {}
    
    for frame_name in tqdm(frame_names, desc="Tracking persons"):
        frame_detections = all_embeddings[frame_name]
        
        frame_assignments[frame_name] = []
        
        for detection in frame_detections:
            embedding = detection['embedding']
            box = detection['box']
            mask_id = detection['mask_id']
            
            # Assign person ID
            person_id = tracker.add_detection(frame_name, embedding, box, mask_id)
            
            frame_assignments[frame_name].append({
                'person_id': person_id,
                'box': box,
                'mask_id': mask_id
            })
    
    # Get final tracks
    tracks = tracker.get_tracks()
    
    # Calculate statistics for each person
    person_stats = []
    
    for track in tracks:
        person_id = track['person_id']
        appearances = track['appearances']
        
        # Get frame indices
        frame_indices = []
        for app in appearances:
            frame_name = app['frame']
            # Extract frame number from "frame_0042"
            frame_num = int(frame_name.split('_')[1])
            frame_indices.append(frame_num)
        
        person_stats.append({
            'person_id': person_id,
            'total_appearances': len(appearances),
            'first_frame': min(frame_indices),
            'last_frame': max(frame_indices),
            'frames': frame_indices
        })
    
    # Sort by first appearance
    person_stats.sort(key=lambda x: x['first_frame'])
    
    # Prepare output
    output_data = {
        'num_unique_persons': len(tracks),
        'similarity_threshold': similarity_threshold,
        'total_frames': len(all_embeddings),
        'persons': person_stats,
        'frame_assignments': frame_assignments
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Custom JSON encoder for numpy types
        def json_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json.dump(output_data, f, indent=2, default=json_encoder)
    
    print(f"\n✓ Reidentification complete!")
    print(f"  Unique persons detected: {len(tracks)}")
    print(f"  Total frames: {len(all_embeddings)}")
    print(f"  Saved to: {output_path}")
    
    # Print summary
    print(f"\nPerson Summary:")
    for stats in person_stats:
        print(f"  Person {stats['person_id']}: "
              f"{stats['total_appearances']} appearances "
              f"(frames {stats['first_frame']}-{stats['last_frame']})")


def main():
    parser = argparse.ArgumentParser(description="Reidentify persons across frames")
    parser.add_argument("video_dir", help="Video directory (e.g., videos/test-video-1)")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for matching (0-1)")
    
    args = parser.parse_args()
    
    # Construct paths
    video_dir = Path(args.video_dir)
    embeddings_path = video_dir / "outputs" / "embeddings.pkl"
    output_path = video_dir / "outputs" / "person_tracks.json"
    
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings file not found: {embeddings_path}")
        print("Run 3_extract_embeddings.py first!")
        sys.exit(1)
    
    # Reidentify persons
    try:
        reidentify_persons(embeddings_path, output_path, 
                          similarity_threshold=args.threshold)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

