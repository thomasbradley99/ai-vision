"""
Player tracking and re-identification using embedding vectors

Combines Roboflow's ByteTrack approach with embedding-based re-ID
"""

import numpy as np
import supervision as sv
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .embeddings import cosine_similarity


class PlayerTracker:
    """
    Track players across frames using ByteTrack + embedding re-identification
    
    This hybrid approach handles:
    - Short-term tracking: ByteTrack for frame-to-frame consistency
    - Long-term re-ID: Embedding matching for players who leave and re-enter
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_track_length: int = 5,
        max_disappeared: int = 30
    ):
        """
        Initialize player tracker
        
        Args:
            similarity_threshold: Min similarity to match embeddings (0-1)
            min_track_length: Min frames before confirming a player
            max_disappeared: Max frames before removing a track
        """
        self.similarity_threshold = similarity_threshold
        self.min_track_length = min_track_length
        self.max_disappeared = max_disappeared
        
        # Initialize ByteTrack for short-term tracking
        self.byte_tracker = sv.ByteTrack(
            minimum_consecutive_frames=3,
            lost_track_buffer=max_disappeared
        )
        
        # Player database: tracker_id -> player info
        self.player_database: Dict[int, Dict] = {}
        
        # Next player ID for new players
        self.next_player_id = 1
        
        # Frame counter
        self.frame_count = 0
    
    def update(
        self,
        detections: sv.Detections,
        embeddings: np.ndarray
    ) -> Tuple[sv.Detections, np.ndarray]:
        """
        Update tracker with new detections and embeddings
        
        Args:
            detections: Player detections
            embeddings: Player embeddings (N, D)
            
        Returns:
            Tuple of (tracked_detections, player_ids)
            - tracked_detections: Detections with tracker_ids
            - player_ids: Unique player IDs (persistent across re-ID)
        """
        self.frame_count += 1
        
        # Step 1: ByteTrack for short-term tracking
        tracked_detections = self.byte_tracker.update_with_detections(detections)
        
        # Step 2: Assign player IDs using embedding re-identification
        player_ids = self._assign_player_ids(tracked_detections, embeddings)
        
        return tracked_detections, player_ids
    
    def _assign_player_ids(
        self,
        detections: sv.Detections,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Assign persistent player IDs using embedding matching
        
        Args:
            detections: Tracked detections (with tracker_ids)
            embeddings: Player embeddings
            
        Returns:
            Array of player IDs
        """
        player_ids = []
        
        for i, tracker_id in enumerate(detections.tracker_id):
            embedding = embeddings[i]
            
            if tracker_id in self.player_database:
                # Known tracker - update embedding
                player_info = self.player_database[tracker_id]
                player_info['embeddings'].append(embedding)
                player_info['last_seen'] = self.frame_count
                player_ids.append(player_info['player_id'])
            else:
                # New tracker - try to match with existing player
                matched_player_id = self._match_embedding(embedding)
                
                if matched_player_id is not None:
                    # Re-identified existing player
                    player_id = matched_player_id
                else:
                    # New player
                    player_id = self.next_player_id
                    self.next_player_id += 1
                
                # Add to database
                self.player_database[tracker_id] = {
                    'player_id': player_id,
                    'embeddings': [embedding],
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                }
                
                player_ids.append(player_id)
        
        # Cleanup old tracks
        self._cleanup_old_tracks()
        
        return np.array(player_ids)
    
    def _match_embedding(self, embedding: np.ndarray) -> Optional[int]:
        """
        Try to match embedding with existing players
        
        Args:
            embedding: Player embedding to match
            
        Returns:
            Player ID if match found, None otherwise
        """
        best_match_id = None
        best_similarity = 0.0
        
        # Group embeddings by player ID
        player_embeddings = defaultdict(list)
        for tracker_info in self.player_database.values():
            player_id = tracker_info['player_id']
            player_embeddings[player_id].extend(tracker_info['embeddings'])
        
        # Compare with each known player
        for player_id, embs in player_embeddings.items():
            # Use average of recent embeddings
            recent_embs = embs[-5:]  # Last 5 appearances
            avg_embedding = np.mean(recent_embs, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            # Calculate similarity
            similarity = cosine_similarity(embedding, avg_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = player_id
        
        # Return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id
        else:
            return None
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently"""
        to_remove = []
        for tracker_id, info in self.player_database.items():
            if self.frame_count - info['last_seen'] > self.max_disappeared:
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            del self.player_database[tracker_id]
    
    def get_player_count(self) -> int:
        """
        Get count of unique players seen
        
        Returns:
            Number of unique players
        """
        # Count unique player IDs
        player_ids = set()
        for info in self.player_database.values():
            player_ids.add(info['player_id'])
        return len(player_ids)
    
    def get_player_stats(self) -> List[Dict]:
        """
        Get statistics for each player
        
        Returns:
            List of player stats dictionaries
        """
        # Group tracks by player ID
        player_tracks = defaultdict(list)
        for tracker_id, info in self.player_database.items():
            player_id = info['player_id']
            player_tracks[player_id].append(info)
        
        stats = []
        for player_id, tracks in player_tracks.items():
            # Aggregate stats across all tracks for this player
            first_seen = min(t['first_seen'] for t in tracks)
            last_seen = max(t['last_seen'] for t in tracks)
            total_appearances = sum(
                len(t['embeddings']) for t in tracks
            )
            
            stats.append({
                'player_id': player_id,
                'first_seen_frame': first_seen,
                'last_seen_frame': last_seen,
                'total_appearances': total_appearances,
                'num_tracks': len(tracks),  # Re-entries
            })
        
        # Sort by first appearance
        stats.sort(key=lambda x: x['first_seen_frame'])
        
        return stats

