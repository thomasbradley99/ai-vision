"""
Main PlayerCounter class - combines all components

Following Roboflow's pattern for sports analytics
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from .detection import PlayerDetector, get_player_crops
from .segmentation import SAM2Segmenter, FallbackSegmenter
from .embeddings import EmbeddingExtractor
from .tracking import PlayerTracker


class PlayerCounter:
    """
    Count unique players in sports videos
    
    Pipeline:
    1. Detect players (YOLO)
    2. Segment players (SAM2)
    3. Extract embeddings (ResNet50/CLIP)
    4. Track & re-identify (ByteTrack + embeddings)
    5. Count unique players
    """
    
    def __init__(
        self,
        detection_model: str = "yolov8n.pt",
        detection_conf: float = 0.5,
        use_sam2: bool = True,
        sam2_checkpoint: Optional[str] = None,
        embedding_model: str = "resnet50",
        reid_threshold: float = 0.7,
        device: str = "cuda",
        fps: float = 2.0,
    ):
        """
        Initialize player counter
        
        Args:
            detection_model: YOLO model name or path
            detection_conf: Detection confidence threshold
            use_sam2: Whether to use SAM2 for segmentation
            sam2_checkpoint: Path to SAM2 checkpoint
            embedding_model: Embedding model ('resnet50' or 'clip')
            reid_threshold: Re-ID similarity threshold
            device: Device to run on ('cuda' or 'cpu')
            fps: Frame sampling rate
        """
        self.fps = fps
        self.device = device
        
        # Initialize components
        print("Initializing player detector...")
        self.detector = PlayerDetector(
            model_name=detection_model,
            confidence=detection_conf,
            device=device
        )
        
        if use_sam2 and sam2_checkpoint:
            print("Initializing SAM2 segmenter...")
            try:
                self.segmenter = SAM2Segmenter(
                    checkpoint_path=sam2_checkpoint,
                    device=device
                )
            except ImportError:
                print("SAM2 not available, using fallback segmenter")
                self.segmenter = FallbackSegmenter()
        else:
            print("Using fallback segmenter (bounding boxes)")
            self.segmenter = FallbackSegmenter()
        
        print(f"Initializing embedding extractor ({embedding_model})...")
        self.embedding_extractor = EmbeddingExtractor(
            model_name=embedding_model,
            device=device
        )
        
        print("Initializing player tracker...")
        self.tracker = PlayerTracker(
            similarity_threshold=reid_threshold
        )
        
        print("✓ PlayerCounter initialized")
    
    def count(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Count unique players in a video
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with results:
            - player_count: Number of unique players
            - stats: Per-player statistics
            - frames_processed: Number of frames analyzed
        """
        print(f"\nProcessing video: {video_path}")
        
        # Get video info
        video_info = sv.VideoInfo.from_video_path(video_path)
        total_frames = video_info.total_frames
        
        # Calculate frame stride based on FPS
        frame_stride = max(1, int(video_info.fps / self.fps))
        
        print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps} fps")
        print(f"Sampling at {self.fps} fps (every {frame_stride} frames)")
        
        # Setup video output if requested
        video_sink = None
        if output_path:
            video_sink = sv.VideoSink(output_path, video_info)
            video_sink.__enter__()
        
        # Process video
        frame_generator = sv.get_video_frames_generator(
            source_path=video_path,
            stride=frame_stride
        )
        
        frames_processed = 0
        
        if show_progress:
            frame_generator = tqdm(
                frame_generator,
                total=total_frames // frame_stride,
                desc="Counting players"
            )
        
        for frame in frame_generator:
            # Detect players
            detections = self.detector.detect(frame)
            
            if len(detections) == 0:
                continue
            
            # Segment players (optional, improves crop quality)
            try:
                masks, detections = self.segmenter.segment(frame, detections)
            except Exception as e:
                print(f"Warning: Segmentation failed: {e}")
                masks = None
            
            # Extract player crops
            crops = get_player_crops(frame, detections)
            
            # Extract embeddings
            embeddings = self.embedding_extractor.extract_batch(crops)
            
            # Track players and assign IDs
            tracked_detections, player_ids = self.tracker.update(
                detections, embeddings
            )
            
            # Annotate frame if saving video
            if video_sink:
                annotated_frame = self._annotate_frame(
                    frame, tracked_detections, player_ids
                )
                video_sink.write_frame(annotated_frame)
            
            frames_processed += 1
        
        if video_sink:
            video_sink.__exit__(None, None, None)
        
        # Get final results
        player_count = self.tracker.get_player_count()
        player_stats = self.tracker.get_player_stats()
        
        results = {
            'player_count': player_count,
            'stats': player_stats,
            'frames_processed': frames_processed,
            'video_info': {
                'width': video_info.width,
                'height': video_info.height,
                'fps': video_info.fps,
                'total_frames': video_info.total_frames,
            }
        }
        
        print(f"\n✓ Processing complete!")
        print(f"Unique players detected: {player_count}")
        print(f"Frames processed: {frames_processed}")
        
        return results
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        player_ids: np.ndarray
    ) -> np.ndarray:
        """
        Annotate frame with player tracking information
        
        Args:
            frame: Input frame
            detections: Player detections
            player_ids: Player IDs
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw ellipses around players
        ellipse_annotator = sv.EllipseAnnotator(
            thickness=2
        )
        annotated = ellipse_annotator.annotate(annotated, detections)
        
        # Draw player IDs
        labels = [f"Player {pid}" for pid in player_ids]
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.BOTTOM_CENTER,
            text_padding=5,
            text_thickness=2
        )
        annotated = label_annotator.annotate(
            annotated, detections, labels=labels
        )
        
        # Draw player count
        player_count = len(set(player_ids))
        cv2.putText(
            annotated,
            f"Unique Players: {player_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return annotated

