"""
Player detection using YOLO models
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import List, Optional


class PlayerDetector:
    """
    Detect players in sports videos using YOLO
    
    Following Roboflow's pattern for sports detection
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cuda",
        imgsz: int = 1280
    ):
        """
        Initialize player detector
        
        Args:
            model_name: YOLO model name or path
            confidence: Detection confidence threshold
            device: Device to run model on ('cuda' or 'cpu')
            imgsz: Input image size for inference
        """
        self.model = YOLO(model_name).to(device=device)
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        
        # Person class ID in COCO dataset
        self.PERSON_CLASS_ID = 0
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Detections object with player bounding boxes
        """
        result = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for person class only
        detections = detections[detections.class_id == self.PERSON_CLASS_ID]
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """
        Detect players in multiple frames (batched for efficiency)
        
        Args:
            frames: List of input frames
            
        Returns:
            List of Detections objects
        """
        results = self.model(
            frames,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=False
        )
        
        all_detections = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.class_id == self.PERSON_CLASS_ID]
            all_detections.append(detections)
        
        return all_detections


def get_player_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract player crops from frame based on detections
    
    Args:
        frame: Input frame
        detections: Player detections
        
    Returns:
        List of cropped player images
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

