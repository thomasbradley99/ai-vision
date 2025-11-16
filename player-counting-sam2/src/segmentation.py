"""
Player segmentation using SAM2
"""

import numpy as np
import supervision as sv
import torch
from typing import List, Optional, Tuple

# SAM2 imports (will be optional)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("WARNING: SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")


class SAM2Segmenter:
    """
    Segment players using SAM2 with YOLO boxes as prompts
    
    This provides more accurate segmentation than bounding boxes alone,
    which is crucial for accurate re-identification
    """
    
    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/sam2_hiera_large.pt",
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = "cuda"
    ):
        """
        Initialize SAM2 segmenter
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint
            model_cfg: SAM2 model configuration
            device: Device to run model on
        """
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 not installed. Install with:\n"
                "pip install 'player-counting-sam2[sam2]'"
            )
        
        self.device = device
        
        # Build SAM2 model
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
    
    def segment(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> Tuple[List[np.ndarray], sv.Detections]:
        """
        Segment players using SAM2 with detection boxes as prompts
        
        Args:
            frame: Input frame
            detections: Player detections from YOLO
            
        Returns:
            Tuple of (masks, refined_detections)
            - masks: List of binary segmentation masks
            - refined_detections: Detections with refined boxes from masks
        """
        # Set image for SAM2
        self.predictor.set_image(frame)
        
        masks = []
        refined_boxes = []
        
        for box in detections.xyxy:
            # Use box as prompt for SAM2
            mask, score, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            
            masks.append(mask[0])
            
            # Get refined bounding box from mask
            refined_box = self._mask_to_box(mask[0])
            refined_boxes.append(refined_box)
        
        # Create refined detections
        refined_detections = sv.Detections(
            xyxy=np.array(refined_boxes),
            mask=np.array(masks),
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=detections.tracker_id if hasattr(detections, 'tracker_id') else None
        )
        
        return masks, refined_detections
    
    def _mask_to_box(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to bounding box
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return np.array([0, 0, 0, 0])
        
        y_min = np.min(pos[0])
        y_max = np.max(pos[0])
        x_min = np.min(pos[1])
        x_max = np.max(pos[1])
        
        return np.array([x_min, y_min, x_max, y_max])


class FallbackSegmenter:
    """
    Fallback segmenter when SAM2 is not available
    Uses bounding boxes as masks
    """
    
    def segment(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> Tuple[List[np.ndarray], sv.Detections]:
        """
        Create rectangular masks from bounding boxes
        
        Args:
            frame: Input frame
            detections: Player detections
            
        Returns:
            Tuple of (masks, detections)
        """
        h, w = frame.shape[:2]
        masks = []
        
        for box in detections.xyxy:
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
        
        return masks, detections

