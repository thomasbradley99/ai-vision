#!/usr/bin/env python3
"""
Replace blue overlay pixels with green color (#016F32)
"""

import cv2
import numpy as np
from pathlib import Path

def hex_to_bgr(hex_color):
    """Convert hex color to BGR (OpenCV format)"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # BGR format for OpenCV

def replace_blue_with_green(image_path, output_path, green_hex="#016F32"):
    """
    Replace blue overlay pixels with green color by detecting blue dominance.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_float = img.astype(np.float32)
    
    b = img_float[:, :, 0]
    g = img_float[:, :, 1]
    r = img_float[:, :, 2]
    
    # Detect blue overlay: blue must be significantly higher than R and G
    blue_dominance = b - np.maximum(r, g)
    mask = (blue_dominance > 40) & (b > 100) & (b < 250)
    
    # Convert green hex to BGR
    green_bgr = hex_to_bgr(green_hex)
    
    # Replace blue pixels with green
    output = img.copy()
    output[mask] = green_bgr
    
    cv2.imwrite(str(output_path), output)
    print(f"Replaced {np.sum(mask)} pixels with green")
    
    return output

if __name__ == "__main__":
    input_path = Path("/home/ubuntu/clann/clann-jujisu/AI_vision/outputs/sam2_frame.png")
    output_path = Path("/home/ubuntu/clann/clann-jujisu/AI_vision/outputs/sam2_frame_green.png")
    
    replace_blue_with_green(input_path, output_path, green_hex="#016F32")
