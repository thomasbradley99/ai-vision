"""
Player embedding extraction for re-identification

Following Roboflow's team classification approach but for person re-ID
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Optional
import cv2


class EmbeddingExtractor:
    """
    Extract appearance embeddings for player re-identification
    
    Similar to Roboflow's TeamClassifier but for unique player counting
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cuda"
    ):
        """
        Initialize embedding extractor
        
        Args:
            model_name: Embedding model ('resnet50', 'clip', or 'osnet')
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        if model_name == "resnet50":
            self._init_resnet50()
        elif model_name == "clip":
            self._init_clip()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _init_resnet50(self):
        """Initialize ResNet50 backbone"""
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        
        # Remove classification layer
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Preprocessing
        self.transform = weights.transforms()
        self.embedding_dim = 2048
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Install with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.transform = self.preprocess
        self.embedding_dim = 512
    
    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding vector from player crop
        
        Args:
            crop: Player crop (BGR format from OpenCV)
            
        Returns:
            Embedding vector (normalized) or None if crop is invalid
        """
        # Check crop size
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return None
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        try:
            img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Warning: Transform failed: {e}")
            return None
        
        # Extract embedding
        with torch.no_grad():
            if self.model_name == "clip":
                embedding = self.model.encode_image(img_tensor)
            else:
                embedding = self.model(img_tensor)
        
        # Convert to numpy and flatten
        embedding = embedding.cpu().numpy().flatten()
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple crops (batched for efficiency)
        
        Args:
            crops: List of player crops
            
        Returns:
            Array of embeddings (N, embedding_dim)
        """
        embeddings = []
        
        for crop in crops:
            emb = self.extract(crop)
            if emb is not None:
                embeddings.append(emb)
            else:
                # Use zero vector for invalid crops
                embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Similarity score in [0, 1] (1 = identical, 0 = orthogonal)
    """
    # Embeddings should already be L2 normalized
    similarity = np.dot(emb1, emb2)
    # Clip to valid range and map to [0, 1]
    similarity = np.clip(similarity, -1.0, 1.0)
    similarity = (similarity + 1.0) / 2.0
    return float(similarity)


def calculate_similarity_matrix(
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate pairwise similarity matrix
    
    Args:
        embeddings: Array of embeddings (N, D)
        
    Returns:
        Similarity matrix (N, N)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
    
    # Map to [0, 1] range
    similarity_matrix = (similarity_matrix + 1.0) / 2.0
    
    return similarity_matrix

