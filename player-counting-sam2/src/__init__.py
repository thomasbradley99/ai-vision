"""
Player Counting with SAM2 & Embedding Vectors

A complete pipeline for counting and tracking unique players in sports videos.
"""

__version__ = "0.1.0"
__author__ = "ClannAI"

from .player_counter import PlayerCounter
from .detection import PlayerDetector
from .segmentation import SAM2Segmenter
from .embeddings import EmbeddingExtractor
from .tracking import PlayerTracker

__all__ = [
    "PlayerCounter",
    "PlayerDetector",
    "SAM2Segmenter",
    "EmbeddingExtractor",
    "PlayerTracker",
]

