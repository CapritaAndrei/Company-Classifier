"""
Models module for company classification.
Handles model loading and embedding generation.
"""

from .embeddings import EmbeddingManager
from .nli import NLIManager

__all__ = ['EmbeddingManager', 'NLIManager'] 