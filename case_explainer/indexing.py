"""
Indexing strategies for efficient nearest neighbor search.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.neighbors import KDTree, BallTree
from abc import ABC, abstractmethod


class IndexStrategy(ABC):
    """Abstract base class for indexing strategies."""
    
    @abstractmethod
    def build(self, X: np.ndarray) -> None:
        """Build the index from data."""
        pass
    
    @abstractmethod
    def query(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query for k nearest neighbors.
        
        Returns:
            distances: Array of distances (k,)
            indices: Array of indices (k,)
        """
        pass


class BruteForceIndex(IndexStrategy):
    """Brute force search - computes all distances."""
    
    def __init__(self):
        self.X = None
    
    def build(self, X: np.ndarray) -> None:
        """Store the training data."""
        self.X = X
    
    def query(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors by computing all distances."""
        from .metrics import compute_all_distances
        
        distances = compute_all_distances(point, self.X)
        # Get indices of k smallest distances
        indices = np.argpartition(distances, min(k, len(distances) - 1))[:k]
        # Sort by distance
        sorted_idx = np.argsort(distances[indices])
        indices = indices[sorted_idx]
        distances = distances[indices]
        
        return distances, indices


class KDTreeIndex(IndexStrategy):
    """K-D Tree for fast nearest neighbor search (best for low dimensions)."""
    
    def __init__(self, leaf_size: int = 30):
        self.leaf_size = leaf_size
        self.tree = None
    
    def build(self, X: np.ndarray) -> None:
        """Build K-D tree."""
        self.tree = KDTree(X, leaf_size=self.leaf_size, metric='euclidean')
    
    def query(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query K-D tree for k nearest neighbors."""
        distances, indices = self.tree.query([point], k=k)
        return distances[0], indices[0]


class BallTreeIndex(IndexStrategy):
    """Ball Tree for nearest neighbor search (better for high dimensions)."""
    
    def __init__(self, leaf_size: int = 30):
        self.leaf_size = leaf_size
        self.tree = None
    
    def build(self, X: np.ndarray) -> None:
        """Build Ball tree."""
        self.tree = BallTree(X, leaf_size=self.leaf_size, metric='euclidean')
    
    def query(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query Ball tree for k nearest neighbors."""
        distances, indices = self.tree.query([point], k=k)
        return distances[0], indices[0]


def create_index(method: str = 'kd_tree', **kwargs) -> IndexStrategy:
    """
    Factory function to create an index.
    
    Args:
        method: One of 'brute', 'kd_tree', 'ball_tree'
        **kwargs: Additional arguments for the index
        
    Returns:
        IndexStrategy instance
    """
    if method == 'brute':
        return BruteForceIndex()
    elif method == 'kd_tree':
        return KDTreeIndex(**kwargs)
    elif method == 'ball_tree':
        return BallTreeIndex(**kwargs)
    else:
        raise ValueError(f"Unknown index method: {method}")
