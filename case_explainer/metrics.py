"""
Distance metrics and correspondence calculation.
"""

import numpy as np
from scipy.spatial import distance as scipy_distance
from typing import List, Tuple, Optional, Dict


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as numpy array
        point2: Second point as numpy array
        
    Returns:
        Euclidean distance as float
    """
    return scipy_distance.euclidean(point1, point2)


def compute_correspondence(
    neighbors: List[Tuple[int, float, int]],
    predicted_class: int,
    distance_weighted: bool = True,
    class_weights: Optional[Dict[int, float]] = None
) -> Tuple[float, str]:
    """
    Quantify agreement between prediction and retrieved neighbors.
    
    Based on refined Method 2 formula from hardware trojan detection pipeline:
    weight(class_c) = sum_{i in neighbors with class c} class_weight_c / (distance_i + 1)^3
    
    Correspondence = weight(predicted_class) / sum(weight(all_classes))
    
    Args:
        neighbors: List of tuples (index, distance, label) for k nearest neighbors
        predicted_class: The predicted class label
        distance_weighted: Whether to weight by inverse cubed distance (default: True)
        class_weights: Optional weights for each class, e.g., {0: 1.0, 1: 2.0}
                      for imbalanced datasets (default: all weights = 1.0)
        
    Returns:
        correspondence: float in [0, 1]
        interpretation: "high" (>0.85), "medium" (0.70-0.85), "low" (<0.70)
    """
    if not neighbors:
        return 0.0, "undefined"
    
    # Default class weights to 1.0 if not provided
    if class_weights is None:
        class_weights = {}
    
    if distance_weighted:
        # Weight by inverse cubed distance (refined formula from pipeline)
        # weight = class_weight / (distance + 1)^3
        class_weight_sums = {}
        for _, dist, label in neighbors:
            weight_multiplier = class_weights.get(label, 1.0)
            weight = weight_multiplier / ((dist + 1.0) ** 3)
            
            if label not in class_weight_sums:
                class_weight_sums[label] = 0.0
            class_weight_sums[label] += weight
        
        total_weight = sum(class_weight_sums.values())
        if total_weight == 0:
            return 0.0, "undefined"
            
        correspondence = class_weight_sums.get(predicted_class, 0.0) / total_weight
    else:
        # Simple voting with class weights
        class_counts = {}
        for _, _, label in neighbors:
            weight = class_weights.get(label, 1.0)
            if label not in class_counts:
                class_counts[label] = 0.0
            class_counts[label] += weight
        
        total_count = sum(class_counts.values())
        if total_count == 0:
            return 0.0, "undefined"
            
        correspondence = class_counts.get(predicted_class, 0.0) / total_count
    
    # Interpret correspondence
    if correspondence >= 0.85:
        interpretation = "high"
    elif correspondence >= 0.70:
        interpretation = "medium"
    else:
        interpretation = "low"
    
    return correspondence, interpretation


def compute_all_distances(point: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances from a point to all points in a dataset.
    
    Args:
        point: Query point as 1D numpy array
        data: Dataset as 2D numpy array (n_samples, n_features)
        
    Returns:
        Array of distances (n_samples,)
    """
    # scipy_distance.cdist is faster than a loop
    return scipy_distance.cdist([point], data, metric='euclidean')[0]
