"""
Distance metrics and correspondence calculation.
"""

import numpy as np
from scipy.spatial import distance as scipy_distance
from typing import List, Tuple, Optional, Dict, Union, Callable


def get_adaptive_distance_penalty(n_features: int) -> float:
    """
    Get dimension-adaptive distance penalty exponent.
    
    Research basis: High-dimensional spaces require gentler penalties
    to maintain discrimination (curse of dimensionality).
    
    Based on precedents:
    - Aggarwal et al. (2001) on distance metrics in high dimensions
    - Kernel methods with adaptive bandwidth
    - Local Outlier Factor density weighting
    
    Args:
        n_features: Number of features/dimensions
        
    Returns:
        Penalty exponent alpha (for weight = 1/(distance+1)^alpha)
    """
    if n_features <= 5:
        return 3.0  # Sharp cubic penalty (original, works great for low-D)
    elif n_features <= 15:
        return 2.0  # Square penalty (moderate dimensions)
    elif n_features <= 30:
        return 1.5  # Moderate penalty (high dimensions)
    elif n_features <= 50:
        return 1.0  # Linear penalty (very high dimensions)
    else:
        return 0.5  # Gentle penalty (extreme dimensions)


def get_distance_penalty_function(
    strategy: Union[str, float, Callable] = 'fixed',
    n_features: Optional[int] = None
) -> Callable[[float], float]:
    """
    Get distance penalty function based on strategy.
    
    Strategies:
        'fixed' or 3.0: Original cubic penalty (default)
        'adaptive': Dimension-adaptive penalty (recommended for high-D)
        'linear' or 1.0: Linear penalty
        'square' or 2.0: Square penalty
        'percentile': Percentile-based normalization within k-neighborhood
        float: Custom fixed exponent
        callable: Custom penalty function
        
    Args:
        strategy: Penalty strategy name, exponent value, or custom function
        n_features: Number of features (required for 'adaptive' strategy)
        
    Returns:
        Function that takes distance and returns weight
        
    Examples:
        >>> penalty_fn = get_distance_penalty_function('fixed')
        >>> weight = penalty_fn(2.5)  # Returns 1/(2.5+1)^3
        
        >>> penalty_fn = get_distance_penalty_function('adaptive', n_features=30)
        >>> weight = penalty_fn(2.5)  # Returns 1/(2.5+1)^1.5 (adaptive for 30D)
    """
    if callable(strategy):
        return strategy
    
    # Determine exponent
    if strategy == 'fixed':
        exponent = 3.0
    elif strategy == 'adaptive':
        if n_features is None:
            raise ValueError("n_features required for 'adaptive' strategy")
        exponent = get_adaptive_distance_penalty(n_features)
    elif strategy == 'linear':
        exponent = 1.0
    elif strategy == 'square':
        exponent = 2.0
    elif strategy == 'cubic':
        exponent = 3.0
    elif isinstance(strategy, (int, float)):
        exponent = float(strategy)
    else:
        raise ValueError(f"Unknown penalty strategy: {strategy}. "
                        f"Use 'fixed', 'adaptive', 'linear', 'square', float, or callable.")
    
    # Return penalty function
    return lambda distance: 1.0 / ((distance + 1.0) ** exponent)


def compute_percentile_normalized_weights(
    neighbors: List[Tuple[int, float, int]]
) -> List[float]:
    """
    Compute percentile-normalized weights for neighbors.
    
    Normalizes distances by the maximum distance within the k-neighborhood,
    making weights comparable across different scales and dimensions.
    
    Args:
        neighbors: List of tuples (index, distance, label)
        
    Returns:
        List of normalized weights
    """
    distances = np.array([dist for _, dist, _ in neighbors])
    max_dist = np.max(distances)
    
    if max_dist == 0:
        # All neighbors at same location
        return [1.0] * len(neighbors)
    
    # Normalize to [0, 1] range
    normalized_dists = distances / max_dist
    
    # Apply gentler penalty on normalized distances
    weights = 1.0 / (normalized_dists + 0.1) ** 2
    
    return weights.tolist()


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
    class_weights: Optional[Dict[int, float]] = None,
    distance_penalty: Union[str, float, Callable] = 'fixed',
    n_features: Optional[int] = None
) -> Tuple[float, str]:
    """
    Quantify agreement between prediction and retrieved neighbors.
    
    Based on refined Method 2 formula from hardware trojan detection pipeline:
    weight(class_c) = sum_{i in neighbors with class c} class_weight_c * penalty(distance_i)
    
    Correspondence = weight(predicted_class) / sum(weight(all_classes))
    
    Args:
        neighbors: List of tuples (index, distance, label) for k nearest neighbors
        predicted_class: The predicted class label
        distance_weighted: Whether to weight by distance (default: True)
        class_weights: Optional weights for each class, e.g., {0: 1.0, 1: 2.0}
                      for imbalanced datasets (default: all weights = 1.0)
        distance_penalty: Penalty strategy - 'fixed' (cubic, default), 'adaptive',
                         'linear', 'square', float exponent, or custom function
        n_features: Number of features (required for 'adaptive' strategy)
        
    Returns:
        correspondence: float in [0, 1]
        interpretation: "high" (>0.85), "medium" (0.70-0.85), "low" (<0.70)
        
    Distance Penalty Strategies:
        'fixed' (default): Cubic penalty 1/(d+1)^3 - original formula
        'adaptive': Dimension-adaptive (recommended for high-D data)
            - Low-D (≤5): cubic penalty (sharp)
            - Medium-D (6-15): square penalty
            - High-D (16-30): moderate penalty (^1.5)
            - Very high-D (31-50): linear penalty
            - Extreme-D (>50): gentle penalty (^0.5)
        'linear': Linear penalty 1/(d+1)
        'square': Square penalty 1/(d+1)^2
        'percentile': Percentile-normalized weights
        float: Custom exponent for 1/(d+1)^exponent
        callable: Custom function distance -> weight
    """
    if not neighbors:
        return 0.0, "undefined"
    
    # Default class weights to 1.0 if not provided
    if class_weights is None:
        class_weights = {}
    
    if distance_weighted:
        # Get distance penalty function
        if distance_penalty == 'percentile':
            # Use percentile normalization
            weights = compute_percentile_normalized_weights(neighbors)
            class_weight_sums = {}
            for (_, _, label), weight in zip(neighbors, weights):
                class_multiplier = class_weights.get(label, 1.0)
                weighted = weight * class_multiplier
                if label not in class_weight_sums:
                    class_weight_sums[label] = 0.0
                class_weight_sums[label] += weighted
        else:
            # Use penalty function strategy
            penalty_fn = get_distance_penalty_function(distance_penalty, n_features)
            class_weight_sums = {}
            for _, dist, label in neighbors:
                class_multiplier = class_weights.get(label, 1.0)
                distance_weight = penalty_fn(dist)
                weight = class_multiplier * distance_weight
                
                if label not in class_weight_sums:
                    class_weight_sums[label] = 0.0
                class_weight_sums[label] += weight
        
        total_weight = sum(class_weight_sums.values())
        if total_weight == 0:
            return 0.0, "undefined"
            
        correspondence = class_weight_sums.get(predicted_class, 0.0) / total_weight
    else:
        # Simple voting with class weights (no distance penalty)
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
