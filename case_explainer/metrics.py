"""
Distance metrics and correspondence calculation.
"""

import numpy as np
from scipy.spatial import distance as scipy_distance
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Default ``(medium, high)`` correspondence thresholds.
DEFAULT_THRESHOLDS: Tuple[float, float] = (0.70, 0.85)


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


def validate_thresholds(thresholds: Sequence[float]) -> Tuple[float, float]:
    """Check and normalize a ``(medium, high)`` correspondence threshold pair.

    Args:
        thresholds: Two values with ``0 <= medium <= high <= 1``. A score at or
            above ``high`` is "high", at or above ``medium`` is "medium", and
            anything lower is "low".

    Returns:
        The thresholds as a ``(medium, high)`` tuple of floats.
    """
    try:
        medium, high = (float(value) for value in thresholds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "correspondence thresholds must be a (medium, high) pair of numbers"
        ) from exc
    if not 0.0 <= medium <= high <= 1.0:
        raise ValueError(
            "correspondence thresholds must satisfy 0 <= medium <= high <= 1, "
            f"got ({medium}, {high})"
        )
    return medium, high


def interpret_correspondence(
    correspondence: float, thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> str:
    """Map a correspondence score to ``"high"``, ``"medium"``, or ``"low"``."""
    medium, high = validate_thresholds(thresholds)
    if correspondence >= high:
        return "high"
    if correspondence >= medium:
        return "medium"
    return "low"


def compute_correspondence(
    neighbors: List[Tuple[int, float, Any]],
    predicted_class: Any,
    distance_weighted: bool = True,
    class_weights: Optional[Dict[Any, float]] = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Tuple[float, str]:
    """
    Quantify agreement between prediction and retrieved neighbors.

    Based on refined Method 2 formula from hardware trojan detection pipeline:
    weight(class_c) = sum_{i in neighbors with class c} class_weight_c / (distance_i + 1)^3

    Correspondence = weight(predicted_class) / sum(weight(all_classes))

    The ``+1`` offset assumes distances on roughly the standardized-feature
    scale. Distances from other retrieval spaces (hidden activations, forest
    proximity) have different scales, so correspondence values are comparable
    within one retrieval space but not across spaces unless the distances have
    been normalized (see ``CaseExplainer(distance_normalization=...)``).

    Args:
        neighbors: List of tuples (index, distance, label) for k nearest neighbors
        predicted_class: The predicted class label
        distance_weighted: Whether to weight by inverse cubed distance (default: True)
        class_weights: Optional weights for each class, e.g., {0: 1.0, 1: 2.0}
                      for imbalanced datasets (default: all weights = 1.0)
        thresholds: ``(medium, high)`` cut points for the interpretation label
                    (default: ``(0.70, 0.85)``). Thresholds are domain-dependent
                    and should be validated per use case.

    Returns:
        A ``(correspondence, interpretation)`` tuple: the score in ``[0, 1]`` and
        ``"high"``, ``"medium"``, or ``"low"`` according to ``thresholds``
        (``"undefined"`` when there are no neighbors).
    """
    thresholds = validate_thresholds(thresholds)
    if not neighbors:
        return 0.0, "undefined"

    # Default class weights to 1.0 if not provided
    if class_weights is None:
        class_weights = {}

    class_weight_sums: Dict[Any, float] = {}
    for _, dist, label in neighbors:
        weight = class_weights.get(label, 1.0)
        if distance_weighted:
            # Weight by inverse cubed distance (refined formula from pipeline)
            weight = weight / ((dist + 1.0) ** 3)
        class_weight_sums[label] = class_weight_sums.get(label, 0.0) + weight

    total_weight = sum(class_weight_sums.values())
    if total_weight == 0:
        return 0.0, "undefined"

    correspondence = class_weight_sums.get(predicted_class, 0.0) / total_weight
    return correspondence, interpret_correspondence(correspondence, thresholds)


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
    return scipy_distance.cdist([point], data, metric="euclidean")[0]
