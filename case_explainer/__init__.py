"""
Case-Explainer: General-Purpose Case-Based Explainability Module

Provides model-agnostic explanations through training set precedent and 
nearest neighbor correspondence.
"""

from .explainer import CaseExplainer
from .explanation import Explanation
from .metrics import compute_correspondence, euclidean_distance

__version__ = "0.1.0"
__all__ = ["CaseExplainer", "Explanation", "compute_correspondence", "euclidean_distance"]
