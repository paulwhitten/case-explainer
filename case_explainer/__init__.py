"""
Case-Explainer: General-Purpose Case-Based Explainability Module

Provides model-agnostic explanations through training set precedent and
nearest neighbor correspondence.
"""

from .explainer import CaseExplainer
from .explanation import Explanation, Neighbor
from .metrics import compute_correspondence
from .activations import (
    ActivationExtractor,
    Predictor,
    SklearnMLPActivationExtractor,
    DecisionTreeActivationExtractor,
    CallableActivationExtractor,
    Features,
    HiddenActivations,
    CustomActivations,
    TreeLeaf,
    ForestProximity,
    Blend,
    SimilarityStrategy,
)

__version__ = "0.3.0"
__all__ = [
    "CaseExplainer",
    "Explanation",
    "Neighbor",
    "compute_correspondence",
    "ActivationExtractor",
    "Predictor",
    "SklearnMLPActivationExtractor",
    "DecisionTreeActivationExtractor",
    "CallableActivationExtractor",
    "Features",
    "HiddenActivations",
    "CustomActivations",
    "TreeLeaf",
    "ForestProximity",
    "Blend",
    "SimilarityStrategy",
]
