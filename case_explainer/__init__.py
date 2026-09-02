"""
Case-Explainer: General-Purpose Case-Based Explainability Module

Provides model-agnostic explanations through training set precedent and
nearest neighbor correspondence.
"""

from .explainer import CaseExplainer
from .explanation import Explanation
from .metrics import compute_correspondence, euclidean_distance
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

import warnings as _warnings

#: Deprecated strategy class names mapped to their preferred replacements.
_DEPRECATED_STRATEGY_ALIASES = {
    "HiddenActivationRetrieval": "HiddenActivations",
    "CustomActivationRetrieval": "CustomActivations",
    "TreeLeafRetrieval": "TreeLeaf",
    "ForestProximityRetrieval": "ForestProximity",
}


def __getattr__(name: str):
    """Serve the deprecated ``*Retrieval`` names with a ``DeprecationWarning``."""
    if name in _DEPRECATED_STRATEGY_ALIASES:
        from . import activations

        _warnings.warn(
            f"{name} is deprecated; use "
            f"{_DEPRECATED_STRATEGY_ALIASES[name]} with the similarity= "
            "parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(activations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.1"
__all__ = [
    "CaseExplainer",
    "Explanation",
    "compute_correspondence",
    "euclidean_distance",
    "ActivationExtractor",
    "Predictor",
    "SklearnMLPActivationExtractor",
    "DecisionTreeActivationExtractor",
    "CallableActivationExtractor",
    # Preferred similarity strategies
    "Features",
    "HiddenActivations",
    "CustomActivations",
    "TreeLeaf",
    "ForestProximity",
    "Blend",
    "SimilarityStrategy",
    # Deprecated aliases (served lazily by __getattr__ with a warning)
    "HiddenActivationRetrieval",
    "CustomActivationRetrieval",
    "TreeLeafRetrieval",
    "ForestProximityRetrieval",
]
