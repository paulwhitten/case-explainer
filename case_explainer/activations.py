"""
Activation-based similarity for case-based explanation.

Implements the Caruana et al. (1999) approach of using a trained model's
internal representations as the distance metric for case retrieval, rather
than raw input features.

Reference:
    Caruana, R., Kangarloo, H., Dionisio, J. D. N., Sinha, U., & Johnson, D.
    (1999). Case-based explanation of non-case-based learning methods.
    Proceedings of AMIA Annual Symposium, 212-215.

Usage::

    # Pure activation-based similarity (Caruana 1999)
    from case_explainer import CaseExplainer, SklearnMLPActivationExtractor

    extractor = SklearnMLPActivationExtractor(use_output_weights=True)
    explainer = CaseExplainer(
        X_train, y_train, k=5,
        model=clf,
        activation_extractor=extractor,
        blend_alpha=0.0,   # 0.0 = pure activations, 1.0 = pure features
    )

    # Hybrid: 30% features + 70% activations
    explainer = CaseExplainer(
        X_train, y_train, k=5,
        model=clf,
        activation_extractor=extractor,
        blend_alpha=0.3,
    )
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ActivationExtractor(ABC):
    """
    Abstract base class for extracting model internal representations.

    Subclass this (or use a pre-built extractor) to enable activation-based
    case retrieval. The extractor must be fitted on the training data before
    it can transform new samples.
    """

    @abstractmethod
    def fit(self, model, X: np.ndarray) -> "ActivationExtractor":
        """
        Bind this extractor to *model* and record any statistics needed
        for normalization (e.g. output-layer weight magnitudes).

        Args:
            model: A trained model.
            X:     Training feature matrix, shape (n_samples, n_features).

        Returns:
            self
        """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract activation vectors for *X*.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Activation matrix, shape (n_samples, n_activations).
        """

    def fit_transform(self, model, X: np.ndarray) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(model, X).transform(X)

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_model", None) is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_hidden_activation(a: np.ndarray, name: str) -> np.ndarray:
    """Apply sklearn MLP hidden-layer activation in-place and return *a*."""
    if name == "relu":
        np.maximum(a, 0, out=a)
    elif name == "tanh":
        np.tanh(a, out=a)
    elif name == "logistic":
        # 1 / (1 + exp(-a))
        np.negative(a, out=a)
        np.exp(a, out=a)
        a += 1.0
        np.reciprocal(a, out=a)
    elif name == "identity":
        pass
    else:
        # Fall back to sklearn internals for any future activation names
        try:
            from sklearn.neural_network._base import ACTIVATIONS  # type: ignore
            ACTIVATIONS[name](a)
        except (ImportError, KeyError):
            logger.warning("Unknown activation '%s'; treating as identity.", name)
    return a


# ---------------------------------------------------------------------------
# sklearn MLP extractor
# ---------------------------------------------------------------------------

class SklearnMLPActivationExtractor(ActivationExtractor):
    """
    Extracts hidden-layer activations from a fitted sklearn
    ``MLPClassifier`` or ``MLPRegressor``.

    Implements the weighted-Euclidean refinement from Caruana et al. (1999),
    Section 5: hidden units with larger output-layer connections receive more
    weight in the distance metric because they contribute more to the
    prediction.

    Args:
        layer:
            Which hidden layer to extract.  ``'last_hidden'`` (default) uses
            the final hidden layer before the output.  An integer selects by
            0-based index among the hidden layers.
        use_output_weights:
            If ``True`` (default), scale each hidden unit's activation by the
            mean absolute magnitude of its output-layer connections, so that
            units important to the prediction dominate the distance metric.
    """

    def __init__(
        self,
        layer: Union[str, int] = "last_hidden",
        use_output_weights: bool = True,
    ):
        self.layer = layer
        self.use_output_weights = use_output_weights
        self._model = None
        self._output_weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, model, X: np.ndarray) -> "SklearnMLPActivationExtractor":
        self._validate(model)
        self._model = model

        if self.use_output_weights and self.layer == "last_hidden":
            # coefs_[-1] shape: (n_hidden_last, n_outputs)
            # Mean absolute weight across output nodes → importance of each unit
            output_coefs = model.coefs_[-1]
            weights = np.mean(np.abs(output_coefs), axis=1)
            # Normalize: preserve total scale while redistributing emphasis
            total = weights.sum()
            if total > 0:
                self._output_weights = weights / total * len(weights)
            else:
                self._output_weights = np.ones(len(weights))

            logger.info(
                "SklearnMLPActivationExtractor: %d hidden units, "
                "output weight range [%.4f, %.4f]",
                len(self._output_weights),
                self._output_weights.min(),
                self._output_weights.max(),
            )
        else:
            # Output-weight scaling requires last_hidden; defer to transform()
            if self.use_output_weights and self.layer != "last_hidden":
                logger.warning(
                    "use_output_weights=True is only supported for layer='last_hidden'. "
                    "Using uniform weights for layer=%r.",
                    self.layer,
                )
            # Lazy: shape is determined by the actual activation output
            self._output_weights = None

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        activations = self._forward_to_hidden(X)
        # Lazy-init uniform weights when shape wasn't known at fit time
        if self._output_weights is None:
            self._output_weights = np.ones(activations.shape[1])
        # Apply output-weight scaling (weighted Euclidean, Caruana §5)
        return activations * self._output_weights

    # ------------------------------------------------------------------
    def _forward_to_hidden(self, X: np.ndarray) -> np.ndarray:
        """Manual forward pass up to the target hidden layer."""
        model = self._model
        # n_layers_ = input + hidden(s) + output
        # coefs_ has (n_layers_ - 1) entries; last entry is hidden→output
        n_hidden_layers = model.n_layers_ - 2  # hidden layers only

        if self.layer == "last_hidden":
            target = n_hidden_layers - 1
        elif isinstance(self.layer, int):
            target = max(0, min(int(self.layer), n_hidden_layers - 1))
        else:
            raise ValueError(
                f"layer must be 'last_hidden' or int, got {self.layer!r}"
            )

        current = X.copy()
        for i in range(target + 1):
            current = current @ model.coefs_[i] + model.intercepts_[i]
            _apply_hidden_activation(current, model.activation)

        return current

    # ------------------------------------------------------------------
    @staticmethod
    def _validate(model) -> None:
        try:
            from sklearn.neural_network import MLPClassifier, MLPRegressor
        except ImportError as exc:
            raise ImportError("scikit-learn is required.") from exc
        if not isinstance(model, (MLPClassifier, MLPRegressor)):
            raise TypeError(
                f"SklearnMLPActivationExtractor requires an sklearn MLP, "
                f"got {type(model).__name__}."
            )
        if not hasattr(model, "coefs_"):
            raise ValueError("Model must be fitted before extracting activations.")


# ---------------------------------------------------------------------------
# Decision-tree / random-forest extractor
# ---------------------------------------------------------------------------

class DecisionTreeActivationExtractor(ActivationExtractor):
    """
    Extracts leaf-node identity from a fitted sklearn
    ``DecisionTreeClassifier``, ``DecisionTreeRegressor``,
    ``RandomForestClassifier``, or ``RandomForestRegressor``.

    The Caruana et al. (1999) paper observes that a decision tree implicitly
    assigns zero distance to training cases that land in the same leaf node
    and non-zero distance to cases in different leaves.  This extractor
    makes that distance metric explicit via one-hot leaf encoding.

    For a **single tree** the output is a one-hot vector of length
    ``max_leaf_id + 1``.  For a **forest** the per-tree one-hot vectors are
    concatenated (one column block per tree).  If the total encoding
    dimensionality would exceed *max_one_hot_dims*, normalized raw leaf IDs
    are returned instead (a warning is logged; using ``metric='hamming'`` in
    ``CaseExplainer`` gives semantically correct results in that case).

    Args:
        max_one_hot_dims:
            Maximum allowed total one-hot dimensions before falling back to
            normalized raw leaf IDs. Default 2000.
    """

    def __init__(self, max_one_hot_dims: int = 2000):
        self.max_one_hot_dims = max_one_hot_dims
        self._model = None
        self._encoding: str = "one_hot"  # or "raw"
        self._is_forest: bool = False
        # one_hot mode
        self._leaf_offsets: np.ndarray | None = None
        self._total_dims: int = 0
        # raw mode
        self._max_per_tree: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, model, X: np.ndarray) -> "DecisionTreeActivationExtractor":
        self._validate(model)
        self._model = model

        leaf_ids = model.apply(np.atleast_2d(X))  # (n_samples,) or (n_samples, n_trees)
        self._is_forest = leaf_ids.ndim == 2

        if self._is_forest:
            max_per_tree = leaf_ids.max(axis=0)  # (n_trees,)
            total_dims = int((max_per_tree + 1).sum())
            if total_dims <= self.max_one_hot_dims:
                self._encoding = "one_hot"
                self._leaf_offsets = np.concatenate(
                    [[0], np.cumsum(max_per_tree + 1)[:-1]]
                ).astype(int)
                self._total_dims = total_dims
            else:
                self._encoding = "raw"
                self._max_per_tree = max_per_tree.astype(float)
                logger.warning(
                    "Forest one-hot encoding requires %d dims (> max_one_hot_dims=%d). "
                    "Falling back to normalized raw leaf IDs. "
                    "Use metric='hamming' in CaseExplainer for best results.",
                    total_dims,
                    self.max_one_hot_dims,
                )
        else:
            max_leaf = int(leaf_ids.max())
            self._encoding = "one_hot"
            self._leaf_offsets = np.array([0], dtype=int)
            self._total_dims = max_leaf + 1

        logger.info(
            "DecisionTreeActivationExtractor: encoding=%s, dims=%d, forest=%s",
            self._encoding,
            self._total_dims if self._encoding == "one_hot" else (
                leaf_ids.shape[1] if self._is_forest else 1
            ),
            self._is_forest,
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.atleast_2d(X)
        leaf_ids = self._model.apply(X)

        if self._encoding == "one_hot":
            return self._to_one_hot(leaf_ids)
        else:
            return self._to_normalized_raw(leaf_ids)

    # ------------------------------------------------------------------
    def _to_one_hot(self, leaf_ids: np.ndarray) -> np.ndarray:
        n_samples = leaf_ids.shape[0]
        result = np.zeros((n_samples, self._total_dims), dtype=np.float32)

        if self._is_forest:
            for t, offset in enumerate(self._leaf_offsets):
                col_idx = offset + leaf_ids[:, t]
                result[np.arange(n_samples), col_idx] = 1.0
        else:
            result[np.arange(n_samples), leaf_ids] = 1.0

        return result

    def _to_normalized_raw(self, leaf_ids: np.ndarray) -> np.ndarray:
        # Normalize each tree's leaf IDs to [0, 1] by the max seen during fit
        return (leaf_ids / np.maximum(self._max_per_tree, 1)).astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _validate(model) -> None:
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )
        except ImportError as exc:
            raise ImportError("scikit-learn is required.") from exc
        valid = (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        if not isinstance(model, valid):
            raise TypeError(
                f"DecisionTreeActivationExtractor requires a sklearn decision "
                f"tree or random forest, got {type(model).__name__}."
            )
        if not hasattr(model, "apply"):
            raise ValueError("Model must be fitted before extracting activations.")


# ---------------------------------------------------------------------------
# Generic callable extractor
# ---------------------------------------------------------------------------

class CallableActivationExtractor(ActivationExtractor):
    """
    Generic extractor for any model type (PyTorch, Keras, XGBoost, etc.).

    The user provides a callable ``fn(model, X) -> np.ndarray`` that accepts
    the fitted model and a feature matrix and returns an activation matrix of
    shape ``(n_samples, n_activations)``.

    Example — PyTorch hook::

        def pytorch_fn(model, X):
            import torch
            cache = []
            handle = model.fc2.register_forward_hook(
                lambda m, i, o: cache.append(o.detach().cpu().numpy())
            )
            with torch.no_grad():
                model(torch.tensor(X, dtype=torch.float32))
            handle.remove()
            return cache[0]

        extractor = CallableActivationExtractor(pytorch_fn)

    Example — XGBoost leaf indices::

        def xgb_leaves(model, X):
            import xgboost as xgb
            return model.get_booster().predict(
                xgb.DMatrix(X), pred_leaf=True
            ).astype(float)

        extractor = CallableActivationExtractor(xgb_leaves)

    Args:
        fn: Callable with signature ``fn(model, X) -> np.ndarray``.
    """

    def __init__(self, fn):
        if not callable(fn):
            raise TypeError("fn must be callable.")
        self._fn = fn
        self._model = None

    def fit(self, model, X: np.ndarray) -> "CallableActivationExtractor":
        self._model = model
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.atleast_2d(X)
        result = self._fn(self._model, X)
        return np.asarray(result, dtype=float)
