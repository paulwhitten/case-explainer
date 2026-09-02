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
    from case_explainer import CaseExplainer, HiddenActivations

    explainer = CaseExplainer(
        X_train, y_train, k=5,
        similarity=HiddenActivations(model=clf),
    )

    # Hybrid: 30% features + 70% activations
    from case_explainer import Blend

    explainer = CaseExplainer(
        X_train, y_train, k=5,
        similarity=Blend(HiddenActivations(model=clf), features=0.3),
    )
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Predictor(Protocol):
    """Minimal structural type for a fitted classifier: it must ``predict``."""

    def predict(self, X: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Public similarity strategies (preferred API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    """Retrieve neighbors in raw (scaled) feature space.

    This is the default behaviour and the feature-space half of the
    Caruana et al. (1999) comparison. Passing ``similarity=Features()`` is
    equivalent to passing no ``similarity`` at all; it exists so the two
    halves of the comparison read symmetrically in user code.
    """


@dataclass(frozen=True)
class HiddenActivations:
    """Retrieve neighbors in the hidden-activation space of an sklearn MLP.

    Implements Caruana et al. (1999): the trained network's internal
    representation becomes the distance metric for case retrieval.

    Args:
        model: A fitted ``MLPClassifier``.
        layer: ``"last_hidden"`` (default), ``"all_hidden"``, or an integer
            hidden-layer index.
        output_weighting: How to weight hidden units by their contribution to
            the output. One of ``"mean_abs"`` (default), ``"predicted_class"``,
            or ``"none"``.
        input_transform: Optional preprocessing applied before the forward pass.

    Wrap this in :class:`Blend` to mix activation space with feature space.
    """

    model: Any
    layer: Union[str, int] = "last_hidden"
    output_weighting: str = "mean_abs"
    input_transform: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.output_weighting not in ("mean_abs", "predicted_class", "none"):
            raise ValueError(
                "output_weighting must be 'mean_abs', 'predicted_class', "
                f"or 'none', got {self.output_weighting!r}"
            )


@dataclass(frozen=True)
class CustomActivations:
    """Retrieve neighbors using a user-provided activation extractor.

    Args:
        model: The fitted model the extractor reads activations from.
        extractor: An :class:`ActivationExtractor` implementation.
    """

    model: Any
    extractor: "ActivationExtractor"


@dataclass(frozen=True)
class TreeLeaf:
    """Retrieve training cases that share a decision-tree leaf.

    Args:
        model: A fitted ``DecisionTreeClassifier``.
        overflow: ``"truncate"`` (default) returns only same-leaf cases;
            ``"nearest"`` backfills from the closest out-of-leaf cases when a
            leaf holds fewer than ``k`` training samples.
        within_leaf: How same-leaf cases are ordered. Only
            ``"feature_distance"`` is supported.
    """

    model: Any
    overflow: str = "truncate"
    within_leaf: str = "feature_distance"


@dataclass(frozen=True)
class ForestProximity:
    """Retrieve training cases by random-forest shared-leaf proximity.

    Args:
        model: A fitted ``RandomForestClassifier``.
    """

    model: Any


@dataclass(frozen=True)
class Blend:
    """Blend an activation strategy with raw feature space.

    Args:
        strategy: A :class:`HiddenActivations` or :class:`CustomActivations`
            instance whose activation space is mixed with features.
        features: Weight given to feature space, in ``[0, 1]``. The remainder
            goes to the wrapped strategy's activation space. The index is built
            on ``[sqrt(features) * X, sqrt(1 - features) * A]``, so
            ``features=0.0`` is pure activations and ``features=1.0`` is pure
            features.
    """

    strategy: Any
    features: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.features <= 1.0:
            raise ValueError(f"features must be in [0, 1], got {self.features}")


#: Union of every public similarity strategy accepted by ``similarity=``.
SimilarityStrategy = Union[
    Features,
    HiddenActivations,
    CustomActivations,
    TreeLeaf,
    ForestProximity,
    Blend,
]


# ---------------------------------------------------------------------------
# Legacy retrieval configuration (deprecated; superseded by the strategies
# above and the ``similarity=`` parameter). Retained as the internal
# representation and for backward compatibility with ``retrieval=``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenActivationRetrieval:
    """Deprecated. Use :class:`HiddenActivations` (with :class:`Blend`).

    ``output_weighting`` accepts ``"mean_abs"`` (the compatibility default),
    ``"predicted_class"``, or ``"none"``. Setting the legacy
    ``use_output_weights`` field to false also selects unweighted activations.
    """

    model: Any
    layer: Union[str, int] = "last_hidden"
    use_output_weights: bool = True
    output_weighting: str = "mean_abs"
    blend_alpha: float = 0.0
    input_transform: Optional[Any] = None


@dataclass(frozen=True)
class CustomActivationRetrieval:
    """Deprecated. Use :class:`CustomActivations` (with :class:`Blend`)."""

    model: Any
    extractor: "ActivationExtractor"
    blend_alpha: float = 0.0


@dataclass(frozen=True)
class TreeLeafRetrieval:
    """Deprecated. Use :class:`TreeLeaf`."""

    model: Any
    overflow: str = "truncate"
    within_leaf: str = "feature_distance"


@dataclass(frozen=True)
class ForestProximityRetrieval:
    """Deprecated. Use :class:`ForestProximity`."""

    model: Any


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

    @property
    def metric_ready(self) -> bool:
        """Whether transform output is already normalized for distance use."""
        return False


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
            Which hidden layer to extract.

            * ``'last_hidden'`` *(default)* — the final hidden layer only.
              This is the most decision-proximal representation and the mode
              described in Caruana et al. (1999).
            * ``'all_hidden'`` — ALL hidden layers are extracted,
              independently standardised, then concatenated with
              layer-position weights so that later (more decision-proximal)
              layers dominate.  Layer *i* (0-indexed) receives weight
              ``sqrt((i+1) / n_hidden_layers)``, so the last layer is always
              weighted 1.0 and the first ``sqrt(1/n)``.
              Use this when you want the k-NN index to account for the full
              representational hierarchy of the network.
            * An integer — selects a single hidden layer by 0-based index.

        use_output_weights:
            If ``True`` (default), scale each unit in the *last* hidden layer
            by its output-layer connection magnitude (Caruana §5). Connections
            are averaged across outputs unless ``output_class`` selects one
            output column. This applies to ``'last_hidden'`` and to the last
            layer when ``'all_hidden'`` is used; it has no effect on earlier
            layers because output-weight importance is only directly readable
            from ``coefs_[-1]``.

        input_transform:
            Optional preprocessing applied before the MLP forward pass. This
            can be a fitted transformer with ``transform`` or a callable.

        output_class:
            Optional zero-based output-column index. When provided, unit
            importance is derived from that output instead of averaging
            absolute connections across outputs.
    """

    def __init__(
        self,
        layer: Union[str, int] = "last_hidden",
        use_output_weights: bool = True,
        input_transform: Optional[Any] = None,
        output_class: Optional[int] = None,
    ):
        self.layer = layer
        self.use_output_weights = use_output_weights
        self.input_transform = input_transform
        self.output_class = output_class
        self._model: Any = None
        # Single-layer modes
        self._output_weights: Any = None
        self._activation_scaler: Any = None
        # all_hidden mode
        self._layer_scalers: Any = None
        self._layer_position_weights: Any = None
        self._all_hidden_output_weights: Any = None

    # ------------------------------------------------------------------
    def fit(self, model, X: np.ndarray) -> "SklearnMLPActivationExtractor":
        self._validate(model)
        self._model = model
        X_model = self._prepare_input(X)

        if self.layer == "all_hidden":
            self._fit_all_hidden(model, X_model)
        else:
            self._validate_layer_index(model)
            from sklearn.preprocessing import StandardScaler

            raw_activations = self._forward_to_hidden(X_model)
            self._activation_scaler = StandardScaler().fit(raw_activations)

        if self.use_output_weights and self.layer == "last_hidden":
            # coefs_[-1] shape: (n_hidden_last, n_outputs)
            # Convert output connections to one importance value per hidden unit.
            output_coefs = model.coefs_[-1]
            weights = self._output_importance(output_coefs)
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
        elif self.layer != "all_hidden":
            # Output-weight scaling requires last_hidden; defer to transform()
            if self.use_output_weights and isinstance(self.layer, int):
                logger.warning(
                    "use_output_weights=True is only applied to the last hidden layer. "
                    "Using uniform weights for layer=%r.",
                    self.layer,
                )
            # Lazy: shape is determined by the actual activation output
            self._output_weights = None

        return self

    # ------------------------------------------------------------------
    def _fit_all_hidden(self, model, X: np.ndarray) -> None:
        """Fit per-layer StandardScalers and layer-position weights."""
        from sklearn.preprocessing import StandardScaler

        n_hidden = model.n_layers_ - 2
        layer_acts = self._forward_all_hidden(X)

        # Independent StandardScaler per layer — each layer contributes
        # unit variance before position weighting is applied.
        self._layer_scalers = []
        for acts in layer_acts:
            scaler = StandardScaler()
            scaler.fit(acts)
            self._layer_scalers.append(scaler)

        # Position weights: w_i = sqrt((i+1) / n_hidden)
        # First layer → sqrt(1/n), last layer → 1.0
        # In Euclidean distance: layer i contributes (i+1)/n_hidden * ||Δa_i||²
        self._layer_position_weights = np.sqrt(
            np.arange(1, n_hidden + 1, dtype=float) / n_hidden
        )

        # Output-weight scaling for the last hidden layer only (Caruana §5)
        if self.use_output_weights:
            output_coefs = model.coefs_[-1]
            weights = self._output_importance(output_coefs)
            total = weights.sum()
            self._all_hidden_output_weights = (
                weights / total * len(weights) if total > 0 else np.ones(len(weights))
            )
        else:
            self._all_hidden_output_weights = None

        logger.info(
            "SklearnMLPActivationExtractor (all_hidden): %d layers, "
            "position weights %s",
            n_hidden,
            np.round(self._layer_position_weights, 3),
        )

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before transform().")
        X = self._prepare_input(X)

        if self.layer == "all_hidden":
            return self._transform_all_hidden(X)

        activations = self._activation_scaler.transform(self._forward_to_hidden(X))
        # Lazy-init uniform weights when shape wasn't known at fit time
        if self._output_weights is None:
            self._output_weights = np.ones(activations.shape[1])
        # Apply output-weight scaling (weighted Euclidean, Caruana §5)
        return activations * np.sqrt(self._output_weights)

    # ------------------------------------------------------------------
    def _transform_all_hidden(self, X: np.ndarray) -> np.ndarray:
        """Transform using all hidden layers with layer-position weighting."""
        layer_acts = self._forward_all_hidden(X)
        n_layers = len(layer_acts)
        parts = []
        for i, (acts, scaler, pos_w) in enumerate(
            zip(layer_acts, self._layer_scalers, self._layer_position_weights)
        ):
            scaled = scaler.transform(acts)
            if i == n_layers - 1 and self._all_hidden_output_weights is not None:
                # Output-weight scaling for the last layer (Caruana §5)
                scaled = scaled * np.sqrt(self._all_hidden_output_weights)
            parts.append(pos_w * scaled)
        return np.hstack(parts)

    @property
    def metric_ready(self) -> bool:
        return True

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Apply optional model preprocessing and return a dense 2D array."""
        X = np.atleast_2d(X)
        if self.input_transform is not None:
            transform = getattr(self.input_transform, "transform", self.input_transform)
            if not callable(transform):
                raise TypeError(
                    "input_transform must be callable or provide transform()."
                )
            X = transform(X)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.atleast_2d(np.asarray(X, dtype=float))

    def _validate_layer_index(self, model) -> None:
        if not isinstance(self.layer, int):
            if self.layer != "last_hidden":
                raise ValueError(
                    "layer must be 'last_hidden', 'all_hidden', or an integer"
                )
            return
        n_hidden_layers = model.n_layers_ - 2
        if not 0 <= self.layer < n_hidden_layers:
            raise ValueError(
                f"layer index must be in [0, {n_hidden_layers - 1}], got {self.layer}"
            )

    def _output_importance(self, output_coefs: np.ndarray) -> np.ndarray:
        if self.output_class is None:
            return np.mean(np.abs(output_coefs), axis=1)
        if not 0 <= self.output_class < output_coefs.shape[1]:
            raise ValueError(
                f"output_class must be in [0, {output_coefs.shape[1] - 1}], "
                f"got {self.output_class}"
            )
        return np.abs(output_coefs[:, self.output_class])

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
            target = self.layer
        else:
            raise ValueError(
                f"layer must be 'last_hidden', 'all_hidden', or int, got {self.layer!r}"
            )

        current = X.copy()
        for i in range(target + 1):
            current = current @ model.coefs_[i] + model.intercepts_[i]
            _apply_hidden_activation(current, model.activation)

        return current

    # ------------------------------------------------------------------
    def _forward_all_hidden(self, X: np.ndarray) -> list:
        """Forward pass returning each hidden layer's activations as a list."""
        model = self._model
        n_hidden = model.n_layers_ - 2
        results = []
        current = X.copy()
        for i in range(n_hidden):
            current = current @ model.coefs_[i] + model.intercepts_[i]
            _apply_hidden_activation(current, model.activation)
            results.append(current.copy())
        return results

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
        self._model: Any = None
        self._encoding: str = "one_hot"  # or "raw"
        self._is_forest: bool = False
        # one_hot mode
        self._leaf_offsets: Any = None
        self._total_dims: int = 0
        # raw mode
        self._max_per_tree: Any = None

    # ------------------------------------------------------------------
    def fit(self, model, X: np.ndarray) -> "DecisionTreeActivationExtractor":
        self._validate(model)
        self._model = model

        leaf_ids = model.apply(np.atleast_2d(X))  # (n_samples,) or (n_samples, n_trees)
        self._is_forest = leaf_ids.ndim == 2

        if self._is_forest:
            max_per_tree = np.array(
                [estimator.tree_.node_count - 1 for estimator in model.estimators_]
            )
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
            max_leaf = model.tree_.node_count - 1
            self._encoding = "one_hot"
            self._leaf_offsets = np.array([0], dtype=int)
            self._total_dims = max_leaf + 1

        logger.info(
            "DecisionTreeActivationExtractor: encoding=%s, dims=%d, forest=%s",
            self._encoding,
            (
                self._total_dims
                if self._encoding == "one_hot"
                else (leaf_ids.shape[1] if self._is_forest else 1)
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
        result: np.ndarray = np.zeros((n_samples, self._total_dims), dtype=np.float32)

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
        self._model: Any = None

    def fit(self, model, X: np.ndarray) -> "CallableActivationExtractor":
        self._model = model
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.atleast_2d(X)
        result = self._fn(self._model, X)
        return np.asarray(result, dtype=float)
