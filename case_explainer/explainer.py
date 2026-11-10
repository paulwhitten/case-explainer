"""
Main CaseExplainer class for case-based explanations.

Based on refined Method 2 (case-based) from hardware trojan detection pipeline.
Uses sklearn's NearestNeighbors for efficient k-NN lookups with pre-built index.
"""

import copy
import logging
import warnings
import numpy as np
import pandas as pd
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .explanation import Explanation, Neighbor
from .metrics import DEFAULT_THRESHOLDS, compute_correspondence, validate_thresholds
from ._compat import removed_parameter_error
from .activations import (
    ActivationExtractor,
    Predictor,
    Features,
    HiddenActivations,
    CustomActivations,
    TreeLeaf,
    ForestProximity,
    Blend,
    SimilarityStrategy,
    SklearnMLPActivationExtractor,
)

logger = logging.getLogger(__name__)

#: Accepted values for ``CaseExplainer(distance_normalization=...)``.
DISTANCE_NORMALIZATIONS = (None, "median_nn")

# Training rows sampled when estimating the typical nearest-neighbor distance.
_REFERENCE_SAMPLE_SIZE = 1000
_FOREST_REFERENCE_SAMPLE_SIZE = 200
# Reference distances below this fraction of the largest sampled distance are
# treated as duplicates when estimating the normalization scale.
_DUPLICATE_RELATIVE_TOLERANCE = 1e-6
# Upper bound on ``n_queries * n_train * n_trees`` booleans held at once when
# computing forest proximities in chunks (~50 MB).
_FOREST_CHUNK_ELEMENTS = 50_000_000
#: Upper bound on the query-by-training distance block TreeLeaf materializes.
_TREE_LEAF_CHUNK_ELEMENTS = 20_000_000


def _label_value(value: Any) -> Any:
    """Convert NumPy scalar labels to their JSON-friendly Python values."""
    return value.item() if isinstance(value, np.generic) else value


def _resolve_similarity(
    similarity: Optional[Union[str, SimilarityStrategy]],
    model: Optional[Any],
) -> SimilarityStrategy:
    """Normalize the public ``similarity`` argument into a strategy object."""
    if similarity is None:
        return Features(model=model)
    if isinstance(similarity, str):
        key = similarity.strip().lower()
        if key in ("features", "feature"):
            return Features(model=model)
        if key in ("activations", "activation", "hidden"):
            if model is None:
                raise ValueError("similarity='activations' requires model= to be set")
            return HiddenActivations(model=model)
        raise ValueError(
            f"Unknown similarity {similarity!r}; expected 'features', "
            "'activations', or a strategy object"
        )
    if isinstance(similarity, Features):
        if similarity.model is None and model is not None:
            return Features(model=model)
        return similarity
    if isinstance(similarity, Blend):
        if not isinstance(similarity.strategy, (HiddenActivations, CustomActivations)):
            raise TypeError("Blend wraps HiddenActivations or CustomActivations")
        return similarity
    if isinstance(similarity, (HiddenActivations, CustomActivations, TreeLeaf)):
        return similarity
    if isinstance(similarity, ForestProximity):
        return similarity
    raise TypeError(
        "similarity must be a strategy object (Features, HiddenActivations, "
        "CustomActivations, TreeLeaf, ForestProximity, Blend) or a string"
    )


def _strategy_model(similarity: SimilarityStrategy) -> Optional[Any]:
    """Return the model carried by a strategy (looking inside ``Blend``)."""
    inner = similarity.strategy if isinstance(similarity, Blend) else similarity
    return getattr(inner, "model", None)


def _hidden_strategy(similarity: SimilarityStrategy) -> Optional[HiddenActivations]:
    """Return the ``HiddenActivations`` strategy in use, if any."""
    inner = similarity.strategy if isinstance(similarity, Blend) else similarity
    return inner if isinstance(inner, HiddenActivations) else None


def _reference_rows(n_samples: int, rng: np.random.RandomState, size: int):
    """Indices of training rows used to estimate the typical neighbor distance."""
    if n_samples <= size:
        return np.arange(n_samples)
    return np.sort(rng.choice(n_samples, size=size, replace=False))


# Queries with at most this many rows run on one thread. With ``n_jobs=-1``
# scikit-learn's tree search splits even a single row across all workers, and
# the scheduling costs an order of magnitude more than the search itself
# (about 13 ms against under 1 ms on a 20k-row index).
_SERIAL_QUERY_ROWS = 1


def _kneighbors(index: NearestNeighbors, vectors, k: int):
    """Query ``index`` for ``k`` neighbors, serially when the query is tiny.

    The serial path uses a shallow copy of the fitted index that shares its
    tree, so no state on the shared index changes and no index is rebuilt.
    """
    if len(vectors) <= _SERIAL_QUERY_ROWS and index.n_jobs not in (None, 1):
        index = copy.copy(index)
        index.n_jobs = 1
    return index.kneighbors(vectors, n_neighbors=k)


def _second_neighbor_distances(index: NearestNeighbors, queries) -> np.ndarray:
    """Distance from each training query to its nearest *other* indexed point."""
    n_indexed = index.n_samples_fit_
    if n_indexed < 2 or len(queries) == 0:
        return np.empty(0)
    distances, _ = _kneighbors(index, queries, 2)
    return distances[:, 1]


# ---------------------------------------------------------------------------
# Internal retrieval spaces. One object per configured similarity strategy;
# each owns its index state and both the single-query and batch query paths.
# ---------------------------------------------------------------------------


class _FeatureSpace:
    """Feature-space retrieval on a shared k-NN index."""

    def __init__(self) -> None:
        self.index: Any = None

    def build(self, explainer: "CaseExplainer", X_train: np.ndarray) -> None:
        self.index = explainer._fit_index(explainer.X_train_scaled)
        explainer.nn_index = self.index

    def label(self, explainer: "CaseExplainer") -> str:
        return "features"

    def metric_label(self, explainer: "CaseExplainer") -> str:
        return explainer.metric

    def query_vectors(self, explainer, X, X_scaled) -> np.ndarray:
        return X_scaled

    def neighbors_batch(self, explainer, X, X_scaled, predicted, k):
        vectors = self.query_vectors(explainer, X, X_scaled)
        distances, indices = _kneighbors(self.index, vectors, k)
        return list(distances), list(indices)

    def reference_distances(self, explainer, rng) -> np.ndarray:
        rows = _reference_rows(explainer.n_samples, rng, _REFERENCE_SAMPLE_SIZE)
        vectors = self.query_vectors(
            explainer, explainer.X_train_original[rows], explainer.X_train_scaled[rows]
        )
        return _second_neighbor_distances(self.index, vectors)


class _ActivationSpace(_FeatureSpace):
    """Activation-space (optionally blended with features) retrieval."""

    def __init__(self, extractor: ActivationExtractor, features_weight: float):
        super().__init__()
        self.extractor = extractor
        self.features_weight = float(features_weight)
        self.scaler: Any = None

    def build(self, explainer, X_train) -> None:
        logger.info(
            "Extracting training activations with %s (features weight %.2f)...",
            type(self.extractor).__name__,
            self.features_weight,
        )
        raw = self.extractor.fit_transform(
            explainer._model, explainer._model_input(X_train)
        )
        # Built-in extractors return normalized, metric-ready coordinates.
        if getattr(self.extractor, "metric_ready", False):
            self.scaler = None
            activations = raw
        else:
            self.scaler = StandardScaler()
            activations = self.scaler.fit_transform(raw)
        index_data = self._combine(explainer.X_train_scaled, activations)
        logger.info(
            "Index space: %s, dims=%d", self.label(explainer), index_data.shape[1]
        )
        self.index = explainer._fit_index(index_data)
        explainer.nn_index = self.index

    def _combine(self, X_scaled: np.ndarray, activations: np.ndarray) -> np.ndarray:
        alpha = self.features_weight
        if alpha == 0.0:
            return activations
        if alpha == 1.0:
            return X_scaled
        return np.hstack(
            [np.sqrt(alpha) * X_scaled, np.sqrt(1.0 - alpha) * activations]
        )

    def label(self, explainer) -> str:
        if self.features_weight == 0.0:
            return "activations"
        if self.features_weight == 1.0:
            return "features"
        return f"hybrid(features={self.features_weight:.2f})"

    def query_vectors(self, explainer, X, X_scaled) -> np.ndarray:
        raw = self.extractor.transform(explainer._model_input(X))
        activations = raw if self.scaler is None else self.scaler.transform(raw)
        return self._combine(X_scaled, activations)


class _ClassActivationSpace(_ActivationSpace):
    """Predicted-class-weighted activation retrieval with per-class indexes."""

    def __init__(self, hidden: HiddenActivations, features_weight: float):
        self.hidden = hidden
        self.features_weight = float(features_weight)
        self.scaler = None
        self.index = None
        self.extractors: Dict[Any, SklearnMLPActivationExtractor] = {}
        self.indexes: Dict[Any, NearestNeighbors] = {}

    def build(self, explainer, X_train) -> None:
        model = self.hidden.model
        if not hasattr(model, "classes_") or not hasattr(model, "coefs_"):
            raise TypeError(
                "predicted_class output weighting requires a fitted sklearn "
                f"MLPClassifier, got {type(model).__name__}"
            )
        output_count = model.coefs_[-1].shape[1]
        classes = list(model.classes_)
        if output_count not in (1, len(classes)):
            raise ValueError("MLP output columns do not match model classes")

        shared: Optional[Tuple[Any, Any]] = None
        for position, class_label in enumerate(classes):
            if output_count == 1 and shared is not None:
                extractor, index = shared
            else:
                extractor = SklearnMLPActivationExtractor(
                    layer=self.hidden.layer,
                    use_output_weights=True,
                    input_transform=self.hidden.input_transform,
                    output_class=0 if output_count == 1 else position,
                )
                activations = extractor.fit_transform(
                    model, explainer._model_input(X_train)
                )
                index = explainer._fit_index(
                    self._combine(explainer.X_train_scaled, activations)
                )
                if output_count == 1:
                    shared = (extractor, index)
            self.extractors[_label_value(class_label)] = extractor
            self.indexes[_label_value(class_label)] = index

    def label(self, explainer) -> str:
        if self.features_weight == 0.0:
            return "activations"
        if self.features_weight == 1.0:
            return "features"
        return f"hybrid(features={self.features_weight:.2f})"

    def _vectors_for(self, explainer, class_label, X, X_scaled) -> np.ndarray:
        activations = self.extractors[class_label].transform(explainer._model_input(X))
        return self._combine(X_scaled, activations)

    def neighbors_batch(self, explainer, X, X_scaled, predicted, k):
        n = len(X)
        distances_out: List[Any] = [None] * n
        indices_out: List[Any] = [None] * n
        groups: Dict[Any, List[int]] = {}
        for row, class_label in enumerate(predicted):
            if class_label not in self.extractors:
                raise ValueError(
                    f"predicted_class {class_label!r} is not in model.classes_"
                )
            groups.setdefault(class_label, []).append(row)
        for class_label, rows in groups.items():
            vectors = self._vectors_for(explainer, class_label, X[rows], X_scaled[rows])
            distances, indices = _kneighbors(self.indexes[class_label], vectors, k)
            for row, dist, idx in zip(rows, distances, indices):
                distances_out[row] = dist
                indices_out[row] = idx
        return distances_out, indices_out

    def reference_distances(self, explainer, rng) -> np.ndarray:
        rows = _reference_rows(explainer.n_samples, rng, _REFERENCE_SAMPLE_SIZE)
        labels = [_label_value(value) for value in explainer.y_train[rows]]
        parts = []
        for class_label in self.indexes:
            member_rows = rows[[label == class_label for label in labels]]
            if len(member_rows) == 0:
                continue
            vectors = self._vectors_for(
                explainer,
                class_label,
                explainer.X_train_original[member_rows],
                explainer.X_train_scaled[member_rows],
            )
            parts.append(_second_neighbor_distances(self.indexes[class_label], vectors))
        return np.concatenate(parts) if parts else np.empty(0)


class _TreeLeafSpace:
    """Same-leaf decision-tree retrieval ordered by feature distance."""

    def __init__(self, strategy: TreeLeaf):
        self.strategy = strategy
        self.training_leaves: Any = None

    def build(self, explainer, X_train) -> None:
        from sklearn.tree import DecisionTreeClassifier

        model = self.strategy.model
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(
                "TreeLeaf requires a fitted sklearn decision tree classifier"
            )
        if not hasattr(model, "tree_"):
            raise ValueError("TreeLeaf model must be fitted")
        if self.strategy.overflow not in ("truncate", "nearest"):
            raise ValueError("overflow must be 'truncate' or 'nearest'")
        if self.strategy.within_leaf != "feature_distance":
            raise ValueError("within_leaf must be 'feature_distance'")
        self.training_leaves = np.asarray(model.apply(explainer._model_input(X_train)))
        explainer.nn_index = None

    def label(self, explainer) -> str:
        return "tree_leaf"

    def metric_label(self, explainer) -> str:
        return explainer.metric

    def query_vectors(self, explainer, X, X_scaled):
        return None

    def _distance_rows(self, explainer, X_scaled):
        """Yield ``(start, distances)`` blocks of shape (chunk, n_train).

        Blocks are produced one at a time so memory stays bounded by the chunk
        size instead of growing with ``n_queries * n_train``.
        """
        n_train = explainer.n_samples
        chunk = max(1, _TREE_LEAF_CHUNK_ELEMENTS // max(1, n_train))
        for start in range(0, len(X_scaled), chunk):
            yield start, pairwise_distances(
                X_scaled[start : start + chunk],
                explainer.X_train_scaled,
                metric=explainer.metric,
            )

    def neighbors_batch(self, explainer, X, X_scaled, predicted, k):
        query_leaves = np.asarray(self.strategy.model.apply(explainer._model_input(X)))
        distances_out, indices_out = [], []
        for start, block in self._distance_rows(explainer, X_scaled):
            for offset, feature_distances in enumerate(block):
                query_leaf = query_leaves[start + offset]
                same_leaf = np.flatnonzero(self.training_leaves == query_leaf)
                same_leaf = same_leaf[
                    np.argsort(feature_distances[same_leaf], kind="stable")
                ]
                indices = same_leaf[:k]
                if len(indices) < k and self.strategy.overflow == "nearest":
                    outside = np.flatnonzero(self.training_leaves != query_leaf)
                    outside = outside[
                        np.argsort(feature_distances[outside], kind="stable")
                    ]
                    indices = np.concatenate([indices, outside[: k - len(indices)]])
                distances_out.append(feature_distances[indices])
                indices_out.append(indices)
        return distances_out, indices_out

    def reference_distances(self, explainer, rng) -> np.ndarray:
        rows = _reference_rows(explainer.n_samples, rng, _REFERENCE_SAMPLE_SIZE)
        index = explainer._fit_index(explainer.X_train_scaled)
        return _second_neighbor_distances(index, explainer.X_train_scaled[rows])


class _ForestProximitySpace:
    """Random-forest shared-leaf proximity retrieval."""

    def __init__(self, strategy: ForestProximity):
        self.strategy = strategy
        self.training_leaves: Any = None

    def build(self, explainer, X_train) -> None:
        from sklearn.ensemble import RandomForestClassifier

        model = self.strategy.model
        if not isinstance(model, RandomForestClassifier):
            raise TypeError(
                "ForestProximity requires a fitted sklearn random forest classifier"
            )
        if not hasattr(model, "estimators_"):
            raise ValueError("ForestProximity model must be fitted")
        if explainer.metric != "euclidean":
            raise ValueError(
                "ForestProximity defines its own distance (1 - shared-leaf "
                f"fraction) and ignores metric=; got metric={explainer.metric!r}. "
                "Leave metric at its default."
            )
        self.training_leaves = np.asarray(model.apply(explainer._model_input(X_train)))
        explainer.nn_index = None

    def label(self, explainer) -> str:
        return "forest_proximity"

    def metric_label(self, explainer) -> str:
        return "shared_leaf_fraction"

    def query_vectors(self, explainer, X, X_scaled):
        return None

    def _distance_rows(self, explainer, X):
        """Yield ``(start, distances)`` blocks of shape (chunk, n_train)."""
        query_leaves = np.asarray(self.strategy.model.apply(explainer._model_input(X)))
        n_train, n_trees = self.training_leaves.shape
        chunk = max(1, _FOREST_CHUNK_ELEMENTS // max(1, n_train * n_trees))
        for start in range(0, len(X), chunk):
            block = query_leaves[start : start + chunk]
            proximity = (block[:, None, :] == self.training_leaves[None, :, :]).mean(
                axis=2
            )
            yield start, 1.0 - proximity

    def neighbors_batch(self, explainer, X, X_scaled, predicted, k):
        distances_out, indices_out = [], []
        for _, block in self._distance_rows(explainer, X):
            for row in block:
                indices = np.argsort(row, kind="stable")[:k]
                distances_out.append(row[indices])
                indices_out.append(indices)
        return distances_out, indices_out

    def reference_distances(self, explainer, rng) -> np.ndarray:
        if explainer.n_samples < 2:
            return np.empty(0)
        rows = _reference_rows(explainer.n_samples, rng, _FOREST_REFERENCE_SAMPLE_SIZE)
        nearest = []
        for start, block in self._distance_rows(
            explainer, explainer.X_train_original[rows]
        ):
            block = block.copy()
            own = rows[start : start + len(block)]
            block[np.arange(len(block)), own] = np.inf  # exclude self
            nearest.append(block.min(axis=1))
        return np.concatenate(nearest)


def _as_label_array(values: Any, name: str) -> np.ndarray:
    """Return ``values`` as a 1-D array holding one class label per sample."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(
                f"{name} must hold one class label per sample, got a DataFrame "
                f"with {values.shape[1]} columns"
            )
        values = values.iloc[:, 0]
    array: np.ndarray = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(
            f"{name} must hold one class label per sample (1-D), got shape "
            f"{array.shape}. If these are class probabilities, pass the "
            "predicted class labels instead."
        )
    return array


def _validate_k(k: Any) -> int:
    """Return ``k`` as a positive ``int`` or raise a clear error."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be a positive integer, got {k!r}")
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    return int(k)


def _normalize_metadata(metadata: Any, n_samples: int) -> Dict[Any, List[Any]]:
    """Copy per-sample metadata into plain lists addressed by row position.

    Pandas objects are read positionally (a Series that went through
    ``train_test_split`` keeps its original index labels, which must not be
    used for lookups) and NumPy values become Python scalars so explanations
    stay JSON-serializable.
    """
    if metadata is None:
        return {}
    if isinstance(metadata, pd.DataFrame):
        metadata = {column: metadata[column] for column in metadata.columns}
    if not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping from field name to per-sample values, "
            f"or a DataFrame; got {type(metadata).__name__}"
        )
    normalized: Dict[Any, List[Any]] = {}
    for key, values in metadata.items():
        if isinstance(values, (pd.Series, pd.Index, np.ndarray)):
            column = values.tolist()
        else:
            column = list(values)
        if len(column) != n_samples:
            raise ValueError(
                f"Metadata '{key}' has {len(column)} items, expected {n_samples}"
            )
        normalized[key] = column
    return normalized


def _align_columns(frame: pd.DataFrame, columns: List[Any], name: str) -> pd.DataFrame:
    """Reorder ``frame`` to the training column order, matching by name.

    A frame whose columns are a plain ``RangeIndex`` carries no names and is
    taken positionally. Any other mismatch is an error: silently comparing
    columns in a different order would mislabel every feature.
    """
    labels = list(frame.columns)
    if labels == columns or isinstance(frame.columns, pd.RangeIndex):
        return frame
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} has duplicate column names")
    wanted = set(columns)
    missing = [c for c in columns if c not in set(labels)]
    unexpected = [c for c in labels if c not in wanted]
    if missing or unexpected:
        raise ValueError(
            f"{name} columns do not match the training columns: "
            f"missing {missing}, unexpected {unexpected}"
        )
    return frame[columns]


def _query_rows(X: Any, columns: Optional[List[Any]], name: str) -> np.ndarray:
    """Return query features as an array, aligned by column name when possible.

    ``columns`` is the training column list when the explainer was built from
    a DataFrame, else None. A DataFrame query is reordered to match; a Series
    is treated as a single row keyed by its index labels. Arrays are taken
    positionally.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    if isinstance(X, pd.DataFrame):
        if columns is not None:
            X = _align_columns(X, columns, name)
        return X.to_numpy()
    return np.asarray(X)


def _normalize_class_names(class_names: Any, y_train: np.ndarray) -> Dict[Any, str]:
    """Return class display names as a ``{label: name}`` dict.

    A mapping is copied. A sequence (for example scikit-learn's
    ``target_names``) is matched to the sorted class labels in ``y_train``.
    """
    if class_names is None:
        return {}
    if isinstance(class_names, Mapping):
        return dict(class_names)
    names = [_label_value(name) for name in class_names]
    unique_labels: np.ndarray = np.unique(y_train)
    labels = unique_labels.tolist()
    if len(names) != len(labels):
        raise ValueError(
            f"class_names has {len(names)} names for {len(labels)} classes in "
            "y_train; pass a {label: name} mapping instead"
        )
    return dict(zip(labels, names))


def _resolve_class_weights(
    class_weights: Optional[Union[Mapping, str]], y_train: np.ndarray
) -> Dict[Any, float]:
    """Return the per-class weight mapping used for correspondence.

    ``None`` means unweighted, a mapping is copied as given, and ``"balanced"``
    derives ``n_samples / (n_classes * count)`` from the training labels.
    """
    if class_weights is None:
        return {}
    if isinstance(class_weights, str):
        if class_weights != "balanced":
            raise ValueError(
                'class_weights must be a mapping, None, or "balanced", '
                f"got {class_weights!r}"
            )
        classes: np.ndarray
        counts: np.ndarray
        classes, counts = np.unique(np.asarray(y_train), return_counts=True)
        total = float(counts.sum())
        return {
            label: total / (len(classes) * count)
            for label, count in zip(classes.tolist(), counts.tolist())
        }
    if not isinstance(class_weights, Mapping):
        raise TypeError(
            'class_weights must be a mapping, None, or "balanced", '
            f"got {type(class_weights).__name__}"
        )
    weights = dict(class_weights)
    for label, weight in weights.items():
        valid = (
            isinstance(weight, (int, float, np.integer, np.floating))
            and not isinstance(weight, bool)
            and np.isfinite(weight)
            and weight >= 0
        )
        if not valid:
            raise ValueError(
                f"class_weights[{label!r}] must be a non-negative finite number, "
                f"got {weight!r}"
            )
    known_labels: np.ndarray = np.unique(np.asarray(y_train))
    known = set(known_labels.tolist())
    unknown = [label for label in weights if _label_value(label) not in known]
    if unknown:
        warnings.warn(
            f"class_weights has entries for labels that do not occur in y_train "
            f"({unknown!r}); they have no effect. Check the label type, for "
            "example 1 versus '1'.",
            UserWarning,
            stacklevel=3,
        )
    return weights


class CaseExplainer:
    """
    General-purpose case-based explainability module.

    Provides model-agnostic explanations through training set precedent
    and nearest neighbor correspondence. Builds k-NN index during initialization
    for fast lookups during explanation.

    Based on refined Method 2 from hardware trojan detection pipeline:
    - Pre-builds NearestNeighbors index on training data
    - Uses distance-weighted correspondence: weight = 1 / (distance + 1)^3
    - Supports class weights for imbalanced datasets
    - Compatible with any classifier (sklearn, XGBoost, etc.)

    Example:
        >>> from case_explainer import CaseExplainer
        >>> explainer = CaseExplainer(X_train, y_train, k=5)
        >>> explanation = explainer.explain_instance(test_sample, model=clf)
        >>> print(f"Correspondence: {explanation.correspondence:.2%}")
        >>> explanation.plot()
    """

    def __init__(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series, List],
        k: int = 5,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[Dict[Any, str]] = None,
        metric: str = "euclidean",
        algorithm: str = "auto",
        scale_data: bool = True,
        class_weights: Optional[Union[Dict[Any, float], str]] = None,
        metadata: Optional[Dict[str, List]] = None,
        n_jobs: int = -1,
        similarity: Optional[Union[str, SimilarityStrategy]] = None,
        model: Optional[Predictor] = None,
        correspondence_thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
        distance_normalization: Optional[str] = None,
        **removed: Any,
    ):
        """
        Initialize CaseExplainer with training data and build k-NN index.

        Args:
            X_train: Training features (n_samples, n_features), as a dense
                array or DataFrame. The index is built on a plain array, but
                the column names of a DataFrame are remembered: later
                DataFrame queries are reordered to match them (a mismatch
                raises ``ValueError``), and the model, ``input_transform``,
                and built-in extractors receive a DataFrame with those columns
                instead of an array. Array queries are taken positionally.
                Custom extractor callables always receive arrays.
            y_train: Training labels, one per sample
            k: Number of nearest neighbors for explanations (default: 5).
                Must be a positive integer.
            feature_names: Names of features (optional). Any sequence with one
                name per feature, such as a list, array, or pandas Index.
            class_names: Display names for classes (optional). Either a
                ``{label: name}`` mapping or a sequence of names matched to the
                sorted class labels, such as scikit-learn's ``target_names``.
            metric: Distance metric for feature, activation, blended, and
                tree-leaf retrieval (default: 'euclidean'). Any metric accepted
                by sklearn's ``NearestNeighbors`` and ``pairwise_distances``.
                ``ForestProximity`` defines its own distance and rejects any
                other value.
            algorithm: k-NN algorithm - 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
            scale_data: Whether to standardize features (recommended: True)
            class_weights: Optional weights for each class in correspondence computation
                          e.g., {0: 1.0, 1: 2.0} to weight class 1 twice as much.
                          Pass ``"balanced"`` to derive inverse-frequency weights
                          from ``y_train`` (``n_samples / (n_classes * count)``,
                          the scikit-learn convention). Recommended for
                          imbalanced data: without weights, a prediction of the
                          majority class that disagrees with one or two
                          minority-class neighbors still scores "high". The
                          resolved weights are stored in ``class_weights``.
            metadata: Optional per-sample metadata, as a dict of sequences
                     (e.g., {'sample_id': [...], 'source': [...], ...}) or a
                     DataFrame with one row per training sample. Values are
                     copied and read by row position, so a pandas Series is
                     matched to ``X_train`` by order and not by its index.
            n_jobs: Number of parallel jobs for k-NN search (-1 = all CPUs).
                Applies to batches; a single-sample query always runs on one
                thread, where it is an order of magnitude faster.
            similarity: How neighbor similarity is computed. This is the single
                knob for retrieval mode. Accepts:

                * ``None`` (default) or ``"features"`` / ``Features()`` — raw
                  feature space.
                * ``"activations"`` — hidden-activation space of ``model``
                  (which must be a fitted ``MLPClassifier``).
                * A strategy object: ``HiddenActivations(...)``,
                  ``CustomActivations(...)``, ``TreeLeaf(...)``,
                  ``ForestProximity(...)``, or ``Blend(strategy, features=...)``
                  to mix an activation space with features.

                The ``model`` lives on the strategy, so an invalid combination
                (e.g. activations without a model) cannot be expressed.
            model: Trained model used for predictions when ``explain_instance``
                is called without ``model=`` or ``predicted_class=``. Required
                for the string shorthand ``similarity="activations"``; for
                strategy objects the strategy's own model is used.
            correspondence_thresholds: ``(medium, high)`` cut points for the
                ``"low"`` / ``"medium"`` / ``"high"`` interpretation label
                (default ``(0.70, 0.85)``). Thresholds are domain-dependent;
                validate them for your use case.
            distance_normalization: ``None`` (default) reports raw distances in
                the retrieval space. ``"median_nn"`` divides every distance by
                the median nearest-neighbor distance among training cases in
                that space, so a distance of 1.0 means "as far as a typical
                training case is from its closest distinct neighbor". Training
                cases that have an exact duplicate are left out of the
                estimate, which uses a random sample of training cases. If
                every sampled case has a duplicate, a warning is issued and
                distances stay unnormalized. This puts distances from
                different retrieval spaces on one scale. The divisor is
                stored as ``distance_scale_``. Neighbor selection is
                unaffected.

        Raises:
            TypeError: For the constructor parameters removed in 0.3.0
                (``activation_extractor``, ``activation_layer``, ``blend_alpha``,
                ``use_output_weights``, ``retrieval``), with a migration hint.
        """
        if removed:
            raise removed_parameter_error(removed)

        # Convert inputs to numpy arrays
        self._columns: Optional[List[Any]] = None
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns.tolist()
            # Remember the columns so models and transforms fitted on a
            # DataFrame keep receiving one (see _model_input).
            self._columns = X_train.columns.tolist()
            X_train = X_train.to_numpy()
        else:
            X_train = np.asarray(X_train)
        if X_train.ndim != 2:
            raise ValueError(
                "X_train must be a dense 2-D array of shape (n_samples, "
                f"n_features), got shape {X_train.shape}"
            )

        y_train = _as_label_array(y_train, "y_train")

        # Store original data
        self.X_train_original = X_train.copy()
        self.y_train = y_train.copy()

        # Validate shapes
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train must have same length "
                f"(got {len(X_train)} and {len(y_train)})"
            )

        self.n_samples, self.n_features = X_train.shape
        self.k = _validate_k(k)
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = [_label_value(name) for name in feature_names]
            if len(self.feature_names) != self.n_features:
                raise ValueError(
                    f"feature_names has {len(self.feature_names)} names for "
                    f"{self.n_features} features"
                )
        self.class_names = _normalize_class_names(class_names, y_train)
        self.metric = metric
        self.algorithm = algorithm
        self.scale_data = scale_data
        self.class_weights = _resolve_class_weights(class_weights, y_train)
        self.metadata = _normalize_metadata(metadata, self.n_samples)
        self.n_jobs = n_jobs
        self.correspondence_thresholds = validate_thresholds(correspondence_thresholds)
        if distance_normalization not in DISTANCE_NORMALIZATIONS:
            raise ValueError(
                f"distance_normalization must be one of {DISTANCE_NORMALIZATIONS}, "
                f"got {distance_normalization!r}"
            )
        self.distance_normalization = distance_normalization

        self.similarity: SimilarityStrategy = _resolve_similarity(similarity, model)
        strategy_model = _strategy_model(self.similarity)
        self._model = strategy_model if strategy_model is not None else model
        self.nn_index: Any = None

        # Scale data if requested
        if scale_data:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None
            self.X_train_scaled = X_train.copy()

        # The selected retrieval space owns index construction and queries.
        self._strategy = self._make_strategy(self.similarity)
        self._strategy.build(self, X_train)

        self.distance_scale_ = 1.0
        if self.distance_normalization == "median_nn":
            self.distance_scale_ = self._estimate_distance_scale()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _fit_index(self, X_for_index: np.ndarray) -> NearestNeighbors:
        """Fit a NearestNeighbors index on ``X_for_index``."""
        logger.info(
            "Building k-NN index (k=%d, metric=%s, algorithm=%s)...",
            self.k,
            self.metric,
            self.algorithm,
        )
        index = NearestNeighbors(
            n_neighbors=min(self.k, self.n_samples),  # Handle k > n_samples
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )
        index.fit(X_for_index)
        logger.info("Index built on %d training samples", self.n_samples)
        return index

    def _make_strategy(self, similarity: SimilarityStrategy):
        """Select the internal retrieval space for the configured strategy."""
        if isinstance(similarity, Features):
            return _FeatureSpace()
        if isinstance(similarity, TreeLeaf):
            return _TreeLeafSpace(similarity)
        if isinstance(similarity, ForestProximity):
            return _ForestProximitySpace(similarity)

        features_weight = 0.0
        inner = similarity
        if isinstance(similarity, Blend):
            features_weight = similarity.features
            inner = similarity.strategy

        if isinstance(inner, HiddenActivations):
            if inner.output_weighting == "predicted_class":
                return _ClassActivationSpace(inner, features_weight)
            extractor = SklearnMLPActivationExtractor(
                layer=inner.layer,
                use_output_weights=inner.output_weighting != "none",
                input_transform=inner.input_transform,
            )
            return _ActivationSpace(extractor, features_weight)
        if isinstance(inner, CustomActivations):
            if not isinstance(inner.extractor, ActivationExtractor):
                raise TypeError(
                    "CustomActivations extractor must implement ActivationExtractor"
                )
            return _ActivationSpace(inner.extractor, features_weight)
        raise TypeError(f"Unsupported similarity strategy {similarity!r}")

    def _estimate_distance_scale(self) -> float:
        """Median nearest-neighbor distance among non-duplicated training cases.

        Distances below ``1e-6`` of the largest sampled distance are treated as
        duplicates (exact repeats or floating-point noise) and those cases are
        left out, so a partly duplicated training set still yields a meaningful
        scale. When every sampled case has a duplicate the scale falls back to
        1.0 with a warning.
        """
        rng = np.random.RandomState(0)
        reference = np.asarray(self._strategy.reference_distances(self, rng), float)
        reference = reference[np.isfinite(reference)]
        if reference.size == 0:
            warnings.warn(
                "Could not estimate a reference distance (need at least two "
                "training cases); distances are left unnormalized.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1.0
        tolerance = _DUPLICATE_RELATIVE_TOLERANCE * float(reference.max())
        distinct = reference[reference > tolerance]
        if distinct.size == 0:
            warnings.warn(
                "All sampled training cases have a duplicate at distance 0; "
                "distances are left unnormalized.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1.0
        duplicate_fraction = 1.0 - distinct.size / reference.size
        scale = float(np.median(distinct))
        logger.info(
            "Distance normalization scale (median_nn): %.6g "
            "(%.0f%% of sampled training cases had a duplicate and were excluded)",
            scale,
            100.0 * duplicate_fraction,
        )
        return scale

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_instance(
        self,
        test_sample: Union[np.ndarray, pd.Series, List],
        test_index: Optional[int] = None,
        true_class: Optional[Any] = None,
        predicted_class: Optional[Any] = None,
        model: Optional[Any] = None,
        k: Optional[int] = None,
        return_provenance: bool = True,
        distance_weighted: bool = True,
    ) -> Explanation:
        """
        Explain a prediction using case-based reasoning with k-NN precedent.

        This method:
        1. Finds k nearest neighbors in the pre-built index
        2. Computes weighted correspondence based on neighbor labels
        3. Returns explanation with neighbor details and correspondence score

        Args:
            test_sample: Sample to explain (n_features,)
            test_index: Index in test set (optional, for tracking)
            true_class: True class label (optional, for validation)
            predicted_class: Predicted class (optional, will use model if not provided)
            model: Trained model with predict() method (optional)
            k: Number of neighbors (optional, uses default from init if not provided)
            return_provenance: Include metadata in explanation
            distance_weighted: Use distance weighting for correspondence

        Returns:
            Explanation object with neighbors and correspondence
        """
        sample: np.ndarray = _query_rows(test_sample, self._columns, "test_sample")
        if sample.ndim == 2 and sample.shape[0] == 1:
            sample = sample[0]
        if sample.ndim != 1:
            raise ValueError(
                f"test_sample must be a single sample, got shape {sample.shape}; "
                "use explain_batch for several samples"
            )
        if len(sample) != self.n_features:
            raise ValueError(
                f"test_sample has {len(sample)} features, expected {self.n_features}"
            )

        X = sample[np.newaxis, :]
        if predicted_class is not None:
            predicted = [_label_value(predicted_class)]
        else:
            predicted = self._predict(X, model)

        return self._explain_rows(
            X,
            predicted,
            [_label_value(true_class)],
            [_label_value(test_index)],
            k,
            return_provenance,
            distance_weighted,
        )[0]

    def explain_batch(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Optional[Union[np.ndarray, pd.Series, List]] = None,
        predictions: Optional[Union[np.ndarray, List]] = None,
        model: Optional[Any] = None,
        k: Optional[int] = None,
        return_provenance: bool = True,
        distance_weighted: bool = True,
    ) -> List[Explanation]:
        """
        Explain multiple predictions efficiently.

        The whole batch is scaled, transformed, and queried against the index
        in vectorized calls (one ``predict`` call and one k-NN query per batch,
        or one per predicted class for predicted-class activation weighting),
        so this is much faster than calling ``explain_instance`` in a loop.
        Results match the per-instance path up to floating-point rounding.
        In activation spaces, exact ties between duplicate training rows
        can resolve to a different duplicate.

        Args:
            X_test: Test samples (n_samples, n_features)
            y_test: True labels (optional)
            predictions: Predicted labels (optional, will use model if not provided)
            model: Trained model (optional)
            k: Number of neighbors (optional, uses default from init)
            return_provenance: Include metadata
            distance_weighted: Use distance weighting

        Returns:
            List of Explanation objects
        """
        X = _query_rows(X_test, self._columns, "X_test")
        if X.ndim in (1, 2) and len(X) == 0:
            X = X.reshape(0, self.n_features)  # an empty batch has no rows
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"X_test must have shape (n_samples, {self.n_features}), "
                f"got {X.shape}"
            )
        n = len(X)

        if y_test is None:
            true_classes: List[Any] = [None] * n
        else:
            y = _as_label_array(y_test, "y_test")
            if len(y) != n:
                raise ValueError(
                    f"y_test has {len(y)} labels for {n} samples in X_test"
                )
            true_classes = [_label_value(value) for value in y]

        predicted: List[Any] = []
        if predictions is not None:
            preds = _as_label_array(predictions, "predictions")
            if len(preds) != n:
                raise ValueError(
                    f"predictions has {len(preds)} labels for {n} samples in X_test"
                )
            predicted = [_label_value(value) for value in preds]

        if n == 0:
            return []
        if predictions is None:
            predicted = self._predict(X, model)

        return self._explain_rows(
            X,
            predicted,
            true_classes,
            list(range(n)),
            k,
            return_provenance,
            distance_weighted,
        )

    def _model_input(self, X: np.ndarray) -> Any:
        """Return ``X`` in the form the user's estimators were fitted on.

        The index works on plain arrays, but a model or ``input_transform``
        fitted on a DataFrame expects one back: scikit-learn pipelines that
        select columns by name fail on arrays, and estimators that recorded
        ``feature_names_in_`` warn on every call. When the explainer was built
        from a DataFrame, rebuild one with the training columns before each
        model call. Array-trained explainers pass arrays through unchanged.
        """
        if self._columns is None:
            return X
        return pd.DataFrame(X, columns=self._columns)

    def _predict(self, X: np.ndarray, model: Optional[Any]) -> List[Any]:
        """Predict class labels for ``X`` with the call-time or bound model."""
        prediction_model = model if model is not None else self._model
        if prediction_model is None:
            raise ValueError("Either predicted_class or model must be provided")
        prediction_input: Any = self._model_input(X)
        hidden = _hidden_strategy(self.similarity)
        if hidden is not None and hidden.input_transform is not None:
            transform = getattr(
                hidden.input_transform, "transform", hidden.input_transform
            )
            prediction_input = transform(prediction_input)
        raw_predictions = np.asarray(prediction_model.predict(prediction_input))
        if raw_predictions.ndim == 0:
            raw_predictions = raw_predictions[np.newaxis]
        raw_predictions = _as_label_array(raw_predictions, "model.predict output")
        if len(raw_predictions) != len(X):
            raise ValueError(
                f"model.predict returned {len(raw_predictions)} predictions for "
                f"{len(X)} samples"
            )
        return [_label_value(value) for value in raw_predictions]

    def _explain_rows(
        self,
        X: np.ndarray,
        predicted: List[Any],
        true_classes: List[Any],
        test_indices: List[Optional[int]],
        k: Optional[int],
        return_provenance: bool,
        distance_weighted: bool,
    ) -> List[Explanation]:
        """Shared explanation path for single and batch queries."""
        if self.scale_data:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()

        k_actual = self.k if k is None else _validate_k(k)
        k_actual = min(k_actual, self.n_samples)  # Handle case where k > n_samples

        distances_list, indices_list = self._strategy.neighbors_batch(
            self, X, X_scaled, predicted, k_actual
        )

        retrieval_space = self._strategy.label(self)
        distance_metric = self._strategy.metric_label(self)
        explanations = []
        for row in range(len(X)):
            distances = np.asarray(distances_list[row], float) / self.distance_scale_
            neighbors = self._build_neighbors(
                indices_list[row], distances, return_provenance
            )
            neighbor_tuples = [(n.index, n.distance, n.label) for n in neighbors]
            correspondence, interpretation = compute_correspondence(
                neighbor_tuples,
                predicted[row],
                distance_weighted=distance_weighted,
                class_weights=self.class_weights,
                thresholds=self.correspondence_thresholds,
            )
            explanations.append(
                Explanation(
                    test_sample=X[row].copy(),
                    test_index=test_indices[row],
                    neighbors=neighbors,
                    predicted_class=predicted[row],
                    true_class=true_classes[row],
                    correspondence=correspondence,
                    correspondence_interpretation=interpretation,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    retrieval_space=retrieval_space,
                    distance_metric=distance_metric,
                    distance_scale=self.distance_scale_,
                )
            )
        return explanations

    def _build_neighbors(self, indices, distances, return_provenance):
        """Assemble Neighbor objects, attaching provenance metadata when present."""
        neighbors = []
        for idx, dist in zip(indices, distances):
            neighbor_metadata = {}
            if return_provenance and self.metadata:
                for key, values in self.metadata.items():
                    neighbor_metadata[key] = values[idx]

            neighbors.append(
                Neighbor(
                    index=int(idx),
                    distance=float(dist),
                    label=_label_value(self.y_train[idx]),
                    features=self.X_train_original[idx].copy(),
                    metadata=neighbor_metadata if neighbor_metadata else None,
                )
            )
        return neighbors

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compute_correspondence(
        self, explanation: Explanation, distance_weighted: bool = True
    ) -> Tuple[float, str]:
        """Compute correspondence for an explanation using the configured weights.

        Applies the ``class_weights`` and ``correspondence_thresholds`` passed to
        the constructor and returns ``(correspondence, interpretation)``. This
        is the same computation ``explain_instance`` runs, exposed as a method
        for reuse (for example, to re-score an explanation with a different
        distance weighting).
        """
        neighbor_tuples = [
            (n.index, n.distance, n.label) for n in explanation.neighbors
        ]
        # Bare name resolves to the module-level function, not this method.
        return compute_correspondence(
            neighbor_tuples,
            explanation.predicted_class,
            distance_weighted=distance_weighted,
            class_weights=self.class_weights,
            thresholds=self.correspondence_thresholds,
        )

    @property
    def retrieval_space(self) -> str:
        """Label of the configured retrieval space (``"features"``, ...)."""
        return self._strategy.label(self)

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training data."""
        unique_classes: np.ndarray
        class_counts: np.ndarray
        unique_classes, class_counts = np.unique(self.y_train, return_counts=True)

        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": len(unique_classes),
            "classes": unique_classes.tolist(),
            "class_counts": dict(zip(unique_classes.tolist(), class_counts.tolist())),
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "scaled": self.scale_data,
            "has_metadata": bool(self.metadata),
            "default_k": self.k,
            "retrieval_space": self.retrieval_space,
            "distance_normalization": self.distance_normalization,
            "distance_scale": self.distance_scale_,
            "correspondence_thresholds": self.correspondence_thresholds,
        }

    def __repr__(self) -> str:
        return (
            f"CaseExplainer(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"k={self.k}, mode='{self.retrieval_space}', "
            f"algorithm='{self.algorithm}')"
        )
