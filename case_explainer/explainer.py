"""
Main CaseExplainer class for case-based explanations.

Based on refined Method 2 (case-based) from hardware trojan detection pipeline.
Uses sklearn's NearestNeighbors for efficient k-NN lookups with pre-built index.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

from .explanation import Explanation, Neighbor
from .metrics import compute_correspondence
from ._compat import LEGACY_ACTIVATION_WARNING
from .activations import (
    ActivationExtractor,
    HiddenActivationRetrieval,
    CustomActivationRetrieval,
    TreeLeafRetrieval,
    ForestProximityRetrieval,
)


def _label_value(value: Any) -> Any:
    """Convert NumPy scalar labels to their JSON-friendly Python values."""
    return value.item() if isinstance(value, np.generic) else value


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
        metric: str = 'euclidean',
        algorithm: str = 'auto',
        scale_data: bool = True,
        class_weights: Optional[Dict[Any, float]] = None,
        metadata: Optional[Dict[str, List]] = None,
        n_jobs: int = -1,
        activation_extractor: Optional[Any] = None,
        model: Optional[Any] = None,
        blend_alpha: float = 0.0,
        activation_layer: Optional[Union[str, int]] = None,
        use_output_weights: bool = True,
        retrieval: Optional[Union[
            HiddenActivationRetrieval,
            CustomActivationRetrieval,
            TreeLeafRetrieval,
            ForestProximityRetrieval,
        ]] = None,
    ):
        """
        Initialize CaseExplainer with training data and build k-NN index.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            k: Number of nearest neighbors for explanations (default: 5)
            feature_names: Names of features (optional)
            class_names: Mapping from class labels to names (optional)
            metric: Distance metric (default: 'euclidean')
            algorithm: k-NN algorithm - 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
            scale_data: Whether to standardize features (recommended: True)
            class_weights: Optional weights for each class in correspondence computation
                          e.g., {0: 1.0, 1: 2.0} to weight class 1 twice as much
            metadata: Optional dict with metadata for each training sample
                     e.g., {'sample_id': [...], 'source': [...], ...}
            n_jobs: Number of parallel jobs for k-NN search (-1 = all CPUs)
            activation_extractor: Optional ActivationExtractor instance. When
                provided, the k-NN index is built on model activations (or a
                blend of activations and features) instead of raw features.
                Implements Caruana et al. (1999).
            model: Trained model required when activation_extractor is set.
                   Used to extract activations from the training data.
            blend_alpha: Controls the feature/activation blend when
                activation_extractor is set.
                A value of 0.0 uses pure activations, 1.0 uses pure features,
                and intermediate values build an index on
                ``[sqrt(alpha)*X_scaled, sqrt(1-alpha)*A_scaled]``.
            activation_layer: Convenience shorthand for
                ``SklearnMLPActivationExtractor(layer=..., use_output_weights=...)``.
                Accepted values: ``'last_hidden'`` (default), ``'all_hidden'``
                (concatenated, position-weighted), or an integer layer index.
                Mutually exclusive with ``activation_extractor``.
                Requires ``model`` (a fitted ``MLPClassifier``) to also be set.
            use_output_weights: When ``activation_layer`` is used, controls
                whether per-unit output-connection weighting (Caruana §5) is
                applied to the last hidden layer.  Ignored when
                ``activation_extractor`` is supplied directly.
            retrieval: Explicit retrieval configuration. Use
                ``HiddenActivationRetrieval`` for an sklearn MLP or
                ``TreeLeafRetrieval`` for same-leaf decision-tree retrieval or
                ``ForestProximityRetrieval`` for shared-leaf forest proximity.
                Mutually exclusive with the legacy activation parameters.
        """
        # Convert inputs to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            X_train = np.asarray(X_train)
        
        if isinstance(y_train, (pd.Series, list)):
            y_train = np.asarray(y_train)
        
        # Store original data
        self.X_train_original = X_train.copy()
        self.y_train = y_train.copy()
        
        # Validate shapes
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train must have same length "
                           f"(got {len(X_train)} and {len(y_train)})")
        
        self.n_samples, self.n_features = X_train.shape
        self.k = k
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features)]
        self.class_names = class_names or {}
        self.metric = metric
        self.algorithm = algorithm
        self.scale_data = scale_data
        self.class_weights = class_weights or {}
        self.metadata = metadata or {}
        self.n_jobs = n_jobs
        self.retrieval = retrieval
        self._tree_retrieval = None
        self._forest_retrieval = None
        self._class_activation_extractors = None
        self._class_nn_indexes = None
        self._pending_class_retrieval = None
        self._model = model
        if retrieval is None and (
            activation_extractor is not None or activation_layer is not None
        ):
            warnings.warn(
                LEGACY_ACTIVATION_WARNING,
                DeprecationWarning,
                stacklevel=2,
            )
        if retrieval is not None:
            if activation_extractor is not None or activation_layer is not None:
                raise ValueError(
                    "retrieval cannot be combined with activation_extractor "
                    "or activation_layer"
                )
            if blend_alpha != 0.0 or use_output_weights is not True:
                raise ValueError(
                    "Set blend_alpha and use_output_weights on the retrieval "
                    "configuration, not CaseExplainer"
                )
            model = retrieval.model
            self._model = model
            if isinstance(retrieval, HiddenActivationRetrieval):
                from .activations import SklearnMLPActivationExtractor
                if retrieval.output_weighting not in (
                    "mean_abs", "predicted_class", "none"
                ):
                    raise ValueError(
                        "output_weighting must be 'mean_abs', "
                        "'predicted_class', or 'none'"
                    )
                if not retrieval.use_output_weights:
                    output_weighting = "none"
                else:
                    output_weighting = retrieval.output_weighting
                if output_weighting == "predicted_class":
                    self._pending_class_retrieval = retrieval
                else:
                    activation_extractor = SklearnMLPActivationExtractor(
                        layer=retrieval.layer,
                        use_output_weights=output_weighting != "none",
                        input_transform=retrieval.input_transform,
                    )
                blend_alpha = retrieval.blend_alpha
            elif isinstance(retrieval, CustomActivationRetrieval):
                if not isinstance(retrieval.extractor, ActivationExtractor):
                    raise TypeError(
                        "CustomActivationRetrieval extractor must implement "
                        "ActivationExtractor"
                    )
                activation_extractor = retrieval.extractor
                blend_alpha = retrieval.blend_alpha
            elif isinstance(retrieval, TreeLeafRetrieval):
                self._validate_tree_retrieval(retrieval)
                self._tree_retrieval = retrieval
            elif isinstance(retrieval, ForestProximityRetrieval):
                self._validate_forest_retrieval(retrieval)
                self._forest_retrieval = retrieval
            else:
                raise TypeError(
                    "retrieval must be HiddenActivationRetrieval, "
                    "CustomActivationRetrieval, "
                    "TreeLeafRetrieval, or ForestProximityRetrieval"
                )
        self.activation_extractor = activation_extractor
        self.blend_alpha = float(blend_alpha)
        if not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")
        if activation_extractor is not None and activation_layer is not None:
            raise ValueError(
                "Provide either activation_extractor or activation_layer, not both"
            )
        if activation_layer is not None:
            from .activations import SklearnMLPActivationExtractor
            activation_extractor = SklearnMLPActivationExtractor(
                layer=activation_layer,
                use_output_weights=use_output_weights,
            )
            self.activation_extractor = activation_extractor
        if activation_extractor is not None and model is None:
            raise ValueError("model must be provided when activation_extractor is set")
        
        # Validate metadata
        if self.metadata:
            for key, values in self.metadata.items():
                if len(values) != self.n_samples:
                    raise ValueError(f"Metadata '{key}' has {len(values)} items, "
                                   f"expected {self.n_samples}")
        
        # Scale data if requested
        if scale_data:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None
            self.X_train_scaled = X_train.copy()

        if self._pending_class_retrieval is not None:
            self._initialize_class_activation_retrieval(
                self._pending_class_retrieval, X_train, model
            )
        
        # --- Build the index data (features, activations, or hybrid blend) ---
        if activation_extractor is not None:
            logger.info(
                "Extracting training activations with %s (blend_alpha=%.2f)...",
                type(activation_extractor).__name__, blend_alpha
            )
            raw_activations = activation_extractor.fit_transform(model, X_train)
            # Built-in extractors return normalized, metric-ready coordinates.
            if getattr(activation_extractor, "metric_ready", False):
                self._act_scaler = None
                A_scaled = raw_activations
            else:
                self._act_scaler = StandardScaler()
                A_scaled = self._act_scaler.fit_transform(raw_activations)

            alpha = self.blend_alpha
            if alpha == 0.0:
                X_for_index = A_scaled
            elif alpha == 1.0:
                X_for_index = self.X_train_scaled
            else:
                X_for_index = np.hstack([
                    np.sqrt(alpha) * self.X_train_scaled,
                    np.sqrt(1.0 - alpha) * A_scaled,
                ])
            logger.info(
                "Index space: %s, dims=%d",
                "activations" if alpha == 0.0 else
                "features" if alpha == 1.0 else
                f"hybrid(alpha={alpha:.2f})",
                X_for_index.shape[1],
            )
        else:
            self._act_scaler = None
            X_for_index = self.X_train_scaled

        if self._tree_retrieval is not None:
            self._training_leaf_ids = np.asarray(
                self._tree_retrieval.model.apply(X_train)
            )
        elif self._forest_retrieval is not None:
            self._training_leaf_ids = np.asarray(
                self._forest_retrieval.model.apply(X_train)
            )

        # Build k-NN index using sklearn's NearestNeighbors
        # This is done once during initialization for efficiency
        logger.info("Building k-NN index (k=%d, metric=%s, algorithm=%s)...", k, metric, algorithm)
        self.nn_index = NearestNeighbors(
            n_neighbors=min(k, self.n_samples),  # Handle case where k > n_samples
            metric=metric,
            algorithm=algorithm,
            n_jobs=n_jobs
        )
        self.nn_index.fit(X_for_index)
        logger.info("Index built on %d training samples", self.n_samples)
    
    def explain_instance(
        self,
        test_sample: Union[np.ndarray, pd.Series, List],
        test_index: Optional[int] = None,
        true_class: Optional[Any] = None,
        predicted_class: Optional[Any] = None,
        model: Optional[Any] = None,
        k: Optional[int] = None,
        return_provenance: bool = True,
        distance_weighted: bool = True
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
        # Convert test sample to numpy array
        if isinstance(test_sample, (pd.Series, list)):
            test_sample = np.asarray(test_sample)
        
        if len(test_sample) != self.n_features:
            raise ValueError(f"test_sample has {len(test_sample)} features, "
                           f"expected {self.n_features}")
        
        # Scale test sample if needed
        if self.scale_data:
            test_sample_scaled = self.scaler.transform([test_sample])[0]
        else:
            test_sample_scaled = test_sample.copy()

        # Build the query vector for the index (mirrors __init__ logic)
        if self._class_activation_extractors is not None:
            test_vec = None
        elif self.activation_extractor is not None:
            raw_act = self.activation_extractor.transform([test_sample])
            A_scaled = (
                raw_act if self._act_scaler is None
                else self._act_scaler.transform(raw_act)
            )
            alpha = self.blend_alpha
            if alpha == 0.0:
                test_vec = A_scaled[0]
            elif alpha == 1.0:
                test_vec = test_sample_scaled
            else:
                test_vec = np.hstack([
                    np.sqrt(alpha) * test_sample_scaled,
                    np.sqrt(1.0 - alpha) * A_scaled[0],
                ])
        else:
            test_vec = test_sample_scaled

        # Get prediction if not provided
        if predicted_class is None:
            prediction_model = model if model is not None else self._model
            if prediction_model is None:
                raise ValueError("Either predicted_class or model must be provided")
            prediction_input = np.atleast_2d(test_sample)
            if isinstance(self.retrieval, HiddenActivationRetrieval):
                transform = self.retrieval.input_transform
                if transform is not None:
                    transform = getattr(transform, "transform", transform)
                    prediction_input = transform(prediction_input)
            predicted_class = _label_value(
                prediction_model.predict(prediction_input)[0]
            )
        else:
            predicted_class = _label_value(predicted_class)
        true_class = _label_value(true_class)
        
        # Query pre-built k-NN index
        k_actual = k if k is not None else self.k
        k_actual = min(k_actual, self.n_samples)  # Handle case where k > n_samples
        
        if self._tree_retrieval is not None:
            distances, indices = self._tree_neighbors(
                test_sample, test_sample_scaled, k_actual
            )
        elif self._forest_retrieval is not None:
            distances, indices = self._forest_neighbors(test_sample, k_actual)
        elif self._class_activation_extractors is not None:
            distances, indices = self._class_activation_neighbors(
                test_sample, test_sample_scaled, predicted_class, k_actual
            )
        else:
            distances, indices = self.nn_index.kneighbors(
                [test_vec],
                n_neighbors=k_actual
            )
            distances = distances[0]
            indices = indices[0]
        
        # Create Neighbor objects
        neighbors = []
        for idx, dist in zip(indices, distances):
            neighbor_metadata = {}
            if return_provenance and self.metadata:
                for key, values in self.metadata.items():
                    neighbor_metadata[key] = values[idx]
            
            neighbor = Neighbor(
                index=int(idx),
                distance=float(dist),
                label=_label_value(self.y_train[idx]),
                features=self.X_train_original[idx].copy(),
                metadata=neighbor_metadata if neighbor_metadata else None
            )
            neighbors.append(neighbor)
        
        # Compute correspondence with optional class weighting
        neighbor_tuples = [(n.index, n.distance, n.label) for n in neighbors]
        correspondence, interpretation = compute_correspondence(
            neighbor_tuples,
            predicted_class,
            distance_weighted=distance_weighted,
            class_weights=self.class_weights
        )
        
        # Create explanation
        explanation = Explanation(
            test_sample=test_sample.copy(),
            test_index=test_index,
            neighbors=neighbors,
            predicted_class=predicted_class,
            true_class=true_class,
            correspondence=correspondence,
            correspondence_interpretation=interpretation,
            feature_names=self.feature_names,
            class_names=self.class_names
        )
        
        return explanation

    def _initialize_class_activation_retrieval(self, retrieval, X_train, model):
        from .activations import SklearnMLPActivationExtractor

        if not hasattr(model, "classes_"):
            raise TypeError(
                "predicted_class output weighting requires an MLPClassifier"
            )
        output_count = model.coefs_[-1].shape[1]
        classes = list(model.classes_)
        if output_count not in (1, len(classes)):
            raise ValueError("MLP output columns do not match model classes")

        self._class_activation_extractors = {}
        self._class_nn_indexes = {}
        shared_binary_extractor = None
        shared_binary_index = None
        for class_position, class_label in enumerate(classes):
            output_class = 0 if output_count == 1 else class_position
            if output_count == 1 and shared_binary_extractor is not None:
                extractor = shared_binary_extractor
                index = shared_binary_index
            else:
                extractor = SklearnMLPActivationExtractor(
                    layer=retrieval.layer,
                    use_output_weights=True,
                    input_transform=retrieval.input_transform,
                    output_class=output_class,
                )
                activations = extractor.fit_transform(model, X_train)
                index_data = self._blend_index_data(
                    activations, retrieval.blend_alpha
                )
                index = NearestNeighbors(
                    n_neighbors=min(self.k, self.n_samples),
                    metric=self.metric,
                    algorithm=self.algorithm,
                    n_jobs=self.n_jobs,
                ).fit(index_data)
                if output_count == 1:
                    shared_binary_extractor = extractor
                    shared_binary_index = index
            self._class_activation_extractors[class_label] = extractor
            self._class_nn_indexes[class_label] = index
        self._pending_class_retrieval = None

    def _blend_index_data(self, activations, alpha):
        if alpha == 0.0:
            return activations
        if alpha == 1.0:
            return self.X_train_scaled
        return np.hstack([
            np.sqrt(alpha) * self.X_train_scaled,
            np.sqrt(1.0 - alpha) * activations,
        ])

    def _class_activation_neighbors(
        self, test_sample, test_sample_scaled, predicted_class, k_actual
    ):
        if predicted_class not in self._class_activation_extractors:
            raise ValueError(
                f"predicted_class {predicted_class!r} is not in model.classes_"
            )
        extractor = self._class_activation_extractors[predicted_class]
        activation = extractor.transform([test_sample])[0]
        alpha = self.blend_alpha
        if alpha == 0.0:
            test_vec = activation
        elif alpha == 1.0:
            test_vec = test_sample_scaled
        else:
            test_vec = np.hstack([
                np.sqrt(alpha) * test_sample_scaled,
                np.sqrt(1.0 - alpha) * activation,
            ])
        distances, indices = self._class_nn_indexes[predicted_class].kneighbors(
            [test_vec], n_neighbors=k_actual
        )
        return distances[0], indices[0]

    @staticmethod
    def _validate_tree_retrieval(retrieval: TreeLeafRetrieval) -> None:
        from sklearn.tree import DecisionTreeClassifier

        if not isinstance(retrieval.model, DecisionTreeClassifier):
            raise TypeError(
                "TreeLeafRetrieval requires a fitted sklearn decision tree "
                "classifier"
            )
        if not hasattr(retrieval.model, "tree_"):
            raise ValueError("TreeLeafRetrieval model must be fitted")
        if retrieval.overflow not in ("truncate", "nearest"):
            raise ValueError("overflow must be 'truncate' or 'nearest'")
        if retrieval.within_leaf != "feature_distance":
            raise ValueError("within_leaf must be 'feature_distance'")

    def _tree_neighbors(self, test_sample, test_sample_scaled, k_actual):
        query_leaf = int(self._tree_retrieval.model.apply([test_sample])[0])
        same_leaf = np.flatnonzero(self._training_leaf_ids == query_leaf)
        feature_distances = np.linalg.norm(
            self.X_train_scaled - test_sample_scaled, axis=1
        )
        same_leaf = same_leaf[np.argsort(feature_distances[same_leaf])]
        indices = same_leaf[:k_actual]

        if len(indices) < k_actual and self._tree_retrieval.overflow == "nearest":
            outside = np.flatnonzero(self._training_leaf_ids != query_leaf)
            outside = outside[np.argsort(feature_distances[outside])]
            indices = np.concatenate([indices, outside[:k_actual - len(indices)]])

        return feature_distances[indices], indices

    @staticmethod
    def _validate_forest_retrieval(retrieval: ForestProximityRetrieval) -> None:
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(retrieval.model, RandomForestClassifier):
            raise TypeError(
                "ForestProximityRetrieval requires a fitted sklearn random "
                "forest classifier"
            )
        if not hasattr(retrieval.model, "estimators_"):
            raise ValueError("ForestProximityRetrieval model must be fitted")

    def _forest_neighbors(self, test_sample, k_actual):
        query_leaves = self._forest_retrieval.model.apply([test_sample])[0]
        distances = 1.0 - np.mean(
            self._training_leaf_ids == query_leaves, axis=1
        )
        indices = np.argsort(distances, kind="stable")[:k_actual]
        return distances[indices], indices
    
    def explain_batch(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Optional[Union[np.ndarray, pd.Series, List]] = None,
        predictions: Optional[Union[np.ndarray, List]] = None,
        model: Optional[Any] = None,
        k: Optional[int] = None,
        return_provenance: bool = True,
        distance_weighted: bool = True
    ) -> List[Explanation]:
        """
        Explain multiple predictions efficiently.
        
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
        # Convert inputs
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        else:
            X_test = np.asarray(X_test)
        
        if y_test is not None:
            if isinstance(y_test, (pd.Series, list)):
                y_test = np.asarray(y_test)
        
        if predictions is not None:
            if isinstance(predictions, list):
                predictions = np.asarray(predictions)
        
        # Generate explanations
        explanations = []
        for i, sample in enumerate(X_test):
            true_class = (
                None if y_test is None else _label_value(y_test[i])
            )
            pred_class = (
                None if predictions is None else _label_value(predictions[i])
            )
            
            explanation = self.explain_instance(
                test_sample=sample,
                test_index=i,
                true_class=true_class,
                predicted_class=pred_class,
                model=model,
                k=k,
                return_provenance=return_provenance,
                distance_weighted=distance_weighted
            )
            explanations.append(explanation)
        
        return explanations
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training data."""
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
            "default_k": self.k
        }
    
    def __repr__(self) -> str:
        has_activation_retrieval = (
            self.activation_extractor is not None
            or self._class_activation_extractors is not None
        )
        mode = (
            "tree_leaf" if self._tree_retrieval is not None
            else "forest_proximity" if self._forest_retrieval is not None
            else "activations" if (has_activation_retrieval and self.blend_alpha == 0.0)
            else f"hybrid(alpha={self.blend_alpha:.2f})" if has_activation_retrieval
            else "features"
        )
        return (f"CaseExplainer(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"k={self.k}, mode='{mode}', algorithm='{self.algorithm}')")
