"""
Main CaseExplainer class for case-based explanations.

Based on refined Method 2 (case-based) from hardware trojan detection pipeline.
Uses sklearn's NearestNeighbors for efficient k-NN lookups with pre-built index.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

from .explanation import Explanation, Neighbor
from .metrics import compute_correspondence


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
        class_names: Optional[Dict[int, str]] = None,
        metric: str = 'euclidean',
        algorithm: str = 'auto',
        scale_data: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        metadata: Optional[Dict[str, List]] = None,
        n_jobs: int = -1
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
        
        # Build k-NN index using sklearn's NearestNeighbors
        # This is done once during initialization for efficiency
        logger.info("Building k-NN index (k=%d, metric=%s, algorithm=%s)...", k, metric, algorithm)
        self.nn_index = NearestNeighbors(
            n_neighbors=min(k, self.n_samples),  # Handle case where k > n_samples
            metric=metric,
            algorithm=algorithm,
            n_jobs=n_jobs
        )
        self.nn_index.fit(self.X_train_scaled)
        logger.info("Index built on %d training samples", self.n_samples)
    
    def explain_instance(
        self,
        test_sample: Union[np.ndarray, pd.Series, List],
        test_index: Optional[int] = None,
        true_class: Optional[int] = None,
        predicted_class: Optional[int] = None,
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
        
        # Get prediction if not provided
        if predicted_class is None:
            if model is None:
                raise ValueError("Either predicted_class or model must be provided")
            predicted_class = int(model.predict([test_sample])[0])
        
        # Query pre-built k-NN index
        k_actual = k if k is not None else self.k
        k_actual = min(k_actual, self.n_samples)  # Handle case where k > n_samples
        
        distances, indices = self.nn_index.kneighbors(
            [test_sample_scaled],
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
                label=int(self.y_train[idx]),
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
            true_class = None if y_test is None else int(y_test[i])
            pred_class = None if predictions is None else int(predictions[i])
            
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
        return (f"CaseExplainer(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"k={self.k}, algorithm='{self.algorithm}')")
