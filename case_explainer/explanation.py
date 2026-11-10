"""
Explanation object for case-based explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any


class Neighbor:
    """Represents a single nearest neighbor."""

    def __init__(
        self,
        index: int,
        distance: float,
        label: Any,
        features: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.distance = distance
        self.label = label
        self.features = features
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        meta_str = f", {self.metadata}" if self.metadata else ""
        return f"Neighbor(index={self.index}, distance={self.distance:.4f}, label={self.label}{meta_str})"


class Explanation:
    """
    Explanation object containing case-based explanation details.
    """

    def __init__(
        self,
        test_sample: np.ndarray,
        test_index: Optional[int],
        neighbors: List[Neighbor],
        predicted_class: Any,
        true_class: Optional[Any],
        correspondence: float,
        correspondence_interpretation: str,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[Dict[Any, str]] = None,
        retrieval_space: str = "features",
        distance_metric: Optional[str] = None,
        distance_scale: float = 1.0,
    ):
        """
        Initialize explanation.

        Args:
            test_sample: The test sample being explained
            test_index: Index in test set (if applicable)
            neighbors: List of Neighbor objects
            predicted_class: Predicted class label
            true_class: True class label (if available)
            correspondence: Correspondence score [0, 1]
            correspondence_interpretation: "high", "medium", or "low"
                (``"undefined"`` when no neighbors were retrieved)
            feature_names: Names of features (optional)
            class_names: Mapping from class labels to names (optional)
            retrieval_space: Which space the neighbors were retrieved in
                (``"features"``, ``"activations"``, ``"hybrid(features=0.30)"``,
                ``"tree_leaf"``, or ``"forest_proximity"``). Neighbor distances
                are measured in this space, so their scale is only comparable
                between explanations that share a retrieval space.
            distance_metric: Name of the distance used within the retrieval
                space (for example ``"euclidean"`` or ``"shared_leaf_fraction"``).
            distance_scale: Divisor applied to raw distances before they were
                stored on the neighbors. ``1.0`` means raw distances; any other
                value (it may be smaller or larger than 1) means the explainer
                normalized them, with
                ``raw_distance = neighbor.distance * distance_scale``.
        """
        self.test_sample = test_sample
        self.test_index = test_index
        self.neighbors = neighbors
        self.predicted_class = predicted_class
        self.true_class = true_class
        self.correspondence = correspondence
        self.correspondence_interpretation = correspondence_interpretation
        self.retrieval_space = retrieval_space
        self.distance_metric = distance_metric
        self.distance_scale = float(distance_scale)
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(test_sample))]
        else:
            self.feature_names = list(feature_names)
        # Convert a class_names sequence to a dict if needed
        self.class_names: Dict[Any, str]
        if class_names is None:
            self.class_names = {}
        elif isinstance(class_names, dict):
            self.class_names = class_names
        else:
            self.class_names = {i: name for i, name in enumerate(class_names)}

    def get_predicted_class_name(self) -> str:
        """Get the predicted class name."""
        return self.class_names.get(self.predicted_class, str(self.predicted_class))

    def get_true_class_name(self) -> Optional[str]:
        """Get the true class name."""
        if self.true_class is None:
            return None
        return self.class_names.get(self.true_class, str(self.true_class))

    def is_correct(self) -> Optional[bool]:
        """Check if prediction matches true label (if available)."""
        if self.true_class is None:
            return None
        return self.predicted_class == self.true_class

    def summary(self) -> str:
        """Generate a text summary of the explanation."""
        lines = []
        lines.append("=" * 60)
        lines.append("CASE-BASED EXPLANATION")
        lines.append("=" * 60)

        if self.test_index is not None:
            lines.append(f"Test sample index: {self.test_index}")

        lines.append(f"Predicted class: {self.get_predicted_class_name()}")

        if self.true_class is not None:
            correct_str = "[OK]" if self.is_correct() else "[X]"
            lines.append(f"True class: {self.get_true_class_name()} {correct_str}")

        lines.append(
            f"Correspondence: {self.correspondence:.2%} ({self.correspondence_interpretation})"
        )
        lines.append(f"Retrieval space: {self.describe_retrieval_space()}")
        lines.append("")
        lines.append(f"Nearest {len(self.neighbors)} neighbors:")
        lines.append("-" * 60)

        for i, neighbor in enumerate(self.neighbors, 1):
            neighbor_class = self.class_names.get(neighbor.label, str(neighbor.label))
            match_str = "*" if neighbor.label == self.predicted_class else " "
            lines.append(
                f"{i}. [{match_str}] Index {neighbor.index}: class {neighbor_class}, "
                f"distance {neighbor.distance:.4f}"
            )

            # Add metadata if available
            if neighbor.metadata:
                for key, value in neighbor.metadata.items():
                    lines.append(f"      {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def describe_retrieval_space(self) -> str:
        """One-line description of the retrieval space and distance units."""
        text = self.retrieval_space
        if self.distance_metric:
            text += f" ({self.distance_metric})"
        if self.distance_scale != 1.0:
            text += f", distances normalized by {self.distance_scale:.4g}"
        return text

    def to_dict(self) -> Dict[str, Any]:
        """Export explanation as dictionary (for JSON serialization)."""
        return {
            "test_index": self.test_index,
            "test_sample": self.test_sample.tolist(),
            "predicted_class": self.predicted_class,
            "predicted_class_name": self.get_predicted_class_name(),
            "true_class": self.true_class,
            "true_class_name": self.get_true_class_name(),
            "is_correct": self.is_correct(),
            "correspondence": float(self.correspondence),
            "correspondence_interpretation": self.correspondence_interpretation,
            "retrieval_space": self.retrieval_space,
            "distance_metric": self.distance_metric,
            "distance_scale": self.distance_scale,
            "neighbors": [
                {
                    "index": n.index,
                    "distance": float(n.distance),
                    "label": n.label,
                    "label_name": self.class_names.get(n.label, str(n.label)),
                    "features": n.features.tolist(),
                    "metadata": n.metadata,
                }
                for n in self.neighbors
            ],
            "feature_names": self.feature_names,
        }

    def plot(
        self,
        plot_type: str = "bar",
        highlight_differences: bool = True,
        show_distances: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Visualize the explanation.

        Args:
            plot_type: 'bar' (default), 'radar', or 'parallel'. Radar and
                parallel coordinates are not implemented yet and fall back
                to the bar plot.
            highlight_differences: Reserved; currently has no effect
            show_distances: Whether to show each neighbor's distance in the
                legend
            save_path: Path to save the figure. When given, the figure is
                written to this file and closed instead of being shown.
            figsize: Figure size
        """
        if plot_type == "bar":
            self._plot_bar(figsize, save_path, show_distances)
        elif plot_type == "radar":
            self._plot_radar(figsize, save_path, show_distances)
        elif plot_type == "parallel":
            self._plot_parallel(figsize, save_path, show_distances)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _plot_bar(
        self,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        show_distances: bool = True,
    ) -> None:
        """Create bar plot comparing features."""
        n_features = len(self.test_sample)
        n_neighbors = len(self.neighbors)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(n_features)
        width = 0.8 / (n_neighbors + 1)

        # Plot test sample
        ax.bar(
            x,
            self.test_sample,
            width,
            label="Test Sample",
            color="red",
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )

        # Plot neighbors
        colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.8, n_neighbors))
        for i, neighbor in enumerate(self.neighbors):
            offset = width * (i + 1)
            match_str = "*" if neighbor.label == self.predicted_class else ""
            legend_label = f"Neighbor {i+1} {match_str}".rstrip()
            if show_distances:
                legend_label += f" (d={neighbor.distance:.3g})"
            ax.bar(
                x + offset,
                neighbor.features,
                width,
                label=legend_label,
                color=colors[i],
                alpha=0.6,
            )

        ax.set_xlabel("Features")
        ax.set_ylabel("Feature Values")
        ax.set_title(
            f"Case-Based Explanation (Correspondence: {self.correspondence:.2%})"
        )
        ax.set_xticks(x + width * n_neighbors / 2)
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def _plot_radar(
        self,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        show_distances: bool = True,
    ) -> None:
        """Create radar plot (not implemented yet - use bar for now)."""
        print("Radar plot not yet implemented, using bar plot instead.")
        self._plot_bar(figsize, save_path, show_distances)

    def _plot_parallel(
        self,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        show_distances: bool = True,
    ) -> None:
        """Create parallel coordinates plot (not implemented yet - use bar for now)."""
        print("Parallel coordinates plot not yet implemented, using bar plot instead.")
        self._plot_bar(figsize, save_path, show_distances)

    def __repr__(self) -> str:
        return (
            f"Explanation(test_index={self.test_index}, "
            f"predicted={self.get_predicted_class_name()}, "
            f"correspondence={self.correspondence:.2%}, "
            f"neighbors={len(self.neighbors)}, "
            f"space='{self.retrieval_space}')"
        )
