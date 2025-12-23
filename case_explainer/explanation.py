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
        label: int,
        features: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
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
        predicted_class: int,
        true_class: Optional[int],
        correspondence: float,
        correspondence_interpretation: str,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[Dict[int, str]] = None
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
            feature_names: Names of features (optional)
            class_names: Mapping from class labels to names (optional)
        """
        self.test_sample = test_sample
        self.test_index = test_index
        self.neighbors = neighbors
        self.predicted_class = predicted_class
        self.true_class = true_class
        self.correspondence = correspondence
        self.correspondence_interpretation = correspondence_interpretation
        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(test_sample))]
        # Convert class_names list to dict if needed
        if isinstance(class_names, list):
            self.class_names = {i: name for i, name in enumerate(class_names)}
        else:
            self.class_names = class_names or {}
    
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
            correct_str = "✓" if self.is_correct() else "✗"
            lines.append(f"True class: {self.get_true_class_name()} {correct_str}")
        
        lines.append(f"Correspondence: {self.correspondence:.2%} ({self.correspondence_interpretation})")
        lines.append("")
        lines.append(f"Nearest {len(self.neighbors)} neighbors:")
        lines.append("-" * 60)
        
        for i, neighbor in enumerate(self.neighbors, 1):
            neighbor_class = self.class_names.get(neighbor.label, str(neighbor.label))
            match_str = "✓" if neighbor.label == self.predicted_class else " "
            lines.append(f"{i}. [{match_str}] Index {neighbor.index}: class {neighbor_class}, "
                        f"distance {neighbor.distance:.4f}")
            
            # Add metadata if available
            if neighbor.metadata:
                for key, value in neighbor.metadata.items():
                    lines.append(f"      {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
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
            "neighbors": [
                {
                    "index": n.index,
                    "distance": float(n.distance),
                    "label": n.label,
                    "label_name": self.class_names.get(n.label, str(n.label)),
                    "features": n.features.tolist(),
                    "metadata": n.metadata
                }
                for n in self.neighbors
            ],
            "feature_names": self.feature_names
        }
    
    def plot(
        self,
        plot_type: str = 'radar',
        highlight_differences: bool = True,
        show_distances: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Visualize the explanation.
        
        Args:
            plot_type: 'radar', 'bar', or 'parallel'
            highlight_differences: Whether to highlight feature differences
            show_distances: Whether to show distance values
            save_path: Path to save figure (if provided)
            figsize: Figure size
        """
        if plot_type == 'bar':
            self._plot_bar(figsize, save_path)
        elif plot_type == 'radar':
            self._plot_radar(figsize, save_path)
        elif plot_type == 'parallel':
            self._plot_parallel(figsize, save_path)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_bar(self, figsize: Tuple[int, int], save_path: Optional[str]) -> None:
        """Create bar plot comparing features."""
        n_features = len(self.test_sample)
        n_neighbors = len(self.neighbors)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_features)
        width = 0.8 / (n_neighbors + 1)
        
        # Plot test sample
        ax.bar(x, self.test_sample, width, label='Test Sample', 
               color='red', alpha=0.8, edgecolor='black', linewidth=2)
        
        # Plot neighbors
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, n_neighbors))
        for i, neighbor in enumerate(self.neighbors):
            offset = width * (i + 1)
            match_str = "✓" if neighbor.label == self.predicted_class else ""
            ax.bar(x + offset, neighbor.features, width,
                   label=f'Neighbor {i+1} {match_str}', 
                   color=colors[i], alpha=0.6)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Feature Values')
        ax.set_title(f'Case-Based Explanation (Correspondence: {self.correspondence:.2%})')
        ax.set_xticks(x + width * n_neighbors / 2)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_radar(self, figsize: Tuple[int, int], save_path: Optional[str]) -> None:
        """Create radar plot (not implemented yet - use bar for now)."""
        print("Radar plot not yet implemented, using bar plot instead.")
        self._plot_bar(figsize, save_path)
    
    def _plot_parallel(self, figsize: Tuple[int, int], save_path: Optional[str]) -> None:
        """Create parallel coordinates plot (not implemented yet - use bar for now)."""
        print("Parallel coordinates plot not yet implemented, using bar plot instead.")
        self._plot_bar(figsize, save_path)
    
    def __repr__(self) -> str:
        return (f"Explanation(test_index={self.test_index}, "
                f"predicted={self.get_predicted_class_name()}, "
                f"correspondence={self.correspondence:.2%}, "
                f"neighbors={len(self.neighbors)})")
