# Case-Explainer: General-Purpose Case-Based Explainability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Provides model-agnostic explanations through training set precedent and nearest neighbor correspondence.

## What is Case-Based Explainability?

While some explainability methods provide feature importance scores, case-based explainability answers: **"Why was this prediction made?"** by showing similar training examples.

Instead of: *"Feature X has importance 0.45"*  
You get: *"This sample is classified as X because it resembles these 5 training examples"*

## Features

- **Model-agnostic**: Works with any classifier (sklearn, XGBoost, neural networks, etc.)
- **Correspondence metric**: Quantifies agreement between prediction and neighbors
- **Multiple indexing strategies**: K-D Tree, Ball Tree, or brute force
- **Automatic scaling**: Optional feature standardization
- **Metadata tracking**: Attach provenance data to training samples
- **Sklearn-compatible API**: Familiar interface for ML practitioners
- **Batch explanations**: Explain multiple predictions efficiently

## Installation

```bash
# From source (current development version)
cd case-explainer
pip install -e .

# Dependencies
pip install numpy scipy scikit-learn matplotlib pandas
```

## Quick Start

```python
from case_explainer import CaseExplainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Create explainer
explainer = CaseExplainer(
    X_train=X_train,
    y_train=y_train,
    feature_names=['sepal_len', 'sepal_width', 'petal_len', 'petal_width'],
    algorithm='auto',           # 'auto', 'kd_tree', 'ball_tree', 'brute'
    distance_penalty='fixed'    # 'fixed', 'adaptive', 'linear', 'square', float
)

# Explain a prediction
explanation = explainer.explain_instance(X_test[0], k=5, model=clf)
print(f"Correspondence: {explanation.correspondence:.2%}")
print(explanation.summary())
```

## Core Concepts

### Correspondence Metric

Quantifies agreement between prediction and retrieved neighbors using distance-weighted voting:

```
w(c) = Σ[penalty_function(distance)] for neighbors with class c
Correspondence = w(predicted_class) / Σ w(all_classes)
```

**Distance Penalty Strategies:**

The distance penalty function controls how neighbor distance affects voting weight. Choose based on data dimensionality:

- **`'fixed'` (default)**: Cubic penalty `1/(d+1)³` - Original formula, works well for low-dimensional data (≤5 features)
- **`'adaptive'` (recommended)**: Auto-adjusts exponent based on dimensionality:
  - Low-D (≤5): cubic (³) - sharp discrimination
  - Medium-D (6-15): square (²) - moderate penalty
  - High-D (16-30): moderate (^1.5) - gentler penalty
  - Very high-D (31-50): linear (¹) - gentle penalty
  - Extreme-D (>50): very gentle (^0.5)
- **`'linear'`**: Linear penalty `1/(d+1)` - Recommended for high-dimensional data (>20 features)
- **`'square'`**: Square penalty `1/(d+1)²` - Balance between cubic and linear
- **`'percentile'`**: Percentile-normalized weights within k-neighborhood (scale-invariant)
- **`float`**: Custom exponent, e.g., `2.5` for `1/(d+1)^2.5`
- **`callable`**: Custom function `distance -> weight` for full control

**Why adaptive penalties?** In high-dimensional spaces (curse of dimensionality), all distances become large. Gentler penalties maintain discrimination by preventing the nearest neighbor from dominating.

**Research precedent:**
- Aggarwal et al. (2001) "On the Surprising Behavior of Distance Metrics in High Dimensional Space"
- Kernel methods with adaptive bandwidth
- Local Outlier Factor density weighting

**Interpretation:**
- **High (≥85%)**: Strong agreement with training precedent
- **Medium (70-85%)**: Moderate agreement
- **Low (<70%)**: Weak agreement, prediction may be uncertain

### Indexing Strategies

- **`auto`**: Let sklearn choose based on data characteristics ⚡ (recommended)
- **`kd_tree`**: Fast for low-dimensional data (<20 features)
- **`ball_tree`**: Better for high-dimensional data (20-50 features)
- **`brute`**: Exact search, good for small datasets (<10k samples)

**Note:** The algorithm parameter specifies the k-NN indexing structure, while distance_penalty controls how neighbor distances affect voting weights. Both impact performance on high-dimensional data.

## Examples

See `quickstart.py` for a complete working example:

```bash
python quickstart.py
```

## Security & Privacy Considerations

**IMPORTANT:** Case-based explanations expose actual training samples as evidence. This can leak sensitive information:

- **Medical domains:** Patient records, diagnoses, treatments
- **Financial domains:** Account details, transaction patterns
- **Security domains:** Attack signatures, system vulnerabilities
- **Personal data:** User behavior, preferences, demographics

**Before using in production with sensitive data:**
1. Implement feature masking for sensitive columns
2. Consider differential privacy mechanisms
3. Apply anonymization to metadata
4. Set up access control and audit logging
5. Review legal/regulatory requirements (GDPR, HIPAA, etc.)

**Privacy protection features are planned for Phase 2.** For now, use only with non-sensitive data or in controlled research environments.

Unlike LIME/SHAP which only show feature importance, case-explainer exposes training sample features. Evaluate whether this trade-off is acceptable for your use case.

---

## API Overview

### CaseExplainer

```python
explainer = CaseExplainer(
    X_train,                # Training features
    y_train,                # Training labels
    k=5,                    # Number of neighbors (default: 5)
    feature_names=None,     # Optional feature names
    class_names=None,       # Optional class names {0: 'cat', 1: 'dog'}
    metric='euclidean',     # Distance metric
    algorithm='auto',       # 'auto', 'kd_tree', 'ball_tree', 'brute'
    scale_data=True,        # Standardize features (recommended)
    class_weights=None,     # Optional class weights {0: 1.0, 1: 2.0}
    distance_penalty='fixed',  # Penalty strategy (see below)
    metadata=None,          # Optional provenance data
    n_jobs=-1               # Parallel jobs (-1 = all CPUs)
)
```

**Distance Penalty Strategies:**

```python
# Default: Fixed cubic penalty (original formula)
explainer = CaseExplainer(X_train, y_train, distance_penalty='fixed')

# Recommended: Adaptive (auto-adjusts for dimensionality)
explainer = CaseExplainer(X_train, y_train, distance_penalty='adaptive')

# High-dimensional data: Linear penalty
explainer = CaseExplainer(X_train, y_train, distance_penalty='linear')

# Custom exponent: 1/(d+1)^2.5
explainer = CaseExplainer(X_train, y_train, distance_penalty=2.5)

# Custom function
def my_penalty(distance):
    return np.exp(-distance / 10.0)
explainer = CaseExplainer(X_train, y_train, distance_penalty=my_penalty)
```

**Choosing a penalty strategy:**
- Low-D (≤5 features): `'fixed'` or `'cubic'`
- Medium-D (6-20 features): `'adaptive'` or `'square'`
- High-D (21-50 features): `'adaptive'` or `'linear'`
- Extreme-D (>50 features): `'adaptive'` (auto-adjusts)
- Unknown/mixed: `'adaptive'` (safe default)

### Explain Single Instance

```python
explanation = explainer.explain_instance(
    test_sample,            # Sample to explain
    k=5,                    # Number of neighbors
    model=clf,              # Trained classifier
    true_class=None,        # Optional true label
    distance_weighted=True  # Use distance weighting
)
```

### Explain Batch

```python
explanations = explainer.explain_batch(
    X_test,                 # Test samples
    k=5,                    # Number of neighbors
    y_test=None,            # Optional true labels
    model=clf               # Trained classifier
)
```

### Explanation Object

```python
explanation.correspondence          # Correspondence score [0, 1]
explanation.correspondence_interpretation  # 'high', 'medium', 'low'
explanation.neighbors               # List of Neighbor objects
explanation.predicted_class         # Predicted class
explanation.is_correct()            # True if prediction matches label
explanation.summary()               # Text summary
explanation.to_dict()               # Export as dictionary
explanation.plot()                  # Visualize (bar plot)
```

## Validated Domains

**Hardware Trojan Detection** (56,959 samples, 5 features, low-dimensional)
- 98.46% correspondence with `distance_penalty='fixed'` (cubic)
- 77.8% gap between correct/incorrect predictions
- Original JETTA paper: 97.4% correspondence

**Breast Cancer Diagnosis** (569 samples, 30 features, high-dimensional)
- 91.70% correspondence with `distance_penalty='fixed'`
- 53.8% gap between correct/incorrect predictions
- All penalty strategies perform within 1% (50-51% gap)

**Credit Card Fraud Detection** (284,807 samples, 30 features, extreme imbalance)
- 98.72% correspondence with `distance_penalty='fixed'` and class_weights={0: 1.0, 1: 107.6}
- 25.8% gap between correct/incorrect predictions
- Slightly better performance with `distance_penalty='linear'` or `'adaptive'` (30.4% gap)
- Demonstrates handling of extreme class imbalance (579:1 ratio)

**Key Findings:**
- Low-dimensional data (≤5 features): Fixed cubic penalty optimal
- High-dimensional data (>20 features): Adaptive or linear penalties improve discrimination
- Extreme imbalance: Class weighting essential, gentler penalties help
- All strategies maintain strong correspondence (>90%) across domains

## Comparison to LIME/SHAP

| Feature | Case-Explainer | LIME | SHAP |
|---------|---------------|------|------|
| **Explanation Type** | Training precedents | Feature importance | Shapley values |
| **Intuition** | "Similar to examples X, Y, Z" | "Feature A is important" | "Feature A contributes +0.3" |
| **Speed** | Very Fast ⚡ | Fast | Slow |
| **Domain Expert Friendly** | Yes (concrete examples) | Moderate | No (math heavy) |
| **Model-Agnostic** | Yes | Yes | Yes |

**When to use Case-Explainer:**
- Domain experts need to verify predictions against known cases
- Precedent-based reasoning is valued (medical, legal, security)
- Fast explanations needed for real-time systems
- Training data has provenance/metadata worth surfacing

## Development Status

### Core Functionality MVP
- [x] CaseExplainer class with sklearn-compatible API
- [x] Correspondence metric with distance weighting
- [x] Multiple indexing strategies (K-D tree, Ball tree, brute force)
- [x] Explanation object with summary and visualization
- [x] Metadata/provenance tracking
- [x] Batch explanation support

### Phase 1: Multi-Domain Validation - COMPLETE ✓
- [x] Hardware trojan detection (validated in JETTA paper, 98.46% correspondence)
- [x] Medical diagnosis (UCI Breast Cancer, 91.70% correspondence)
- [x] Fraud detection (Credit Card Fraud, 98.72% correspondence, extreme imbalance)
- [x] Distance penalty parameterization (fixed, adaptive, linear, square, percentile, custom)
- [x] Benchmarking (time, memory, correspondence across strategies)

### Phase 2: Documentation & Polish - PLANNED
- [ ] API reference (Sphinx)
- [ ] Tutorial notebooks (4 domains)
- [ ] Comparison guide (vs LIME/SHAP)
- [ ] Code coverage >90%

### Phase 4: Release & Distribution - PLANNED
- [ ] PyPI package
- [ ] GitHub Pages documentation
- [ ] CI/CD pipeline
- [ ] Zenodo DOI

## Citation

If you use this module in academic work, please cite:

```bibtex
@software{case_explainer2025,
  author = {Whitten, Paul and Wolff, Francis and Papachristou, Chris},
  title = {Case-Explainer: General-Purpose Case-Based Explainability},
  year = {2025},
  url = {https://github.com/paulwhitten/case-explainer}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! This is currently in active development (Phase 1).

**Priority areas:**
- Additional distance metrics (Manhattan, Cosine, Mahalanobis)
- Approximate nearest neighbors (Annoy, FAISS) for large-scale data (>1M samples)
- Learned metric spaces (Siamese networks, triplet loss)
- Intrinsic dimensionality estimation for automatic penalty selection
- Radar and parallel coordinate visualizations
- More comprehensive unit tests
- Privacy-preserving mechanisms (differential privacy, feature masking)

## Contact

Questions? Issues? Open a GitHub issue or contact pcw@case.edu.

## Acknowledgments

- Inspired by Caruana et al. (1999) "Case-based explanation of non-case-based learning"
- Validated on hardware trojan detection research
- Built with scikit-learn, scipy, and matplotlib
