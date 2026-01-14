# Case-Explainer: General-Purpose Case-Based Explainability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://paulwhitten.github.io/case-explainer/)

Provides model-agnostic explanations through training set precedent and nearest neighbor correspondence.

**[Read the full documentation](https://paulwhitten.github.io/case-explainer/)**

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
    index_method='kd_tree'
)

# Explain a prediction
explanation = explainer.explain_instance(X_test[0], k=5, model=clf)
print(f"Correspondence: {explanation.correspondence:.2%}")
print(explanation.summary())
```

## Core Concepts

### Correspondence Metric

Quantifies agreement between prediction and retrieved neighbors using inverse squared distance weighting:

```
w(c) = Σ[1/(distance² + 1)] for neighbors with class c
Correspondence = w(predicted_class) / Σ w(all_classes)
```

**Example Interpretation Thresholds** (domain-dependent, not universal standards):
- **High (≥85%)**: Strong agreement with training precedent
- **Medium (70-85%)**: Moderate agreement
- **Low (<70%)**: Weak agreement, prediction may be uncertain

*Note: These thresholds are illustrative examples. Appropriate thresholds should be determined empirically for each specific domain and use case based on validation studies.*

### Indexing Strategies

- **`kd_tree`**: Fast for low-dimensional data (<20 features)
- **`ball_tree`**: Better for high-dimensional data
- **`brute`**: Exact search for small datasets (<10k samples)

## Examples

See `quickstart.py` for a complete working example:

```bash
python quickstart.py
```

### Benchmarking

Comprehensive performance benchmarks across multiple datasets:

```bash
python benchmark.py              # Full benchmark including MNIST
python benchmark.py --no-mnist   # Skip MNIST (faster)
python benchmark.py --help       # See all options
```

Results (single run on reference hardware):
- **Speed**: 14-37 ms per explanation depending on dataset size
- **Memory**: <1 MB to 131 MB (scales with data size and dimensionality)
- **Correspondence**: 87-100% neighbor agreement across validated domains
- **Scalability**: Tested up to 200k training samples

**Note on Correspondence**: This metric measures agreement between predictions and retrieved neighbors, not prediction accuracy or quality. High correspondence indicates consistency with training data patterns, not necessarily correct predictions.

### Documentation

**[View full API documentation online](https://paulwhitten.github.io/case-explainer/)**

Build and view documentation locally:

```bash
# Build documentation
cd docs
make html

# View documentation locally
python3 -m http.server 8000 --directory docs/_build/html
# Then open http://localhost:8000 in your browser
```

The documentation includes:
- Complete API reference for all classes and functions
- Usage examples and code snippets
- Theory and mathematical foundations
- Configuration guides and best practices

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
    feature_names=None,     # Optional feature names
    class_names=None,       # Optional class names {0: 'cat', 1: 'dog'}
    index_method='kd_tree', # Indexing strategy
    scale_data=True,        # Standardize features
    metadata=None           # Optional provenance data
)
```

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

**Hardware Trojan Detection** (56,959 samples, 5 features)
- 99.9% average correspondence across indexing methods
- High neighbor agreement on imbalanced security data
- 25.7 ms/sample explanation time (single run, reference hardware)

**Credit Card Fraud Detection** (284,807 samples, 30 features)
- 100% average correspondence (complete agreement with retrieved neighbors)
- Highly imbalanced dataset (268:1 normal:fraud ratio)
- 36.4 ms/sample explanation time (single run, reference hardware)

**Medical Diagnosis - Breast Cancer** (569 samples, 30 features)
- 93.3% average correspondence
- Correct predictions: 96.2% correspondence vs 47.3% for incorrect predictions
- 25.9 ms/sample explanation time (single run, reference hardware)

**Also Validated On:**
- Iris (92.7%), Wine (91.8%), Digits (94.9%), MNIST (87.5%)
- See `benchmark.py` for full results across 7 datasets

*Note: Correspondence measures neighbor agreement, not prediction quality. High correspondence with incorrect predictions indicates the model has learned incorrect patterns in the training data.*

## When to Use Case-Based Explainability

**Case-Explainer is well-suited for scenarios where:**
- Domain experts need to verify predictions against known training cases
- Precedent-based reasoning is valued (medical diagnosis, legal decisions, security analysis)
- Concrete examples are more intuitive than feature importance scores
- Training data has provenance or metadata worth surfacing to users
- Fast explanation generation is needed for real-time or interactive systems

**Alternative approaches (LIME, SHAP) may be preferable when:**
- Feature contributions are more relevant than training precedents
- Training data cannot be exposed due to privacy/security constraints
- Model debugging requires understanding feature-level behavior

*Note: No formal benchmarking comparison between case-explainer and LIME/SHAP has been conducted. The choice of explainability method should be based on domain requirements and use case constraints.*

## Limitations

**Privacy and Security**
- Exposes actual training samples, which may contain sensitive information
- Not suitable for sensitive data without additional privacy protection mechanisms
- Privacy-preserving features are planned for future releases

**Correspondence Metric**
- Measures neighbor agreement, not prediction correctness or quality
- High correspondence can occur with incorrect predictions if training data contains systematic errors
- Thresholds for "high/medium/low" must be validated per domain

**Performance Benchmarks**
- Timing and memory results are from single runs on reference hardware
- No statistical error bars or confidence intervals provided
- Results may vary significantly on different hardware and with different parameters

**Scalability**
- Memory usage scales linearly with training set size
- Very large datasets (>1M samples) may require approximate nearest neighbor methods (not yet implemented)

**Interpretability**
- Assumes users can meaningfully interpret feature values of retrieved neighbors
- Multi-feature patterns may be difficult to assess without domain expertise
- High-dimensional data may require dimensionality reduction for effective interpretation

## Development Status

### Core Functionality MVP
- [x] CaseExplainer class with sklearn-compatible API
- [x] Correspondence metric with distance weighting
- [x] Multiple indexing strategies (K-D tree, Ball tree, brute force)
- [x] Explanation object with summary and visualization
- [x] Metadata/provenance tracking
- [x] Batch explanation support

### Phase 1: Multi-Domain Validation
- [x] Hardware trojan detection (validated in JETTA paper)
- [x] Medical diagnosis (UCI Breast Cancer)
- [x] Fraud detection (Credit Card Fraud)
- [x] Benchmarking (time, memory, correspondence)

### Phase 2: Documentation - IN PROGRESS
- [x] API reference
- [x] Tutorial notebooks (4 domains)
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
- Additional distance metrics (Manhattan, Cosine)
- Approximate nearest neighbors (Annoy, FAISS) for large-scale data
- Radar and parallel coordinate visualizations
- More comprehensive unit tests

## Contact

Questions? Issues? Open a GitHub issue or contact pcw@case.edu.

## Acknowledgments

- Inspired by Caruana et al. (1999) "Case-based explanation of non-case-based learning"
- Validated on hardware trojan detection research
- Built with scikit-learn, scipy, and matplotlib
