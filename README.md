# Case-Explainer: General-Purpose Case-Based Explainability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/case-explainer.svg)](https://pypi.org/project/case-explainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://paulwhitten.github.io/case-explainer/)
[![CI](https://github.com/paulwhitten/case-explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/paulwhitten/case-explainer/actions/workflows/ci.yml)

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
- **Model-informed similarity**: Retrieve cases from a neural network's learned representation or a decision tree's matching leaf

## Installation

```bash
pip install case-explainer
```

Or, to install the latest development version from source:

```bash
git clone https://github.com/paulwhitten/case-explainer.git
cd case-explainer
pip install -e .

# With development/test dependencies (pytest, pytest-cov, etc.)
pip install -e ".[dev]"
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
    algorithm='kd_tree'
)

# Explain a prediction
explanation = explainer.explain_instance(X_test[0], k=5, model=clf)
print(f"Correspondence: {explanation.correspondence:.2%}")
print(explanation.summary())
```

## Core Concepts

### Traditional Feature-Space Precedent

By default, Case-Explainer retrieves training cases that are nearest to the
query in the original feature space. Numeric features are standardized by
default so a feature with a large numerical range does not dominate Euclidean
distance. Set `scale_data=False` only when the inputs are already on a
meaningful common scale or when custom scaling has been applied.

```python
feature_explainer = CaseExplainer(
    X_train,
    y_train,
    feature_names=feature_names,
    scale_data=True,
)

explanation = feature_explainer.explain_instance(
    X_test[0],
    model=classifier,
)
```

This model-agnostic explanation answers: "Which observed training samples have
the most similar measured attributes?" It is useful when input features have a
domain meaning that a reviewer can inspect directly. Feature-space proximity
does not imply that the model used those features in the same way, and it can
be misleading when irrelevant, redundant, or high-dimensional inputs dominate
the distance.

The returned neighbors retain their original feature values, labels, indexes,
and optional metadata. Correspondence then measures how strongly their labels
agree with the model prediction; it does not measure feature importance.

### Correspondence Metric

Quantifies agreement between prediction and retrieved neighbors using inverse-cubed distance weighting:

```text
w(c) = sum[ 1 / (distance + 1)^3 ] for neighbors with class c
Correspondence = w(predicted_class) / sum( w(all_classes) )
```

The `+1` offset in the denominator prevents division by zero when a test sample is identical to a training sample (distance = 0). In that case the weight is simply `1 / 1 = 1`.

**Example Interpretation Thresholds** (domain-dependent, not universal standards):

- **High (≥85%)**: Strong agreement with training precedent
- **Medium (70-85%)**: Moderate agreement
- **Low (<70%)**: Weak agreement, prediction may be uncertain

*Note: These thresholds are illustrative examples. Appropriate thresholds should be determined empirically for each specific domain and use case based on validation studies.*

### Indexing Strategies

- **`kd_tree`**: Fast for low-dimensional data (<20 features)
- **`ball_tree`**: Better for high-dimensional data
- **`brute`**: Exact search for small datasets (<10k samples)

### Activation-Based Similarity (Neural Networks)

For sklearn `MLPClassifier` models, you can retrieve neighbors by hidden-layer
activations instead of raw input features. The resulting cases show consequences
of the model's learned representation. They do not expose the model's complete
internal reasoning.

```python
from sklearn.neural_network import MLPClassifier
from case_explainer import CaseExplainer, HiddenActivationRetrieval

mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), random_state=42)
mlp.fit(X_train, y_train)

# Last hidden layer, based on Caruana et al. (1999)
explainer = CaseExplainer(
    X_train, y_train,
    retrieval=HiddenActivationRetrieval(model=mlp),
)

# All hidden layers with position weighting, a library extension
explainer_deep = CaseExplainer(
    X_train, y_train,
    retrieval=HiddenActivationRetrieval(model=mlp, layer="all_hidden"),
)

# Hybrid: blend features (30%) and activations (70%)
explainer_hybrid = CaseExplainer(
    X_train, y_train,
    retrieval=HiddenActivationRetrieval(model=mlp, blend_alpha=0.3),
)

# Multiclass: weight units using the predicted class's output connections
explainer_class_weighted = CaseExplainer(
    X_train, y_train,
    retrieval=HiddenActivationRetrieval(
        model=mlp,
        output_weighting="predicted_class",
    ),
)

# The construction model is reused for prediction.
explanation = explainer.explain_instance(X_test[0])
```

If the MLP was trained on transformed inputs, pass its fitted transformer as
`input_transform`. The explainer applies it both when extracting activations
and when asking the model for a prediction.

The legacy `activation_extractor`, `activation_layer`, `use_output_weights`,
and `blend_alpha` constructor path remains supported through version 0.2 and
emits `DeprecationWarning`. It will be removed no earlier than version 0.3.

For sklearn random forests, use shared-leaf proximity rather than treating
leaf identifiers as numeric coordinates:

```python
from case_explainer import ForestProximityRetrieval

forest_explainer = CaseExplainer(
    X_train, y_train,
    retrieval=ForestProximityRetrieval(model=forest),
)
```

See [notebooks/02_breast_cancer_tutorial.ipynb](notebooks/02_breast_cancer_tutorial.ipynb) for a full worked comparison.

### Choosing a Retrieval Space

| Retrieval space | Meaning of a similar case | Best suited to | Main limitation |
| --- | --- | --- | --- |
| Features (default) | Nearby observed input attributes | Model-agnostic review and domain-readable measurements | May not reflect the model's learned notion of similarity |
| Hidden activations | Nearby learned neural-network representations | Inspecting precedents that the MLP represents similarly | Model-specific and less directly interpretable |
| Feature/activation hybrid | Nearby under a weighted combination of both spaces | Balancing domain similarity with model representation | The blend weight is an analyst choice that requires validation |
| Decision-tree leaf | Cases following the same tree path to a leaf | Exact precedent within a fitted decision tree | A leaf may contain fewer than `k` retained cases |
| Random-forest proximity | Cases sharing leaves across many trees | Model-informed precedent for fitted random forests | Requires comparison with the retained case base at query time |

All modes retrieve actual training cases. The retrieval space changes what
"like samples" means; the explanation and correspondence interfaces remain the
same. Use feature retrieval when observed attributes define the comparison you
want to defend. Use model-informed retrieval when the model's internal
partitioning or representation is the relevant basis for precedent.

## Examples

See `quickstart.py` for a complete working example:

```bash
python quickstart.py
```

### Tutorial Notebooks

Interactive Jupyter notebooks for each validated domain:

- [Iris Classification](notebooks/01_iris_tutorial.ipynb) - Introductory multi-class example
- [Breast Cancer Diagnosis](notebooks/02_breast_cancer_tutorial.ipynb) - Medical diagnosis domain
- [Fraud Detection](notebooks/03_fraud_detection_tutorial.ipynb) - Financial security with extreme class imbalance
- [Hardware Trojan Detection](notebooks/04_hardware_trojan_tutorial.ipynb) - Large-scale security domain

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

Privacy-preserving features are not yet available. Use only with non-sensitive data or in controlled research environments.

Unlike LIME/SHAP which only show feature importance, case-explainer exposes training sample features. Evaluate whether this trade-off is acceptable for your use case.

---

## API Overview

### CaseExplainer

```python
explainer = CaseExplainer(
    X_train,                    # Training features
    y_train,                    # Training labels
    feature_names=None,         # Optional feature names
    class_names=None,           # Optional class names {0: 'cat', 1: 'dog'}
    algorithm='kd_tree',        # Indexing strategy
    scale_data=True,            # Standardize features
    metadata=None,              # Optional provenance data
    # Explicit strategy object for hidden activations, tree leaves, or forests
    retrieval=None,
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

- 99.3% average correspondence across indexing methods
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

### Comparison with LIME and SHAP

| Aspect | Case-Explainer | LIME | SHAP |
| -------- | --------------- | ------ | ------ |
| Explanation type | Training precedents (similar cases) | Local surrogate model (feature importance) | Shapley values (feature importance) |
| Output | k nearest neighbors + correspondence score | Per-feature importance for one prediction | Per-feature importance (local and global) |
| Privacy risk | High -- exposes actual training samples | Low -- uses synthetic perturbations | Low -- no sample exposure |
| Speed (pipeline, HW trojan) | ~13 ms/sample | ~25 ms/sample | ~1 ms/sample (TreeSHAP) |
| Model-agnostic | Yes | Yes | Yes (KernelSHAP); tree-specific variants are faster |
| Best for | Precedent-based reasoning, domain expert verification | Local feature contributions, model debugging | Global + local feature analysis, theoretical guarantees |

*Timing from the hardware trojan detection pipeline (XGBoost classifier, 5 features, ~57k samples). SHAP uses TreeSHAP which exploits tree structure for speed; KernelSHAP (model-agnostic) is substantially slower. LIME and Case-Explainer speeds are model-agnostic. Results will vary with dataset size, dimensionality, and hardware.*

## Limitations

### Privacy and Security

- Exposes actual training samples, which may contain sensitive information
- Not suitable for sensitive data without additional privacy protection mechanisms
- Privacy-preserving features are planned for future releases

### Correspondence Metric Limitations

- Measures neighbor agreement, not prediction correctness or quality
- High correspondence can occur with incorrect predictions if training data contains systematic errors
- Thresholds for "high/medium/low" must be validated per domain

### Performance Benchmark Limitations

- Timing and memory results are from single runs on reference hardware
- No statistical error bars or confidence intervals provided
- Results may vary significantly on different hardware and with different parameters

### Scalability Limitations

- Memory usage scales linearly with training set size
- Very large datasets (>1M samples) may require approximate nearest neighbor methods (not yet implemented)

### Interpretability Limitations

- Assumes users can meaningfully interpret feature values of retrieved neighbors
- Multi-feature patterns may be difficult to assess without domain expertise
- High-dimensional data may require dimensionality reduction for effective interpretation

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

Contributions welcome! Core functionality and release infrastructure are complete.

**Priority areas:**

- Additional distance metrics (Manhattan, Cosine)
- Approximate nearest neighbors (Annoy, FAISS) for large-scale data
- Radar and parallel coordinate visualizations
- More comprehensive unit tests

## Contact

Questions? Issues? Open a GitHub issue or contact <pcw@case.edu>.

## Acknowledgments

- Inspired by Caruana et al. (1999) "Case-based explanation of non-case-based learning"
- Validated on hardware trojan detection research
- Built with scikit-learn, scipy, and matplotlib
