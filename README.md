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
- **Pluggable k-NN backend**: sklearn's `auto`, `kd_tree`, `ball_tree`, or `brute` via `algorithm=`
- **Automatic scaling**: Optional feature standardization
- **Metadata tracking**: Attach provenance data to training samples
- **Sklearn-compatible API**: Familiar interface for ML practitioners
- **Batch explanations**: Vectorized `explain_batch` with one model call and one k-NN query per batch
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

### k-NN Backend

The `algorithm=` argument is passed to sklearn's `NearestNeighbors`:

- **`auto`** (default): Let sklearn choose based on the data
- **`kd_tree`**: Fast for low-dimensional data (<20 features)
- **`ball_tree`**: Better for high-dimensional data
- **`brute`**: Exact search for small datasets (<5k samples)

### Activation-Based Similarity (Neural Networks)

For sklearn `MLPClassifier` models, you can retrieve neighbors by hidden-layer
activations instead of raw input features. The resulting cases show consequences
of the model's learned representation. They do not expose the model's complete
internal reasoning.

```python
from sklearn.neural_network import MLPClassifier
from case_explainer import CaseExplainer, HiddenActivations, Blend

mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), random_state=42)
mlp.fit(X_train, y_train)

# Last hidden layer, based on Caruana et al. (1999)
explainer = CaseExplainer(
    X_train, y_train,
    similarity=HiddenActivations(model=mlp),
)

# All hidden layers with position weighting, a library extension
explainer_deep = CaseExplainer(
    X_train, y_train,
    similarity=HiddenActivations(model=mlp, layer="all_hidden"),
)

# Hybrid: blend features (30%) and activations (70%)
explainer_hybrid = CaseExplainer(
    X_train, y_train,
    similarity=Blend(HiddenActivations(model=mlp), features=0.3),
)

# Multiclass: weight units using the predicted class's output connections
explainer_class_weighted = CaseExplainer(
    X_train, y_train,
    similarity=HiddenActivations(
        model=mlp,
        output_weighting="predicted_class",
    ),
)

# The construction model is reused for prediction.
explanation = explainer.explain_instance(X_test[0])
```

Binding the model to `Features` lets the input-space and activation-space halves
of the comparison share one call signature:

```python
from case_explainer import Features

input_space = CaseExplainer(X_train, y_train, similarity=Features(model=mlp))
activation_space = CaseExplainer(X_train, y_train, similarity=HiddenActivations(model=mlp))

for space in (input_space, activation_space):
    space.explain_instance(X_test[0])   # identical call, no per-strategy kwargs
```

If the MLP was trained on transformed inputs, pass its fitted transformer as
`input_transform`. The explainer applies it both when extracting activations
and when asking the model for a prediction.

Version 0.3.0 removed the 0.2.x `activation_extractor`, `activation_layer`,
`use_output_weights`, `blend_alpha`, and `retrieval=` constructor parameters.
Passing one raises `TypeError` with the equivalent `similarity=` strategy; see
the [migration guide](https://paulwhitten.github.io/case-explainer/migration.html).

For sklearn random forests, use shared-leaf proximity rather than treating
leaf identifiers as numeric coordinates:

```python
from case_explainer import ForestProximity

forest_explainer = CaseExplainer(
    X_train, y_train,
    similarity=ForestProximity(model=forest),
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

Every explanation records where its neighbors came from:
`explanation.retrieval_space` (for example `"features"`, `"activations"`,
`"hybrid(features=0.30)"`, `"tree_leaf"`, `"forest_proximity"`),
`explanation.distance_metric`, and `explanation.distance_scale`. They appear in
`summary()` and `to_dict()`, so the meaning of a neighbor's distance travels
with the explanation.

Neighbor distances are reported in the retrieval space that produced them, so
their scale differs between modes and between datasets: feature and
hidden-activation distances can differ by a factor of two or more in either
direction, and forest proximity distances lie in `[0, 1]`. Because correspondence weights neighbors by
`1 / (distance + 1)^3`, raw correspondence values are comparable within one
retrieval space but not across spaces. To compare across spaces, pass
`distance_normalization="median_nn"`: every distance is divided by the median
distance from a training case to its nearest distinct training case in that
space (stored as `explainer.distance_scale_`), so `1.0` means "as far as a
typical training case is from its closest distinct neighbor". Neighbor
selection is unchanged.

The `metric=` constructor argument applies to feature, activation, blended, and
tree-leaf retrieval (tree-leaf uses it to order same-leaf cases in standardized
feature space). Forest proximity defines its own distance and rejects any other
`metric`.

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

- **Speed**: 14-37 ms for a single `explain_instance` call depending on dataset size; `explain_batch` amortizes this to well under 1 ms per sample in feature space
- **Memory**: <1 MB to 131 MB (scales with data size and dimensionality)
- **Correspondence**: 87-100% mean neighbor agreement across validated domains (the 100% is credit-card fraud measured on rows with no fraud; see [Validated Domains](#validated-domains))
- **Scalability**: Tested up to 200k training samples

**Note on Correspondence**: This metric measures agreement between predictions and retrieved neighbors, not prediction accuracy or quality. High correspondence indicates consistency with training data patterns, not necessarily correct predictions.

### Documentation

**[View full API documentation online](https://paulwhitten.github.io/case-explainer/)**

Build and view documentation locally:

```bash
# Build documentation
cd docs
make html

# View documentation locally (still inside docs/)
python3 -m http.server 8000 --directory _build/html
# Then open http://localhost:8000 in your browser
```

The documentation includes:

- Complete API reference for all classes and functions
- Usage examples and code snippets
- Theory and mathematical foundations
- Configuration guides and best practices
- A guide to interpreting and calibrating correspondence, with per-label reliability figures
- A data exposure and privacy page describing what explanations and saved explainers disclose

## Security & Privacy Considerations

**Case-explainer has no privacy-preserving features, and none are planned.**
An explanation is a set of real training records. Showing them is the method,
so the library cannot hide them and still explain the prediction.

**What an explanation discloses.** Each retrieved neighbor carries:

- the complete, unscaled feature values of one training row
- that row's position in the training set, its label, and its exact distance from the query
- every metadata field supplied for that row, unless `return_provenance=False` is passed

`to_dict()` exports all of it. `summary()` prints each neighbor's row position,
label, distance, and metadata, and `plot()` draws the neighbors' feature values.

**Treat the following as training data:**

- **Explanations.** Anyone who can read an explanation can read the training
  rows in it. A neighbor whose feature values equal the query confirms that
  the queried record is in the training set.
- **Saved explainers.** A pickled `CaseExplainer` contains the whole training
  set: raw features, labels, and metadata. Store and share it under the same
  controls as the data itself.
- **Query access.** A service that explains arbitrary inputs lets a caller
  retrieve the training rows nearest to any point they choose. Do not offer
  one to people who may not see the data.

**Use case-explainer when** the audience for the explanations is already
entitled to see the training records, such as the analysts, clinicians, or
engineers who work with that data. This matters most for patient records,
financial transactions, security telemetry, and personal data. If the training
data cannot be shown to the people who need the explanation, use a method that
does not expose it, such as LIME or SHAP.

**Steps that remain your responsibility:**

1. Leave direct identifiers out of the features and out of `metadata`
2. Pass `return_provenance=False` where metadata should not travel with an explanation
3. Put access control and audit logging around wherever explanations and saved explainers are served or stored
4. Confirm the legal basis for disclosing training records (GDPR, HIPAA, and similar) before deployment

Hiding selected columns in the output would not make explanations private.
Row positions, labels, and near-zero distances still reveal whether a record
is in the training set, which is why the library does not present masking as a
privacy feature.

## API Overview

### CaseExplainer

```python
explainer = CaseExplainer(
    X_train,                    # Training features (array or DataFrame)
    y_train,                    # Training labels
    feature_names=None,         # Optional feature names (default: DataFrame columns)
    class_names=None,           # Optional class names {0: 'cat', 1: 'dog'}
    k=5,                        # Default number of neighbors
    algorithm='auto',           # k-NN algorithm: 'auto', 'kd_tree', 'ball_tree', 'brute'
    metric='euclidean',         # Distance metric (not for ForestProximity)
    scale_data=True,            # Standardize features
    class_weights=None,         # Per-class weights for correspondence: dict or 'balanced'
    metadata=None,              # Optional provenance data
    # Retrieval space: None/'features', or HiddenActivations(...), TreeLeaf(...),
    # ForestProximity(...), Blend(...)
    similarity=None,
    model=None,                 # Prediction model for feature-space retrieval
    correspondence_thresholds=(0.70, 0.85),  # (medium, high) cut points
    distance_normalization=None,             # or "median_nn"
)
```

A DataFrame `X_train` does more than supply feature names. The explainer
indexes a plain array, but it remembers the columns and rebuilds a DataFrame
with them right before every call into your model, your `input_transform`, and
the built-in extractors. Models fitted on a DataFrame therefore keep working:
a pipeline that selects columns by name does not fail on an array, and
scikit-learn does not warn about missing feature names. DataFrame queries are
reordered to the training column order by name, and a frame with missing or
extra columns raises `ValueError`. Array queries are taken positionally. See
the [DataFrame guide](https://paulwhitten.github.io/case-explainer/api/explainer.html#working-with-dataframes).

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
explanation.retrieval_space         # 'features', 'activations', 'tree_leaf', ...
explanation.distance_metric         # e.g. 'euclidean', 'shared_leaf_fraction'
explanation.distance_scale          # 1.0 unless distances were normalized
explanation.neighbors               # List of Neighbor objects
explanation.predicted_class         # Predicted class
explanation.is_correct()            # True if prediction matches label
explanation.summary()               # Text summary
explanation.to_dict()               # Export as dictionary
explanation.plot()                  # Visualize (bar plot)
```

## Validated Domains

Feature-space results from `validate_similarity_modes.py` with the 0.3.0
defaults (k=5, 70/30 stratified split, unweighted correspondence). Every
retrieval space is compared in the table that follows.

**Hardware Trojan Detection** (56,959 samples, 5 features, 269:1 imbalance)

- 99.3% mean correspondence on 2,000 test circuits that include all 63 test trojans
- Correct predictions: 99.7% correspondence vs 83.2% for incorrect predictions (51.5% with `class_weights="balanced"`)
- 0.06 ms/sample batch explanation time (single run, development machine)

**Credit Card Fraud Detection** (284,807 samples, 30 features, 578:1 imbalance)

- 99.3% mean correspondence on 2,000 test transactions that include all 148 test frauds
- Correct predictions: 99.6% correspondence vs 84.3% for incorrect predictions (66.9% with `class_weights="balanced"`)
- 0.15 ms/sample batch explanation time (single run, development machine)
- Releases before 0.3.0 reported 100% here. That figure is reproducible, but
  it was measured on the first 100 test rows, which contain no fraud at all,
  so it said nothing about fraud. The 36 frauds the model misses are where
  correspondence matters; see [Label Calibration](#label-calibration).

**Medical Diagnosis - Breast Cancer** (569 samples, 30 features)

- 92.9% mean correspondence
- Correct predictions: 96.2% correspondence vs 45.6% for incorrect predictions
- 0.11 ms/sample batch explanation time (single run, development machine)

**Also Validated On:**

- Iris (92.7%), Wine (91.8%), Digits (94.9%), MNIST (87.5%)
- See `benchmark.py` for full results across 7 datasets

*Note: Correspondence measures neighbor agreement, not prediction quality. High correspondence with incorrect predictions indicates the model has learned incorrect patterns in the training data.*

### Retrieval Spaces Compared

Produced by `validate_similarity_modes.py` (70/30 stratified split,
`random_state=42`, k=5). Hardware trojan and credit-card fraud explain a
2,000-sample test subset that keeps every minority-class test row (63 trojans,
148 frauds) and fills the rest from the majority class, so the accuracy and
AUC columns are measured on real positives. "Mean corr." is the mean
correspondence over the explained test
samples, raw and with `distance_normalization="median_nn"`. "Correct /
incorrect" is the normalized mean correspondence split by whether the model's
prediction was right. "Corr. AUC" is the ROC AUC of correspondence as a
predictor of a correct prediction, taken from the normalized run. Models are a
100-tree random forest, a depth-5 decision tree, and a (64, 32) MLP; the
tree-leaf rows use `TreeLeaf(overflow="nearest")`. Timings are single-run
batch times on the development machine. Full results, including
nearest-neighbor distances and normalization scales, are in
`benchmarks/similarity_validation.csv`.

| Domain | Retrieval space | Model | Accuracy | Mean corr. (raw) | Mean corr. (normalized) | Correct / incorrect (normalized) | Corr. AUC | ms/sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | features | random_forest | 88.9% | 92.7% | 92.8% | 97.2% / 58.3% | 0.96 | 0.62 |
| iris | activations | mlp | 91.1% | 98.2% | 98.2% | 100.0% / 79.3% | 0.62 | 0.38 |
| iris | activations_predicted_class | mlp | 91.1% | 97.3% | 97.3% | 99.4% / 75.7% | 0.74 | 0.09 |
| iris | hybrid_features_0.3 | mlp | 91.1% | 98.2% | 98.2% | 100.0% / 79.5% | 0.62 | 0.06 |
| iris | tree_leaf | decision_tree | 93.3% | 91.8% | 91.5% | 94.1% / 55.6% | 0.79 | 0.05 |
| iris | forest_proximity | random_forest | 88.9% | 96.3% | 97.1% | 100.0% / 74.0% | 0.90 | 0.62 |
| breast_cancer | features | random_forest | 93.6% | 92.9% | 92.9% | 96.1% / 46.0% | 0.93 | 0.11 |
| breast_cancer | activations | mlp | 97.7% | 99.3% | 99.3% | 99.8% / 78.6% | 0.87 | 0.03 |
| breast_cancer | activations_predicted_class | mlp | 97.7% | 99.3% | 99.3% | 99.8% / 78.6% | 0.87 | 0.04 |
| breast_cancer | hybrid_features_0.3 | mlp | 97.7% | 98.6% | 98.6% | 99.3% / 70.1% | 0.87 | 0.04 |
| breast_cancer | tree_leaf | decision_tree | 93.0% | 97.3% | 97.3% | 98.8% / 77.8% | 0.73 | 0.04 |
| breast_cancer | forest_proximity | random_forest | 93.6% | 98.8% | 98.8% | 99.3% / 91.1% | 0.62 | 0.26 |
| hardware_trojan | features | random_forest | 97.5% | 99.3% | 99.4% | 99.8% / 83.6% | 0.75 | 0.06 |
| hardware_trojan | activations | mlp | 97.2% | 99.1% | 99.1% | 99.8% / 76.2% | 0.77 | 0.10 |
| hardware_trojan | activations_predicted_class | mlp | 97.2% | 99.1% | 99.1% | 99.8% / 76.2% | 0.77 | 0.10 |
| hardware_trojan | hybrid_features_0.3 | mlp | 97.2% | 99.1% | 99.1% | 99.8% / 76.1% | 0.79 | 0.10 |
| hardware_trojan | tree_leaf | decision_tree | 97.2% | 99.1% | 99.1% | 99.8% / 76.0% | 0.76 | 1.52 |
| hardware_trojan | forest_proximity | random_forest | 97.5% | 99.5% | 99.5% | 99.8% / 86.3% | 0.71 | 4.60 |
| fraud | features | random_forest | 98.2% | 99.3% | 99.3% | 99.6% / 84.3% | 0.66 | 0.15 |
| fraud | activations | mlp | 98.0% | 99.0% | 99.1% | 99.5% / 79.2% | 0.67 | 0.20 |
| fraud | activations_predicted_class | mlp | 98.0% | 99.0% | 99.1% | 99.5% / 79.2% | 0.67 | 0.19 |
| fraud | hybrid_features_0.3 | mlp | 98.0% | 99.1% | 99.1% | 99.6% / 78.6% | 0.66 | 0.23 |
| fraud | tree_leaf | decision_tree | 98.1% | 99.6% | 99.6% | 99.7% / 91.6% | 0.61 | 18.54 |
| fraud | forest_proximity | random_forest | 98.2% | 99.7% | 99.7% | 99.8% / 91.4% | 0.60 | 31.22 |

What the table shows:

- **Model-informed spaces report higher correspondence, but it is less
  diagnostic.** Hidden-activation, tree-leaf, and forest-proximity retrieval
  find neighbors the model already treats alike, so they agree with the
  prediction even when it is wrong. On breast cancer, feature-space
  correspondence separates correct from incorrect predictions with AUC 0.93,
  forest proximity with 0.62. Feature space is the most diagnostic space on
  Iris and breast cancer; on the two imbalanced domains every space lands
  within a few hundredths of the others.
- **Normalization changes correspondence little in most spaces** because
  correspondence is a ratio among the same k neighbors. It matters where raw
  distances are far from the feature scale (forest proximity on Iris moves from
  96.3% to 97.1%), and it makes neighbor distances readable across spaces: a
  normalized distance of 1.0 is a typical distinct-neighbor gap in any space.
- **On the imbalanced domains, every error is a missed positive, and the missed
  positives still look like their neighbors.** With all 63 trojans and all 148
  frauds in the explained subset, the random forest catches 15 trojans and 112
  frauds with at most one false alarm. The missed positives receive about 84%
  feature-space correspondence (above 90% for fraud in tree-leaf and
  forest-proximity space) because their nearest training precedents are
  negatives, which is exactly why the model missed them. Correspondence therefore says "this
  prediction agrees with precedent," not "this prediction is right"; on these
  domains its AUC as an error detector is only 0.60 to 0.79, and the useful
  signal is the small set of low-correspondence cases worth a second look.
  [Label Calibration](#label-calibration) shows how to make that set sharper.
- **Hardware trojan precedents are mostly exact duplicates.** Only 2,356 of the
  39,871 training rows are distinct, so most test circuits have identical
  matches in the training set and the raw nearest-neighbor distances are near
  zero. The useful explanation there is the identical precedents themselves,
  and the identical-but-negative precedents of the 48 missed trojans show the
  five features cannot separate them.

### Label Calibration

How often each interpretation label is right, in feature space with the
default thresholds (0.70, 0.85). Each cell gives the number of explained
predictions that received the label and the share of them the model got right.
`balanced` rows use `class_weights="balanced"`; the neighbors are the same,
only the scoring changes.

| Domain | Class weights | Mean corr. on incorrect | Corr. AUC | Labeled high | Labeled medium | Labeled low |
| --- | --- | --- | --- | --- | --- | --- |
| iris | none | 59.7% | 0.97 | 36 (100.0% correct) | 2 (50.0% correct) | 7 (42.9% correct) |
| iris | balanced | 59.7% | 0.97 | 36 (100.0% correct) | 2 (50.0% correct) | 7 (42.9% correct) |
| breast_cancer | none | 45.6% | 0.93 | 142 (99.3% correct) | 14 (85.7% correct) | 15 (46.7% correct) |
| breast_cancer | balanced | 42.5% | 0.93 | 139 (99.3% correct) | 12 (100.0% correct) | 20 (50.0% correct) |
| hardware_trojan | none | 83.2% | 0.75 | 1,954 (98.8% correct) | 30 (56.7% correct) | 16 (25.0% correct) |
| hardware_trojan | balanced | 51.5% | 0.75 | 1,958 (98.7% correct) | 0 | 42 (42.9% correct) |
| fraud | none | 84.3% | 0.66 | 1,967 (98.7% correct) | 12 (83.3% correct) | 21 (57.1% correct) |
| fraud | balanced | 66.9% | 0.66 | 1,984 (98.8% correct) | 0 | 16 (25.0% correct) |

- **The default thresholds are well ordered.** On the balanced domains a
  "high" label is right 99% to 100% of the time and a "low" label about half
  the time or less, so the labels can be used as they are.
- **On imbalanced data, use `class_weights="balanced"`.** Unweighted, a rare
  neighbor counts no more than a common one, so a missed fraud with one or two
  fraud neighbors still scores "medium" or "high", and a correctly caught fraud
  with a few legitimate neighbors scores "low". Balanced weights let any
  minority-class neighbor decide. On fraud the "low" label goes from 9 errors
  in 21 cases to 12 errors in 16, the ambiguous "medium" band empties, and mean
  correspondence on the model's errors falls from 84% to 67%. On hardware
  trojan about the same errors are flagged either way (25 of 49 unweighted, 24
  with weights); the weights only merge "medium" into "low".
- **Weights relabel far more than they re-rank.** The AUC moves by less than
  0.01 in every space with weights, because correspondence already ordered
  most of these cases correctly.
- **Some errors cannot be flagged by any calibration.** 24 of the 36 missed
  frauds and 24 of the 48 missed trojans have five neighbors that are all
  negatives, so they score 100% under every weighting and every threshold. The
  model and its precedents are wrong together. A "high" label means the
  prediction is consistent with the training data, and on these subsets 1.2%
  to 1.3% of "high" predictions are still wrong.
- **These subsets over-represent positives** (about 43 times the natural rate
  for fraud, 8.5 times for trojan) so that errors can be measured at all.
  Per-label accuracy on a natural stream of cases would be higher.

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

- Exposes actual training samples by design: feature values, labels, row positions, and metadata
- Suitable only where the audience for explanations may see the training data
- A saved explainer contains the full training set and must be handled as training data
- No privacy-preserving mode exists or is planned; see [Security & Privacy Considerations](#security--privacy-considerations)

### Correspondence Metric Limitations

- Measures neighbor agreement, not prediction correctness or quality
- High correspondence can occur with incorrect predictions if training data contains systematic errors
- A wrong prediction whose neighbors all share its class scores 100%; on the imbalanced reference domains this covers about half to two thirds of the model's errors
- On imbalanced data, unweighted correspondence under-counts minority-class neighbors; use `class_weights="balanced"`
- Thresholds for "high/medium/low" must be validated per domain (set them with `correspondence_thresholds=`)
- Raw correspondence is not comparable across retrieval spaces because distance scales differ; use `distance_normalization="median_nn"` when comparing

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

- Approximate nearest neighbors (Annoy, FAISS) for large-scale data
- Radar and parallel coordinate visualizations
- Disclosure-control options, such as choosing which features and metadata appear in explanations (these limit what is displayed and are not a privacy guarantee)
- Alternative distance-normalization schemes and threshold calibration studies

## Contact

Questions? Issues? Open a GitHub issue or contact <pcw@case.edu>.

## Acknowledgments

- Inspired by Caruana et al. (1999) "Case-based explanation of non-case-based learning"
- Validated on hardware trojan detection research
- Built with scikit-learn, scipy, and matplotlib
