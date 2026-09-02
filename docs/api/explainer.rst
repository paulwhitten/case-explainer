CaseExplainer
=============

The main class for creating case-based explanations.

.. currentmodule:: case_explainer

.. autoclass:: CaseExplainer
   :show-inheritance:

Core Methods
------------

Building the Explainer
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: CaseExplainer.__init__

Explaining Predictions
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: CaseExplainer.explain_instance

.. automethod:: CaseExplainer.explain_batch

Similarity Strategies
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Features
    :members:

.. autoclass:: HiddenActivations
    :members:

.. autoclass:: CustomActivations
    :members:

.. autoclass:: TreeLeaf
    :members:

.. autoclass:: ForestProximity
    :members:

.. autoclass:: Blend
    :members:

Deprecated Retrieval Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: HiddenActivationRetrieval
    :members:

.. autoclass:: CustomActivationRetrieval
    :members:

.. autoclass:: TreeLeafRetrieval
    :members:

.. autoclass:: ForestProximityRetrieval
    :members:

Example Usage
-------------

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

   from case_explainer import CaseExplainer
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   # Load and split data
   data = load_breast_cancer()
   X_train, X_test, y_train, y_test = train_test_split(
       data.data, data.target, test_size=0.3, random_state=42
   )

   # Train classifier
   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)

   # Create explainer
   explainer = CaseExplainer(
       X_train=X_train,
       y_train=y_train,
       feature_names=data.feature_names,
       algorithm='ball_tree',
       scale_data=True
   )

   # Explain a prediction
   explanation = explainer.explain_instance(
       test_sample=X_test[0],
       k=5,
       model=clf,
       true_class=y_test[0]
   )

   print(f"Correspondence: {explanation.correspondence:.2%}")
   print(f"Predicted class: {explanation.predicted_class}")
   print(f"Correct: {explanation.is_correct()}")

Batch Explanations
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Explain multiple predictions at once
   explanations = explainer.explain_batch(
       X_test[:100],
       k=5,
       y_test=y_test[:100],
       model=clf
   )

   # Analyze correspondence distribution
   correspondences = [exp.correspondence for exp in explanations]
   correct_corr = [exp.correspondence for exp in explanations if exp.is_correct()]
   incorrect_corr = [exp.correspondence for exp in explanations if not exp.is_correct()]

   print(f"Mean correspondence: {sum(correspondences)/len(correspondences):.2%}")
   print(f"Correct predictions: {sum(correct_corr)/len(correct_corr):.2%}")
   print(f"Incorrect predictions: {sum(incorrect_corr)/len(incorrect_corr):.2%}")

Working with Metadata
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Attach metadata to training samples
   metadata = {
       'sample_id': [f"patient_{i}" for i in range(len(X_train))],
       'date': ['2024-01-01'] * len(X_train),
       'source': ['hospital_A'] * len(X_train)
   }

   explainer = CaseExplainer(
       X_train=X_train,
       y_train=y_train,
       metadata=metadata,
       algorithm='ball_tree'
   )

   # Access metadata in explanations
   explanation = explainer.explain_instance(X_test[0], k=5, model=clf)
   for neighbor in explanation.neighbors:
       print(f"Neighbor {neighbor.index}: {neighbor.metadata}")

Configuration Options
---------------------

Algorithm Selection
^^^^^^^^^^^^^^^^^^^

Choose the indexing algorithm based on your data characteristics:

* **kd_tree**: Best for low-dimensional data (<20 features), fastest for small to medium datasets
* **ball_tree**: Better for high-dimensional data (>20 features), good all-around choice
* **brute**: Exact search, only recommended for small datasets (<5k samples)
* **auto**: Let scikit-learn choose based on data characteristics (default)

.. code-block:: python

   # For low-dimensional data
   explainer = CaseExplainer(X_train, y_train, algorithm='kd_tree')

   # For high-dimensional data
   explainer = CaseExplainer(X_train, y_train, algorithm='ball_tree')

   # For very small datasets
   explainer = CaseExplainer(X_train, y_train, algorithm='brute')

Feature Scaling
^^^^^^^^^^^^^^^

Feature scaling is recommended to prevent features with large ranges from dominating distance calculations:

.. code-block:: python

   # With scaling (recommended)
   explainer = CaseExplainer(X_train, y_train, scale_data=True)

   # Without scaling (if features are already normalized)
   explainer = CaseExplainer(X_train, y_train, scale_data=False)

Class Weights
^^^^^^^^^^^^^

For imbalanced datasets, you can weight classes differently in correspondence computation:

.. code-block:: python

   # Weight minority class more heavily
   explainer = CaseExplainer(
       X_train, y_train,
       class_weights={0: 1.0, 1: 5.0}  # Weight class 1 five times more
   )

Notes
-----

**Performance Considerations**

* Index building time is O(n log n) for tree-based methods
* Query time is O(log n) for tree-based methods, O(n) for brute force
* Memory usage scales with dataset size and dimensionality
* Use ``n_jobs=-1`` to parallelize nearest neighbor search

**Correspondence Interpretation**

* **High (≥85%)**: Strong agreement with training precedent, high confidence
* **Medium (70-85%)**: Moderate agreement, reasonable confidence
* **Low (<70%)**: Weak agreement, prediction may be uncertain or unusual

Activation-Based Similarity
----------------------------

When your model is a neural network, you can optionally build the k-NN index on
the model's **hidden-layer activations** instead of raw input features, an approach
described by Caruana et al. (1999). The retrieved cases show consequences of the
model's learned representation. They do not expose its complete internal reasoning.

Quick Start
^^^^^^^^^^^

.. code-block:: python

   from sklearn.neural_network import MLPClassifier
   from case_explainer import CaseExplainer, HiddenActivations

   mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), random_state=42)
   mlp.fit(X_train, y_train)

   # Pure activation-based similarity, using the last hidden layer
   explainer = CaseExplainer(
       X_train, y_train,
       similarity=HiddenActivations(
           model=mlp,
           layer="last_hidden",
       ),
   )

   # All-layer aggregation is a library extension.
   explainer_deep = CaseExplainer(
       X_train, y_train,
       similarity=HiddenActivations(model=mlp, layer="all_hidden"),
   )

   explanation = explainer.explain_instance(X_test[0])

Output weighting
^^^^^^^^^^^^^^^^

``output_weighting="mean_abs"`` is the compatibility default. For multiclass
models, it averages each hidden unit's absolute connections across output
classes and uses one retrieval index for every query.

``output_weighting="predicted_class"`` weights hidden units using only the
predicted class's output connections. The explainer builds one index per class
and selects the matching index after prediction. sklearn class labels are
mapped through ``model.classes_``, so labels do not need to be contiguous or
zero-based. Binary sklearn MLP classifiers have one output column and therefore
use the same connection magnitudes for both labels.

.. code-block:: python

    similarity = HiddenActivations(
         model=mlp,
         output_weighting="predicted_class",
    )
    explainer = CaseExplainer(X_train, y_train, similarity=similarity)

Set ``output_weighting="none"`` to use standardized activations without output
connection weighting.

Layer modes
^^^^^^^^^^^

``'last_hidden'``
    Uses the last hidden layer, based on Caruana (1999).
    ``output_weighting="mean_abs"`` applies Section 5 scaling.

``'all_hidden'``
    Concatenates independently standardized hidden layers and scales layer
    :math:`i` by :math:`w_i = \sqrt{(i+1)/n}`. This is a library extension with
    :math:`d^2 = \sum_i \frac{i+1}{n} \|\Delta a_i\|^2`.

``int``
    Selects a specific hidden layer by 0-based index. Out-of-range indices raise
    ``ValueError``.

Using an explicit extractor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For full control, wrap a
:class:`~case_explainer.SklearnMLPActivationExtractor` in
:class:`~case_explainer.CustomActivations`:

.. code-block:: python

   from case_explainer import (
       CaseExplainer, CustomActivations, SklearnMLPActivationExtractor,
   )

   ext = SklearnMLPActivationExtractor(layer="all_hidden", use_output_weights=True)
   explainer = CaseExplainer(
       X_train, y_train,
       similarity=CustomActivations(model=mlp, extractor=ext),
   )

Feature/activation hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~case_explainer.Blend` mixes feature-space and activation-space
distances:

.. code-block:: python

   from case_explainer import Blend, HiddenActivations

   # 30% features, 70% activations
   explainer = CaseExplainer(
       X_train, y_train,
       similarity=Blend(HiddenActivations(model=mlp), features=0.3),
   )

:math:`d^2 = \alpha\,d_\text{feat}^2 + (1-\alpha)\,d_\text{act}^2`

``features=0.0`` is pure activations; ``features=1.0`` is pure features
(identical to feature retrieval). Hybrid retrieval is a library extension and
is not defined by Caruana et al. (1999).

Model preprocessing
^^^^^^^^^^^^^^^^^^^

When an MLP was trained on transformed features, provide the fitted transformer:

.. code-block:: python

   similarity = HiddenActivations(
       model=mlp,
       input_transform=fitted_scaler,
   )
   explainer = CaseExplainer(X_train, y_train, similarity=similarity)

The transform is applied before activation extraction and prediction. The
``scale_data`` parameter remains responsible only for feature-space retrieval.

Decision-tree leaves
^^^^^^^^^^^^^^^^^^^^

For a fitted sklearn decision tree, paper-faithful retrieval returns only cases
from the query's leaf. If fewer than ``k`` cases occupy that leaf, the default
behavior returns fewer cases rather than crossing the leaf boundary.
If the retained case base contains no case from the query leaf, the explanation
contains no neighbors and reports correspondence as undefined.

.. code-block:: python

   from case_explainer import TreeLeaf

   explainer = CaseExplainer(
       X_train, y_train,
       similarity=TreeLeaf(model=tree, overflow="truncate"),
   )

Set ``overflow="nearest"`` to explicitly fill the remaining positions with
feature-nearest cases from other leaves.

Random-forest proximity
^^^^^^^^^^^^^^^^^^^^^^^

For a fitted sklearn random forest, proximity is the fraction of trees in
which two samples reach the same leaf. ``ForestProximity`` retrieves
the cases with the smallest distance

.. math::

    d(x,z) = 1 - \frac{1}{T}\sum_{t=1}^{T}
    \mathbf{1}\{\ell_t(x)=\ell_t(z)\}.

.. code-block:: python

    from case_explainer import ForestProximity

    explainer = CaseExplainer(
         X_train, y_train,
         similarity=ForestProximity(model=forest),
    )

This strategy compares leaf identities directly. It does not treat tree node
identifiers as numeric coordinates.

Convenience aliases
^^^^^^^^^^^^^^^^^^^

The ``activation_extractor``, ``activation_layer``, ``use_output_weights``,
and ``blend_alpha`` constructor path, along with the ``retrieval=`` parameter,
remains supported but emits ``DeprecationWarning``. New code should use
``similarity=`` with the strategy classes (:class:`Features`,
:class:`HiddenActivations`, :class:`CustomActivations`, :class:`TreeLeaf`,
:class:`ForestProximity`, :class:`Blend`) because it keeps one similarity
strategy together and makes preprocessing ownership explicit.

See Also
--------

* :class:`Explanation`: The explanation object returned by ``explain_instance``
* :mod:`case_explainer.metrics`: Correspondence and distance metrics
