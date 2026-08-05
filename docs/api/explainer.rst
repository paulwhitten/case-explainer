CaseExplainer
=============

The main class for creating case-based explanations.

.. currentmodule:: case_explainer

.. autoclass:: CaseExplainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Core Methods
------------

Building the Explainer
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: CaseExplainer.__init__

Explaining Predictions
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: CaseExplainer.explain_instance

.. automethod:: CaseExplainer.explain_batch

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
the model's **hidden-layer activations** instead of raw input features — an approach
described by Caruana et al. (1999).  Activation-based retrieval captures the model's
*internal reasoning* rather than surface feature proximity.

Quick Start — convenience shorthand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.neural_network import MLPClassifier
   from case_explainer import CaseExplainer

   mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), random_state=42)
   mlp.fit(X_train, y_train)

   # Pure activation-based similarity (Caruana 1999) — last hidden layer
   explainer = CaseExplainer(
       X_train, y_train,
       activation_layer="last_hidden",   # convenience shorthand
       use_output_weights=True,          # Caruana §5 output-weight scaling
       model=mlp,
   )

   # All hidden layers with layer-position weighting
   explainer_deep = CaseExplainer(
       X_train, y_train,
       activation_layer="all_hidden",    # later layers weighted more heavily
       model=mlp,
   )

   explanation = explainer.explain_instance(X_test[0], model=mlp)

Layer modes
^^^^^^^^^^^

+----------------------+----------------------------------------------------------+
| ``activation_layer`` | Description                                              |
+======================+==========================================================+
| ``'last_hidden'``    | Last hidden layer only — default Caruana (1999) method.  |
|                      | ``use_output_weights=True`` applies §5 scaling.          |
+----------------------+----------------------------------------------------------+
| ``'all_hidden'``     | Concatenates all hidden layers, each independently       |
|                      | standardised, then scaled by                             |
|                      | :math:`w_i = \sqrt{(i+1)/n}`.  Later layers dominate:   |
|                      | :math:`d^2 = \sum_i \frac{i+1}{n} \|\Delta a_i\|^2`     |
+----------------------+----------------------------------------------------------+
| ``int``              | Specific hidden layer by 0-based index.                  |
+----------------------+----------------------------------------------------------+

Using an explicit extractor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For full control, pass a :class:`~case_explainer.SklearnMLPActivationExtractor`
directly:

.. code-block:: python

   from case_explainer import CaseExplainer, SklearnMLPActivationExtractor

   ext = SklearnMLPActivationExtractor(layer="all_hidden", use_output_weights=True)
   explainer = CaseExplainer(
       X_train, y_train,
       activation_extractor=ext,
       model=mlp,
   )

Feature/activation hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^

``blend_alpha`` mixes feature-space and activation-space distances:

.. code-block:: python

   # 30% features, 70% activations
   explainer = CaseExplainer(
       X_train, y_train,
       activation_layer="last_hidden",
       model=mlp,
       blend_alpha=0.3,
   )

:math:`d^2 = \alpha\,d_\text{feat}^2 + (1-\alpha)\,d_\text{act}^2`

``blend_alpha=0.0`` is pure activations; ``blend_alpha=1.0`` is pure features
(identical to not setting ``activation_layer`` at all).

See Also
--------

* :class:`Explanation`: The explanation object returned by ``explain_instance``
* :mod:`case_explainer.metrics`: Correspondence and distance metrics
