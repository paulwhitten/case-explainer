CaseExplainer
=============

The main class for creating case-based explanations.

.. warning::

   Explanations return real training rows, and a pickled ``CaseExplainer``
   contains the whole training set. Use the library where the audience for the
   explanations may see the training data. See :doc:`/data_exposure`.

.. module:: case_explainer

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

.. autodata:: SimilarityStrategy

.. autoclass:: Predictor
    :members:

Activation Extractors
^^^^^^^^^^^^^^^^^^^^^

Extractors turn inputs into the vectors that activation-based retrieval
indexes. The strategies above build them for you; use them directly with
:class:`CustomActivations`.

.. autoclass:: ActivationExtractor
    :members:

.. autoclass:: SklearnMLPActivationExtractor
    :members:

.. autoclass:: DecisionTreeActivationExtractor
    :members:

.. autoclass:: CallableActivationExtractor
    :members:

Utilities
^^^^^^^^^

.. automethod:: CaseExplainer.compute_correspondence

.. automethod:: CaseExplainer.get_training_info

.. autoattribute:: CaseExplainer.retrieval_space

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

   # Leave metadata out of an explanation
   explanation = explainer.explain_instance(
       X_test[0], k=5, model=clf, return_provenance=False
   )

Metadata travels with every explanation and with a saved explainer, so keep
direct identifiers out of it unless the readers of the explanations may see
them. ``return_provenance=False`` removes metadata only; feature values,
labels, and row positions are still returned.

Working with DataFrames
^^^^^^^^^^^^^^^^^^^^^^^

Pass a ``pandas.DataFrame`` as ``X_train`` and the explainer remembers its
column names. They become the default ``feature_names``, and they also decide
what the model sees from then on.

**Why the explainer turns arrays back into frames.** The k-NN index works on
plain NumPy arrays, so every query is reduced to an array internally. A model
or transformer fitted on a DataFrame, however, expects a DataFrame back: a
scikit-learn pipeline whose ``ColumnTransformer`` selects columns by name
raises on an array, and any estimator that recorded ``feature_names_in_``
warns about missing feature names on every call. To keep those models working
exactly as they do in the user's own code, the explainer rebuilds a DataFrame
with the training columns immediately before each call to ``model.predict``,
``model.apply`` (tree spaces), the ``input_transform`` of
:class:`HiddenActivations`, and the built-in activation extractors. The index
itself never changes; only the object handed to the user's estimator does.
Explainers built from an array pass arrays through unchanged.

.. code-block:: python

   import pandas as pd
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import make_pipeline

   X_train = pd.DataFrame(X, columns=names)
   pipeline = make_pipeline(
       ColumnTransformer([("scaled", StandardScaler(), names[:4])]),
       LogisticRegression(),
   ).fit(X_train, y_train)

   explainer = CaseExplainer(X_train, y_train, model=pipeline)
   explanation = explainer.explain_instance(X_train.iloc[0])

**Queries are aligned by column name.** A DataFrame query is reordered to the
training column order before anything else happens, so column order in the
caller's frame does not matter. A ``Series`` row (``frame.iloc[i]``) is
treated the same way using its index labels. A frame with missing or extra
columns raises ``ValueError`` naming the offending columns, since silently
comparing columns in the wrong order would mislabel every feature. A frame
whose columns are a plain ``RangeIndex`` carries no names and is taken
positionally, as arrays are.

Three cases to know:

* An explainer built from an array has no column names to align to, so a
  DataFrame query is taken positionally and its column order is not checked.
* The rebuilt frame carries the training column names but not the original
  per-column dtypes; values arrive as the numeric dtype of the query array.
* A custom extractor passed through :class:`CustomActivations` always
  receives a plain 2-D array, so existing callables (for example ones that
  build a ``torch.tensor``) keep working. Rebuild a frame inside the callable
  if the model needs names: ``pd.DataFrame(X, columns=explainer.feature_names)``.

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

   # Or derive inverse-frequency weights from y_train
   explainer = CaseExplainer(X_train, y_train, class_weights="balanced")
   print(explainer.class_weights)  # e.g. {0: 0.5, 1: 289.8}

``"balanced"`` uses ``n_samples / (n_classes * count)``, the scikit-learn
convention. It is recommended whenever one class is rare. Without weights, a
majority-class prediction whose five neighbors include one or two minority
cases still scores "medium" or "high", even though a minority neighbor is
strong evidence on imbalanced data. Weighting does not change which neighbors
are retrieved, only how their agreement is scored. It cannot help when every
neighbor shares the (wrong) predicted class. :doc:`/calibration` shows how the
labels behave on the reference domains with and without weights.

Notes
-----

**Performance Considerations**

* Index building time is O(n log n) for tree-based methods
* Query time is O(log n) for tree-based methods, O(n) for brute force
* Memory usage scales with dataset size and dimensionality
* ``n_jobs=-1`` (the default) parallelizes nearest-neighbor search for
  batches. Single-sample queries always run on one thread: splitting one row
  across workers costs about 13 ms of scheduling against under 1 ms of search
  on a tree index, so ``explain_instance`` skips it.

**Correspondence Interpretation**

* **High (≥85%)**: Strong agreement with training precedent
* **Medium (70-85%)**: Moderate agreement
* **Low (<70%)**: Weak agreement, prediction may be uncertain or unusual

The labels describe agreement with the training data, not the probability that
the prediction is correct. A wrong prediction whose neighbors all share its
class scores 100%. The cut points are set with ``correspondence_thresholds=``.
See :doc:`/calibration` for how often each label is right on the reference
domains.

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
If the explainer was built from a DataFrame, the transform receives a
DataFrame with the training columns, so a scaler fitted on named columns works
without warnings (see `Working with DataFrames`_).

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

Distance metric by retrieval space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``metric=`` names the distance used inside the retrieval space. Feature,
activation, and blended retrieval pass it to sklearn's ``NearestNeighbors``;
:class:`TreeLeaf` uses it to order same-leaf cases in standardized feature
space. :class:`ForestProximity` defines its own distance (one minus the
shared-leaf fraction) and raises ``ValueError`` for any other ``metric``.

.. _comparing-correspondence-across-spaces:

Comparing correspondence across retrieval spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neighbor distances are reported in the retrieval space that produced them.
Their scale differs between spaces and between datasets: feature and
hidden-activation distances can differ by a factor of two or more in either
direction, and forest proximities lie in ``[0, 1]``. Because
correspondence weights neighbors by :math:`1/(d+1)^3`, raw correspondence
values are comparable within one space but not across spaces.

Pass ``distance_normalization="median_nn"`` to divide every distance by the
median nearest-neighbor distance among training cases in the active space.
Training cases that have an exact duplicate are left out of the estimate; if
every sampled case has one, distances stay unnormalized and a warning is
issued:

.. code-block:: python

   explainer = CaseExplainer(
       X_train, y_train,
       similarity=HiddenActivations(model=mlp),
       distance_normalization="median_nn",
   )
   explainer.distance_scale_        # the divisor that was applied
   explanation = explainer.explain_instance(x)
   explanation.distance_scale       # same value, carried on the explanation
   explanation.retrieval_space      # "activations"

A normalized distance of ``1.0`` means "as far as a typical training case is
from its closest distinct neighbor". Neighbor selection is unaffected; only the
reported distances and therefore the correspondence weighting change. Use
``correspondence_thresholds=(medium, high)`` to set the interpretation cut
points for your domain.

Normalization makes distances comparable; it does not make every space equally
informative. Model-informed spaces retrieve cases the model already treats
alike, and on the balanced reference domains their correspondence is higher
and separates correct from incorrect predictions less well than feature space.
See :doc:`/calibration`.

Removed parameters
^^^^^^^^^^^^^^^^^^

The 0.2.x ``activation_extractor``, ``activation_layer``,
``use_output_weights``, ``blend_alpha``, and ``retrieval`` parameters were
removed in 0.3.0 and raise ``TypeError`` with a migration hint. See
:doc:`/migration`.

See Also
--------

* :class:`Explanation`: The explanation object returned by ``explain_instance``
* :doc:`metrics`: Correspondence and distance metrics
