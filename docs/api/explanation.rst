Explanation Objects
===================

Classes for representing and visualizing explanations.

.. currentmodule:: case_explainer

Explanation
-----------

.. autoclass:: Explanation
   :show-inheritance:

Key Attributes
^^^^^^^^^^^^^^

.. attribute:: Explanation.correspondence

   The correspondence score between 0 and 1, indicating agreement between the prediction and retrieved neighbors.
   Higher values indicate stronger agreement with training precedent.

.. attribute:: Explanation.correspondence_interpretation

   Human-readable interpretation of correspondence: 'high', 'medium', or 'low'
   according to the explainer's ``correspondence_thresholds`` (default: high at
   or above 0.85, medium at or above 0.70), or 'undefined' when no neighbors
   were retrieved.

.. attribute:: Explanation.retrieval_space

   Which space the neighbors were retrieved in: ``'features'``,
   ``'activations'``, ``'hybrid(features=0.30)'``, ``'tree_leaf'``, or
   ``'forest_proximity'``. Neighbor distances are measured in this space.

.. attribute:: Explanation.distance_metric

   Name of the distance used inside the retrieval space, for example
   ``'euclidean'`` or ``'shared_leaf_fraction'``.

.. attribute:: Explanation.distance_scale

   Divisor applied to raw distances before they were stored on the neighbors.
   ``1.0`` means raw distances; with ``distance_normalization="median_nn"`` it
   is the median nearest-neighbor distance among training cases, and
   ``raw_distance = neighbor.distance * distance_scale``.

.. attribute:: Explanation.neighbors

   List of :class:`~case_explainer.explanation.Neighbor` objects representing the k nearest training samples.

.. attribute:: Explanation.predicted_class

   The predicted class for the explained instance.

.. attribute:: Explanation.true_class

   The true class of the explained instance (if provided).

Methods
^^^^^^^

.. automethod:: Explanation.is_correct

.. automethod:: Explanation.describe_retrieval_space

.. automethod:: Explanation.summary

.. automethod:: Explanation.to_dict

.. automethod:: Explanation.plot

Neighbor
--------

.. autoclass:: case_explainer.explanation.Neighbor
   :members:
   :undoc-members:
   :show-inheritance:

Key Attributes
^^^^^^^^^^^^^^

.. attribute:: Neighbor.index

   Index of the neighbor in the training set.

.. attribute:: Neighbor.distance

   Distance from the test sample to this neighbor.

.. attribute:: Neighbor.label

   Class label of the neighbor.

.. attribute:: Neighbor.features

   The complete, unscaled feature values of the training row.

.. attribute:: Neighbor.metadata

   Metadata dictionary for this training sample. Empty when no metadata was
   supplied or the explanation was requested with ``return_provenance=False``.

.. note::

   A neighbor is a real training record. Anyone who can read an explanation,
   its summary, its plot, or its exported dictionary can read the training
   rows in it. See :doc:`/data_exposure`.

Example Usage
-------------

Accessing Explanation Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from case_explainer import CaseExplainer
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   # Setup
   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
   
   clf = RandomForestClassifier()
   clf.fit(X_train, y_train)
   
   explainer = CaseExplainer(X_train, y_train)
   explanation = explainer.explain_instance(X_test[0], k=5, model=clf)

   # Access explanation properties
   print(f"Correspondence: {explanation.correspondence:.2%}")
   print(f"Interpretation: {explanation.correspondence_interpretation}")
   print(f"Predicted class: {explanation.predicted_class}")
   print(f"Is correct: {explanation.is_correct()}")

Inspecting Neighbors
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Examine each neighbor
   for i, neighbor in enumerate(explanation.neighbors):
       print(f"\nNeighbor {i+1}:")
       print(f"  Training index: {neighbor.index}")
       print(f"  Distance: {neighbor.distance:.3f}")
       print(f"  Label: {neighbor.label}")
       print(f"  Features: {neighbor.features[:3]}...")  # First 3 features
       
       if neighbor.metadata:
           print(f"  Metadata: {neighbor.metadata}")

Exporting Explanations
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Convert to dictionary for JSON serialization
   exp_dict = explanation.to_dict()
   
   import json
   with open('explanation.json', 'w') as f:
       json.dump(exp_dict, f, indent=2)

   # The dictionary contains:
   # - correspondence: float
   # - correspondence_interpretation: str
   # - retrieval_space, distance_metric, distance_scale
   # - predicted_class: int
   # - true_class: int or None
   # - is_correct: bool or None
   # - predicted_class_name, true_class_name: display names
   # - test_index: position of the query in its batch, or None
   # - test_sample: the query's feature values
   # - neighbors: list of dicts with index, distance, label, label_name,
   #   features, metadata
   # - feature_names

The exported file contains the query and the full feature values and metadata
of each retrieved training row. Store it under the same controls as the
training data.

Generating Summaries
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get human-readable summary
   summary = explanation.summary()
   print(summary)
   
   # Example output (k=3, with class names and metadata):
   # ============================================================
   # CASE-BASED EXPLANATION
   # ============================================================
   # Predicted class: malignant
   # True class: malignant [OK]
   # Correspondence: 100.00% (high)
   # Retrieval space: features (euclidean)
   #
   # Nearest 3 neighbors:
   # ------------------------------------------------------------
   # 1. [*] Index 94: class malignant, distance 2.8589
   #       sample_id: patient_94
   # 2. [*] Index 97: class malignant, distance 3.4230
   #       sample_id: patient_97
   # 3. [*] Index 196: class malignant, distance 3.6326
   #       sample_id: patient_196
   # ============================================================

A ``[*]`` marks a neighbor whose class matches the prediction. The summary
lists each neighbor's row position, class, distance, and metadata; it does not
print feature values.

Visualizing Explanations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Grouped bar plot of feature values: the test sample next to each neighbor
   explanation.plot()

   # Save the plot to a file instead of showing it
   explanation.plot(save_path='explanation.png')

The plot draws the feature values of the test sample and of every retrieved
neighbor, so it works best with a modest number of features. ``plot_type``
accepts ``'radar'`` and ``'parallel'``, which are not implemented yet and fall
back to the bar plot.

Batch Analysis
^^^^^^^^^^^^^^

.. code-block:: python

   # Analyze multiple explanations
   explanations = explainer.explain_batch(X_test[:50], k=5, y_test=y_test[:50], model=clf)
   
   # Correspondence by class
   from collections import defaultdict
   corr_by_class = defaultdict(list)
   
   for exp in explanations:
       corr_by_class[exp.predicted_class].append(exp.correspondence)
   
   for cls, corrs in corr_by_class.items():
       mean_corr = sum(corrs) / len(corrs)
       print(f"Class {cls}: {mean_corr:.2%} avg correspondence ({len(corrs)} samples)")

   # High vs low correspondence predictions
   high_corr = [exp for exp in explanations if exp.correspondence >= 0.85]
   low_corr = [exp for exp in explanations if exp.correspondence < 0.70]
   
   print(f"\nHigh correspondence (≥85%): {len(high_corr)} samples")
   print(f"Low correspondence (<70%): {len(low_corr)} samples")
   
   # Accuracy by correspondence level
   high_acc = sum(1 for exp in high_corr if exp.is_correct()) / len(high_corr)
   low_acc = sum(1 for exp in low_corr if exp.is_correct()) / len(low_corr) if low_corr else 0
   
   print(f"High correspondence accuracy: {high_acc:.2%}")
   print(f"Low correspondence accuracy: {low_acc:.2%}")

See Also
--------

* :class:`CaseExplainer`: The main explainer class that generates these explanations
* :mod:`case_explainer.metrics`: The correspondence computation functions
