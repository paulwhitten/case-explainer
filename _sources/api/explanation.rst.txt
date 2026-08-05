Explanation Objects
===================

Classes for representing and visualizing explanations.

.. currentmodule:: case_explainer

Explanation
-----------

.. autoclass:: Explanation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Attributes
^^^^^^^^^^^^^^

.. attribute:: Explanation.correspondence

   The correspondence score between 0 and 1, indicating agreement between the prediction and retrieved neighbors.
   Higher values indicate stronger agreement with training precedent.

.. attribute:: Explanation.correspondence_interpretation

   Human-readable interpretation of correspondence: 'high' (≥85%), 'medium' (70-85%), or 'low' (<70%).

.. attribute:: Explanation.neighbors

   List of :class:`Neighbor` objects representing the k nearest training samples.

.. attribute:: Explanation.predicted_class

   The predicted class for the explained instance.

.. attribute:: Explanation.true_class

   The true class of the explained instance (if provided).

Methods
^^^^^^^

.. automethod:: Explanation.is_correct

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

   Feature values of the neighbor.

.. attribute:: Neighbor.metadata

   Optional metadata dictionary for this training sample.

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
   # - predicted_class: int
   # - true_class: int or None
   # - is_correct: bool or None
   # - neighbors: list of dicts with index, distance, label, features

Generating Summaries
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get human-readable summary
   summary = explanation.summary()
   print(summary)
   
   # Example output:
   # Explanation for test sample:
   #   Predicted class: 1
   #   True class: 1 (CORRECT)
   #   Correspondence: 94.3% (high)
   #
   # Nearest neighbors:
   #   1. Index 42, distance 0.123, label 1
   #   2. Index 67, distance 0.234, label 1
   #   3. Index 15, distance 0.289, label 1
   #   4. Index 89, distance 0.345, label 0
   #   5. Index 23, distance 0.401, label 1

Visualizing Explanations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create bar plot of neighbor distances and labels
   explanation.plot()
   
   # Save plot to file
   import matplotlib.pyplot as plt
   explanation.plot()
   plt.savefig('explanation.png', dpi=300, bbox_inches='tight')
   plt.close()

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
