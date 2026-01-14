Metrics
=======

Distance and correspondence metrics for case-based explanations.

.. currentmodule:: case_explainer.metrics

Functions
---------

compute_correspondence
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_correspondence

euclidean_distance
^^^^^^^^^^^^^^^^^^

.. autofunction:: euclidean_distance

Theory
------

Correspondence Metric
^^^^^^^^^^^^^^^^^^^^^

The correspondence metric quantifies the agreement between a prediction and retrieved nearest neighbors.
It uses inverse distance weighting to give more importance to closer neighbors.

Mathematical Definition
"""""""""""""""""""""""

For a test sample with predicted class ``c_pred``, given k nearest neighbors:

.. math::

   w_i = \\frac{1}{(d_i + 1)^3}

where :math:`d_i` is the distance to neighbor i.

The weight for each class c is:

.. math::

   W(c) = \\sum_{i: y_i = c} w_i

The correspondence score is:

.. math::

   \\text{Correspondence} = \\frac{W(c_{pred})}{\\sum_{c} W(c)}

Interpretation
""""""""""""""

* **1.0 (100%)**: All neighbors have the same class as the prediction (perfect agreement)
* **0.5 (50%)**: Neighbors are equally split between classes (no agreement)
* **0.0 (0%)**: All neighbors have different classes than the prediction (complete disagreement)

**Practical Ranges:**

* **High (≥0.85)**: Strong agreement, high confidence in prediction
* **Medium (0.70-0.85)**: Moderate agreement, reasonable confidence
* **Low (<0.70)**: Weak agreement, uncertain prediction or unusual sample

Distance Weighting
^^^^^^^^^^^^^^^^^^

The cubed inverse distance weighting scheme has several desirable properties:

1. **Emphasizes closest neighbors**: Closest neighbors have exponentially more influence
2. **Stable at distance=0**: The +1 term prevents division by zero
3. **Smooth falloff**: Neighbors further away contribute less but not zero

Alternative weighting schemes can be implemented by modifying the weight function.

Example Usage
-------------

Computing Correspondence
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from case_explainer.metrics import compute_correspondence
   import numpy as np

   # Example: 5 neighbors with their distances and labels
   neighbor_distances = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
   neighbor_labels = np.array([1, 1, 1, 0, 1])
   predicted_class = 1

   # Compute correspondence
   correspondence = compute_correspondence(
       neighbor_distances=neighbor_distances,
       neighbor_labels=neighbor_labels,
       predicted_class=predicted_class
   )

   print(f"Correspondence: {correspondence:.2%}")
   # Output: Correspondence: 94.5%

Distance Computation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from case_explainer.metrics import euclidean_distance
   import numpy as np

   sample1 = np.array([1.0, 2.0, 3.0])
   sample2 = np.array([1.5, 2.5, 3.5])

   distance = euclidean_distance(sample1, sample2)
   print(f"Distance: {distance:.3f}")
   # Output: Distance: 0.866

Understanding Correspondence Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from case_explainer.metrics import compute_correspondence

   # Perfect agreement: all neighbors same class
   distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
   labels = np.array([1, 1, 1, 1, 1])
   corr = compute_correspondence(distances, labels, predicted_class=1)
   print(f"Perfect agreement: {corr:.2%}")  # 100%

   # Complete disagreement: all neighbors different class
   labels = np.array([0, 0, 0, 0, 0])
   corr = compute_correspondence(distances, labels, predicted_class=1)
   print(f"Complete disagreement: {corr:.2%}")  # 0%

   # Mixed: 3 same class, 2 different
   labels = np.array([1, 1, 1, 0, 0])
   corr = compute_correspondence(distances, labels, predicted_class=1)
   print(f"Mixed (3:2): {corr:.2%}")  # ~80-90% depending on distances

   # Effect of distance: closer neighbors matter more
   # Close neighbors same class
   distances_close = np.array([0.1, 0.2, 0.3, 1.0, 1.5])
   labels_mixed = np.array([1, 1, 1, 0, 0])
   corr_close = compute_correspondence(distances_close, labels_mixed, predicted_class=1)
   
   # Far neighbors same class
   distances_far = np.array([1.0, 1.5, 2.0, 0.1, 0.2])
   labels_mixed = np.array([1, 1, 1, 0, 0])
   corr_far = compute_correspondence(distances_far, labels_mixed, predicted_class=1)
   
   print(f"Close neighbors same class: {corr_close:.2%}")  # Higher
   print(f"Far neighbors same class: {corr_far:.2%}")      # Lower

Class Weights
^^^^^^^^^^^^^

For imbalanced datasets, you can adjust correspondence using class weights:

.. code-block:: python

   # Without class weights
   distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
   labels = np.array([1, 1, 0, 0, 0])  # Minority class 1
   
   corr = compute_correspondence(distances, labels, predicted_class=1)
   print(f"No weights: {corr:.2%}")
   
   # With class weights (weight minority class more)
   corr_weighted = compute_correspondence(
       distances, labels, predicted_class=1,
       class_weights={0: 1.0, 1: 3.0}
   )
   print(f"With weights: {corr_weighted:.2%}")  # Higher

Notes
-----

**Performance**

* Correspondence computation is O(k) where k is the number of neighbors
* Distance computation is O(d) where d is the number of features
* Both operations are highly vectorized using NumPy for efficiency

**Numerical Stability**

* The +1 term in the weight function prevents division by zero
* All distance computations use double precision floats
* Correspondence is always in the range [0, 1]

See Also
--------

* :class:`CaseExplainer`: Uses these metrics for generating explanations
* :class:`Explanation`: Contains computed correspondence scores
