Metrics
=======

Distance and correspondence metrics for case-based explanations.

.. module:: case_explainer.metrics

Functions
---------

compute_correspondence
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_correspondence

interpret_correspondence
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: interpret_correspondence

validate_thresholds
^^^^^^^^^^^^^^^^^^^

.. autofunction:: validate_thresholds

euclidean_distance
^^^^^^^^^^^^^^^^^^

.. autofunction:: euclidean_distance

compute_all_distances
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_all_distances

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

   w_i = \frac{1}{(d_i + 1)^3}

where :math:`d_i` is the distance to neighbor i.

The weight for each class c is:

.. math::

   W(c) = \sum_{i: y_i = c} w_i

The correspondence score is:

.. math::

   \text{Correspondence} = \frac{W(c_{pred})}{\sum_{c} W(c)}

Interpretation
""""""""""""""

* **1.0 (100%)**: All neighbors have the same class as the prediction (complete neighbor agreement)
* **0.5 (50%)**: Neighbors are equally split between classes (no agreement)
* **0.0 (0%)**: All neighbors have different classes than the prediction (complete disagreement)

**Important**: Correspondence measures neighbor agreement, not prediction correctness. High correspondence 
can occur with incorrect predictions if the training data contains systematic errors or learned incorrect patterns.

**Default Interpretation Thresholds** (domain-dependent, not universal standards):

* **High (≥0.85)**: Strong agreement with retrieved neighbors
* **Medium (0.70-0.85)**: Moderate agreement with retrieved neighbors
* **Low (<0.70)**: Weak agreement, potentially unusual sample or inconsistent neighbors

Pass ``thresholds=(medium, high)`` to ``compute_correspondence`` or
``correspondence_thresholds=(medium, high)`` to ``CaseExplainer`` to use cut
points validated for your domain.

On the reference domains the default cut points are well ordered: a ``high``
label is right about 99% of the time and a ``low`` label about half the time
or less. A wrong prediction whose neighbors all share its class still scores
1.0, so ``high`` means "consistent with the training data" and not "verified".
:doc:`/calibration` gives the per-label figures and a recipe for checking them
on your own data.

Distance scale and retrieval space
""""""""""""""""""""""""""""""""""

The ``+1`` offset in the weight assumes distances on roughly the standardized
feature scale. Distances in other spaces are on different scales: forest
proximities lie in ``[0, 1]``, where the weighting flattens toward plain
voting, and hidden-activation distances can be larger or smaller than feature
distances depending on the dataset. Raw correspondence values are therefore
not comparable across retrieval spaces. Construct the
explainer with ``distance_normalization="median_nn"`` to divide distances by
the median distance from a training case to its nearest distinct training case
in the active space before weighting.

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

   # Example: 5 neighbors as (training_index, distance, label) tuples
   neighbors = [(12, 0.1, 1), (40, 0.2, 1), (7, 0.3, 1), (91, 0.5, 0), (3, 0.8, 1)]

   correspondence, interpretation = compute_correspondence(
       neighbors, predicted_class=1
   )
   print(f"Correspondence: {correspondence:.2%} ({interpretation})")
   # Output: Correspondence: 86.85% (high)

   # Domain-specific interpretation thresholds
   _, strict = compute_correspondence(
       neighbors, predicted_class=1, thresholds=(0.90, 0.95)
   )
   # strict == "low"

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

   from case_explainer.metrics import compute_correspondence

   def neighbors(distances, labels):
       return list(zip(range(len(distances)), distances, labels))

   distances = [0.1, 0.2, 0.3, 0.4, 0.5]

   # Complete agreement: all neighbors same class
   corr, _ = compute_correspondence(neighbors(distances, [1, 1, 1, 1, 1]), 1)
   print(f"Complete neighbor agreement: {corr:.2%}")  # 100%

   # Complete disagreement: all neighbors different class
   corr, _ = compute_correspondence(neighbors(distances, [0, 0, 0, 0, 0]), 1)
   print(f"Complete disagreement: {corr:.2%}")  # 0%

   # Mixed: 3 same class, 2 different
   corr, _ = compute_correspondence(neighbors(distances, [1, 1, 1, 0, 0]), 1)
   print(f"Mixed (3:2): {corr:.2%}")  # ~70% depending on distances

   # Effect of distance: closer neighbors matter more
   labels_mixed = [1, 1, 1, 0, 0]
   corr_close, _ = compute_correspondence(
       neighbors([0.1, 0.2, 0.3, 1.0, 1.5], labels_mixed), 1
   )
   corr_far, _ = compute_correspondence(
       neighbors([1.0, 1.5, 2.0, 0.1, 0.2], labels_mixed), 1
   )
   print(f"Close neighbors same class: {corr_close:.2%}")  # Higher
   print(f"Far neighbors same class: {corr_far:.2%}")      # Lower

Class Weights
^^^^^^^^^^^^^

For imbalanced datasets, you can adjust correspondence using class weights:

.. code-block:: python

   # Without class weights
   neighbors = [(0, 0.1, 1), (1, 0.2, 1), (2, 0.3, 0), (3, 0.4, 0), (4, 0.5, 0)]

   corr, label = compute_correspondence(neighbors, predicted_class=1)
   print(f"No weights: {corr:.2%} ({label})")       # 54.38% (low)

   # With class weights (weight minority class 1 more)
   corr_weighted, label = compute_correspondence(
       neighbors, predicted_class=1, class_weights={0: 1.0, 1: 3.0}
   )
   print(f"With weights: {corr_weighted:.2%} ({label})")  # 78.15% (medium)

Weights matter most when one class is rare. Here the model predicts the common
class 0, but one of the five neighbors belongs to rare class 1:

.. code-block:: python

   neighbors = [(0, 0.2, 0), (1, 0.3, 0), (2, 0.3, 1), (3, 0.4, 0), (4, 0.5, 0)]

   corr, label = compute_correspondence(neighbors, predicted_class=0)
   print(f"{corr:.2%} ({label})")   # 78.83% (medium)

   # Inverse-frequency weights for a class that is about 0.2% of the data
   corr, label = compute_correspondence(
       neighbors, predicted_class=0, class_weights={0: 0.5, 1: 290.0}
   )
   print(f"{corr:.2%} ({label})")   # 0.64% (low)

``CaseExplainer(class_weights="balanced")`` computes these inverse-frequency
weights from the training labels. See :doc:`/calibration` for their measured
effect on the reference domains.

Notes
-----

**Performance**

* Correspondence computation is O(k) where k is the number of neighbors; it is
  a plain Python loop over the k neighbors
* Distance computation is O(d) where d is the number of features and uses
  SciPy's compiled routines

**Numerical Stability**

* The +1 term in the weight function prevents division by zero
* All distance computations use double precision floats
* Correspondence is always in the range [0, 1]

See Also
--------

* :class:`~case_explainer.CaseExplainer`: Uses these metrics for generating explanations
* :class:`~case_explainer.explanation.Explanation`: Contains computed correspondence scores
