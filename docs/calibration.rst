Interpreting and Calibrating Correspondence
===========================================

Correspondence measures how strongly the retrieved training cases agree with a
prediction. It does not measure whether the prediction is right. This page
shows how reliable the ``high``, ``medium``, and ``low`` labels are on the
reference domains, what to configure on imbalanced data, and what no
configuration can fix.

All figures come from ``validate_similarity_modes.py`` and are stored in
``benchmarks/similarity_validation.csv``.

How reliable the labels are
---------------------------

The table gives, for feature-space retrieval with the default thresholds
``(0.70, 0.85)``, how many explained predictions received each label and the
share of them the model got right. ``balanced`` rows use
``class_weights="balanced"``. The neighbors are the same in both rows; only the
scoring changes.

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 10 18 18 18

   * - Domain
     - Class weights
     - Mean corr. on incorrect
     - Corr. AUC
     - Labeled high
     - Labeled medium
     - Labeled low
   * - iris
     - none
     - 59.7%
     - 0.97
     - 36 (100.0% correct)
     - 2 (50.0% correct)
     - 7 (42.9% correct)
   * - iris
     - balanced
     - 59.7%
     - 0.97
     - 36 (100.0% correct)
     - 2 (50.0% correct)
     - 7 (42.9% correct)
   * - breast_cancer
     - none
     - 45.6%
     - 0.93
     - 142 (99.3% correct)
     - 14 (85.7% correct)
     - 15 (46.7% correct)
   * - breast_cancer
     - balanced
     - 42.5%
     - 0.93
     - 139 (99.3% correct)
     - 12 (100.0% correct)
     - 20 (50.0% correct)
   * - hardware_trojan
     - none
     - 83.2%
     - 0.75
     - 1,954 (98.8% correct)
     - 30 (56.7% correct)
     - 16 (25.0% correct)
   * - hardware_trojan
     - balanced
     - 51.5%
     - 0.75
     - 1,958 (98.7% correct)
     - 0
     - 42 (42.9% correct)
   * - fraud
     - none
     - 84.3%
     - 0.66
     - 1,967 (98.7% correct)
     - 12 (83.3% correct)
     - 21 (57.1% correct)
   * - fraud
     - balanced
     - 66.9%
     - 0.66
     - 1,984 (98.8% correct)
     - 0
     - 16 (25.0% correct)

"Corr. AUC" is the ROC AUC of correspondence as a predictor of a correct
prediction. The hardware trojan and credit-card fraud rows are measured on
2,000 test samples that include every positive in the test split (63 trojans,
148 frauds). Positives are therefore over-represented, by about 8.5 times for
trojan and 43 times for fraud, and per-label accuracy on a natural stream of
cases would be higher.

The default thresholds are well ordered. On the balanced domains a ``high``
label is right 99% to 100% of the time and a ``low`` label about half the time
or less, so the labels can be used as they are.

Imbalanced data: use balanced class weights
-------------------------------------------

Without weights, a rare-class neighbor counts no more than a common-class one.
On imbalanced data that mislabels two kinds of prediction:

* A missed positive with one or two positive neighbors among five still scores
  ``medium`` or ``high``, although a positive neighbor is strong evidence. Of
  1,852 correctly classified legitimate transactions, only 4 have any fraud
  neighbor.
* A correctly caught positive with a few negative neighbors scores ``low``.

Pass ``class_weights="balanced"`` to weight each class by
``n_samples / (n_classes * count)``:

.. code-block:: python

   explainer = CaseExplainer(X_train, y_train, class_weights="balanced")
   explainer.class_weights     # e.g. {0: 0.5, 1: 289.8}

On credit-card fraud the ``low`` label goes from 9 errors in 21 cases to 12
errors in 16, the ambiguous ``medium`` band empties, and mean correspondence on
the model's errors falls from 84% to 67%. On hardware trojan about the same
errors are flagged either way (25 of 49 unweighted, 24 with weights); the
weights only merge ``medium`` into ``low``. The fraud and hardware trojan
tutorial notebooks walk through this comparison on their own test samples;
the trojan one, drawn at random rather than enriched with positives, also shows
the extra review flags balanced weights raise on correct negative predictions
that have a positive neighbor.

Weights relabel predictions far more than they re-rank them. The AUC moves by
less than 0.01 in every retrieval space, and the retrieved neighbors do not
change.

What calibration cannot fix
---------------------------

A wrong prediction whose neighbors all share its class scores 100% under every
weighting and every threshold. The model and its precedents are wrong
together.

On the reference domains this is half to two thirds of the errors: 24 of the
36 missed frauds and 24 of the 48 missed trojans have five negative neighbors.
As a result 1.2% to 1.3% of ``high`` predictions on those subsets are wrong.

Read a ``high`` label as "consistent with the training data". Use the ``low``
label as a list of predictions worth a second look, and do not treat ``high``
as verification.

Choosing thresholds for your domain
-----------------------------------

The thresholds are configurable with ``correspondence_thresholds=(medium,
high)``. Before changing them, measure how the current labels behave on
held-out data with known outcomes:

.. code-block:: python

   import numpy as np

   explanations = explainer.explain_batch(X_val, y_test=y_val, model=clf)
   labels = np.array([e.correspondence_interpretation for e in explanations])
   correct = np.array([e.is_correct() for e in explanations])

   for label in ("high", "medium", "low"):
       mask = labels == label
       if mask.any():
           print(f"{label}: {mask.sum()} predictions, {correct[mask].mean():.1%} correct")

Raise the thresholds if ``high`` predictions are wrong more often than the
application tolerates. If the validation set has few errors, include every
available minority-class case; a uniform sample of a heavily imbalanced set
can contain no positives at all.

Comparing retrieval spaces
--------------------------

Model-informed spaces (hidden activations, tree leaves, forest proximity)
retrieve cases the model already treats alike, so the neighbors tend to agree
with the prediction even when it is wrong. On the two balanced domains this
makes their correspondence higher and less diagnostic than feature space: on
breast cancer, feature-space correspondence separates correct from incorrect
predictions with AUC 0.93 and forest proximity with 0.62.

The pattern does not hold everywhere. On the two imbalanced domains every
space reports about 99% correspondence and the AUCs lie within 0.08 of each
other, with hidden activations slightly ahead of feature space. Check the
spaces on your own data before choosing one for error detection.

Raw distances are also on different scales in different spaces. Pass
``distance_normalization="median_nn"`` when distances or correspondence values
from different spaces are compared; see
:ref:`comparing-correspondence-across-spaces`.

Reproducing the figures
-----------------------

.. code-block:: bash

   python validate_similarity_modes.py

Credit-card fraud needs ``creditcard.csv`` in the project root. The script
writes ``benchmarks/similarity_validation.csv`` with one row per domain,
retrieval space, and variant.
