Migrating to 0.3.0
==================

Case Explainer 0.3.0 removes the constructor parameters and classes that 0.2.x
deprecated. Passing a removed parameter raises ``TypeError`` with a hint naming
the replacement, so an outdated call fails immediately rather than silently
changing behavior.

Removed constructor parameters
------------------------------

.. list-table::
   :header-rows: 1

   * - Removed in 0.3.0
     - Use instead
   * - ``activation_extractor=ext, model=m``
     - ``similarity=CustomActivations(model=m, extractor=ext)``
   * - ``activation_layer=layer, model=mlp``
     - ``similarity=HiddenActivations(model=mlp, layer=layer)``
   * - ``use_output_weights=False``
     - ``similarity=HiddenActivations(..., output_weighting="none")``
   * - ``blend_alpha=a``
     - ``similarity=Blend(strategy, features=a)``
   * - ``retrieval=HiddenActivationRetrieval(...)``
     - ``similarity=HiddenActivations(...)``
   * - ``retrieval=CustomActivationRetrieval(...)``
     - ``similarity=CustomActivations(...)``
   * - ``retrieval=TreeLeafRetrieval(...)``
     - ``similarity=TreeLeaf(...)``
   * - ``retrieval=ForestProximityRetrieval(...)``
     - ``similarity=ForestProximity(...)``

The ``model=`` constructor argument remains. It supplies the prediction model
for feature-space retrieval and for the ``similarity="activations"`` string
shorthand; strategy objects carry their own model.

Before and after
----------------

Hidden-activation retrieval with a feature blend:

.. code-block:: python

   # 0.2.x (deprecated)
   explainer = CaseExplainer(
       X_train, y_train,
       model=mlp,
       activation_layer="last_hidden",
       blend_alpha=0.25,
   )

   # 0.3.0
   from case_explainer import Blend, HiddenActivations

   explainer = CaseExplainer(
       X_train, y_train,
       similarity=Blend(
           HiddenActivations(model=mlp, layer="last_hidden"),
           features=0.25,
       ),
   )

A custom extractor:

.. code-block:: python

   # 0.2.x (deprecated)
   explainer = CaseExplainer(
       X_train, y_train,
       model=model,
       activation_extractor=CallableActivationExtractor(fn),
   )

   # 0.3.0
   from case_explainer import CallableActivationExtractor, CustomActivations

   explainer = CaseExplainer(
       X_train, y_train,
       similarity=CustomActivations(
           model=model, extractor=CallableActivationExtractor(fn)
       ),
   )

Unweighted activations:

.. code-block:: python

   # 0.2.x (deprecated)
   CaseExplainer(X_train, y_train, model=mlp, activation_layer=0, use_output_weights=False)

   # 0.3.0
   CaseExplainer(
       X_train, y_train,
       similarity=HiddenActivations(model=mlp, layer=0, output_weighting="none"),
   )

Other changes to check
----------------------

* ``from case_explainer import euclidean_distance`` no longer works; import it
  from ``case_explainer.metrics``.
* ``case_explainer.indexing`` has been removed. Use the ``algorithm=``
  constructor argument (``'auto'``, ``'kd_tree'``, ``'ball_tree'``,
  ``'brute'``), which is passed to sklearn's ``NearestNeighbors``.
* ``ForestProximity`` now rejects ``metric=`` values other than the default,
  and ``TreeLeaf`` now honors ``metric=`` when ordering same-leaf cases.
* ``explain_batch`` calls ``model.predict`` once per batch instead of once per
  sample. A model wrapper that returns a single prediction regardless of input
  size must be updated to return one prediction per row.
* The hybrid retrieval label in ``repr(explainer)`` reads
  ``hybrid(features=0.30)`` instead of ``hybrid(alpha=0.30)``.
* ``Explanation.to_dict()`` gains ``retrieval_space``, ``distance_metric``,
  and ``distance_scale`` keys, and ``get_training_info()`` gains
  ``retrieval_space``, ``distance_normalization``, ``distance_scale``, and
  ``correspondence_thresholds``. Code that asserts an exact key set needs
  updating.
* ``class_weights`` accepts ``"balanced"`` in addition to a mapping. Any other
  string, including the empty string, now raises ``ValueError``. On imbalanced
  data, switching to ``class_weights="balanced"`` changes correspondence
  scores and labels (not the retrieved neighbors); see :doc:`calibration`.
* Arguments are validated more strictly. ``k`` must be a positive integer,
  ``feature_names`` must have one name per feature, and labels or predictions
  must be one value per sample. A model whose ``predict`` returns class
  probabilities needs a wrapper that returns class labels.
* ``HiddenActivations`` with an integer ``layer`` that names the last hidden
  layer now applies output weighting, as ``layer="last_hidden"`` does. Pass
  ``output_weighting="none"`` to keep the previous unweighted retrieval.
* An explainer built from a DataFrame now passes a DataFrame with the training
  columns to ``model.predict``, ``model.apply``, ``input_transform``, and the
  built-in extractors, where 0.2.x passed bare arrays. Estimators fitted on a
  DataFrame gain from this; an estimator fitted on arrays and used with a
  DataFrame-built explainer now receives frames and may emit scikit-learn's
  "X has feature names" warning. Build the explainer from the same kind of
  input the model was fitted on. DataFrame queries are also reordered to the
  training columns by name, and a frame with missing or unexpected columns
  raises ``ValueError`` instead of being compared positionally. Custom
  extractor callables still receive arrays.
* The documentation no longer describes privacy-preserving features as
  planned. Explanations and saved explainers contain training data and must be
  handled as such; see :doc:`data_exposure`.
