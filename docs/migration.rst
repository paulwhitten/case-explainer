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
