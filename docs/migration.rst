Legacy Activation Migration
===========================

Case Explainer 0.2 supports the legacy activation constructor parameters while
emitting :class:`DeprecationWarning`. Version 0.3.0 will remove
``activation_extractor``, ``activation_layer``, ``blend_alpha``, and
``use_output_weights`` from :class:`case_explainer.CaseExplainer`.

Replace the shorthand constructor:

.. code-block:: python

   explainer = CaseExplainer(
       X_train,
       y_train,
       model=mlp,
       activation_layer="last_hidden",
       blend_alpha=0.25,
   )

with an explicit similarity strategy:

.. code-block:: python

   explainer = CaseExplainer(
       X_train,
       y_train,
       similarity=Blend(
           HiddenActivations(model=mlp, layer="last_hidden"),
           features=0.25,
       ),
   )

For a custom extractor, migrate its behavior into an explicit similarity
strategy before upgrading to 0.3.0:

.. code-block:: python

   explainer = CaseExplainer(
       X_train,
       y_train,
       similarity=Blend(
           CustomActivations(
               model=model,
               extractor=CallableActivationExtractor(custom_activation_function),
           ),
           features=0.25,
       ),
   )

``CustomActivations`` accepts any ``ActivationExtractor`` implementation and,
wrapped in :class:`~case_explainer.Blend`, preserves the existing
feature/activation blend geometry without relying on deprecated constructor
parameters.

Renamed ``retrieval=`` parameter
--------------------------------

The ``retrieval=`` parameter and the ``*Retrieval`` configuration classes
introduced in 0.2 are deprecated in favor of ``similarity=`` and the
intent-named strategy classes. The mapping is direct:

.. list-table::
   :header-rows: 1

   * - Deprecated
     - Preferred
   * - ``retrieval=HiddenActivationRetrieval(...)``
     - ``similarity=HiddenActivations(...)``
   * - ``retrieval=CustomActivationRetrieval(...)``
     - ``similarity=CustomActivations(...)``
   * - ``retrieval=TreeLeafRetrieval(...)``
     - ``similarity=TreeLeaf(...)``
   * - ``retrieval=ForestProximityRetrieval(...)``
     - ``similarity=ForestProximity(...)``
   * - ``...Retrieval(..., blend_alpha=a)``
     - ``similarity=Blend(strategy, features=a)``

The ``use_output_weights`` field is replaced by ``output_weighting``, where
``use_output_weights=False`` becomes ``output_weighting="none"``.