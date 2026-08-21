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

with an explicit retrieval strategy:

.. code-block:: python

   explainer = CaseExplainer(
       X_train,
       y_train,
       retrieval=HiddenActivationRetrieval(
           model=mlp,
           layer="last_hidden",
           blend_alpha=0.25,
       ),
   )

For a custom extractor, migrate its behavior into an explicit retrieval
strategy before upgrading to 0.3.0:

.. code-block:: python

   explainer = CaseExplainer(
       X_train,
       y_train,
       retrieval=CustomActivationRetrieval(
           model=model,
           extractor=CallableActivationExtractor(custom_activation_function),
           blend_alpha=0.25,
       ),
   )

``CustomActivationRetrieval`` accepts any ``ActivationExtractor`` implementation
and preserves the existing feature/activation blend geometry without relying on
deprecated constructor parameters.