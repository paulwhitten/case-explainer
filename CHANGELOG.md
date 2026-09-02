---
title: Case Explainer Changelog
description: Release history and migration schedule for Case Explainer
---

## 0.2.1 (2026-09-02)

* Added an optional ``model`` field to ``Features`` so a prediction model can be
  bound once at construction, giving the input-space and activation-space halves
  of a comparison an identical ``explain_instance`` call signature. Call-time
  ``model=`` and ``predicted_class=`` continue to work and take precedence.
* Added ``CaseExplainer.compute_correspondence(explanation)``, a discoverable
  method that scores an explanation using the constructor's ``class_weights``
  (the module-level function remains available).

## 0.2.0 (2026-09-02)

* Added the unified ``similarity=`` constructor parameter as the single knob
  for retrieval mode, accepting a string shorthand (``"features"``,
  ``"activations"``) or a strategy object
* Added intent-named similarity strategies: ``Features``, ``HiddenActivations``,
  ``CustomActivations``, ``TreeLeaf``, ``ForestProximity``, and ``Blend``
* Added predicted-class activation weighting for multiclass classifiers
* Added support for arbitrary hashable classifier labels
* Added a PEP 561 ``py.typed`` marker so downstream users receive the package's
  type hints
* Deprecated the ``retrieval=`` parameter and the ``*Retrieval`` configuration
  classes in favor of ``similarity=`` and the strategy classes above
* Deprecated the legacy `activation_extractor`, `activation_layer`,
  `blend_alpha`, and `use_output_weights` constructor parameters

The deprecated parameters and classes remain functional throughout the `0.2.x`
series. Use `similarity=` with the strategy classes for new integrations.

### Removal checklist for 0.3.0

* Remove the deprecated constructor parameters (`activation_extractor`,
  `activation_layer`, `blend_alpha`, `use_output_weights`) and the
  ``retrieval=`` parameter along with the ``*Retrieval`` classes
* Remove tests that assert legacy and strategy-based equivalence
* Retain a clear `TypeError` for removed parameter usage when practical
* Update examples and API reference pages to show only ``similarity=``
* Consider moving ``compute_correspondence`` and ``euclidean_distance`` into the
  ``case_explainer.metrics`` submodule and dropping them from the top-level
  ``__all__`` (breaking; batch with this release)
* Add a migration note with before-and-after constructor examples
* Verify the package version, runtime version, and Sphinx release stay aligned

The runtime warning contract is centralized in ``case_explainer._compat`` and
the migration examples are maintained in ``docs/migration.rst``. Complete the
checklist only when preparing the 0.3.0 release; version 0.2.x must continue to
accept the deprecated parameters.

## 0.1.1

* Published the initial general-purpose feature-space case explainer
