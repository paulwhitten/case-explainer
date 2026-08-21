---
title: Case Explainer Changelog
description: Release history and migration schedule for Case Explainer
---

## 0.2.0

* Added explicit hidden-activation, decision-tree leaf, and random-forest
  proximity retrieval strategies
* Added predicted-class activation weighting for multiclass classifiers
* Added support for arbitrary hashable classifier labels
* Added ``CustomActivationRetrieval`` as the explicit replacement for custom
  legacy activation extractors
* Deprecated the legacy `activation_extractor`, `activation_layer`,
  `blend_alpha`, and `use_output_weights` constructor parameters

The legacy activation parameters remain functional throughout the `0.2.x`
series. Use `HiddenActivationRetrieval` for new integrations.

### Removal checklist for 0.3.0

* Remove the deprecated constructor parameters and compatibility translation
* Remove tests that assert legacy and strategy-based retrieval equivalence
* Retain a clear `TypeError` for removed parameter usage when practical
* Update examples and API reference pages to show only retrieval strategies
* Add a migration note with before-and-after constructor examples
* Verify the package version, runtime version, and Sphinx release stay aligned

The runtime warning contract is centralized in ``case_explainer._compat`` and
the migration examples are maintained in ``docs/migration.rst``. Complete the
checklist only when preparing the 0.3.0 release; version 0.2.x must continue to
accept the deprecated parameters.

## 0.1.1

* Published the initial general-purpose feature-space case explainer
