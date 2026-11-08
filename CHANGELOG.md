---
title: Case Explainer Changelog
description: Release history and migration schedule for Case Explainer
---

## 0.3.0 (2026-11-08)

Breaking release that completes the ``similarity=`` migration started in 0.2.0.

### Removed

* The deprecated constructor parameters ``activation_extractor``,
  ``activation_layer``, ``blend_alpha``, ``use_output_weights``, and
  ``retrieval``. Passing any of them now raises ``TypeError`` with a migration
  hint pointing at the equivalent ``similarity=`` strategy.
* The ``HiddenActivationRetrieval``, ``CustomActivationRetrieval``,
  ``TreeLeafRetrieval``, and ``ForestProximityRetrieval`` classes. The
  ``Features``, ``HiddenActivations``, ``CustomActivations``, ``TreeLeaf``,
  ``ForestProximity``, and ``Blend`` dataclasses are now the internal
  representation as well as the public API; ``CaseExplainer.similarity`` holds
  the resolved strategy.
* The unused ``case_explainer.indexing`` module. Nearest-neighbor search has
  always gone through sklearn's ``NearestNeighbors``; the ``algorithm=``
  constructor argument is the supported knob.
* ``euclidean_distance`` from the top-level namespace. It and
  ``compute_all_distances`` remain available from ``case_explainer.metrics``.

### Added

* ``Explanation.retrieval_space``, ``Explanation.distance_metric``, and
  ``Explanation.distance_scale`` record how the neighbors were retrieved, so
  the meaning of ``Neighbor.distance`` travels with the explanation. They
  appear in ``summary()``, ``to_dict()``, and ``repr()``;
  ``Explanation.describe_retrieval_space()`` renders them as one line.
  ``CaseExplainer.retrieval_space`` exposes the same label.
* ``CaseExplainer(correspondence_thresholds=(medium, high))`` makes the
  ``"low"`` / ``"medium"`` / ``"high"`` interpretation cut points
  configurable. ``compute_correspondence`` accepts ``thresholds=``, and
  ``case_explainer.metrics`` gains ``interpret_correspondence``,
  ``validate_thresholds``, and ``DEFAULT_THRESHOLDS``.
* ``CaseExplainer(distance_normalization="median_nn")`` divides every reported
  distance by the median distance from a training case to its nearest distinct
  training case in the active retrieval space (duplicates are excluded; the
  divisor is stored as ``distance_scale_``). Neighbor selection is unchanged;
  correspondence becomes comparable across retrieval spaces. The default
  remains raw distances.
* ``validate_similarity_modes.py`` re-runs the reference domains in every
  retrieval space, with raw and normalized distances, and writes
  ``benchmarks/similarity_validation.csv``.

### Changed

* ``explain_batch`` is vectorized: one scaler/extractor transform, one
  ``predict`` call, and one k-NN query per batch (one per predicted class for
  predicted-class activation weighting; chunked proximity for forests).
  Results are identical to ``explain_instance``; feature-space batches run
  about two orders of magnitude faster. ``explain_batch`` now validates the
  lengths of ``y_test`` and ``predictions`` and the number of predictions a
  model returns.
* ``metric=`` is honored by ``TreeLeaf`` when ordering same-leaf cases
  (previously always Euclidean). ``ForestProximity`` raises ``ValueError`` for
  any ``metric`` other than the default, since it defines its own distance.
* ``explain_instance`` accepts a single-row 2-D array as well as a 1-D sample.
* Hybrid retrieval is labelled ``hybrid(features=0.30)`` (was
  ``hybrid(alpha=0.30)``) in ``repr`` and ``retrieval_space``.
* ``get_training_info()`` reports ``retrieval_space``,
  ``distance_normalization``, ``distance_scale``, and
  ``correspondence_thresholds``.

## 0.2.1 (2026-09-02)

* Added an optional ``model`` field to ``Features`` so a prediction model can be
  bound once at construction, giving the input-space and activation-space halves
  of a comparison an identical ``explain_instance`` call signature. Call-time
  ``model=`` and ``predicted_class=`` continue to work and take precedence.
* Added ``CaseExplainer.compute_correspondence(explanation)``, a discoverable
  method that scores an explanation using the constructor's ``class_weights``
  (the module-level function remains available).
* Changed the default ``Explanation.plot()`` type from ``"radar"`` to
  ``"bar"``; radar and parallel plots are not implemented and previously fell
  back to the bar plot by default.
* Documentation: the README API overview now shows ``similarity=`` instead of
  the deprecated ``retrieval=``; documented that neighbor distance scales and
  therefore correspondence values are not comparable across retrieval spaces,
  and that ``metric=`` applies to feature and activation retrieval only;
  corrected the hardware trojan correspondence and feature count in the docs.
* CI: test on Python 3.13 and 3.14.

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

The removal checklist for these parameters was completed in 0.3.0; see that
entry and ``docs/migration.rst`` for before-and-after examples.

## 0.1.1

* Published the initial general-purpose feature-space case explainer
