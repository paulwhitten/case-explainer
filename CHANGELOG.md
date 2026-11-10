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
  the resolved strategy. The ``retrieval``, ``activation_extractor``, and
  ``blend_alpha`` attributes of an explainer are gone with them.
* The unused ``case_explainer.indexing`` module. Nearest-neighbor search has
  always gone through sklearn's ``NearestNeighbors``; the ``algorithm=``
  constructor argument is the supported knob.
* ``euclidean_distance`` from the top-level namespace. It and
  ``compute_all_distances`` remain available from ``case_explainer.metrics``.

### Added

* ``Explanation.retrieval_space``, ``Explanation.distance_metric``, and
  ``Explanation.distance_scale`` record how the neighbors were retrieved, so
  the meaning of ``Neighbor.distance`` travels with the explanation. They
  appear in ``summary()`` and ``to_dict()``, and ``repr()`` shows the space;
  ``Explanation.describe_retrieval_space()`` renders them as one line.
  ``CaseExplainer.retrieval_space`` exposes the same label.
* ``Neighbor`` is exported from the top-level package.
* ``CaseExplainer(correspondence_thresholds=(medium, high))`` makes the
  ``"low"`` / ``"medium"`` / ``"high"`` interpretation cut points
  configurable. ``compute_correspondence`` accepts ``thresholds=``, and
  ``case_explainer.metrics`` gains ``interpret_correspondence``,
  ``validate_thresholds``, and ``DEFAULT_THRESHOLDS``.
* ``CaseExplainer(distance_normalization="median_nn")`` divides every reported
  distance by the median nearest-neighbor distance among training cases in the
  active retrieval space (the divisor is stored as ``distance_scale_``).
  Training cases with an exact duplicate are left out of the estimate; if every
  sampled case has one, distances stay unnormalized and a warning is issued.
  Neighbor selection is unchanged; distances from different retrieval spaces
  land on one scale. The default remains raw distances.
* ``CaseExplainer(class_weights="balanced")`` derives inverse-frequency class
  weights from ``y_train`` (``n_samples / (n_classes * count)``). Recommended
  for imbalanced data, where unweighted correspondence lets a majority-class
  prediction score "medium" or "high" despite minority-class neighbors. The
  resolved mapping is stored in ``class_weights``; any other string raises
  ``ValueError``.
* ``validate_similarity_modes.py`` re-runs the reference domains (Iris, breast
  cancer, hardware trojan, credit-card fraud) in every retrieval space, with
  raw distances, normalized distances, and balanced class weights, and writes
  ``benchmarks/similarity_validation.csv``. Each row also reports how many
  predictions received each interpretation label and how many of those were
  correct. On the imbalanced domains the 2,000-row test subset keeps every
  minority-class sample, so accuracy and the correspondence AUC are measured
  on real positives rather than on the handful a uniform sample would contain.
* ``class_names`` accepts a sequence of names matched to the sorted class
  labels (for example scikit-learn's ``target_names``) as well as a mapping.
  ``metadata`` accepts a DataFrame as well as a dict of sequences.
* ``Explanation.plot(show_distances=True)`` now shows each neighbor's distance
  in the legend; the parameter was previously accepted and ignored.
* DataFrame support. An explainer built from a DataFrame remembers the
  training columns, aligns later DataFrame or Series queries to them by name
  (missing or extra columns raise ``ValueError``), and hands the model,
  ``input_transform``, and built-in extractors a DataFrame with those columns
  instead of a bare array. Pipelines that select columns by name now work as
  ``model=``, and scikit-learn no longer warns about missing feature names.
  Custom extractor callables still receive arrays.

### Changed

* ``explain_batch`` is vectorized: one scaler/extractor transform, one
  ``predict`` call, and one k-NN query per batch (one per predicted class for
  predicted-class activation weighting; chunked proximity for forests).
  Results match ``explain_instance`` up to floating-point rounding (in
  activation spaces an exact tie between duplicate training rows can resolve
  to a different duplicate); feature-space batches run about two orders of
  magnitude faster. An empty batch returns an empty list. ``explain_batch`` now validates the
  lengths of ``y_test`` and ``predictions`` and the number of predictions a
  model returns.
* ``metric=`` is honored by ``TreeLeaf`` when ordering same-leaf cases
  (previously always Euclidean). ``ForestProximity`` raises ``ValueError`` for
  any ``metric`` other than the default, since it defines its own distance.
* ``explain_instance`` accepts a single-row 2-D array as well as a 1-D sample.
* Inputs are validated earlier and with clearer errors. ``k`` must be a
  positive integer, at construction and per call (a negative ``k`` used to
  return most of the training set in tree and forest spaces).
  ``feature_names`` must have one name per feature. ``X_train`` must be 2-D.
  ``y_train``, ``y_test``, ``predictions``, and a model's ``predict`` output
  must hold one label per sample; a single column is flattened and anything
  wider, such as class probabilities, raises ``ValueError``. ``class_weights``
  must map to non-negative finite numbers, and entries for labels absent from
  ``y_train`` produce a warning. ``TreeLeaf`` validates ``overflow`` when the
  strategy is created.
* ``HiddenActivations(layer=<index of the last hidden layer>)`` now applies
  output weighting exactly as ``layer="last_hidden"`` does. Previously any
  integer layer silently used uniform weights.
* ``TreeLeaf`` batches compute distances one block at a time, so memory no
  longer grows with ``n_queries * n_train``.
* Single-sample queries run the k-NN search on one thread regardless of
  ``n_jobs``. With the default ``n_jobs=-1``, ``explain_instance`` on a tree
  index spent about 13 ms scheduling workers for under 1 ms of search; it now
  takes about 1 ms. Batches still use all requested workers. The retrieval
  performance baseline in ``benchmarks/`` was regenerated to match.
* ``Explanation.plot(save_path=...)`` closes its figure after saving.
* ``Blend(..., features=1.0)`` is labelled ``features`` for every output
  weighting.
* Hybrid retrieval is labelled ``hybrid(features=0.30)`` (was
  ``hybrid(alpha=0.30)``) in ``repr`` and ``retrieval_space``.
* ``get_training_info()`` reports ``retrieval_space``,
  ``distance_normalization``, ``distance_scale``, and
  ``correspondence_thresholds``.
* The README's validated-domain figures are regenerated from
  ``validate_similarity_modes.py``. The credit-card fraud figure of 100%
  correspondence is replaced: it was measured on 100 test rows containing no
  fraud. The fraud imbalance ratio is corrected to 578:1 (the old figure
  belonged to the hardware trojan data, which is 269:1).
* The README and documentation no longer say privacy-preserving features are
  planned. They now state what an explanation and a saved explainer disclose
  (real training rows, labels, row positions, metadata), that the library is
  for audiences entitled to see the training data, and why output masking
  would not be a privacy guarantee. New documentation pages: "Data Exposure
  and Privacy" and "Interpreting and Calibrating Correspondence".

### Fixed

* ``feature_names`` or ``class_names`` given as a NumPy array or pandas Index
  (for example ``load_breast_cancer().feature_names``) raised ``ValueError``.
* ``metadata`` values given as a pandas Series were looked up by index label
  instead of row position, so a Series that had been shuffled or split
  returned another row's metadata or raised ``KeyError``. Metadata is now
  copied into plain lists by position, and NumPy values become Python scalars.
* ``Explanation.to_dict()`` was not JSON-serializable when ``test_index`` or a
  metadata value was a NumPy scalar.
* A bound model whose ``len()`` is zero was dropped, and an unfitted forest
  raised an unrelated ``AttributeError`` instead of the intended message.
* ``benchmark.py`` looked for ``creditcard.csv`` at a hardcoded absolute path
  on the author's machine; it now looks next to the script.
* The iris and breast cancer tutorial notebooks failed to run: one passed a
  ``boxplot`` argument that matplotlib removed, the other read a private
  explainer attribute. The fraud tutorial labelled every feature with its
  neighbor's name. All four notebooks now execute on this release, and a test
  guards them against removed or private API.
* The fraud and hardware trojan tutorials now demonstrate
  ``class_weights="balanced"`` alongside the default weighting they use for
  their main path, showing which missed positives the balanced score flags,
  which it cannot, and the extra review flags it produces on a random sample.
  A test checks that the executed comparison stays in both notebooks.
* The three core formulas on the metrics documentation page rendered as raw
  text, and several cross-references did not resolve.
* ``MANIFEST.in`` packaged every Markdown file under ``docs/``; it now names
  the two that belong in the source distribution.

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
