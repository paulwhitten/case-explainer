---
title: Retrieval Benchmark Baselines
description: Reproducible reference results for activation and forest retrieval
---

## Reference environment

The checked-in baseline was captured on 2026-11-10 with:

* Debian GNU/Linux 13 on x86-64
* Python 3.12.14
* scikit-learn 1.9.1
* 300 and 600 generated samples (240 and 480 training samples)
* 20 and 40 features
* 2, 3, and 5 classes
* 32 hidden units
* 50 forest estimators
* 2 untimed warm-up queries per strategy
* 15 trials of 3 timed queries per configuration (CI candidates use 5)

The baseline was regenerated for 0.3.0 because single-sample queries now run
on one thread, which changed how small-batch latency scales with training
size. The earlier baseline's scaling ratios were dominated by worker
scheduling overhead rather than search time.

Each matrix row changes one dimension from the first sample, feature, and class
value. Timings are local reference measurements, not cross-machine performance
guarantees.

## Reproduce the baseline

```bash
python benchmark.py \
  --retrieval-strategies \
  --retrieval-matrix \
  --retrieval-samples 300 \
  --retrieval-features 20 40 \
  --retrieval-classes 2 3 5 \
  --batch-size 3 \
  --retrieval-warmups 2 \
  --retrieval-repetitions 15 \
  --output benchmarks/retrieval_baseline.csv
```

The output records initialization time, mean query latency, and peak Python
memory for predicted-class activation and exact forest-proximity retrieval.
Query latency is the median trial time per explanation, and
``query_time_std_ms`` records dispersion across timed trials.

## Regression policy

CI regenerates the same experiment matrix and compares normalized scaling
relationships instead of absolute timings. For each strategy, query latency
and peak memory are divided by that run's smallest experiment. This allows
different runner speeds while still detecting disproportionate growth as
sample, feature, or class counts increase.

Query scaling permits a 50% relative margin plus three times the combined
coefficient of variation from the baseline and candidate trials. Memory
scaling permits a 25% relative margin. Initialization scaling permits a 100%
relative margin because fit time is currently measured once and has no trial
dispersion. Missing experiments, duplicate keys, fewer than three repetitions,
and non-finite, negative, or zero-valued ratio metrics fail the gate.

Run the same comparison locally after generating a candidate:

```bash
python benchmark.py \
  --baseline benchmarks/retrieval_baseline.csv \
  --check-regression retrieval_candidate.csv
```

CI uploads ``retrieval_candidate.csv`` on every run so a failed relationship
can be inspected without rerunning the benchmark.

## Retrieval-space validation

``similarity_validation.csv`` records every retrieval space on the reference
domains (Iris, UCI Breast Cancer, hardware trojan detection; credit-card fraud
when ``creditcard.csv`` is present). Each space is run three ways, identified
by the ``normalization`` and ``class_weights`` columns: raw distances,
``median_nn``-normalized distances, and raw distances with
``class_weights="balanced"``. Each row reports the model's accuracy on the
explained samples, mean correspondence, mean correspondence on correct and
incorrect predictions, the ROC AUC of correspondence as a predictor of
correctness, the mean distance to the nearest retrieved case, the normalization
scale, and batch time per sample. The ``n_high``/``acc_high``,
``n_medium``/``acc_medium``, and ``n_low``/``acc_low`` columns give, for each
interpretation label at the default thresholds, how many predictions received
it and the share of those the model got right.

Regenerate it with:

```bash
python validate_similarity_modes.py            # writes benchmarks/similarity_validation.csv
python validate_similarity_modes.py --domains iris breast_cancer --max-test 500
```

Splits are 70/30 stratified with ``random_state=42``. Large domains explain a
2,000-row test subset that keeps every minority-class test sample and fills the
remainder with a random sample of the majority class, so the imbalanced domains
are judged on all of their positives. Credit-card fraud needs ``creditcard.csv``
in the project root (the Kaggle ``mlg-ulb/creditcardfraud`` release, or the
identical OpenML dataset 1597 with its ``Time`` column retained). Timings are
single-run local measurements.
