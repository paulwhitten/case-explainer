---
title: Retrieval Benchmark Baselines
description: Reproducible reference results for activation and forest retrieval
---

## Reference environment

The checked-in baseline was captured on 2026-08-05 with:

* macOS 26.5.2 on x86-64
* Python 3.9.6
* scikit-learn 1.6.1
* 300 and 600 generated samples (240 and 480 training samples)
* 20 and 40 features
* 2, 3, and 5 classes
* 32 hidden units
* 50 forest estimators
* 2 untimed warm-up queries per strategy
* 5 trials of 3 timed queries per configuration

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
