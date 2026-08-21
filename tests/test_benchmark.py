"""Focused tests for retrieval strategy benchmark orchestration."""

import pandas as pd
import pytest

from benchmark import (
    benchmark_retrieval_matrix,
    compare_retrieval_results,
    save_retrieval_results,
)


def test_retrieval_matrix_varies_one_dimension_at_a_time(tmp_path):
    results = benchmark_retrieval_matrix(
        sample_sizes=[80, 100],
        feature_counts=[6, 8],
        class_counts=[2, 3],
        hidden_width=4,
        n_estimators=2,
        n_queries=1,
        warmup_queries=1,
        repetitions=3,
    )
    shapes = {
        (result.n_samples, result.n_features, result.n_classes)
        for result in results
    }
    assert len(results) == 8
    assert len(shapes) == 4
    assert all(result.repetitions == 3 for result in results)
    assert all(result.query_time > 0 for result in results)
    assert all(result.query_time_std >= 0 for result in results)

    output = tmp_path / "retrieval.csv"
    save_retrieval_results(results, str(output))
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "strategy,n_samples,n_features,n_classes,fit_time_s,"
        "query_time_ms,query_time_std_ms,repetitions,peak_memory_mb"
    )


def _write_comparison_csv(
    path, query_times=(1.0, 2.0), fit_times=(0.1, 0.2)
):
    rows = []
    for strategy in ("activation_predicted_class", "forest_proximity"):
        for n_samples, query_time, fit_time in zip(
            (100, 200), query_times, fit_times
        ):
            rows.append({
                "strategy": strategy,
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "query_time_ms": query_time,
                "query_time_std_ms": 0.01,
                "repetitions": 5,
                "peak_memory_mb": n_samples / 100,
                "fit_time_s": fit_time,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_regression_comparison_uses_relative_scaling(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_comparison_csv(baseline)
    _write_comparison_csv(candidate, query_times=(10.0, 20.0))
    assert compare_retrieval_results(str(baseline), str(candidate)) == []


def test_regression_comparison_reports_scaling_regression(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_comparison_csv(baseline)
    _write_comparison_csv(candidate, query_times=(1.0, 4.0))
    findings = compare_retrieval_results(str(baseline), str(candidate))
    assert len(findings) == 2
    assert all("query scaling regressed" in finding for finding in findings)


def test_regression_comparison_rejects_missing_experiments(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_comparison_csv(baseline)
    _write_comparison_csv(candidate)
    frame = pd.read_csv(candidate).iloc[:-1]
    frame.to_csv(candidate, index=False)
    with pytest.raises(ValueError, match="experiment keys differ"):
        compare_retrieval_results(str(baseline), str(candidate))


def test_regression_comparison_reports_fit_scaling_regression(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_comparison_csv(baseline)
    _write_comparison_csv(candidate, fit_times=(0.1, 0.5))
    findings = compare_retrieval_results(str(baseline), str(candidate))
    assert len(findings) == 2
    assert all("fit scaling regressed" in finding for finding in findings)