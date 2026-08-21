#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for case-explainer.

Tests performance across multiple datasets with varying characteristics:
- Iris: Small (150 samples, 4 features) - baseline
- Wine: Small (178 samples, 13 features) - more features
- Breast Cancer: Medium (569 samples, 30 features) - medical domain
- Digits: Medium (1797 samples, 64 features) - high dimensional
- MNIST subset: Large (10k+ samples, 784 features) - image data
- Hardware Trojan: Very Large (56k+ samples, 5 features) - real-world

Metrics collected:
- Fit time (explainer initialization)
- Single explanation time
- Batch explanation time
- Memory usage
- Correspondence scores
- Accuracy metrics

Compares indexing methods: kd_tree, ball_tree, brute force
"""

import sys
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, fetch_openml
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add case_explainer to path
sys.path.insert(0, os.path.dirname(__file__))
from case_explainer import (
    CaseExplainer,
    ForestProximityRetrieval,
    HiddenActivationRetrieval,
)


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single configuration."""
    dataset_name: str
    n_samples: int
    n_features: int
    n_classes: int
    index_method: str
    
    # Timing metrics (seconds)
    fit_time: float
    single_explain_time: float
    batch_explain_time: float
    time_per_explanation: float
    
    # Memory metrics (MB)
    memory_usage: float
    
    # Quality metrics
    mean_correspondence: float
    std_correspondence: float
    accuracy: float
    
    # Correspondence by correctness
    correct_correspondence: Optional[float] = None
    incorrect_correspondence: Optional[float] = None


@dataclass
class RetrievalStrategyBenchmark:
    """Timing and peak-memory results for one retrieval strategy."""

    strategy: str
    n_samples: int
    n_features: int
    n_classes: int
    fit_time: float
    query_time: float
    query_time_std: float
    repetitions: int
    peak_memory_mb: float


def benchmark_retrieval_strategies(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 3,
    hidden_width: int = 32,
    n_estimators: int = 50,
    n_queries: int = 25,
    warmup_queries: int = 2,
    repetitions: int = 5,
) -> List[RetrievalStrategyBenchmark]:
    """Benchmark query-dependent retrieval modes on deterministic data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(n_classes, n_features // 2),
        n_redundant=0,
        n_classes=n_classes,
        random_state=42,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_width,),
        max_iter=200,
        random_state=42,
    ).fit(X_train, y_train)
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)
    configurations = [
        (
            "activation_predicted_class",
            HiddenActivationRetrieval(
                model=mlp, output_weighting="predicted_class"
            ),
        ),
        ("forest_proximity", ForestProximityRetrieval(model=forest)),
    ]

    results = []
    query_count = min(n_queries, len(X_test))
    for strategy, retrieval in configurations:
        tracemalloc.start()
        start_time = time.perf_counter()
        explainer = CaseExplainer(X_train, y_train, retrieval=retrieval)
        fit_time = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        warmup_count = min(warmup_queries, query_count)
        for sample in X_test[:warmup_count]:
            explainer.explain_instance(sample)

        trial_times = []
        for _ in range(repetitions):
            start_time = time.perf_counter()
            for sample in X_test[:query_count]:
                explainer.explain_instance(sample)
            trial_times.append(
                (time.perf_counter() - start_time) / query_count
            )
        results.append(RetrievalStrategyBenchmark(
            strategy=strategy,
            n_samples=len(X_train),
            n_features=n_features,
            n_classes=n_classes,
            fit_time=fit_time,
            query_time=float(np.median(trial_times)),
            query_time_std=float(np.std(trial_times)),
            repetitions=repetitions,
            peak_memory_mb=peak / 1024 / 1024,
        ))
    return results


def benchmark_retrieval_matrix(
    sample_sizes: Iterable[int],
    feature_counts: Iterable[int],
    class_counts: Iterable[int],
    hidden_width: int = 32,
    n_estimators: int = 50,
    n_queries: int = 10,
    warmup_queries: int = 2,
    repetitions: int = 5,
) -> List[RetrievalStrategyBenchmark]:
    """Benchmark retrieval strategies across independent scaling dimensions."""
    sample_sizes = list(sample_sizes)
    feature_counts = list(feature_counts)
    class_counts = list(class_counts)
    if not sample_sizes or not feature_counts or not class_counts:
        raise ValueError("benchmark dimensions must not be empty")

    baseline = (sample_sizes[0], feature_counts[0], class_counts[0])
    configurations = {baseline}
    configurations.update(
        (value, baseline[1], baseline[2]) for value in sample_sizes
    )
    configurations.update(
        (baseline[0], value, baseline[2]) for value in feature_counts
    )
    configurations.update(
        (baseline[0], baseline[1], value) for value in class_counts
    )

    results = []
    for n_samples, n_features, n_classes in sorted(configurations):
        results.extend(benchmark_retrieval_strategies(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            hidden_width=hidden_width,
            n_estimators=n_estimators,
            n_queries=n_queries,
            warmup_queries=warmup_queries,
            repetitions=repetitions,
        ))
    return results


def save_retrieval_results(
    results: List[RetrievalStrategyBenchmark], filename: str
) -> None:
    """Write retrieval strategy benchmark results as machine-readable CSV."""
    pd.DataFrame([
        {
            "strategy": result.strategy,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "n_classes": result.n_classes,
            "fit_time_s": result.fit_time,
            "query_time_ms": result.query_time * 1000,
            "query_time_std_ms": result.query_time_std * 1000,
            "repetitions": result.repetitions,
            "peak_memory_mb": result.peak_memory_mb,
        }
        for result in results
    ]).to_csv(filename, index=False)


def compare_retrieval_results(
    baseline_file: str,
    candidate_file: str,
    relative_tolerance: float = 0.5,
    uncertainty_multiplier: float = 3.0,
    fit_tolerance: float = 1.0,
    memory_tolerance: float = 0.25,
) -> List[str]:
    """Compare normalized benchmark relationships across different machines."""
    key_columns = ["strategy", "n_samples", "n_features", "n_classes"]
    required_columns = key_columns + [
        "query_time_ms",
        "query_time_std_ms",
        "repetitions",
        "peak_memory_mb",
        "fit_time_s",
    ]
    frames = {
        "baseline": pd.read_csv(baseline_file),
        "candidate": pd.read_csv(candidate_file),
    }
    for name, frame in frames.items():
        missing = set(required_columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} benchmark is missing columns: {sorted(missing)}"
            )
        if frame.duplicated(key_columns).any():
            raise ValueError(f"{name} benchmark contains duplicate experiment keys")
        numeric = frame[required_columns[1:]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError(f"{name} benchmark contains invalid metric values")
        positive_columns = ["query_time_ms", "peak_memory_mb", "fit_time_s"]
        if (frame[positive_columns] <= 0).any().any():
            raise ValueError(f"{name} benchmark contains zero-valued metrics")
        if (frame["repetitions"] < 3).any():
            raise ValueError(f"{name} benchmark requires at least 3 repetitions")

    baseline = frames["baseline"].set_index(key_columns).sort_index()
    candidate = frames["candidate"].set_index(key_columns).sort_index()
    if not baseline.index.equals(candidate.index):
        missing = baseline.index.difference(candidate.index).tolist()
        extra = candidate.index.difference(baseline.index).tolist()
        raise ValueError(
            f"benchmark experiment keys differ; missing={missing}, extra={extra}"
        )

    findings = []
    for strategy in baseline.index.get_level_values("strategy").unique():
        baseline_strategy = baseline.xs(strategy, level="strategy")
        candidate_strategy = candidate.xs(strategy, level="strategy")
        anchor = baseline_strategy.index[0]
        baseline_anchor = baseline_strategy.loc[anchor]
        candidate_anchor = candidate_strategy.loc[anchor]
        for experiment in baseline_strategy.index[1:]:
            baseline_row = baseline_strategy.loc[experiment]
            candidate_row = candidate_strategy.loc[experiment]
            baseline_ratio = (
                baseline_row["query_time_ms"]
                / baseline_anchor["query_time_ms"]
            )
            candidate_ratio = (
                candidate_row["query_time_ms"]
                / candidate_anchor["query_time_ms"]
            )
            relative_uncertainty = uncertainty_multiplier * np.sqrt(
                (baseline_row["query_time_std_ms"]
                 / baseline_row["query_time_ms"]) ** 2
                + (candidate_row["query_time_std_ms"]
                   / candidate_row["query_time_ms"]) ** 2
                + (baseline_anchor["query_time_std_ms"]
                   / baseline_anchor["query_time_ms"]) ** 2
                + (candidate_anchor["query_time_std_ms"]
                   / candidate_anchor["query_time_ms"]) ** 2
            )
            allowed_ratio = baseline_ratio * (
                1.0 + relative_tolerance + relative_uncertainty
            )
            if candidate_ratio > allowed_ratio:
                findings.append(
                    f"{strategy} query scaling regressed at {experiment}: "
                    f"{candidate_ratio:.2f}x vs {baseline_ratio:.2f}x baseline"
                )

            baseline_memory_ratio = (
                baseline_row["peak_memory_mb"]
                / baseline_anchor["peak_memory_mb"]
            )
            candidate_memory_ratio = (
                candidate_row["peak_memory_mb"]
                / candidate_anchor["peak_memory_mb"]
            )
            if candidate_memory_ratio > baseline_memory_ratio * (
                1.0 + memory_tolerance
            ):
                findings.append(
                    f"{strategy} memory scaling regressed at {experiment}: "
                    f"{candidate_memory_ratio:.2f}x vs "
                    f"{baseline_memory_ratio:.2f}x baseline"
                )

            baseline_fit_ratio = (
                baseline_row["fit_time_s"] / baseline_anchor["fit_time_s"]
            )
            candidate_fit_ratio = (
                candidate_row["fit_time_s"] / candidate_anchor["fit_time_s"]
            )
            if candidate_fit_ratio > baseline_fit_ratio * (1.0 + fit_tolerance):
                findings.append(
                    f"{strategy} fit scaling regressed at {experiment}: "
                    f"{candidate_fit_ratio:.2f}x vs "
                    f"{baseline_fit_ratio:.2f}x baseline"
                )
    return findings


class DatasetLoader:
    """Load and prepare datasets for benchmarking."""
    
    @staticmethod
    def load_iris() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load Iris dataset (baseline small dataset)."""
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
        )
        return X_train, X_test, y_train, y_test, list(data.feature_names), "Iris"
    
    @staticmethod
    def load_wine() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load Wine dataset (small with more features)."""
        data = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
        )
        return X_train, X_test, y_train, y_test, list(data.feature_names), "Wine"
    
    @staticmethod
    def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load Breast Cancer dataset (medium size, medical domain)."""
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
        )
        return X_train, X_test, y_train, y_test, list(data.feature_names), "Breast Cancer"
    
    @staticmethod
    def load_digits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load Digits dataset (medium size, high dimensional)."""
        data = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
        )
        feature_names = [f"pixel_{i}" for i in range(data.data.shape[1])]
        return X_train, X_test, y_train, y_test, feature_names, "Digits (8x8)"
    
    @staticmethod
    def load_mnist_subset(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load MNIST subset (large dataset, high dimensional images)."""
        print(f"  Fetching MNIST (first {n_samples} samples, may take a moment)...")
        try:
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X = mnist.data.values if hasattr(mnist.data, 'values') else mnist.data
            y = mnist.target.values.astype(int) if hasattr(mnist.target, 'values') else mnist.target.astype(int)
            
            # Take first n_samples
            X = X[:n_samples]
            y = y[:n_samples]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
            return X_train, X_test, y_train, y_test, feature_names, f"MNIST (subset {n_samples})"
        except Exception as e:
            print(f"  Warning: Could not load MNIST: {e}")
            return None, None, None, None, None, None
    
    @staticmethod
    def load_fraud_detection() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load Credit Card Fraud dataset (large, highly imbalanced)."""
        fraud_csv = "/home/pcw/devel/i9_developer/case-explainer/creditcard.csv"
        
        if not os.path.exists(fraud_csv):
            print(f"  Warning: Fraud detection data not found at {fraud_csv}")
            return None, None, None, None, None, None
        
        df = pd.read_csv(fraud_csv)
        X = df.drop('Class', axis=1).values
        y = df['Class'].values.astype(int)
        feature_names = df.drop('Class', axis=1).columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names, "Credit Card Fraud"
    
    @staticmethod
    def load_hardware_trojan() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """Load Hardware Trojan dataset (very large real-world dataset)."""
        # Path to local dataset (tracked with Git LFS)
        data_file = os.path.join(os.path.dirname(__file__), "assets/data/hardware_trojan.csv")
        
        if not os.path.exists(data_file):
            print(f"  Warning: Hardware trojan data not found at {data_file}")
            return None, None, None, None, None, None
        
        # Load full dataset and split
        df = pd.read_csv(data_file)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(int)
        
        # Split into train/test (70/30 split, stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        feature_names = list(df.columns[:-1])
        return X_train, X_test, y_train, y_test, feature_names, "Hardware Trojan"


class Benchmarker:
    """Run benchmarks on case-explainer."""
    
    def __init__(self, k: int = 5, n_batch_samples: int = 100):
        """
        Initialize benchmarker.
        
        Args:
            k: Number of neighbors for explanations
            n_batch_samples: Number of samples for batch explanation timing
        """
        self.k = k
        self.n_batch_samples = n_batch_samples
        self.results: List[BenchmarkResult] = []
    
    def benchmark_configuration(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        dataset_name: str,
        index_method: str = 'kd_tree'
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark a single dataset + index method configuration.
        
        Returns:
            BenchmarkResult or None if benchmark failed
        """
        print(f"\n  Testing {index_method} index...")
        
        try:
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Measure fit time and memory
            tracemalloc.start()
            start_time = time.perf_counter()
            
            explainer = CaseExplainer(
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                algorithm=index_method,
                scale_data=True
            )
            
            fit_time = time.perf_counter() - start_time
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            # Measure single explanation time
            start_time = time.perf_counter()
            single_explanation = explainer.explain_instance(X_test[0], k=self.k, model=clf)
            single_explain_time = time.perf_counter() - start_time
            
            # Measure batch explanation time
            n_batch = min(self.n_batch_samples, len(X_test))
            start_time = time.perf_counter()
            batch_explanations = explainer.explain_batch(
                X_test[:n_batch], 
                k=self.k, 
                y_test=y_test[:n_batch],
                model=clf
            )
            batch_explain_time = time.perf_counter() - start_time
            time_per_explanation = batch_explain_time / n_batch
            
            # Calculate correspondence statistics
            correspondences = [exp.correspondence for exp in batch_explanations]
            mean_corr = np.mean(correspondences)
            std_corr = np.std(correspondences)
            
            # Correspondence by correctness
            correct_corr = np.mean([exp.correspondence for exp in batch_explanations if exp.is_correct()])
            incorrect_corr = np.mean([exp.correspondence for exp in batch_explanations if not exp.is_correct()])
            
            result = BenchmarkResult(
                dataset_name=dataset_name,
                n_samples=len(X_train),
                n_features=X_train.shape[1],
                n_classes=len(np.unique(y_train)),
                index_method=index_method,
                fit_time=fit_time,
                single_explain_time=single_explain_time,
                batch_explain_time=batch_explain_time,
                time_per_explanation=time_per_explanation,
                memory_usage=memory_mb,
                mean_correspondence=mean_corr,
                std_correspondence=std_corr,
                accuracy=accuracy,
                correct_correspondence=correct_corr,
                incorrect_correspondence=incorrect_corr
            )
            
            print(f"    Fit time: {fit_time:.3f}s | Explain time: {time_per_explanation*1000:.2f}ms/sample | Correspondence: {mean_corr:.1%}")
            
            return result
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def run_all_benchmarks(
        self, 
        include_mnist: bool = True,
        include_hardware: bool = True,
        index_methods: Optional[List[str]] = None
    ):
        """
        Run benchmarks on all datasets.
        
        Args:
            include_mnist: Whether to include MNIST (slow)
            include_hardware: Whether to include hardware trojan data
            index_methods: List of index methods to test (default: all applicable)
        """
        if index_methods is None:
            index_methods = ['kd_tree', 'ball_tree', 'brute']
        
        datasets = [
            ('load_iris', DatasetLoader.load_iris),
            ('load_wine', DatasetLoader.load_wine),
            ('load_breast_cancer', DatasetLoader.load_breast_cancer),
            ('load_digits', DatasetLoader.load_digits),
            ('load_fraud_detection', DatasetLoader.load_fraud_detection),
        ]
        
        if include_mnist:
            datasets.append(('load_mnist_subset', DatasetLoader.load_mnist_subset))
        
        if include_hardware:
            datasets.append(('load_hardware_trojan', DatasetLoader.load_hardware_trojan))
        
        print("=" * 80)
        print("CASE-EXPLAINER BENCHMARK SUITE")
        print("=" * 80)
        
        for dataset_func_name, dataset_func in datasets:
            print(f"\n{'=' * 80}")
            print(f"Dataset: {dataset_func_name}")
            print(f"{'=' * 80}")
            
            # Load dataset
            result = dataset_func()
            if result[0] is None:
                print("  Skipping (data not available)")
                continue
            
            X_train, X_test, y_train, y_test, feature_names, dataset_name = result
            
            print(f"  Samples: {len(X_train):,} train, {len(X_test):,} test")
            print(f"  Features: {X_train.shape[1]}")
            print(f"  Classes: {len(np.unique(y_train))}")
            
            # Determine which index methods to use based on dataset characteristics
            n_features = X_train.shape[1]
            n_samples = len(X_train)
            
            methods_to_test = []
            for method in index_methods:
                if method == 'brute' and n_samples > 5000:
                    print(f"  Skipping {method} (too slow for {n_samples:,} samples)")
                    continue
                if method == 'kd_tree' and n_features > 20:
                    print(f"  Skipping {method} (not efficient for {n_features} features)")
                    continue
                methods_to_test.append(method)
            
            # Run benchmarks for each index method
            for index_method in methods_to_test:
                result = self.benchmark_configuration(
                    X_train, X_test, y_train, y_test,
                    feature_names, dataset_name, index_method
                )
                if result:
                    self.results.append(result)
        
        print(f"\n{'=' * 80}")
        print("BENCHMARK COMPLETE")
        print(f"{'=' * 80}\n")
    
    def print_summary(self):
        """Print summary table of all benchmark results."""
        if not self.results:
            print("No benchmark results to display.")
            return
        
        print("\n" + "=" * 120)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 120)
        
        # Create DataFrame for easy formatting
        data = []
        for r in self.results:
            data.append({
                'Dataset': r.dataset_name,
                'Samples': f"{r.n_samples:,}",
                'Features': r.n_features,
                'Index': r.index_method,
                'Fit (s)': f"{r.fit_time:.3f}",
                'Explain (ms)': f"{r.time_per_explanation * 1000:.2f}",
                'Memory (MB)': f"{r.memory_usage:.1f}",
                'Accuracy': f"{r.accuracy:.1%}",
                'Correspondence': f"{r.mean_correspondence:.1%}",
                'Corr (Correct)': f"{r.correct_correspondence:.1%}" if r.correct_correspondence else "N/A",
                'Corr (Wrong)': f"{r.incorrect_correspondence:.1%}" if r.incorrect_correspondence else "N/A"
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("=" * 120)
        
        # Key insights
        print("\nKEY INSIGHTS:")
        print("-" * 80)
        
        # Fastest index method per dataset
        print("\n1. Fastest Index Method by Dataset:")
        for dataset_name in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset_name]
            explain_times = [float(t) for t in subset['Explain (ms)']]
            fastest_idx = np.argmin(explain_times)
            fastest = subset.iloc[fastest_idx]
            print(f"   {dataset_name:20s}: {fastest['Index']:10s} ({fastest['Explain (ms)']} ms/sample)")
        
        # Correspondence trends
        print("\n2. Correspondence Quality:")
        for dataset_name in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset_name]
            corr_values = [float(c.strip('%'))/100 for c in subset['Correspondence']]
            mean_corr = np.mean(corr_values)
            print(f"   {dataset_name:20s}: {mean_corr:.1%} (avg across index methods)")
        
        # Memory usage
        print("\n3. Memory Usage:")
        for dataset_name in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset_name]
            mem_values = [float(m) for m in subset['Memory (MB)']]
            max_mem = np.max(mem_values)
            print(f"   {dataset_name:20s}: {max_mem:.1f} MB (max)")
        
        print("-" * 80)
    
    def save_results(self, filename: str = "benchmark_results.csv"):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save.")
            return
        
        data = []
        for r in self.results:
            data.append({
                'dataset': r.dataset_name,
                'n_samples': r.n_samples,
                'n_features': r.n_features,
                'n_classes': r.n_classes,
                'index_method': r.index_method,
                'fit_time_s': r.fit_time,
                'single_explain_time_s': r.single_explain_time,
                'batch_explain_time_s': r.batch_explain_time,
                'time_per_explanation_ms': r.time_per_explanation * 1000,
                'memory_mb': r.memory_usage,
                'mean_correspondence': r.mean_correspondence,
                'std_correspondence': r.std_correspondence,
                'accuracy': r.accuracy,
                'correct_correspondence': r.correct_correspondence,
                'incorrect_correspondence': r.incorrect_correspondence
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")


def main():
    """Run the benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark case-explainer performance')
    parser.add_argument('--no-mnist', action='store_true', help='Skip MNIST dataset')
    parser.add_argument('--no-hardware', action='store_true', help='Skip hardware trojan dataset')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors (default: 5)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for timing (default: 100)')
    parser.add_argument('--index-methods', nargs='+', choices=['kd_tree', 'ball_tree', 'brute'],
                        help='Index methods to test (default: all applicable)')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file (default: benchmark_results.csv)')
    parser.add_argument(
        '--retrieval-strategies', action='store_true',
        help='Benchmark predicted-class activation and forest proximity retrieval'
    )
    parser.add_argument(
        '--retrieval-samples', type=int, default=2000,
        help='Synthetic sample count for retrieval strategy benchmarks'
    )
    parser.add_argument(
        '--retrieval-matrix', action='store_true',
        help='Vary sample, feature, and class counts independently'
    )
    parser.add_argument(
        '--retrieval-features', type=int, nargs='+', default=[20, 50],
        help='Feature counts for the retrieval matrix'
    )
    parser.add_argument(
        '--retrieval-classes', type=int, nargs='+', default=[2, 3, 5],
        help='Class counts for the retrieval matrix'
    )
    parser.add_argument(
        '--retrieval-warmups', type=int, default=2,
        help='Untimed warm-up queries per strategy'
    )
    parser.add_argument(
        '--retrieval-repetitions', type=int, default=5,
        help='Timed trials used to report median latency and standard deviation'
    )
    parser.add_argument(
        '--check-regression', metavar='CANDIDATE_CSV',
        help='Compare normalized candidate scaling against --baseline'
    )
    parser.add_argument(
        '--baseline', default='benchmarks/retrieval_baseline.csv',
        help='Reference CSV for --check-regression'
    )
    
    args = parser.parse_args()

    if args.check_regression:
        findings = compare_retrieval_results(args.baseline, args.check_regression)
        if findings:
            print("\n".join(findings))
            raise SystemExit(1)
        print("Retrieval performance regression check passed.")
        return

    if args.retrieval_strategies:
        if args.retrieval_matrix:
            results = benchmark_retrieval_matrix(
                sample_sizes=[args.retrieval_samples, args.retrieval_samples * 2],
                feature_counts=args.retrieval_features,
                class_counts=args.retrieval_classes,
                n_queries=args.batch_size,
                warmup_queries=args.retrieval_warmups,
                repetitions=args.retrieval_repetitions,
            )
        else:
            results = benchmark_retrieval_strategies(
                n_samples=args.retrieval_samples,
                n_queries=args.batch_size,
                warmup_queries=args.retrieval_warmups,
                repetitions=args.retrieval_repetitions,
            )
        print("\nRETRIEVAL STRATEGY BENCHMARKS")
        for result in results:
            print(
                f"{result.strategy:28s} fit={result.fit_time:.3f}s "
                f"query={result.query_time * 1000:.2f}ms "
                f"std={result.query_time_std * 1000:.2f}ms "
                f"peak={result.peak_memory_mb:.1f}MB"
            )
        save_retrieval_results(results, args.output)
        print(f"\nResults saved to: {args.output}")
        return
    
    benchmarker = Benchmarker(k=args.k, n_batch_samples=args.batch_size)
    
    benchmarker.run_all_benchmarks(
        include_mnist=not args.no_mnist,
        include_hardware=not args.no_hardware,
        index_methods=args.index_methods
    )
    
    benchmarker.print_summary()
    benchmarker.save_results(args.output)


if __name__ == '__main__':
    main()
