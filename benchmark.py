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
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, fetch_openml
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add case_explainer to path
sys.path.insert(0, os.path.dirname(__file__))
from case_explainer import CaseExplainer


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
        pipeline_dir = "/home/pcw/devel/i9_developer/explainable_hw_trojan_detection_pipeline"
        train_csv = f"{pipeline_dir}/data/processed/train.csv"
        test_csv = f"{pipeline_dir}/data/processed/test.csv"
        
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"  Warning: Hardware trojan data not found at {pipeline_dir}")
            return None, None, None, None, None, None
        
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values.astype(int)
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values.astype(int)
        
        feature_names = [f"metric_{i}" for i in range(X_train.shape[1])]
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
    
    args = parser.parse_args()
    
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
