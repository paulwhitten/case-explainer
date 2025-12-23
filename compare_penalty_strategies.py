#!/usr/bin/env python3
"""
Compare distance penalty strategies across all three validation domains.

Tests: Hardware Trojans (5D), Breast Cancer (30D), Fraud Detection (30D)
to see how different penalty strategies affect correspondence discrimination.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, os.path.dirname(__file__))
from case_explainer import CaseExplainer


def test_hardware_trojans(strategies):
    """Test on hardware trojan dataset (5D, low-dimensional)."""
    print("="*70)
    print("DOMAIN 1: HARDWARE TROJAN DETECTION (5 dimensions)")
    print("="*70)
    
    # Load from pipeline
    data_path = '../explainable_hw_trojan_detection_pipeline/data/processed/method1_training_data_combined.csv'
    if not os.path.exists(data_path):
        print("  [SKIP] Hardware trojan data not found")
        return None
    
    df = pd.read_csv(data_path)
    X = df[['total_gates', 'total_wires', 'num_inputs', 'num_outputs', 'gate_wire_ratio']].values
    y = df['label'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    correct_mask = predictions == y_test
    
    print(f"  Data: {len(X_train)} train, {len(X_test)} test, {X.shape[1]} features")
    print(f"  Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"  Correct: {np.sum(correct_mask)}, Incorrect: {np.sum(~correct_mask)}")
    print()
    
    results = {}
    for name, strategy in strategies:
        explainer = CaseExplainer(X_train, y_train, k=5, scale_data=True,
                                 class_weights={0: 1.0, 1: 275.0},
                                 distance_penalty=strategy, n_jobs=-1)
        explanations = explainer.explain_batch(X_test[:200], y_test=y_test[:200], 
                                               predictions=predictions[:200], model=clf)
        corrs = np.array([e.correspondence for e in explanations])
        mask = correct_mask[:200]
        
        gap = np.mean(corrs[mask]) - np.mean(corrs[~mask]) if np.sum(~mask) > 0 else 0.0
        results[name] = {'gap': gap, 'std': np.std(corrs), 'mean': np.mean(corrs)}
        print(f"  {name:15} -> gap: {gap:.4f} ({gap*100:.1f}%), std: {np.std(corrs):.4f}")
    
    return results


def test_breast_cancer(strategies):
    """Test on breast cancer dataset (30D, high-dimensional)."""
    print("\n" + "="*70)
    print("DOMAIN 2: BREAST CANCER DIAGNOSIS (30 dimensions)")
    print("="*70)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    correct_mask = predictions == y_test
    
    print(f"  Data: {len(X_train)} train, {len(X_test)} test, {X.shape[1]} features")
    print(f"  Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"  Correct: {np.sum(correct_mask)}, Incorrect: {np.sum(~correct_mask)}")
    print()
    
    results = {}
    for name, strategy in strategies:
        explainer = CaseExplainer(X_train, y_train, k=5, scale_data=True,
                                 distance_penalty=strategy, n_jobs=-1)
        explanations = explainer.explain_batch(X_test, y_test=y_test, 
                                               predictions=predictions, model=clf)
        corrs = np.array([e.correspondence for e in explanations])
        
        gap = np.mean(corrs[correct_mask]) - np.mean(corrs[~correct_mask])
        results[name] = {'gap': gap, 'std': np.std(corrs), 'mean': np.mean(corrs)}
        print(f"  {name:15} -> gap: {gap:.4f} ({gap*100:.1f}%), std: {np.std(corrs):.4f}")
    
    return results


def test_fraud_detection(strategies):
    """Test on fraud detection dataset (30D, extreme imbalance)."""
    print("\n" + "="*70)
    print("DOMAIN 3: CREDIT CARD FRAUD DETECTION (30 dimensions)")
    print("="*70)
    
    if not os.path.exists('creditcard.csv'):
        print("  [SKIP] Fraud detection data not found (creditcard.csv)")
        return None
    
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values.astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Sample test set with fraud cases
    fraud_idx = np.where(y_test == 1)[0][:50]
    legit_idx = np.random.choice(np.where(y_test == 0)[0], 150, replace=False)
    sample_idx = np.concatenate([fraud_idx, legit_idx])
    
    X_test_sample = X_test[sample_idx]
    y_test_sample = y_test[sample_idx]
    predictions = clf.predict(X_test_sample)
    correct_mask = predictions == y_test_sample
    
    print(f"  Data: {len(X_train)} train, {len(sample_idx)} test (sampled), {X.shape[1]} features")
    print(f"  Accuracy: {accuracy_score(y_test_sample, predictions):.4f}")
    print(f"  Correct: {np.sum(correct_mask)}, Incorrect: {np.sum(~correct_mask)}")
    print()
    
    results = {}
    for name, strategy in strategies:
        explainer = CaseExplainer(X_train, y_train, k=5, scale_data=True,
                                 class_weights={0: 1.0, 1: 107.6},
                                 distance_penalty=strategy, n_jobs=-1)
        explanations = explainer.explain_batch(X_test_sample, y_test=y_test_sample,
                                               predictions=predictions, model=clf)
        corrs = np.array([e.correspondence for e in explanations])
        
        gap = np.mean(corrs[correct_mask]) - np.mean(corrs[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0
        results[name] = {'gap': gap, 'std': np.std(corrs), 'mean': np.mean(corrs)}
        print(f"  {name:15} -> gap: {gap:.4f} ({gap*100:.1f}%), std: {np.std(corrs):.4f}")
    
    return results


def main():
    """Compare distance penalty strategies across all domains."""
    print("="*70)
    print("DISTANCE PENALTY STRATEGY COMPARISON")
    print("="*70)
    print()
    print("Testing strategies:")
    
    strategies = [
        ('fixed', 'fixed'),
        ('adaptive', 'adaptive'),
        ('linear', 'linear'),
        ('square', 'square'),
        ('percentile', 'percentile'),
    ]
    
    for name, _ in strategies:
        print(f"  - {name}")
    print()
    
    # Run tests
    hw_results = test_hardware_trojans(strategies)
    bc_results = test_breast_cancer(strategies)
    fraud_results = test_fraud_detection(strategies)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BEST STRATEGY PER DOMAIN")
    print("="*70)
    
    if hw_results:
        best_hw = max(hw_results.items(), key=lambda x: x[1]['gap'])
        print(f"\nHardware Trojans (5D):")
        print(f"  Best: {best_hw[0]} with gap={best_hw[1]['gap']:.4f} ({best_hw[1]['gap']*100:.1f}%)")
    
    if bc_results:
        best_bc = max(bc_results.items(), key=lambda x: x[1]['gap'])
        print(f"\nBreast Cancer (30D):")
        print(f"  Best: {best_bc[0]} with gap={best_bc[1]['gap']:.4f} ({best_bc[1]['gap']*100:.1f}%)")
    
    if fraud_results:
        best_fraud = max(fraud_results.items(), key=lambda x: x[1]['gap'])
        print(f"\nFraud Detection (30D, extreme imbalance):")
        print(f"  Best: {best_fraud[0]} with gap={best_fraud[1]['gap']:.4f} ({best_fraud[1]['gap']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print()
    print("1. LOW-D (≤5 features): Use 'fixed' (cubic) - sharp discrimination")
    print("2. MEDIUM-D (6-20 features): Use 'adaptive' or 'square'")
    print("3. HIGH-D (21-50 features): Use 'adaptive' or 'linear'")
    print("4. EXTREME-D (>50 features): Use 'adaptive' (auto-adjusts)")
    print()
    print("The 'adaptive' strategy is RECOMMENDED as the new default:")
    print("  - Maintains performance on low-D data")
    print("  - Improves discrimination on high-D data")
    print("  - Automatically adjusts based on dimensionality")


if __name__ == "__main__":
    main()
