#!/usr/bin/env python3
"""
Validate case-explainer on hardware trojan detection dataset.

This replicates the Method 2 (case-based) results from the JETTA paper,
which achieved 97.4% correspondence on 56,959 samples.

Expected results:
- Training samples: ~45,000
- Test samples: ~11,000
- Features: 5 (gate-level metrics)
- Classes: 2 (trojan=1, normal=0)
- Class imbalance: ~268:1 (normal:trojan)
- Target correspondence: ~97.4%
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Add case_explainer to path
sys.path.insert(0, os.path.dirname(__file__))
from case_explainer import CaseExplainer

# Paths to pipeline data
PIPELINE_DIR = "/home/pcw/devel/i9_developer/explainable_hw_trojan_detection_pipeline"
TRAIN_CSV = f"{PIPELINE_DIR}/data/processed/train.csv"
TEST_CSV = f"{PIPELINE_DIR}/data/processed/test.csv"

def load_data():
    """Load training and test data."""
    print("Loading hardware trojan detection data...")
    
    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values.astype(int)
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(int)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training class distribution: {np.bincount(y_train)}")
    print(f"  Test class distribution: {np.bincount(y_test)}")
    
    # Calculate imbalance ratio
    train_normal = np.sum(y_train == 0)
    train_trojan = np.sum(y_train == 1)
    imbalance = train_normal / train_trojan if train_trojan > 0 else float('inf')
    print(f"  Imbalance ratio (normal:trojan): {imbalance:.1f}:1")
    
    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train):
    """Train RandomForest classifier."""
    print("\nTraining RandomForest classifier...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    return clf

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    """Evaluate classifier performance."""
    print("\nEvaluating classifier...")
    
    # Training accuracy
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test accuracy
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    # Per-class metrics
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=['Normal', 'Trojan']))
    
    return test_pred

def validate_case_explainer(X_train, y_train, X_test, y_test, predictions, clf):
    """Validate case-explainer with hardware trojan data."""
    print("\n" + "="*70)
    print("VALIDATING CASE-EXPLAINER")
    print("="*70)
    
    # Feature names (from JETTA paper)
    feature_names = [
        'avg_fanin',
        'avg_fanout', 
        'avg_depth',
        'controllability',
        'observability'
    ]
    
    class_names = ['Normal', 'Trojan']
    
    # Create explainer with class weighting (trojan 2x more important)
    print("\nCreating CaseExplainer...")
    print("  Using class_weights={0: 1.0, 1: 2.0} to weight trojans 2x")
    
    start_time = time.time()
    explainer = CaseExplainer(
        X_train=X_train,
        y_train=y_train,
        k=5,
        feature_names=feature_names,
        class_names=class_names,
        algorithm='auto',
        scale_data=True,
        class_weights={0: 1.0, 1: 2.0},  # Weight trojans higher
        n_jobs=-1
    )
    index_time = time.time() - start_time
    print(f"  Index built in {index_time:.2f} seconds")
    
    # Generate explanations for all test samples
    print("\nGenerating explanations for all test samples...")
    start_time = time.time()
    
    explanations = explainer.explain_batch(
        X_test,
        y_test=y_test,
        predictions=predictions,
        model=clf
    )
    
    explain_time = time.time() - start_time
    print(f"  Generated {len(explanations):,} explanations in {explain_time:.2f} seconds")
    print(f"  Average time per explanation: {explain_time/len(explanations)*1000:.2f} ms")
    
    # Analyze correspondence
    correspondences = [e.correspondence for e in explanations]
    avg_corr = np.mean(correspondences)
    
    print(f"\n{'='*70}")
    print(f"CORRESPONDENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Average correspondence: {avg_corr:.4f} ({avg_corr*100:.2f}%)")
    print(f"Min correspondence: {np.min(correspondences):.4f}")
    print(f"Max correspondence: {np.max(correspondences):.4f}")
    print(f"Std correspondence: {np.std(correspondences):.4f}")
    
    # Correspondence by correctness
    correct_mask = predictions == y_test
    correct_corr = [e.correspondence for i, e in enumerate(explanations) if correct_mask[i]]
    incorrect_corr = [e.correspondence for i, e in enumerate(explanations) if not correct_mask[i]]
    
    print(f"\nBy prediction correctness:")
    print(f"  Correct predictions ({len(correct_corr):,}): avg = {np.mean(correct_corr):.4f}")
    if len(incorrect_corr) > 0:
        print(f"  Incorrect predictions ({len(incorrect_corr):,}): avg = {np.mean(incorrect_corr):.4f}")
    
    # Correspondence by class
    normal_mask = y_test == 0
    trojan_mask = y_test == 1
    
    normal_corr = [e.correspondence for i, e in enumerate(explanations) if normal_mask[i]]
    trojan_corr = [e.correspondence for i, e in enumerate(explanations) if trojan_mask[i]]
    
    print(f"\nBy true class:")
    print(f"  Normal gates ({len(normal_corr):,}): avg = {np.mean(normal_corr):.4f}")
    print(f"  Trojan gates ({len(trojan_corr):,}): avg = {np.mean(trojan_corr):.4f}")
    
    # JETTA paper comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON TO JETTA PAPER")
    print(f"{'='*70}")
    print(f"Paper result: 97.4% correspondence")
    print(f"This result:  {avg_corr*100:.2f}% correspondence")
    
    diff = abs(avg_corr - 0.974) * 100
    if diff < 2.0:
        print(f"✅ VALIDATED: Within 2% of paper results (diff={diff:.2f}%)")
    else:
        print(f"⚠️  DEVIATION: More than 2% difference (diff={diff:.2f}%)")
    
    # Show example explanation
    print(f"\n{'='*70}")
    print(f"EXAMPLE EXPLANATION")
    print(f"{'='*70}")
    
    # Find an interesting example (high correspondence, correct prediction)
    high_corr_idx = np.argmax(correspondences)
    example = explanations[high_corr_idx]
    
    print(f"\nTest sample #{high_corr_idx}:")
    print(example.summary())
    
    return explanations

def main():
    """Main validation script."""
    print("="*70)
    print("HARDWARE TROJAN DETECTION: CASE-EXPLAINER VALIDATION")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train classifier
    clf = train_classifier(X_train, y_train)
    
    # Evaluate classifier
    predictions = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
    
    # Validate case-explainer
    explanations = validate_case_explainer(
        X_train, y_train, X_test, y_test, predictions, clf
    )
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
