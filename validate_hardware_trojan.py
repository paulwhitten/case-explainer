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
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
    """Load training and test data, split training into train/val."""
    print("Loading hardware trojan detection data...")
    
    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)
    X_trainval = train_df.iloc[:, :-1].values
    y_trainval = train_df.iloc[:, -1].values.astype(int)
    
    # Split training into train (80%) and validation (20%)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(int)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(X_train):,} (64%)")
    print(f"  Validation samples: {len(X_val):,} (16%)")
    print(f"  Test samples: {len(X_test):,} (20%)")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training class distribution: {np.bincount(y_train)}")
    print(f"  Validation class distribution: {np.bincount(y_val)}")
    print(f"  Test class distribution: {np.bincount(y_test)}")
    
    # Calculate imbalance ratio
    train_normal = np.sum(y_train == 0)
    train_trojan = np.sum(y_train == 1)
    imbalance = train_normal / train_trojan if train_trojan > 0 else float('inf')
    print(f"  Imbalance ratio (normal:trojan): {imbalance:.1f}:1")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def cross_validate_model(X_trainval, y_trainval):
    """Perform cross-validation to assess model stability."""
    print("\nPerforming 5-fold cross-validation...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,              # Regularization
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_trainval, y_trainval, cv=cv, scoring='accuracy')
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Min: {cv_scores.min():.4f}, Max: {cv_scores.max():.4f}")
    
    return cv_scores

def train_classifier(X_train, y_train, X_val, y_val):
    """Train RandomForest classifier with regularization."""
    print("\nTraining RandomForest classifier with regularization...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,              # Limit tree depth
        min_samples_split=10,      # Require 10 samples to split
        min_samples_leaf=4,        # Require 4 samples per leaf
        max_features='sqrt',       # Use sqrt(n_features) per split
        class_weight='balanced',   # Handle imbalanced data
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    print(f"  Hyperparameters: max_depth=15, min_samples_split=10, min_samples_leaf=4")
    
    # Evaluate on training and validation
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    
    # Check for overfitting
    overfit_gap = train_acc - val_acc
    if overfit_gap > 0.05:
        print(f"  WARNING: Overfitting detected: {overfit_gap:.2%} gap between train and val")
    else:
        print(f"  Good generalization: {overfit_gap:.2%} gap between train and val")
    
    return clf

def evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate classifier performance on all splits."""
    print("\nEvaluating classifier on all data splits...")
    
    # Training accuracy
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Validation accuracy
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Test accuracy
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    # Confusion matrix (test set)
    cm = confusion_matrix(y_test, test_pred)
    print(f"\nTest Set Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    # Per-class metrics
    print("\nTest Set Classification Report:")
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
        print(f"VALIDATED: Within 2% of paper results (diff={diff:.2f}%)")
    else:
        print(f"DEVIATION: More than 2% difference (diff={diff:.2f}%)")
    
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
    
    # Load data with train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Cross-validate to check stability
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    cv_scores = cross_validate_model(X_trainval, y_trainval)
    
    # Train classifier with regularization
    clf = train_classifier(X_train, y_train, X_val, y_val)
    
    # Evaluate classifier on all splits
    predictions = evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Validate case-explainer (on test set only)
    explanations = validate_case_explainer(
        X_train, y_train, X_test, y_test, predictions, clf
    )
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    correspondences = [e.correspondence for e in explanations]
    print(f"  Test correspondence: {np.mean(correspondences):.4f} ± {np.std(correspondences):.4f}")

if __name__ == "__main__":
    main()
