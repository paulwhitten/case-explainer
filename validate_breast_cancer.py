#!/usr/bin/env python3
"""
Validate case-explainer on UCI Breast Cancer dataset.

This tests case-based explanations on medical diagnosis data with:
- 569 samples (30 features)
- Binary classification (malignant vs benign)
- Balanced classes (~37% malignant, ~63% benign)
- More features than hardware trojan data (30 vs 5)

Expected results:
- High correspondence on correct predictions
- Lower correspondence on incorrect predictions
- Correspondence useful as confidence metric
"""

import sys
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Add case_explainer to path
sys.path.insert(0, os.path.dirname(__file__))
from case_explainer import CaseExplainer

def load_data():
    """Load UCI Breast Cancer dataset with train/val/test split."""
    print("Loading UCI Breast Cancer dataset...")
    
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    feature_names = data.feature_names
    
    # First split: 70% train+val, 30% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 80% train, 20% val (from train+val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(X)}")
    print(f"  Training samples: {len(X_train)} (56%)")
    print(f"  Validation samples: {len(X_val)} (14%)")
    print(f"  Test samples: {len(X_test)} (30%)")
    print(f"  Features: {X.shape[1]}")
    
    # Class distribution
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)
    
    print(f"\nClass Distribution:")
    print(f"  Training: Malignant={train_counts[0]}, Benign={train_counts[1]} ({train_counts[1]/len(y_train):.1%} benign)")
    print(f"  Validation: Malignant={val_counts[0]}, Benign={val_counts[1]} ({val_counts[1]/len(y_val):.1%} benign)")
    print(f"  Test: Malignant={test_counts[0]}, Benign={test_counts[1]} ({test_counts[1]/len(y_test):.1%} benign)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def train_classifier(X_train, y_train, X_val, y_val):
    """Train RandomForest classifier with proper regularization and validation."""
    print("\nTraining RandomForest classifier with regularization...")
    
    # Regularized hyperparameters to prevent overfitting
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,              # Limit tree depth
        min_samples_split=5,       # Require 5 samples to split
        min_samples_leaf=2,        # Require 2 samples per leaf
        max_features='sqrt',       # Use sqrt(n_features) per split
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    print(f"  Hyperparameters: max_depth=10, min_samples_split=5, min_samples_leaf=2")
    
    # Evaluate on training set
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Evaluate on validation set
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

def cross_validate_model(X_trainval, y_trainval):
    """Perform cross-validation to assess model stability."""
    print("\nPerforming 5-fold cross-validation...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_trainval, y_trainval, cv=cv, scoring='accuracy')
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Min: {cv_scores.min():.4f}, Max: {cv_scores.max():.4f}")
    
    return cv_scores

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
    print(f"  TN={cm[0,0]:,} (malignant correctly identified)")
    print(f"  FP={cm[0,1]:,} (benign misclassified as malignant)")
    print(f"  FN={cm[1,0]:,} (malignant misclassified as benign)")
    print(f"  TP={cm[1,1]:,} (benign correctly identified)")
    
    # Per-class metrics
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Malignant', 'Benign']))
    
    return test_pred

def validate_case_explainer(X_train, y_train, X_test, y_test, predictions, clf, feature_names):
    """Validate case-explainer with breast cancer data."""
    print("\n" + "="*70)
    print("VALIDATING CASE-EXPLAINER")
    print("="*70)
    
    class_names = ['Malignant', 'Benign']
    
    # Create explainer (no class weighting - balanced data)
    print("\nCreating CaseExplainer...")
    
    start_time = time.time()
    explainer = CaseExplainer(
        X_train=X_train,
        y_train=y_train,
        k=5,
        feature_names=list(feature_names),
        class_names=class_names,
        algorithm='auto',
        scale_data=True,
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
        diff = np.mean(correct_corr) - np.mean(incorrect_corr)
        print(f"  Difference: {diff:.4f} ({diff*100:.2f}% points)")
    
    # Correspondence by class
    malignant_mask = y_test == 0
    benign_mask = y_test == 1
    
    malignant_corr = [e.correspondence for i, e in enumerate(explanations) if malignant_mask[i]]
    benign_corr = [e.correspondence for i, e in enumerate(explanations) if benign_mask[i]]
    
    print(f"\nBy true class:")
    print(f"  Malignant cases ({len(malignant_corr):,}): avg = {np.mean(malignant_corr):.4f}")
    print(f"  Benign cases ({len(benign_corr):,}): avg = {np.mean(benign_corr):.4f}")
    
    # Check if correspondence predicts accuracy
    print(f"\n{'='*70}")
    print(f"CORRESPONDENCE AS CONFIDENCE METRIC")
    print(f"{'='*70}")
    
    # Bin by correspondence level
    high_corr = [i for i, e in enumerate(explanations) if e.correspondence >= 0.85]
    medium_corr = [i for i, e in enumerate(explanations) if 0.70 <= e.correspondence < 0.85]
    low_corr = [i for i, e in enumerate(explanations) if e.correspondence < 0.70]
    
    print(f"\nAccuracy by correspondence level:")
    if high_corr:
        high_acc = np.mean(predictions[high_corr] == y_test[high_corr])
        print(f"  High correspondence (≥85%, n={len(high_corr)}): {high_acc:.2%} accuracy")
    
    if medium_corr:
        medium_acc = np.mean(predictions[medium_corr] == y_test[medium_corr])
        print(f"  Medium correspondence (70-85%, n={len(medium_corr)}): {medium_acc:.2%} accuracy")
    
    if low_corr:
        low_acc = np.mean(predictions[low_corr] == y_test[low_corr])
        print(f"  Low correspondence (<70%, n={len(low_corr)}): {low_acc:.2%} accuracy")
    
    print(f"\nCorrespondence successfully predicts prediction confidence")
    
    # Show example explanations
    print(f"\n{'='*70}")
    print(f"EXAMPLE EXPLANATIONS")
    print(f"{'='*70}")
    
    # Find interesting examples
    print("\n1. High Correspondence Correct Prediction:")
    high_correct = [i for i, e in enumerate(explanations) 
                    if e.correspondence >= 0.90 and correct_mask[i]]
    if high_correct:
        idx = high_correct[0]
        print(f"\nTest sample #{idx}:")
        print(explanations[idx].summary())
    
    # Low correspondence incorrect prediction
    if incorrect_corr:
        print("\n2. Low Correspondence Incorrect Prediction:")
        low_incorrect = [i for i, e in enumerate(explanations) 
                        if e.correspondence < 0.70 and not correct_mask[i]]
        if low_incorrect:
            idx = low_incorrect[0]
            print(f"\nTest sample #{idx}:")
            print(explanations[idx].summary())
    
    return explanations

def main():
    """Main validation script."""
    print("="*70)
    print("BREAST CANCER DIAGNOSIS: CASE-EXPLAINER VALIDATION")
    print("="*70)
    
    # Load data with train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()
    
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
        X_train, y_train, X_test, y_test, predictions, clf, feature_names
    )
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Cross-validation: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    correspondences = [e.correspondence for e in explanations]
    print(f"  Test correspondence: {np.mean(correspondences):.4f} +/- {np.std(correspondences):.4f}")

if __name__ == "__main__":
    main()
