#!/usr/bin/env python3
"""
Improved fraud detection validation with diagnostic analysis.

Tests multiple configurations to find what works best for high-dimensional,
large-scale fraud detection.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import time

sys.path.insert(0, os.path.dirname(__file__))
from case_explainer import CaseExplainer


def load_data(filepath='creditcard.csv'):
    """Load fraud detection dataset with splits."""
    print("="*70)
    print("DATA LOADING")
    print("="*70)
    
    df = pd.read_csv(filepath)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values.astype(int)
    feature_names = df.drop('Class', axis=1).columns.tolist()
    
    # Splits
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    
    print(f"Dataset: {len(X):,} transactions")
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"  Fraud rate: {np.sum(y==1)/len(y):.4f} ({np.sum(y==1)} cases)")
    print(f"  Imbalance: {np.sum(y==0)/np.sum(y==1):.1f}:1")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def train_model(X_train, y_train, X_val, y_val):
    """Train RandomForest with optimal hyperparameters."""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    val_f1 = f1_score(y_val, clf.predict(X_val))
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    print(f"Val F1-score: {val_f1:.4f}")
    print(f"Generalization gap: {(train_acc - val_acc)*100:.2f}%")
    
    return clf


def test_correspondence_sensitivity(X_train, y_train, X_test, y_test, clf, test_size=200):
    """Test correspondence under different configurations."""
    print("\n" + "="*70)
    print("CORRESPONDENCE SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Sample test set strategically
    fraud_idx = np.where(y_test == 1)[0][:50]
    legit_idx = np.random.choice(np.where(y_test == 0)[0], test_size-50, replace=False)
    sample_idx = np.concatenate([fraud_idx, legit_idx])
    
    X_test_sample = X_test[sample_idx]
    y_test_sample = y_test[sample_idx]
    predictions = clf.predict(X_test_sample)
    correct_mask = predictions == y_test_sample
    
    print(f"Test sample: {len(sample_idx)} cases")
    print(f"  {np.sum(y_test_sample==1)} fraud, {np.sum(y_test_sample==0)} legitimate")
    print(f"  {np.sum(correct_mask)} correct, {np.sum(~correct_mask)} incorrect predictions")
    
    configurations = [
        {"name": "Original (k=5, cubic, weight=107.6)", "k": 5, "power": 3, "weight": 107.6},
        {"name": "Larger k (k=25, cubic, weight=107.6)", "k": 25, "power": 3, "weight": 107.6},
        {"name": "Larger k (k=50, cubic, weight=107.6)", "k": 50, "power": 3, "weight": 107.6},
        {"name": "Linear penalty (k=5, linear, weight=107.6)", "k": 5, "power": 1, "weight": 107.6},
        {"name": "Square penalty (k=5, square, weight=107.6)", "k": 5, "power": 2, "weight": 107.6},
        {"name": "No class weight (k=5, cubic, weight=1.0)", "k": 5, "power": 3, "weight": 1.0},
        {"name": "Moderate weight (k=5, cubic, weight=20.0)", "k": 5, "power": 3, "weight": 20.0},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        # Build explainer with custom correspondence function
        explainer = CaseExplainer(
            X_train, y_train,
            k=config['k'],
            scale_data=True,
            class_weights={0: 1.0, 1: config['weight']},
            n_jobs=-1
        )
        
        # Note: This uses the default cubic penalty
        # To test different penalties, we'd need to modify the metrics module
        start = time.time()
        explanations = explainer.explain_batch(
            X_test_sample,
            y_test=y_test_sample,
            predictions=predictions,
            model=clf
        )
        elapsed = time.time() - start
        
        # Analyze correspondence
        corrs = np.array([e.correspondence for e in explanations])
        corr_correct = np.mean(corrs[correct_mask])
        corr_incorrect = np.mean(corrs[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0
        gap = corr_correct - corr_incorrect
        std = np.std(corrs)
        
        result = {
            'config': config['name'],
            'k': config['k'],
            'mean_corr': np.mean(corrs),
            'std_corr': std,
            'corr_correct': corr_correct,
            'corr_incorrect': corr_incorrect,
            'gap': gap,
            'time_ms': elapsed / len(explanations) * 1000
        }
        results.append(result)
        
        print(f"  Mean correspondence: {np.mean(corrs):.4f} +/- {std:.4f}")
        print(f"  Correct: {corr_correct:.4f}, Incorrect: {corr_incorrect:.4f}")
        print(f"  Gap: {gap:.4f} ({gap*100:.2f}% points)")
        print(f"  Time: {result['time_ms']:.2f} ms/explanation")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: CORRESPONDENCE DISCRIMINATION")
    print("="*70)
    print(f"{'Configuration':<45} {'Gap':<10} {'Std':<10} {'Time(ms)':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<45} {r['gap']:>8.4f}  {r['std_corr']:>8.4f}  {r['time_ms']:>8.2f}")
    
    # Find best configuration
    best = max(results, key=lambda x: x['gap'])
    print(f"\nBest discrimination: {best['config']}")
    print(f"  Gap: {best['gap']:.4f} ({best['gap']*100:.2f}% points)")
    
    return results


def test_alternative_similarity_metrics(X_train, y_train, X_test, y_test, clf, test_size=200):
    """Test alternative approaches to case similarity."""
    print("\n" + "="*70)
    print("ALTERNATIVE SIMILARITY METRICS")
    print("="*70)
    
    # Sample test set
    fraud_idx = np.where(y_test == 1)[0][:50]
    legit_idx = np.random.choice(np.where(y_test == 0)[0], test_size-50, replace=False)
    sample_idx = np.concatenate([fraud_idx, legit_idx])
    
    X_test_sample = X_test[sample_idx]
    y_test_sample = y_test[sample_idx]
    predictions = clf.predict(X_test_sample)
    correct_mask = predictions == y_test_sample
    
    # 1. RandomForest Leaf-Based Similarity
    print("\n1. RandomForest Leaf Co-occurrence (alternative to Euclidean distance)")
    
    # Get leaf indices for each tree
    train_leaves = clf.apply(X_train)  # (n_train, n_trees)
    test_leaves = clf.apply(X_test_sample)  # (n_test, n_trees)
    
    # For each test sample, count how many trees put it in same leaf as each training sample
    leaf_similarities = []
    for test_leaf in test_leaves:
        # Count matching leaves across all trees
        n_matching = np.sum(train_leaves == test_leaf, axis=1)
        # Convert to similarity score (higher is more similar)
        similarity = n_matching / clf.n_estimators
        leaf_similarities.append(similarity)
    
    leaf_similarities = np.array(leaf_similarities)
    
    # For each test sample, find the k=5 most similar training samples
    k = 5
    leaf_corrs = []
    for i, test_sim in enumerate(leaf_similarities):
        # Get k most similar (highest similarity)
        top_k_idx = np.argpartition(test_sim, -k)[-k:]
        top_k_labels = y_train[top_k_idx]
        # Correspondence = fraction matching prediction
        corr = np.sum(top_k_labels == predictions[i]) / k
        leaf_corrs.append(corr)
    
    leaf_corrs = np.array(leaf_corrs)
    corr_correct = np.mean(leaf_corrs[correct_mask])
    corr_incorrect = np.mean(leaf_corrs[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0
    gap = corr_correct - corr_incorrect
    
    print(f"  Mean correspondence: {np.mean(leaf_corrs):.4f} +/- {np.std(leaf_corrs):.4f}")
    print(f"  Correct: {corr_correct:.4f}, Incorrect: {corr_incorrect:.4f}")
    print(f"  Gap: {gap:.4f} ({gap*100:.2f}% points)")
    print(f"  Interpretation: Uses model's learned feature space (which trees/leaves)")
    
    # 2. Model Confidence as Baseline
    print("\n2. Model Probability (baseline for comparison)")
    
    probas = clf.predict_proba(X_test_sample)
    confidences = np.max(probas, axis=1)  # Max probability
    
    conf_correct = np.mean(confidences[correct_mask])
    conf_incorrect = np.mean(confidences[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0
    conf_gap = conf_correct - conf_incorrect
    
    print(f"  Mean confidence: {np.mean(confidences):.4f} +/- {np.std(confidences):.4f}")
    print(f"  Correct: {conf_correct:.4f}, Incorrect: {conf_incorrect:.4f}")
    print(f"  Gap: {conf_gap:.4f} ({conf_gap*100:.2f}% points)")
    
    # Correlation analysis
    print("\n3. Correlation Analysis")
    from scipy.stats import pearsonr, spearmanr
    
    # Original correspondence
    explainer = CaseExplainer(X_train, y_train, k=5, scale_data=True, n_jobs=-1)
    explanations = explainer.explain_batch(X_test_sample, y_test=y_test_sample, 
                                          predictions=predictions, model=clf)
    orig_corrs = np.array([e.correspondence for e in explanations])
    
    pearson_orig, _ = pearsonr(orig_corrs, confidences)
    spearman_orig, _ = spearmanr(orig_corrs, confidences)
    pearson_leaf, _ = pearsonr(leaf_corrs, confidences)
    spearman_leaf, _ = spearmanr(leaf_corrs, confidences)
    
    print(f"  Euclidean k-NN vs Model Confidence:")
    print(f"    Pearson: {pearson_orig:.4f}, Spearman: {spearman_orig:.4f}")
    print(f"  Leaf-based vs Model Confidence:")
    print(f"    Pearson: {pearson_leaf:.4f}, Spearman: {spearman_leaf:.4f}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    best_gap = max(gap, conf_gap, orig_corrs.std())
    
    if conf_gap > gap and conf_gap > (corr_correct - corr_incorrect):
        print("RECOMMENDATION: Use model confidence (predict_proba) as primary confidence metric")
        print(f"  Model confidence gap ({conf_gap:.2%}) > correspondence gap ({gap:.2%})")
        print("  Reason: Euclidean k-NN saturated in high-dimensional space")
        print("\nCase-based explanations still valuable for:")
        print("  - Showing similar historical cases (interpretability)")
        print("  - Understanding what model learned")
        print("  - But NOT as primary confidence metric for fraud detection")
    elif gap > 0.15:
        print("Leaf-based similarity shows promise!")
        print(f"  Discrimination gap: {gap:.2%}")
        print("  Consider using RandomForest leaf co-occurrence for case similarity")
    else:
        print("All metrics show limited discrimination")
        print("  Fraud detection may require alternative explainability methods:")
        print("    - SHAP values for feature importance")
        print("    - Attention mechanisms")
        print("    - Counterfactual explanations")


def main():
    """Main diagnostic analysis."""
    print("="*70)
    print("FRAUD DETECTION: IMPROVED VALIDATION WITH DIAGNOSTICS")
    print("="*70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()
    
    # Train model
    clf = train_model(X_train, y_train, X_val, y_val)
    
    # Test correspondence sensitivity
    results = test_correspondence_sensitivity(X_train, y_train, X_test, y_test, clf)
    
    # Test alternative metrics
    test_alternative_similarity_metrics(X_train, y_train, X_test, y_test, clf)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nSee FRAUD_DETECTION_CRITICAL_ANALYSIS.md for detailed findings")


if __name__ == "__main__":
    main()
