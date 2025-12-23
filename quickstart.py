"""
Quickstart example for case-explainer module.

Demonstrates basic usage with the Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from case_explainer import CaseExplainer

def main():
    print("=" * 60)
    print("Case-Explainer Quickstart Example")
    print("=" * 60)
    print()
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Train a classifier
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2%}")
    print()
    
    # Create CaseExplainer
    print("Creating CaseExplainer...")
    explainer = CaseExplainer(
        X_train=X_train,
        y_train=y_train,
        k=5,  # Build index with k=5 neighbors
        feature_names=feature_names,
        class_names=class_names,
        algorithm='auto',  # Let sklearn choose best algorithm
        scale_data=True
    )
    
    print(explainer)
    print()
    
    # Explain a single prediction
    print("Explaining first test sample...")
    test_sample = X_test[0]
    explanation = explainer.explain_instance(
        test_sample=test_sample,
        model=clf,
        true_class=y_test[0]
    )
    
    print(explanation.summary())
    print()
    
    # Explain multiple predictions
    print("Explaining all test samples...")
    predictions = clf.predict(X_test)
    explanations = explainer.explain_batch(
        X_test=X_test,
        y_test=y_test,
        predictions=predictions
    )
    
    # Compute average correspondence
    correspondences = [exp.correspondence for exp in explanations]
    avg_correspondence = np.mean(correspondences)
    
    print(f"Average correspondence: {avg_correspondence:.2%}")
    print(f"Min correspondence: {min(correspondences):.2%}")
    print(f"Max correspondence: {max(correspondences):.2%}")
    print()
    
    # Analyze correspondence by correctness
    correct_explanations = [exp for exp in explanations if exp.is_correct()]
    incorrect_explanations = [exp for exp in explanations if not exp.is_correct()]
    
    if correct_explanations:
        correct_corr = np.mean([exp.correspondence for exp in correct_explanations])
        print(f"Correct predictions ({len(correct_explanations)}): "
              f"avg correspondence = {correct_corr:.2%}")
    
    if incorrect_explanations:
        incorrect_corr = np.mean([exp.correspondence for exp in incorrect_explanations])
        print(f"Incorrect predictions ({len(incorrect_explanations)}): "
              f"avg correspondence = {incorrect_corr:.2%}")
    
    print()
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
