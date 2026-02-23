"""Shared fixtures for case-explainer tests."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def iris_data():
    """Small Iris dataset split for testing."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def iris_clf(iris_data):
    """Trained RandomForest on Iris."""
    X_train, _, y_train, _ = iris_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture
def binary_data():
    """Small binary classification dataset."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.randn(10, 3)
    y_test = (X_test[:, 0] > 0).astype(int)
    return X_train, X_test, y_train, y_test
