"""Tests for case_explainer.explainer module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from case_explainer.explainer import CaseExplainer
from case_explainer.explanation import Explanation


# --- Fixtures ---

@pytest.fixture
def simple_data():
    """Simple 2-class 2-feature dataset."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.1],
        [10.2, 10.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def explainer(simple_data):
    """CaseExplainer with simple data, no scaling."""
    X, y = simple_data
    return CaseExplainer(X, y, k=3, scale_data=False)


@pytest.fixture
def explainer_scaled(simple_data):
    """CaseExplainer with scaling."""
    X, y = simple_data
    return CaseExplainer(X, y, k=3, scale_data=True)


@pytest.fixture
def mock_model():
    """Model that predicts class 0 for everything."""
    m = MagicMock()
    m.predict.return_value = np.array([0])
    return m


# --- __init__ ---

class TestInit:
    def test_basic_init(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=3)
        assert ex.n_samples == 6
        assert ex.n_features == 2
        assert ex.k == 3

    def test_dataframe_input(self, simple_data):
        X, y = simple_data
        df = pd.DataFrame(X, columns=["feat_a", "feat_b"])
        ex = CaseExplainer(df, y, k=2)
        assert ex.feature_names == ["feat_a", "feat_b"]
        assert ex.n_features == 2

    def test_series_labels(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, pd.Series(y), k=2)
        assert len(ex.y_train) == 6

    def test_list_labels(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y.tolist(), k=2)
        assert len(ex.y_train) == 6

    def test_default_feature_names(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, feature_names=None)
        assert ex.feature_names == ["feature_0", "feature_1"]

    def test_custom_feature_names(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, feature_names=["x", "y"])
        assert ex.feature_names == ["x", "y"]

    def test_class_names(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, class_names={0: "A", 1: "B"})
        assert ex.class_names == {0: "A", 1: "B"}

    def test_class_weights(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, class_weights={0: 1.0, 1: 2.0})
        assert ex.class_weights == {0: 1.0, 1: 2.0}

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CaseExplainer(np.zeros((5, 2)), np.zeros(3), k=2)

    def test_metadata_valid(self, simple_data):
        X, y = simple_data
        meta = {"name": ["a", "b", "c", "d", "e", "f"]}
        ex = CaseExplainer(X, y, k=2, metadata=meta)
        assert ex.metadata == meta

    def test_metadata_invalid_length(self, simple_data):
        X, y = simple_data
        with pytest.raises(ValueError, match="Metadata"):
            CaseExplainer(X, y, k=2, metadata={"name": ["a", "b"]})

    def test_scaling(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, scale_data=True)
        assert ex.scaler is not None
        assert ex.X_train_scaled.shape == X.shape

    def test_no_scaling(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, scale_data=False)
        assert ex.scaler is None
        np.testing.assert_array_equal(ex.X_train_scaled, X)

    def test_k_larger_than_n(self):
        """k > n_samples should still work (clamped)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        ex = CaseExplainer(X, y, k=100, scale_data=False)
        assert ex.k == 100  # stored as-is, clamped at query time

    def test_algorithm_stored(self, simple_data):
        X, y = simple_data
        ex = CaseExplainer(X, y, k=2, algorithm='brute')
        assert ex.algorithm == 'brute'


# --- explain_instance ---

class TestExplainInstance:
    def test_with_model(self, explainer, mock_model):
        exp = explainer.explain_instance(np.array([0.05, 0.05]), model=mock_model)
        assert isinstance(exp, Explanation)
        assert exp.predicted_class == 0
        mock_model.predict.assert_called_once()

    def test_with_predicted_class(self, explainer):
        exp = explainer.explain_instance(
            np.array([0.05, 0.05]),
            predicted_class=0,
        )
        assert exp.predicted_class == 0

    def test_list_input(self, explainer):
        exp = explainer.explain_instance([0.05, 0.05], predicted_class=0)
        assert isinstance(exp, Explanation)

    def test_series_input(self, explainer):
        exp = explainer.explain_instance(pd.Series([0.05, 0.05]), predicted_class=0)
        assert isinstance(exp, Explanation)

    def test_wrong_feature_count_raises(self, explainer):
        with pytest.raises(ValueError, match="features"):
            explainer.explain_instance(np.array([0.0, 0.0, 0.0]), predicted_class=0)

    def test_no_model_no_predicted_raises(self, explainer):
        with pytest.raises(ValueError, match="predicted_class or model"):
            explainer.explain_instance(np.array([0.0, 0.0]))

    def test_k_override(self, explainer):
        exp = explainer.explain_instance(np.array([0.0, 0.0]), predicted_class=0, k=2)
        assert len(exp.neighbors) == 2

    def test_defaults_to_init_k(self, explainer):
        exp = explainer.explain_instance(np.array([0.0, 0.0]), predicted_class=0)
        assert len(exp.neighbors) == 3

    def test_correspondence_high_for_matching_cluster(self, explainer):
        """Query close to class-0 cluster -> high correspondence."""
        exp = explainer.explain_instance(np.array([0.05, 0.05]), predicted_class=0, k=3)
        assert exp.correspondence > 0.8
        assert exp.correspondence_interpretation == "high"

    def test_true_class_stored(self, explainer):
        exp = explainer.explain_instance(
            np.array([0.0, 0.0]), predicted_class=0, true_class=1
        )
        assert exp.true_class == 1

    def test_test_index_stored(self, explainer):
        exp = explainer.explain_instance(
            np.array([0.0, 0.0]), predicted_class=0, test_index=99
        )
        assert exp.test_index == 99

    def test_neighbors_sorted_by_distance(self, explainer):
        exp = explainer.explain_instance(np.array([0.0, 0.0]), predicted_class=0)
        dists = [n.distance for n in exp.neighbors]
        assert dists == sorted(dists)

    def test_return_provenance_false(self, simple_data):
        X, y = simple_data
        meta = {"tag": ["a", "b", "c", "d", "e", "f"]}
        ex = CaseExplainer(X, y, k=2, metadata=meta, scale_data=False)
        exp = ex.explain_instance(
            np.array([0.0, 0.0]), predicted_class=0, return_provenance=False
        )
        for n in exp.neighbors:
            assert n.metadata is None or n.metadata == {}

    def test_return_provenance_true_with_metadata(self, simple_data):
        X, y = simple_data
        meta = {"tag": ["a", "b", "c", "d", "e", "f"]}
        ex = CaseExplainer(X, y, k=2, metadata=meta, scale_data=False)
        exp = ex.explain_instance(
            np.array([0.0, 0.0]), predicted_class=0, return_provenance=True
        )
        has_meta = any(n.metadata for n in exp.neighbors)
        assert has_meta

    def test_scaled_explainer(self, explainer_scaled):
        exp = explainer_scaled.explain_instance(
            np.array([0.0, 0.0]), predicted_class=0
        )
        assert isinstance(exp, Explanation)

    def test_distance_weighted_off(self, explainer):
        exp = explainer.explain_instance(
            np.array([0.0, 0.0]), predicted_class=0, distance_weighted=False
        )
        assert 0.0 <= exp.correspondence <= 1.0


# --- explain_batch ---

class TestExplainBatch:
    def test_batch_length(self, explainer):
        X_test = np.array([[0.0, 0.0], [10.0, 10.0]])
        preds = np.array([0, 1])
        exps = explainer.explain_batch(X_test, predictions=preds)
        assert len(exps) == 2

    def test_batch_with_model(self, explainer, mock_model):
        X_test = np.array([[0.0, 0.0], [10.0, 10.0]])
        exps = explainer.explain_batch(X_test, model=mock_model)
        assert len(exps) == 2

    def test_batch_with_real_classifier(self, simple_data):
        """explain_batch with a real trained model (not mocked)."""
        from sklearn.tree import DecisionTreeClassifier
        X, y = simple_data
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        ex = CaseExplainer(X, y, k=3, scale_data=False)
        X_test = np.array([[0.05, 0.05], [10.05, 10.05]])
        y_test = np.array([0, 1])
        exps = ex.explain_batch(X_test, model=clf, y_test=y_test)
        assert len(exps) == 2
        # Model should predict correctly on these obvious points
        assert exps[0].predicted_class == 0
        assert exps[1].predicted_class == 1
        # Correspondence should be high for well-separated clusters
        for exp in exps:
            assert exp.correspondence > 0.8
            assert exp.is_correct() is True

    def test_batch_with_y_test(self, explainer):
        X_test = np.array([[0.0, 0.0]])
        y_test = np.array([0])
        preds = np.array([0])
        exps = explainer.explain_batch(X_test, y_test=y_test, predictions=preds)
        assert exps[0].true_class == 0

    def test_batch_dataframe(self, explainer):
        df = pd.DataFrame([[0.0, 0.0], [10.0, 10.0]], columns=["a", "b"])
        preds = [0, 1]
        exps = explainer.explain_batch(df, predictions=preds)
        assert len(exps) == 2

    def test_batch_series_labels(self, explainer):
        X_test = np.array([[0.0, 0.0]])
        exps = explainer.explain_batch(
            X_test, y_test=pd.Series([0]), predictions=[0]
        )
        assert exps[0].true_class == 0

    def test_batch_indices_sequential(self, explainer):
        X_test = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]])
        preds = [0, 1, 0]
        exps = explainer.explain_batch(X_test, predictions=preds)
        for i, exp in enumerate(exps):
            assert exp.test_index == i


# --- Integration with real classifier ---

class TestIntegrationRealClassifier:
    """End-to-end tests using actual sklearn classifiers (no mocks)."""

    def test_explain_with_real_model(self, simple_data):
        """explain_instance with a real trained classifier."""
        from sklearn.tree import DecisionTreeClassifier
        X, y = simple_data
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        ex = CaseExplainer(X, y, k=3, scale_data=False)
        exp = ex.explain_instance(np.array([0.05, 0.05]), model=clf)
        assert isinstance(exp, Explanation)
        assert exp.predicted_class == 0  # close to class-0 cluster
        assert exp.correspondence > 0.8

    def test_batch_with_real_model(self, simple_data):
        """explain_batch with a real trained classifier."""
        from sklearn.tree import DecisionTreeClassifier
        X, y = simple_data
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        ex = CaseExplainer(X, y, k=3, scale_data=False)
        X_test = np.array([[0.05, 0.05], [10.05, 10.05]])
        exps = ex.explain_batch(X_test, model=clf, y_test=np.array([0, 1]))
        assert len(exps) == 2
        assert exps[0].predicted_class == 0
        assert exps[1].predicted_class == 1
        assert exps[0].is_correct() is True
        assert exps[1].is_correct() is True

    def test_iris_end_to_end(self, iris_data, iris_clf):
        """Full pipeline with Iris dataset and RandomForest."""
        X_train, X_test, y_train, y_test = iris_data
        ex = CaseExplainer(
            X_train, y_train, k=5,
            feature_names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'],
            class_names={0: 'setosa', 1: 'versicolor', 2: 'virginica'},
        )
        # Single explanation
        exp = ex.explain_instance(
            X_test[0], model=iris_clf, true_class=int(y_test[0])
        )
        assert isinstance(exp, Explanation)
        assert 0.0 <= exp.correspondence <= 1.0
        assert exp.correspondence_interpretation in ('high', 'medium', 'low')
        assert len(exp.neighbors) == 5
        assert exp.feature_names == ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

    def test_iris_batch(self, iris_data, iris_clf):
        """Batch explanation on Iris with accuracy check."""
        X_train, X_test, y_train, y_test = iris_data
        ex = CaseExplainer(X_train, y_train, k=5)
        exps = ex.explain_batch(
            X_test[:10], model=iris_clf, y_test=y_test[:10]
        )
        assert len(exps) == 10
        # RandomForest on Iris should get most right
        correct = sum(1 for e in exps if e.is_correct())
        assert correct >= 8  # at least 80% on easy dataset
        # Correct predictions should have higher correspondence on average
        corr_correct = [e.correspondence for e in exps if e.is_correct()]
        corr_wrong = [e.correspondence for e in exps if not e.is_correct()]
        if corr_wrong:  # might be empty if all correct
            assert np.mean(corr_correct) >= np.mean(corr_wrong)

    def test_iris_summary_and_dict(self, iris_data, iris_clf):
        """Verify summary and to_dict work with real Iris data."""
        X_train, X_test, y_train, y_test = iris_data
        ex = CaseExplainer(
            X_train, y_train, k=3,
            class_names={0: 'setosa', 1: 'versicolor', 2: 'virginica'},
        )
        exp = ex.explain_instance(X_test[0], model=iris_clf)
        summary = exp.summary()
        assert 'CASE-BASED EXPLANATION' in summary
        assert 'Correspondence' in summary
        d = exp.to_dict()
        assert isinstance(d['correspondence'], float)
        assert len(d['neighbors']) == 3
        assert d['predicted_class_name'] in ('setosa', 'versicolor', 'virginica')


# --- get_training_info ---

class TestGetTrainingInfo:
    def test_keys(self, explainer):
        info = explainer.get_training_info()
        expected = {
            "n_samples", "n_features", "n_classes", "classes",
            "class_counts", "feature_names", "class_names",
            "algorithm", "metric", "scaled", "has_metadata", "default_k",
        }
        assert expected == set(info.keys())

    def test_values(self, explainer):
        info = explainer.get_training_info()
        assert info["n_samples"] == 6
        assert info["n_features"] == 2
        assert info["n_classes"] == 2
        assert info["default_k"] == 3
        assert info["scaled"] is False


# --- __repr__ ---

class TestRepr:
    def test_repr(self, explainer):
        r = repr(explainer)
        assert "n_samples=6" in r
        assert "n_features=2" in r
        assert "k=3" in r
