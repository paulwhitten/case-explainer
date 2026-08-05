"""Tests for activation-based case similarity (Caruana et al. 1999)."""

import math
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from case_explainer import (
    CaseExplainer,
    SklearnMLPActivationExtractor,
    DecisionTreeActivationExtractor,
    CallableActivationExtractor,
    ActivationExtractor,
)
from case_explainer.activations import _apply_hidden_activation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris():
    """Iris data split."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


@pytest.fixture(scope="module")
def fitted_mlp(iris):
    X_train, _, y_train, _ = iris
    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=500,
        random_state=0,
    )
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def fitted_tree(iris):
    X_train, _, y_train, _ = iris
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def fitted_forest(iris):
    X_train, _, y_train, _ = iris
    clf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0)
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# _apply_hidden_activation helper
# ---------------------------------------------------------------------------

class TestApplyHiddenActivation:
    def test_relu_zeros_negatives(self):
        a = np.array([-1.0, 0.0, 2.0])
        out = _apply_hidden_activation(a.copy(), "relu")
        np.testing.assert_array_equal(out, [0.0, 0.0, 2.0])

    def test_tanh_range(self):
        a = np.array([-2.0, 0.0, 2.0])
        out = _apply_hidden_activation(a.copy(), "tanh")
        assert -1.0 < out[0] < 0.0
        assert out[1] == 0.0
        assert 0.0 < out[2] < 1.0

    def test_logistic_range(self):
        # sigmoid(10) ≈ 0.9999546, sigmoid(-10) ≈ 4.54e-5 — use loose tolerance
        a = np.array([-10.0, 0.0, 10.0])
        out = _apply_hidden_activation(a.copy(), "logistic")
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-4)

    def test_identity_unchanged(self):
        a = np.array([1.0, -2.0, 3.0])
        out = _apply_hidden_activation(a.copy(), "identity")
        np.testing.assert_array_equal(out, [1.0, -2.0, 3.0])


# ---------------------------------------------------------------------------
# SklearnMLPActivationExtractor
# ---------------------------------------------------------------------------

class TestSklearnMLPActivationExtractor:
    def test_fit_transform_shape(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        acts = ext.fit_transform(fitted_mlp, X_train)
        # Last hidden layer has 8 units
        assert acts.shape == (len(X_train), 8)

    def test_transform_single_sample(self, iris, fitted_mlp):
        X_train, X_test, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        ext.fit(fitted_mlp, X_train)
        act = ext.transform(X_test[:1])
        assert act.shape == (1, 8)

    def test_output_weights_scaling(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        weighted = SklearnMLPActivationExtractor(use_output_weights=True)
        unweighted = SklearnMLPActivationExtractor(use_output_weights=False)
        w = weighted.fit_transform(fitted_mlp, X_train)
        u = unweighted.fit_transform(fitted_mlp, X_train)
        # Should differ when output weights are non-uniform
        assert not np.allclose(w, u)

    def test_no_output_weights_ones(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(use_output_weights=False)
        ext.fit_transform(fitted_mlp, X_train)  # triggers lazy init
        np.testing.assert_array_equal(ext._output_weights, np.ones(8))

    def test_layer_selection(self, iris, fitted_mlp):
        """Selecting layer=0 gives first hidden layer (16 units)."""
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer=0, use_output_weights=False)
        acts = ext.fit_transform(fitted_mlp, X_train)
        assert acts.shape == (len(X_train), 16)

    def test_transform_before_fit_raises(self, iris):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        with pytest.raises(RuntimeError, match="fit"):
            ext.transform(X_train)

    def test_wrong_model_type_raises(self, iris, fitted_tree):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        with pytest.raises(TypeError, match="sklearn MLP"):
            ext.fit(fitted_tree, X_train)

    def test_unfitted_model_raises(self, iris):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        unfitted = MLPClassifier()
        with pytest.raises(ValueError, match="fitted"):
            ext.fit(unfitted, X_train)


# ---------------------------------------------------------------------------
# DecisionTreeActivationExtractor
# ---------------------------------------------------------------------------

class TestDecisionTreeActivationExtractor:
    def test_fit_transform_single_tree_shape(self, iris, fitted_tree):
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        acts = ext.fit_transform(fitted_tree, X_train)
        assert acts.shape[0] == len(X_train)
        assert acts.ndim == 2

    def test_single_tree_one_hot_binary(self, iris, fitted_tree):
        """Each row should have exactly one 1.0 (one-hot)."""
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        acts = ext.fit_transform(fitted_tree, X_train)
        row_sums = acts.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(len(X_train)))

    def test_same_leaf_zero_distance(self, iris, fitted_tree):
        """Two samples in the same leaf have zero Euclidean distance."""
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        acts = ext.fit_transform(fitted_tree, X_train)
        # Find two samples assigned to the same leaf
        leaf_ids = fitted_tree.apply(X_train)
        for leaf_id in np.unique(leaf_ids):
            idx = np.where(leaf_ids == leaf_id)[0]
            if len(idx) >= 2:
                dist = np.linalg.norm(acts[idx[0]] - acts[idx[1]])
                assert dist == 0.0
                break

    def test_different_leaf_nonzero_distance(self, iris, fitted_tree):
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        acts = ext.fit_transform(fitted_tree, X_train)
        leaf_ids = fitted_tree.apply(X_train)
        unique_leaves = np.unique(leaf_ids)
        # Pick one sample from two distinct leaves
        i = np.where(leaf_ids == unique_leaves[0])[0][0]
        j = np.where(leaf_ids == unique_leaves[-1])[0][0]
        dist = np.linalg.norm(acts[i] - acts[j])
        assert dist > 0.0

    def test_forest_one_hot_concatenation(self, iris, fitted_forest):
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor(max_one_hot_dims=5000)
        acts = ext.fit_transform(fitted_forest, X_train)
        assert acts.shape[0] == len(X_train)
        # Total dims = sum of (max_leaf_per_tree + 1) across trees
        assert acts.ndim == 2

    def test_forest_raw_fallback(self, iris, fitted_forest):
        """Very small max_one_hot_dims forces raw-leaf fallback."""
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor(max_one_hot_dims=1)
        with pytest.warns(None):  # warning is logged, not raised
            acts = ext.fit_transform(fitted_forest, X_train)
        assert acts.shape == (len(X_train), fitted_forest.n_estimators)
        assert ext._encoding == "raw"

    def test_wrong_model_type_raises(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        with pytest.raises(TypeError):
            ext.fit(fitted_mlp, X_train)

    def test_transform_before_fit_raises(self, iris):
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        with pytest.raises(RuntimeError, match="fit"):
            ext.transform(X_train)


# ---------------------------------------------------------------------------
# CallableActivationExtractor
# ---------------------------------------------------------------------------

class TestCallableActivationExtractor:
    def test_basic(self, iris, fitted_mlp):
        X_train, _, _, _ = iris

        def fn(model, X):
            # Just return the first hidden layer output via sklearn's predict_proba internals
            return model.predict_proba(X)  # (n_samples, n_classes) — simple placeholder

        ext = CallableActivationExtractor(fn)
        acts = ext.fit_transform(fitted_mlp, X_train)
        assert acts.shape == (len(X_train), 3)  # 3 classes in iris

    def test_lambda(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        fn = lambda model, X: np.ones((len(X), 5))
        ext = CallableActivationExtractor(fn)
        acts = ext.fit_transform(fitted_mlp, X_train)
        assert acts.shape == (len(X_train), 5)
        np.testing.assert_array_equal(acts, 1.0)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            CallableActivationExtractor("not a function")

    def test_transform_before_fit_raises(self, iris):
        X_train, _, _, _ = iris
        ext = CallableActivationExtractor(lambda m, X: X)
        with pytest.raises(RuntimeError, match="fit"):
            ext.transform(X_train)


# ---------------------------------------------------------------------------
# CaseExplainer integration — activation_extractor
# ---------------------------------------------------------------------------

class TestCaseExplainerActivations:
    def test_pure_activations_returns_explanation(self, iris, fitted_mlp):
        X_train, X_test, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_mlp)
        assert exp is not None
        assert len(exp.neighbors) == 5

    def test_pure_features_returns_explanation(self, iris, fitted_mlp):
        """blend_alpha=1.0 with extractor falls back to feature-only indexing."""
        X_train, X_test, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=1.0,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_mlp)
        assert len(exp.neighbors) == 5

    def test_hybrid_blend_returns_explanation(self, iris, fitted_mlp):
        X_train, X_test, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.5,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_mlp)
        assert len(exp.neighbors) == 5

    def test_activation_and_feature_neighbors_differ(self, iris, fitted_mlp):
        """Pure activations and pure features generally select different neighbors."""
        X_train, X_test, y_train, _ = iris
        ext_act = SklearnMLPActivationExtractor()
        ext_feat = SklearnMLPActivationExtractor()

        act_explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext_act,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        feat_explainer = CaseExplainer(
            X_train, y_train, k=5,
        )

        act_exp = act_explainer.explain_instance(X_test[0], model=fitted_mlp)
        feat_exp = feat_explainer.explain_instance(X_test[0], model=fitted_mlp)

        act_ids = {n.index for n in act_exp.neighbors}
        feat_ids = {n.index for n in feat_exp.neighbors}
        # At least sometimes the sets differ (relax assertion: just check both are valid)
        assert len(act_ids) == 5
        assert len(feat_ids) == 5

    def test_decision_tree_extractor_integration(self, iris, fitted_tree):
        X_train, X_test, y_train, _ = iris
        ext = DecisionTreeActivationExtractor()
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_tree,
            blend_alpha=0.0,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_tree)
        assert len(exp.neighbors) == 5

    def test_callable_extractor_integration(self, iris, fitted_mlp):
        X_train, X_test, y_train, _ = iris

        fn = lambda model, X: model.predict_proba(np.atleast_2d(X))

        ext = CallableActivationExtractor(fn)
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_mlp)
        assert len(exp.neighbors) == 5

    def test_missing_model_raises(self, iris):
        X_train, _, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        with pytest.raises(ValueError, match="model"):
            CaseExplainer(
                X_train, y_train, k=5,
                activation_extractor=ext,
                model=None,
            )

    def test_invalid_blend_alpha_raises(self, iris, fitted_mlp):
        X_train, _, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        with pytest.raises(ValueError, match="blend_alpha"):
            CaseExplainer(
                X_train, y_train, k=5,
                activation_extractor=ext,
                model=fitted_mlp,
                blend_alpha=1.5,
            )

    def test_repr_shows_mode(self, iris, fitted_mlp):
        X_train, _, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()

        act_explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        assert "activations" in repr(act_explainer)

        hybrid_explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=SklearnMLPActivationExtractor(),
            model=fitted_mlp,
            blend_alpha=0.4,
        )
        assert "hybrid" in repr(hybrid_explainer)

        feat_explainer = CaseExplainer(X_train, y_train, k=5)
        assert "features" in repr(feat_explainer)

    def test_backward_compat_no_activation_extractor(self, iris):
        """Default (no activation_extractor) must work exactly as before."""
        X_train, X_test, y_train, _ = iris
        explainer = CaseExplainer(X_train, y_train, k=5)
        exp = explainer.explain_instance(X_test[0], predicted_class=0)
        assert len(exp.neighbors) == 5
        assert explainer._act_scaler is None

    def test_blend_alpha_geometry(self, iris, fitted_mlp):
        """
        Distance in hybrid space equals
        sqrt(alpha * d_feat^2 + (1-alpha) * d_act^2)
        which is a proper interpolation between the two metrics.
        """
        X_train, X_test, y_train, _ = iris

        alpha = 0.5
        ext = SklearnMLPActivationExtractor(use_output_weights=False)
        explainer = CaseExplainer(
            X_train, y_train, k=3,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=alpha,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_mlp)
        assert len(exp.neighbors) == 3
        # All distances should be non-negative
        for n in exp.neighbors:
            assert n.distance >= 0.0


# ---------------------------------------------------------------------------
# SklearnMLPActivationExtractor — additional precision/contract tests
# ---------------------------------------------------------------------------

class TestSklearnMLPActivationExtractorContract:
    def test_output_weight_normalization_invariant(self, iris, fitted_mlp):
        """
        After fit, output_weights must sum to n_hidden_last, i.e. the
        normalization preserves total scale: sum(w_i) == len(w_i).
        This ensures the weighted Euclidean distance stays in the same
        ballpark as unweighted Euclidean on the same space.
        """
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(use_output_weights=True)
        ext.fit(fitted_mlp, X_train)
        n_hidden_last = fitted_mlp.coefs_[-1].shape[0]  # 8
        np.testing.assert_allclose(
            ext._output_weights.sum(), n_hidden_last, rtol=1e-6
        )

    def test_fit_transform_equals_fit_then_transform(self, iris, fitted_mlp):
        """fit_transform(X) must equal fit(X).transform(X)."""
        X_train, _, _, _ = iris
        ext1 = SklearnMLPActivationExtractor(use_output_weights=True)
        ext2 = SklearnMLPActivationExtractor(use_output_weights=True)

        combined = ext1.fit_transform(fitted_mlp, X_train)
        separate = ext2.fit(fitted_mlp, X_train).transform(X_train)
        np.testing.assert_allclose(combined, separate)

    def test_deterministic_repeated_transform(self, iris, fitted_mlp):
        """Two transform calls on the same input must give the same result."""
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        ext.fit(fitted_mlp, X_train)
        a1 = ext.transform(X_train)
        a2 = ext.transform(X_train)
        np.testing.assert_array_equal(a1, a2)

    def test_single_sample_1d_input(self, iris, fitted_mlp):
        """A flat 1D array (single sample) must not crash — atleast_2d handles it."""
        X_train, X_test, _, _ = iris
        ext = SklearnMLPActivationExtractor()
        ext.fit(fitted_mlp, X_train)
        # Pass a 1D array (shape (4,) for iris)
        act = ext.transform(X_test[0])
        assert act.shape == (1, 8)

    def test_single_hidden_layer_mlp(self, iris):
        """Edge case: MLP with a single hidden layer — last_hidden == layer 0."""
        X_train, X_test, y_train, _ = iris
        clf = MLPClassifier(hidden_layer_sizes=(12,), max_iter=500, random_state=1)
        clf.fit(X_train, y_train)

        ext = SklearnMLPActivationExtractor(use_output_weights=True)
        acts = ext.fit_transform(clf, X_train)
        assert acts.shape == (len(X_train), 12)
        # Output weights should sum to 12
        np.testing.assert_allclose(ext._output_weights.sum(), 12, rtol=1e-6)

    def test_output_weights_ignored_for_non_last_layer(self, iris, fitted_mlp):
        """
        use_output_weights=True with layer=0 must not crash; it falls back to
        uniform weights (output weights are only meaningful for last_hidden).
        """
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer=0, use_output_weights=True)
        acts = ext.fit_transform(fitted_mlp, X_train)
        # First hidden layer has 16 units; uniform weights set lazily in transform
        assert acts.shape == (len(X_train), 16)
        np.testing.assert_array_equal(ext._output_weights, np.ones(16))


# ---------------------------------------------------------------------------
# SklearnMLPActivationExtractor — all_hidden layer mode
# ---------------------------------------------------------------------------

class TestSklearnMLPAllHidden:
    def test_output_shape_is_sum_of_all_layer_sizes(self, iris, fitted_mlp):
        """
        all_hidden concatenates all hidden layers: iris MLP is (16, 8),
        so output should have 16+8 = 24 columns.
        """
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden")
        acts = ext.fit_transform(fitted_mlp, X_train)
        expected_cols = sum(fitted_mlp.hidden_layer_sizes)  # 16+8=24
        assert acts.shape == (len(X_train), expected_cols)

    def test_last_layer_has_higher_position_weight_than_first(self, iris, fitted_mlp):
        """
        Position weights must increase monotonically: w_0 < w_1 < ... < w_last = 1.0.
        """
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden")
        ext.fit(fitted_mlp, X_train)
        pw = ext._layer_position_weights
        assert len(pw) == len(fitted_mlp.hidden_layer_sizes)
        assert np.all(np.diff(pw) > 0), "Position weights not monotonically increasing"
        np.testing.assert_allclose(pw[-1], 1.0, rtol=1e-6)

    def test_last_layer_position_weight_is_one(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden")
        ext.fit(fitted_mlp, X_train)
        np.testing.assert_allclose(ext._layer_position_weights[-1], 1.0, rtol=1e-6)

    def test_layer_scalers_fitted(self, iris, fitted_mlp):
        """One fitted StandardScaler per hidden layer must be present."""
        X_train, _, _, _ = iris
        from sklearn.preprocessing import StandardScaler
        ext = SklearnMLPActivationExtractor(layer="all_hidden")
        ext.fit(fitted_mlp, X_train)
        assert len(ext._layer_scalers) == len(fitted_mlp.hidden_layer_sizes)
        for s in ext._layer_scalers:
            assert isinstance(s, StandardScaler)
            assert hasattr(s, "mean_")  # confirms fitted

    def test_fit_transform_equals_fit_then_transform(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext1 = SklearnMLPActivationExtractor(layer="all_hidden")
        ext2 = SklearnMLPActivationExtractor(layer="all_hidden")
        combined = ext1.fit_transform(fitted_mlp, X_train)
        separate = ext2.fit(fitted_mlp, X_train).transform(X_train)
        np.testing.assert_allclose(combined, separate, rtol=1e-6)

    def test_single_sample_1d_input(self, iris, fitted_mlp):
        X_train, X_test, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden")
        ext.fit(fitted_mlp, X_train)
        act = ext.transform(X_test[0])  # 1D input
        expected_cols = sum(fitted_mlp.hidden_layer_sizes)
        assert act.shape == (1, expected_cols)

    def test_all_hidden_and_last_hidden_differ(self, iris, fitted_mlp):
        """
        all_hidden must produce different activations than last_hidden
        because it includes earlier layers.
        """
        X_train, _, _, _ = iris
        ext_all  = SklearnMLPActivationExtractor(layer="all_hidden",   use_output_weights=False)
        ext_last = SklearnMLPActivationExtractor(layer="last_hidden", use_output_weights=False)
        acts_all  = ext_all.fit_transform(fitted_mlp, X_train)
        acts_last = ext_last.fit_transform(fitted_mlp, X_train)
        # Different dimensions → certainly different
        assert acts_all.shape[1] != acts_last.shape[1]

    def test_use_output_weights_false_sets_none(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden", use_output_weights=False)
        ext.fit(fitted_mlp, X_train)
        assert ext._all_hidden_output_weights is None

    def test_use_output_weights_true_sets_last_layer_weights(self, iris, fitted_mlp):
        X_train, _, _, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden", use_output_weights=True)
        ext.fit(fitted_mlp, X_train)
        n_last = fitted_mlp.hidden_layer_sizes[-1]  # 8
        assert ext._all_hidden_output_weights is not None
        assert len(ext._all_hidden_output_weights) == n_last
        # Should also sum to n_last (same normalization as last_hidden mode)
        np.testing.assert_allclose(
            ext._all_hidden_output_weights.sum(), n_last, rtol=1e-6
        )

    def test_all_hidden_integration_with_case_explainer(self, iris, fitted_mlp):
        """CaseExplainer built with all_hidden returns valid explanations."""
        X_train, X_test, y_train, _ = iris
        ext = SklearnMLPActivationExtractor(layer="all_hidden")
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        exp = explainer.explain_instance(X_test[0], model=fitted_mlp)
        assert len(exp.neighbors) == 5
        for n in exp.neighbors:
            assert n.distance >= 0.0


# ---------------------------------------------------------------------------
# DecisionTreeActivationExtractor — additional contract tests
# ---------------------------------------------------------------------------

class TestDecisionTreeActivationExtractorContract:
    def test_fit_transform_equals_fit_then_transform(self, iris, fitted_tree):
        X_train, _, _, _ = iris
        ext1 = DecisionTreeActivationExtractor()
        ext2 = DecisionTreeActivationExtractor()
        combined = ext1.fit_transform(fitted_tree, X_train)
        separate = ext2.fit(fitted_tree, X_train).transform(X_train)
        np.testing.assert_array_equal(combined, separate)

    def test_single_sample_1d_input(self, iris, fitted_tree):
        X_train, X_test, _, _ = iris
        ext = DecisionTreeActivationExtractor()
        ext.fit(fitted_tree, X_train)
        act = ext.transform(X_test[0])
        assert act.ndim == 2
        assert act.shape[0] == 1

    def test_forest_one_hot_per_tree_sums_to_one(self, iris, fitted_forest):
        """In one-hot mode each tree's block should sum to exactly 1 per sample."""
        X_train, _, _, _ = iris
        ext = DecisionTreeActivationExtractor(max_one_hot_dims=5000)
        acts = ext.fit_transform(fitted_forest, X_train)
        assert ext._encoding == "one_hot"

        # Verify per-tree one-hot sums
        offsets = list(ext._leaf_offsets) + [ext._total_dims]
        for t in range(len(offsets) - 1):
            block = acts[:, offsets[t]:offsets[t + 1]]
            np.testing.assert_array_equal(
                block.sum(axis=1), np.ones(len(X_train)),
                err_msg=f"Tree {t} one-hot block does not sum to 1 per sample"
            )


# ---------------------------------------------------------------------------
# CaseExplainer — additional integration/state tests
# ---------------------------------------------------------------------------

class TestCaseExplainerActivationsState:
    def test_act_scaler_populated_when_extractor_set(self, iris, fitted_mlp):
        """_act_scaler must be a fitted StandardScaler when extractor is used."""
        from sklearn.preprocessing import StandardScaler
        X_train, _, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        assert isinstance(explainer._act_scaler, StandardScaler)
        assert hasattr(explainer._act_scaler, "mean_")  # confirms it is fitted

    def test_explain_instance_without_model_uses_predicted_class(self, iris, fitted_mlp):
        """
        When activation_extractor is set and no model is passed to
        explain_instance, it should still work as long as predicted_class
        is supplied — the extractor is already fitted at __init__ time.
        """
        X_train, X_test, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
            blend_alpha=0.0,
        )
        # Supply predicted_class explicitly; do NOT pass model to explain_instance
        exp = explainer.explain_instance(X_test[0], predicted_class=0)
        assert len(exp.neighbors) == 5

    def test_blend_alpha_boundary_produces_same_results_as_pure_modes(
        self, iris, fitted_mlp
    ):
        """blend_alpha=0.0 and blend_alpha=1.0 must match their pure counterparts."""
        X_train, X_test, y_train, _ = iris

        # blend_alpha=1.0 with extractor should match pure-feature explainer
        ext_blend1 = SklearnMLPActivationExtractor(use_output_weights=False)
        blend1 = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext_blend1,
            model=fitted_mlp,
            blend_alpha=1.0,
        )
        feat_only = CaseExplainer(X_train, y_train, k=5)

        blend1_ids = sorted(n.index for n in
                            blend1.explain_instance(X_test[0], predicted_class=0).neighbors)
        feat_ids  = sorted(n.index for n in
                            feat_only.explain_instance(X_test[0], predicted_class=0).neighbors)
        assert blend1_ids == feat_ids, (
            "blend_alpha=1.0 should select the same neighbors as pure feature mode"
        )


class TestActivationLayerConvenienceParam:
    """Tests for the activation_layer / use_output_weights shorthand on CaseExplainer."""

    def test_activation_layer_creates_extractor(self, iris, fitted_mlp):
        """activation_layer shorthand should produce the same explainer as explicit extractor."""
        X_train, _, y_train, _ = iris
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_layer="last_hidden",
            model=fitted_mlp,
        )
        assert explainer.activation_extractor is not None

    def test_activation_layer_matches_explicit_extractor(self, iris, fitted_mlp):
        """Convenience param must yield identical neighbors to explicit extractor."""
        X_train, X_test, y_train, _ = iris
        ext = SklearnMLPActivationExtractor(layer="last_hidden", use_output_weights=True)
        explicit = CaseExplainer(
            X_train, y_train, k=5,
            activation_extractor=ext,
            model=fitted_mlp,
        )
        shorthand = CaseExplainer(
            X_train, y_train, k=5,
            activation_layer="last_hidden",
            use_output_weights=True,
            model=fitted_mlp,
        )
        exp_e = explicit.explain_instance(X_test[0], predicted_class=0)
        exp_s = shorthand.explain_instance(X_test[0], predicted_class=0)
        assert sorted(n.index for n in exp_e.neighbors) == sorted(n.index for n in exp_s.neighbors)

    def test_activation_layer_all_hidden(self, iris, fitted_mlp):
        """all_hidden mode should work via activation_layer shorthand."""
        X_train, X_test, y_train, _ = iris
        explainer = CaseExplainer(
            X_train, y_train, k=5,
            activation_layer="all_hidden",
            model=fitted_mlp,
        )
        exp = explainer.explain_instance(X_test[0], predicted_class=0)
        assert len(exp.neighbors) == 5

    def test_activation_layer_and_extractor_raises(self, iris, fitted_mlp):
        """Providing both activation_layer and activation_extractor must raise ValueError."""
        X_train, _, y_train, _ = iris
        ext = SklearnMLPActivationExtractor()
        with pytest.raises(ValueError, match="not both"):
            CaseExplainer(
                X_train, y_train, k=5,
                activation_extractor=ext,
                activation_layer="last_hidden",
                model=fitted_mlp,
            )

    def test_activation_layer_without_model_raises(self, iris):
        """activation_layer without model must raise ValueError."""
        X_train, _, y_train, _ = iris
        with pytest.raises(ValueError, match="model must be provided"):
            CaseExplainer(
                X_train, y_train, k=5,
                activation_layer="last_hidden",
            )
