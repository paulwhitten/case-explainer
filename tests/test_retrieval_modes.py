"""Cross-cutting tests for 0.3.0 retrieval behaviour.

Covers batch/instance equivalence across every retrieval space, ``metric=``
handling in tree and forest modes, the retrieval-space fields on
``Explanation``, configurable correspondence thresholds, distance
normalization, and balanced class weights.
"""

import json

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

import case_explainer
from case_explainer import (
    Blend,
    CaseExplainer,
    Features,
    ForestProximity,
    HiddenActivations,
    TreeLeaf,
)
from case_explainer.metrics import (
    DEFAULT_THRESHOLDS,
    compute_correspondence,
    interpret_correspondence,
    validate_thresholds,
)


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


@pytest.fixture(scope="module")
def forest(cancer):
    X_train, _, y_train, _ = cancer
    return RandomForestClassifier(n_estimators=25, random_state=0).fit(X_train, y_train)


@pytest.fixture(scope="module")
def tree(cancer):
    X_train, _, y_train, _ = cancer
    return DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)


@pytest.fixture(scope="module")
def scaled_mlp(cancer):
    X_train, _, y_train, _ = cancer
    scaler = StandardScaler().fit(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=800, random_state=0)
    mlp.fit(scaler.transform(X_train), y_train)
    return mlp, scaler


def _strategies(forest, tree, scaled_mlp):
    mlp, scaler = scaled_mlp
    return {
        "features": Features(model=forest),
        "activations": HiddenActivations(model=mlp, input_transform=scaler),
        "class_weighted": HiddenActivations(
            model=mlp, input_transform=scaler, output_weighting="predicted_class"
        ),
        "hybrid": Blend(
            HiddenActivations(model=mlp, input_transform=scaler), features=0.3
        ),
        "tree_leaf": TreeLeaf(model=tree, overflow="nearest"),
        "tree_leaf_truncate": TreeLeaf(model=tree),
        "forest_proximity": ForestProximity(model=forest),
    }


# ---------------------------------------------------------------------------
# Batch path equals per-instance path in every retrieval space
# ---------------------------------------------------------------------------


class TestBatchEquivalence:
    @pytest.mark.parametrize(
        "name",
        [
            "features",
            "activations",
            "class_weighted",
            "hybrid",
            "tree_leaf",
            "tree_leaf_truncate",
            "forest_proximity",
        ],
    )
    def test_batch_matches_instance(self, cancer, forest, tree, scaled_mlp, name):
        X_train, X_test, y_train, y_test = cancer
        similarity = _strategies(forest, tree, scaled_mlp)[name]
        explainer = CaseExplainer(X_train, y_train, k=5, similarity=similarity)
        X_sub, y_sub = X_test[:25], y_test[:25]

        batch = explainer.explain_batch(X_sub, y_test=y_sub)
        assert len(batch) == len(X_sub)
        for i, (sample, label) in enumerate(zip(X_sub, y_sub)):
            single = explainer.explain_instance(sample, test_index=i, true_class=label)
            assert [n.index for n in batch[i].neighbors] == [
                n.index for n in single.neighbors
            ]
            np.testing.assert_allclose(
                [n.distance for n in batch[i].neighbors],
                [n.distance for n in single.neighbors],
            )
            assert batch[i].predicted_class == single.predicted_class
            assert batch[i].correspondence == pytest.approx(single.correspondence)
            assert batch[i].test_index == i
            assert batch[i].true_class == int(label)
            assert batch[i].retrieval_space == single.retrieval_space

    def test_batch_uses_one_predict_call(self, cancer):
        X_train, X_test, y_train, _ = cancer
        calls = []

        class Counting:
            def predict(self, X):
                calls.append(len(X))
                return np.zeros(len(X), dtype=int)

        explainer = CaseExplainer(X_train, y_train, k=3)
        explainer.explain_batch(X_test[:40], model=Counting())
        assert calls == [40]

    def test_batch_rejects_wrong_prediction_count(self, cancer):
        X_train, X_test, y_train, _ = cancer

        class Broken:
            def predict(self, X):
                return np.zeros(1, dtype=int)

        explainer = CaseExplainer(X_train, y_train, k=3)
        with pytest.raises(ValueError, match="predictions for"):
            explainer.explain_batch(X_test[:4], model=Broken())

    def test_batch_validates_label_lengths(self, cancer):
        X_train, X_test, y_train, _ = cancer
        explainer = CaseExplainer(X_train, y_train, k=3)
        with pytest.raises(ValueError, match="y_test has"):
            explainer.explain_batch(X_test[:4], y_test=[0, 1], predictions=[0] * 4)
        with pytest.raises(ValueError, match="predictions has"):
            explainer.explain_batch(X_test[:4], predictions=[0, 1])

    def test_batch_validates_feature_count(self, cancer):
        X_train, X_test, y_train, _ = cancer
        explainer = CaseExplainer(X_train, y_train, k=3)
        with pytest.raises(ValueError, match="X_test must have shape"):
            explainer.explain_batch(X_test[:4, :5], predictions=[0] * 4)

    def test_instance_accepts_single_row_matrix(self, cancer, forest):
        X_train, X_test, y_train, _ = cancer
        explainer = CaseExplainer(X_train, y_train, k=3)
        flat = explainer.explain_instance(X_test[0], model=forest)
        row = explainer.explain_instance(X_test[:1], model=forest)
        assert [n.index for n in flat.neighbors] == [n.index for n in row.neighbors]

    def test_instance_rejects_wrong_feature_count(self, cancer):
        X_train, X_test, y_train, _ = cancer
        explainer = CaseExplainer(X_train, y_train, k=3)
        with pytest.raises(ValueError, match="has 5 features, expected 30"):
            explainer.explain_instance(X_test[0, :5], predicted_class=0)


# ---------------------------------------------------------------------------
# metric= handling
# ---------------------------------------------------------------------------


class TestMetricHandling:
    def test_tree_leaf_honors_metric(self, cancer, tree):
        X_train, X_test, y_train, _ = cancer
        euclid = CaseExplainer(X_train, y_train, k=5, similarity=TreeLeaf(model=tree))
        manhattan = CaseExplainer(
            X_train, y_train, k=5, metric="manhattan", similarity=TreeLeaf(model=tree)
        )
        d_e = [n.distance for n in euclid.explain_instance(X_test[0]).neighbors]
        d_m = [n.distance for n in manhattan.explain_instance(X_test[0]).neighbors]
        assert not np.allclose(d_e, d_m)
        # Manhattan distance is never smaller than Euclidean distance.
        assert all(m >= e for m, e in zip(sorted(d_m), sorted(d_e)))
        assert manhattan.explain_instance(X_test[0]).distance_metric == "manhattan"

    def test_tree_leaf_euclidean_matches_norm(self, cancer, tree):
        X_train, X_test, y_train, _ = cancer
        explainer = CaseExplainer(
            X_train, y_train, k=5, similarity=TreeLeaf(model=tree)
        )
        exp = explainer.explain_instance(X_test[0])
        q = explainer.scaler.transform(X_test[:1])[0]
        for n in exp.neighbors:
            expected = np.linalg.norm(explainer.X_train_scaled[n.index] - q)
            assert n.distance == pytest.approx(expected, rel=1e-6, abs=1e-9)

    def test_forest_rejects_non_default_metric(self, cancer, forest):
        X_train, _, y_train, _ = cancer
        with pytest.raises(ValueError, match="ForestProximity defines its own"):
            CaseExplainer(
                X_train,
                y_train,
                metric="manhattan",
                similarity=ForestProximity(model=forest),
            )

    def test_forest_reports_its_metric(self, cancer, forest):
        X_train, X_test, y_train, _ = cancer
        explainer = CaseExplainer(
            X_train, y_train, k=5, similarity=ForestProximity(model=forest)
        )
        exp = explainer.explain_instance(X_test[0])
        assert exp.retrieval_space == "forest_proximity"
        assert exp.distance_metric == "shared_leaf_fraction"
        assert all(0.0 <= n.distance <= 1.0 for n in exp.neighbors)

    def test_feature_space_honors_metric(self, cancer, forest):
        X_train, X_test, y_train, _ = cancer
        a = CaseExplainer(X_train, y_train, k=5).explain_instance(
            X_test[0], model=forest
        )
        b = CaseExplainer(X_train, y_train, k=5, metric="manhattan").explain_instance(
            X_test[0], model=forest
        )
        assert [n.distance for n in a.neighbors] != [n.distance for n in b.neighbors]
        assert b.distance_metric == "manhattan"


# ---------------------------------------------------------------------------
# Explanation records its retrieval space
# ---------------------------------------------------------------------------


class TestExplanationRetrievalFields:
    def test_fields_in_dict_summary_and_repr(self, cancer, forest, scaled_mlp):
        X_train, X_test, y_train, _ = cancer
        mlp, scaler = scaled_mlp
        explainer = CaseExplainer(
            X_train,
            y_train,
            k=5,
            similarity=Blend(
                HiddenActivations(model=mlp, input_transform=scaler), features=0.3
            ),
        )
        exp = explainer.explain_instance(X_test[0])
        d = exp.to_dict()
        assert d["retrieval_space"] == "hybrid(features=0.30)"
        assert d["distance_metric"] == "euclidean"
        assert d["distance_scale"] == 1.0
        json.dumps(d)
        assert "Retrieval space: hybrid(features=0.30) (euclidean)" in exp.summary()
        assert "space='hybrid(features=0.30)'" in repr(exp)

    def test_explainer_exposes_retrieval_space(self, cancer, forest):
        X_train, _, y_train, _ = cancer
        assert CaseExplainer(X_train, y_train).retrieval_space == "features"
        assert (
            CaseExplainer(
                X_train, y_train, similarity=ForestProximity(model=forest)
            ).retrieval_space
            == "forest_proximity"
        )

    def test_explanation_defaults(self):
        exp = case_explainer.Explanation(
            test_sample=np.zeros(2),
            test_index=None,
            neighbors=[],
            predicted_class=0,
            true_class=None,
            correspondence=0.0,
            correspondence_interpretation="undefined",
        )
        assert exp.retrieval_space == "features"
        assert exp.distance_metric is None
        assert exp.distance_scale == 1.0
        assert exp.describe_retrieval_space() == "features"


# ---------------------------------------------------------------------------
# Configurable correspondence thresholds
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_defaults(self):
        assert DEFAULT_THRESHOLDS == (0.70, 0.85)
        assert interpret_correspondence(0.85) == "high"
        assert interpret_correspondence(0.70) == "medium"
        assert interpret_correspondence(0.69) == "low"

    def test_custom_thresholds_in_function(self):
        neighbors = [(0, 0.1, 1), (1, 0.2, 1), (2, 0.3, 0)]
        corr, default_interp = compute_correspondence(neighbors, predicted_class=1)
        _, strict = compute_correspondence(
            neighbors, predicted_class=1, thresholds=(0.9, 0.99)
        )
        _, lenient = compute_correspondence(
            neighbors, predicted_class=1, thresholds=(0.1, 0.5)
        )
        assert 0.5 < corr < 0.9
        assert default_interp == "medium"
        assert strict == "low"
        assert lenient == "high"

    @pytest.mark.parametrize("bad", [(0.9, 0.5), (-0.1, 0.5), (0.5, 1.5), (0.5,), "x"])
    def test_invalid_thresholds_raise(self, bad):
        with pytest.raises(ValueError, match="thresholds"):
            validate_thresholds(bad)

    def test_explainer_thresholds(self, cancer, forest):
        X_train, X_test, y_train, _ = cancer
        strict = CaseExplainer(
            X_train, y_train, k=5, correspondence_thresholds=(0.99, 0.999)
        )
        exp = strict.explain_instance(X_test[0], model=forest)
        assert strict.correspondence_thresholds == (0.99, 0.999)
        if exp.correspondence < 0.99:
            assert exp.correspondence_interpretation == "low"
        # The method re-scores with the same thresholds.
        assert strict.compute_correspondence(exp) == (
            exp.correspondence,
            exp.correspondence_interpretation,
        )
        with pytest.raises(ValueError, match="thresholds"):
            CaseExplainer(X_train, y_train, correspondence_thresholds=(0.9, 0.1))


# ---------------------------------------------------------------------------
# Distance normalization
# ---------------------------------------------------------------------------


class TestDistanceNormalization:
    @pytest.mark.parametrize(
        "name",
        [
            "features",
            "activations",
            "class_weighted",
            "hybrid",
            "tree_leaf",
            "forest_proximity",
        ],
    )
    def test_median_nn_scales_distances_without_changing_neighbors(
        self, cancer, forest, tree, scaled_mlp, name
    ):
        X_train, X_test, y_train, _ = cancer
        similarity = _strategies(forest, tree, scaled_mlp)[name]
        raw = CaseExplainer(X_train, y_train, k=5, similarity=similarity)
        normalized = CaseExplainer(
            X_train,
            y_train,
            k=5,
            similarity=similarity,
            distance_normalization="median_nn",
        )
        scale = normalized.distance_scale_
        assert scale > 0.0
        assert raw.distance_scale_ == 1.0
        for sample in X_test[:10]:
            a = raw.explain_instance(sample)
            b = normalized.explain_instance(sample)
            assert [n.index for n in a.neighbors] == [n.index for n in b.neighbors]
            np.testing.assert_allclose(
                [n.distance for n in b.neighbors],
                np.array([n.distance for n in a.neighbors]) / scale,
            )
            assert b.distance_scale == scale
            assert "normalized by" in b.summary()

    def test_median_nn_makes_typical_neighbor_distance_about_one(self, cancer, forest):
        """After normalization a training case's nearest other case sits near 1."""
        X_train, _, y_train, _ = cancer
        explainer = CaseExplainer(
            X_train, y_train, k=2, distance_normalization="median_nn"
        )
        second = [
            explainer.explain_instance(x, predicted_class=0).neighbors[1].distance
            for x in X_train[:200]
        ]
        assert 0.5 < np.median(second) < 2.0

    def test_median_nn_aligns_scales_across_spaces(self, cancer, forest, scaled_mlp):
        """Raw scales differ by an order of magnitude; normalized ones do not."""
        X_train, X_test, y_train, _ = cancer
        mlp, scaler = scaled_mlp
        spaces = [
            Features(model=forest),
            HiddenActivations(model=mlp, input_transform=scaler),
        ]

        def mean_nearest(normalization):
            out = []
            for similarity in spaces:
                explainer = CaseExplainer(
                    X_train,
                    y_train,
                    k=1,
                    similarity=similarity,
                    distance_normalization=normalization,
                )
                out.append(
                    np.mean(
                        [
                            e.neighbors[0].distance
                            for e in explainer.explain_batch(X_test[:50])
                        ]
                    )
                )
            return out

        raw_features, raw_activations = mean_nearest(None)
        norm_features, norm_activations = mean_nearest("median_nn")
        raw_ratio = max(raw_features, raw_activations) / min(
            raw_features, raw_activations
        )
        norm_ratio = max(norm_features, norm_activations) / min(
            norm_features, norm_activations
        )
        assert raw_ratio > 3.0
        assert norm_ratio < raw_ratio
        assert norm_ratio < 2.0

    def test_invalid_normalization_raises(self, cancer):
        X_train, _, y_train, _ = cancer
        with pytest.raises(ValueError, match="distance_normalization"):
            CaseExplainer(X_train, y_train, distance_normalization="zscore")

    def test_single_training_case_warns_and_skips(self):
        with pytest.warns(RuntimeWarning, match="at least two"):
            explainer = CaseExplainer(
                np.array([[0.0, 1.0]]),
                np.array([0]),
                scale_data=False,
                distance_normalization="median_nn",
            )
        assert explainer.distance_scale_ == 1.0

    def test_duplicates_are_excluded_from_the_scale(self):
        X = np.array([[0.0], [0.0], [0.0], [5.0], [5.0], [5.0], [9.0]])
        y = np.zeros(len(X), dtype=int)
        explainer = CaseExplainer(
            X, y, k=2, scale_data=False, distance_normalization="median_nn"
        )
        assert explainer.distance_scale_ == pytest.approx(4.0)

    def test_near_duplicates_from_float_noise_are_excluded(self):
        """Noise-level distances must not become the normalization scale."""
        rng = np.random.RandomState(0)
        base = rng.randn(40, 3)
        # 30 of the 40 cases get a copy perturbed at floating-point noise level;
        # the other 10 have a genuinely distinct nearest neighbor.
        noisy_copies = base[:30] + 1e-9 * rng.randn(30, 3)
        X = np.vstack([base, noisy_copies])
        y = np.zeros(len(X), dtype=int)
        explainer = CaseExplainer(
            X, y, k=2, scale_data=False, distance_normalization="median_nn"
        )
        assert explainer.distance_scale_ > 0.1

    def test_all_duplicates_warns_and_skips(self):
        X = np.zeros((6, 2))
        y = np.zeros(6, dtype=int)
        with pytest.warns(RuntimeWarning, match="duplicate"):
            explainer = CaseExplainer(
                X, y, k=2, scale_data=False, distance_normalization="median_nn"
            )
        assert explainer.distance_scale_ == 1.0

    def test_training_info_reports_normalization(self, cancer):
        X_train, _, y_train, _ = cancer
        info = CaseExplainer(
            X_train, y_train, distance_normalization="median_nn"
        ).get_training_info()
        assert info["distance_normalization"] == "median_nn"
        assert info["distance_scale"] > 0
        assert info["correspondence_thresholds"] == (0.70, 0.85)


# ---------------------------------------------------------------------------
# Public surface after the 0.3.0 cleanup
# ---------------------------------------------------------------------------


class _ConstantModel:
    """Minimal predictor that always returns one label."""

    def __init__(self, label):
        self.label = label

    def predict(self, X):
        return np.full(len(X), self.label)


def _imbalanced_points():
    """100 majority and 4 minority points; the origin has 4 + 1 near neighbors."""
    rng = np.random.RandomState(0)
    far_majority = rng.normal(loc=20.0, scale=1.0, size=(96, 2))
    near_majority = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    near_minority = np.array([[0.7, 0.7]])
    far_minority = np.array([[-20.0, -20.0], [-21.0, -20.0], [-20.0, -21.0]])
    X = np.vstack([far_majority, near_majority, near_minority, far_minority])
    y = np.array([0] * 100 + [1] * 4)
    return X, y


class TestClassWeights:
    def test_default_is_unweighted(self):
        X, y = _imbalanced_points()
        assert CaseExplainer(X, y).class_weights == {}

    def test_mapping_is_copied(self):
        X, y = _imbalanced_points()
        weights = {0: 1.0, 1: 5.0}
        explainer = CaseExplainer(X, y, class_weights=weights)
        assert explainer.class_weights == weights
        assert explainer.class_weights is not weights

    def test_balanced_matches_sklearn(self):
        X, y = _imbalanced_points()
        explainer = CaseExplainer(X, y, class_weights="balanced")
        expected = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        assert set(explainer.class_weights) == {0, 1}
        assert explainer.class_weights[0] == pytest.approx(expected[0])
        assert explainer.class_weights[1] == pytest.approx(expected[1])

    def test_balanced_on_balanced_data_changes_nothing(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(60, 3))
        y = np.array([0, 1, 2] * 20)
        model = _ConstantModel(1)
        plain = CaseExplainer(X, y, k=5, model=model)
        balanced = CaseExplainer(X, y, k=5, model=model, class_weights="balanced")
        assert set(balanced.class_weights.values()) == {1.0}
        for a, b in zip(plain.explain_batch(X[:10]), balanced.explain_batch(X[:10])):
            assert b.correspondence == pytest.approx(a.correspondence)

    def test_balanced_exposes_a_minority_neighbor(self):
        """One rare-class neighbor among five must not leave the score high."""
        X, y = _imbalanced_points()
        model = _ConstantModel(0)
        query = np.zeros(2)
        plain = CaseExplainer(X, y, k=5, model=model, scale_data=False)
        balanced = CaseExplainer(
            X, y, k=5, model=model, scale_data=False, class_weights="balanced"
        )
        unweighted = plain.explain_instance(query)
        weighted = balanced.explain_instance(query)

        assert [n.index for n in weighted.neighbors] == [
            n.index for n in unweighted.neighbors
        ]
        assert sorted(n.label for n in unweighted.neighbors) == [0, 0, 0, 0, 1]
        assert unweighted.correspondence > 0.70
        assert unweighted.correspondence_interpretation != "low"
        assert weighted.correspondence < 0.30
        assert weighted.correspondence_interpretation == "low"

    def test_balanced_supports_a_minority_prediction(self):
        """A rare-class prediction backed by its rare-class neighbor scores up."""
        X, y = _imbalanced_points()
        model = _ConstantModel(1)
        query = np.zeros(2)
        plain = CaseExplainer(X, y, k=5, model=model, scale_data=False)
        balanced = CaseExplainer(
            X, y, k=5, model=model, scale_data=False, class_weights="balanced"
        )
        assert (
            balanced.explain_instance(query).correspondence
            > plain.explain_instance(query).correspondence
        )

    def test_balanced_with_string_labels(self):
        X, y = _imbalanced_points()
        labels = np.where(y == 1, "fraud", "legit")
        explainer = CaseExplainer(
            X,
            labels,
            k=5,
            model=_ConstantModel("legit"),
            scale_data=False,
            class_weights="balanced",
        )
        assert set(explainer.class_weights) == {"fraud", "legit"}
        assert explainer.class_weights["fraud"] > explainer.class_weights["legit"]
        explanation = explainer.explain_instance(np.zeros(2))
        assert explanation.correspondence_interpretation == "low"

    def test_batch_matches_instance_with_balanced_weights(self):
        X, y = _imbalanced_points()
        explainer = CaseExplainer(
            X, y, k=5, model=_ConstantModel(0), class_weights="balanced"
        )
        queries = np.array([[0.0, 0.0], [20.0, 20.0], [-20.0, -20.0]])
        batch = explainer.explain_batch(queries)
        for row, explanation in zip(queries, batch):
            single = explainer.explain_instance(row)
            assert explanation.correspondence == pytest.approx(single.correspondence)

    @pytest.mark.parametrize("bad", ["auto", "Balanced", ""])
    def test_unknown_string_raises(self, bad):
        X, y = _imbalanced_points()
        with pytest.raises(ValueError, match="balanced"):
            CaseExplainer(X, y, class_weights=bad)


class TestPublicSurface:
    def test_distance_helpers_live_in_metrics_only(self):
        from case_explainer import metrics

        assert callable(metrics.euclidean_distance)
        assert callable(metrics.compute_all_distances)
        assert "euclidean_distance" not in case_explainer.__all__
        assert not hasattr(case_explainer, "euclidean_distance")

    def test_indexing_module_removed(self):
        with pytest.raises(ImportError):
            __import__("case_explainer.indexing")

    def test_version(self):
        assert case_explainer.__version__ == "0.3.0"
