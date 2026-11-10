"""Regression tests for input handling found in the 0.3.0 pre-release review.

Covers array-like names, positional metadata, label and prediction shapes,
``k`` validation, empty batches, class-weight validation, bounded memory in
tree-leaf retrieval, and DataFrame column handling.
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.datasets import load_breast_cancer, make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from case_explainer import (  # noqa: E402
    Blend,
    CaseExplainer,
    CustomActivations,
    ForestProximity,
    HiddenActivations,
    TreeLeaf,
)
from case_explainer import explainer as explainer_module  # noqa: E402
from case_explainer.activations import (  # noqa: E402
    CallableActivationExtractor,
    SklearnMLPActivationExtractor,
)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    return X, y


@pytest.fixture(scope="module")
def logistic(data):
    X, y = data
    return LogisticRegression(max_iter=500).fit(X, y)


@pytest.fixture(scope="module")
def mlp(data):
    X, y = data
    return MLPClassifier(hidden_layer_sizes=(8, 6), max_iter=400, random_state=0).fit(
        X, y
    )


@pytest.fixture(scope="module")
def tree(data):
    X, y = data
    return DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def forest(data):
    X, y = data
    return RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0).fit(
        X, y
    )


class TestNames:
    def test_documented_basic_example_accepts_array_feature_names(self):
        """sklearn's ``feature_names`` is an ndarray; it used to raise."""
        bunch = load_breast_cancer()
        explainer = CaseExplainer(
            X_train=bunch.data,
            y_train=bunch.target,
            feature_names=bunch.feature_names,
            class_names=bunch.target_names,
        )
        assert explainer.feature_names == list(bunch.feature_names)
        assert all(type(name) is str for name in explainer.feature_names)
        assert explainer.class_names == {0: "malignant", 1: "benign"}

    def test_pandas_index_feature_names(self, data):
        X, y = data
        explainer = CaseExplainer(X, y, feature_names=pd.Index(list("abcdef")))
        assert explainer.feature_names == list("abcdef")

    def test_feature_names_are_copied(self, data):
        X, y = data
        names = list("abcdef")
        explainer = CaseExplainer(X, y, feature_names=names)
        names[0] = "changed"
        assert explainer.feature_names[0] == "a"

    def test_wrong_number_of_feature_names_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="feature_names has 2 names for 6"):
            CaseExplainer(X, y, feature_names=["a", "b"])

    def test_class_names_mapping_is_copied(self, data):
        X, y = data
        names = {0: "zero", 1: "one", 2: "two"}
        explainer = CaseExplainer(X, y, class_names=names)
        names[0] = "changed"
        assert explainer.class_names[0] == "zero"

    def test_class_names_sequence_follows_sorted_labels(self, data):
        X, y = data
        labels = np.array(["c", "a", "b"])[y]
        explainer = CaseExplainer(X, labels, class_names=["A", "B", "C"])
        assert explainer.class_names == {"a": "A", "b": "B", "c": "C"}

    def test_class_names_sequence_of_wrong_length_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="class_names has 2 names for 3"):
            CaseExplainer(X, y, class_names=["only", "two"])


class TestMetadata:
    def test_series_metadata_is_read_by_position(self, data):
        """A shuffled Series index must not redirect the provenance lookup."""
        X, y = data
        rng = np.random.RandomState(1)
        order = rng.permutation(len(X))
        ids = pd.Series([f"s{i}" for i in range(len(X))], index=order + 1000)
        explainer = CaseExplainer(X, y, metadata={"sid": ids})
        explanation = explainer.explain_instance(X[17], predicted_class=int(y[17]))
        nearest = explanation.neighbors[0]
        assert nearest.index == 17
        assert nearest.metadata == {"sid": "s17"}

    def test_dataframe_metadata(self, data):
        X, y = data
        frame = pd.DataFrame(
            {"sid": [f"s{i}" for i in range(len(X))], "site": np.arange(len(X)) % 3}
        )
        explainer = CaseExplainer(X, y, metadata=frame)
        nearest = explainer.explain_instance(X[5], predicted_class=0).neighbors[0]
        assert nearest.metadata == {"sid": "s5", "site": 2}

    def test_array_metadata_is_json_serializable(self, data):
        X, y = data
        explainer = CaseExplainer(X, y, metadata={"row": np.arange(len(X))})
        explanation = explainer.explain_instance(X[3], predicted_class=0)
        assert type(explanation.neighbors[0].metadata["row"]) is int
        json.dumps(explanation.to_dict())

    def test_metadata_is_copied(self, data):
        X, y = data
        ids = [f"s{i}" for i in range(len(X))]
        explainer = CaseExplainer(X, y, metadata={"sid": ids})
        ids[4] = "MUTATED"
        nearest = explainer.explain_instance(X[4], predicted_class=0).neighbors[0]
        assert nearest.metadata == {"sid": "s4"}

    def test_short_metadata_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="Metadata 'sid' has 3 items"):
            CaseExplainer(X, y, metadata={"sid": ["a", "b", "c"]})

    def test_non_mapping_metadata_raises(self, data):
        X, y = data
        with pytest.raises(TypeError, match="metadata must be a mapping"):
            CaseExplainer(X, y, metadata=["a"] * len(X))

    def test_numpy_test_index_is_json_serializable(self, data, logistic):
        X, y = data
        explainer = CaseExplainer(X, y)
        for index in np.arange(2):
            explanation = explainer.explain_instance(
                X[index], model=logistic, test_index=index
            )
            assert type(explanation.test_index) is int
            json.dumps(explanation.to_dict())


class TestLabelShapes:
    def test_column_vector_labels_are_flattened(self, data):
        X, y = data
        explainer = CaseExplainer(X, y.reshape(-1, 1))
        assert explainer.y_train.shape == (len(X),)
        explainer.explain_instance(X[0], predicted_class=0)

    def test_single_column_dataframe_labels(self, data):
        X, y = data
        frame = pd.DataFrame({"y": y}, index=np.arange(len(y)) + 500)
        explainer = CaseExplainer(X, frame)
        assert explainer.explain_instance(X[9], predicted_class=0).neighbors[
            0
        ].label == int(y[9])

    def test_tuple_labels(self, data):
        X, y = data
        assert CaseExplainer(X, tuple(y)).n_samples == len(X)

    def test_two_dimensional_labels_raise(self, data):
        X, y = data
        with pytest.raises(ValueError, match="y_train must hold one class label"):
            CaseExplainer(X, np.eye(3)[y])

    def test_one_dimensional_X_train_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="X_train must be a dense 2-D array"):
            CaseExplainer(X[:, 0], y)

    def test_probability_output_from_model_raises_clearly(self, data, logistic):
        X, y = data

        class Probabilities:
            def predict(self, inputs):
                return logistic.predict_proba(inputs)

        explainer = CaseExplainer(X, y)
        with pytest.raises(ValueError, match="class probabilities"):
            explainer.explain_instance(X[0], model=Probabilities())
        with pytest.raises(ValueError, match="model.predict output"):
            explainer.explain_batch(X[:4], model=Probabilities())

    def test_column_vector_predictions_are_accepted(self, data, logistic):
        X, y = data
        explainer = CaseExplainer(X, y)
        flat = explainer.explain_batch(X[:5], predictions=logistic.predict(X[:5]))
        column = explainer.explain_batch(
            X[:5], predictions=logistic.predict(X[:5]).reshape(-1, 1)
        )
        assert [e.correspondence for e in column] == [e.correspondence for e in flat]

    def test_multi_row_sample_message(self, data):
        X, y = data
        with pytest.raises(ValueError, match="single sample.*explain_batch"):
            CaseExplainer(X, y).explain_instance(X[:3], predicted_class=0)


class TestKValidation:
    @pytest.mark.parametrize("bad", [0, -1, -3])
    def test_non_positive_k_raises_at_construction(self, data, bad):
        X, y = data
        with pytest.raises(ValueError, match="k must be a positive integer"):
            CaseExplainer(X, y, k=bad)

    @pytest.mark.parametrize("bad", [5.0, "5", None, True])
    def test_non_integer_k_raises_at_construction(self, data, bad):
        X, y = data
        with pytest.raises(TypeError, match="k must be a positive integer"):
            CaseExplainer(X, y, k=bad)

    def test_bad_per_call_k_raises_in_every_space(self, data, mlp, tree, forest):
        X, y = data
        strategies = [
            None,
            HiddenActivations(mlp),
            TreeLeaf(tree),
            ForestProximity(forest),
        ]
        for strategy in strategies:
            explainer = CaseExplainer(X, y, similarity=strategy)
            for bad in (0, -1):
                with pytest.raises(ValueError, match="k must be a positive integer"):
                    explainer.explain_instance(X[0], predicted_class=0, k=bad)
            with pytest.raises(TypeError, match="k must be a positive integer"):
                explainer.explain_batch(X[:2], predictions=[0, 0], k=2.0)

    def test_numpy_integer_k_is_accepted(self, data):
        X, y = data
        explainer = CaseExplainer(X, y, k=np.int64(4))
        assert explainer.k == 4 and type(explainer.k) is int
        explanation = explainer.explain_instance(X[0], predicted_class=0, k=np.int32(2))
        assert len(explanation.neighbors) == 2


class TestEmptyBatch:
    def test_empty_batch_returns_empty_list_in_every_space(
        self, data, mlp, tree, forest
    ):
        X, y = data
        strategies = [
            None,
            HiddenActivations(mlp),
            HiddenActivations(mlp, output_weighting="predicted_class"),
            Blend(HiddenActivations(mlp), features=0.3),
            TreeLeaf(tree),
            ForestProximity(forest),
        ]
        for strategy in strategies:
            explainer = CaseExplainer(X, y, similarity=strategy)
            assert explainer.explain_batch(np.empty((0, 6))) == []
            assert explainer.explain_batch([], predictions=[]) == []
            assert explainer.explain_batch(pd.DataFrame(X[:0]), y_test=[]) == []

    def test_empty_batch_still_checks_lengths(self, data):
        X, y = data
        with pytest.raises(ValueError, match="predictions has 2 labels for 0"):
            CaseExplainer(X, y).explain_batch(np.empty((0, 6)), predictions=[0, 1])


class TestClassWeightValidation:
    def test_non_mapping_raises(self, data):
        X, y = data
        with pytest.raises(TypeError, match="class_weights must be a mapping"):
            CaseExplainer(X, y, class_weights=[1.0, 2.0, 3.0])

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "2", None])
    def test_invalid_weight_raises(self, data, bad):
        X, y = data
        with pytest.raises(ValueError, match="non-negative finite number"):
            CaseExplainer(X, y, class_weights={0: bad})

    def test_unknown_label_warns(self, data):
        X, y = data
        with pytest.warns(UserWarning, match="do not occur in y_train"):
            CaseExplainer(X, y, class_weights={"1": 10.0})

    def test_known_labels_do_not_warn(self, data):
        X, y = data
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CaseExplainer(X, y, class_weights={np.int64(0): 2.0, 1: 1.0})


class TestModelHandling:
    def test_unfitted_forest_gives_the_intended_error(self, data):
        X, y = data
        with pytest.raises(ValueError, match="ForestProximity model must be fitted"):
            CaseExplainer(X, y, similarity=ForestProximity(RandomForestClassifier()))

    def test_bound_model_with_zero_length_is_kept(self, data, logistic):
        X, y = data

        class EmptyLooking:
            def __len__(self):
                return 0

            def predict(self, inputs):
                return logistic.predict(inputs)

        explainer = CaseExplainer(X, y, model=EmptyLooking())
        assert explainer.explain_instance(X[0]).predicted_class == int(
            logistic.predict(X[:1])[0]
        )

    def test_predicted_class_weighting_needs_an_mlp(self, data, forest):
        X, y = data
        with pytest.raises(TypeError, match="requires a fitted sklearn MLPClassifier"):
            CaseExplainer(
                X,
                y,
                similarity=HiddenActivations(
                    forest, output_weighting="predicted_class"
                ),
            )

    def test_tree_leaf_validates_overflow_at_construction(self, tree):
        with pytest.raises(ValueError, match="overflow must be"):
            TreeLeaf(tree, overflow="bogus")

    def test_pure_feature_blend_is_labelled_features(self, data, mlp):
        X, y = data
        for weighting in ("mean_abs", "predicted_class"):
            explainer = CaseExplainer(
                X,
                y,
                similarity=Blend(
                    HiddenActivations(mlp, output_weighting=weighting), features=1.0
                ),
            )
            assert explainer.retrieval_space == "features"


class TestIntegerLastLayer:
    def test_integer_index_of_last_layer_matches_last_hidden(self, data, mlp):
        """``layer=1`` on a two-hidden-layer MLP is the last hidden layer."""
        X, _ = data
        by_name = SklearnMLPActivationExtractor(layer="last_hidden")
        by_index = SklearnMLPActivationExtractor(layer=1)
        np.testing.assert_allclose(
            by_index.fit_transform(mlp, X), by_name.fit_transform(mlp, X)
        )
        assert not np.allclose(by_index._output_weights, 1.0)

    def test_earlier_integer_layer_keeps_uniform_weights(self, data, mlp):
        X, _ = data
        extractor = SklearnMLPActivationExtractor(layer=0)
        extractor.fit_transform(mlp, X)
        np.testing.assert_array_equal(extractor._output_weights, np.ones(8))

    def test_explainer_retrieval_matches_for_last_layer_index(self, data, mlp):
        X, y = data
        by_name = CaseExplainer(X, y, similarity=HiddenActivations(mlp))
        by_index = CaseExplainer(X, y, similarity=HiddenActivations(mlp, layer=1))
        queries = X[:10] + 0.01
        for a, b in zip(
            by_name.explain_batch(queries, predictions=y[:10]),
            by_index.explain_batch(queries, predictions=y[:10]),
        ):
            assert [n.index for n in a.neighbors] == [n.index for n in b.neighbors]


class TestSerialSingleQuery:
    """Single-row k-NN queries bypass joblib without touching the shared index."""

    def test_single_query_uses_one_thread(self, data, logistic, monkeypatch):
        X, y = data
        explainer = CaseExplainer(X, y, model=logistic, algorithm="kd_tree", n_jobs=-1)
        seen = []
        original = explainer_module.NearestNeighbors.kneighbors

        def spy(self, *args, **kwargs):
            seen.append(self.n_jobs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(explainer_module.NearestNeighbors, "kneighbors", spy)
        explainer.explain_instance(X[0])
        explainer.explain_batch(X[:2])
        assert seen == [1, -1]
        assert explainer.nn_index.n_jobs == -1  # the shared index is untouched

    def test_serial_and_parallel_results_agree(self, data, mlp):
        X, y = data
        for similarity in (
            None,
            HiddenActivations(mlp, output_weighting="predicted_class"),
        ):
            explainer = CaseExplainer(
                X, y, model=mlp, similarity=similarity, algorithm="ball_tree", n_jobs=-1
            )
            single = [explainer.explain_instance(row) for row in X[:6]]
            batch = explainer.explain_batch(X[:6])
            for a, b in zip(single, batch):
                assert [n.index for n in a.neighbors] == [n.index for n in b.neighbors]
                assert a.correspondence == pytest.approx(b.correspondence)

    def test_explicit_n_jobs_one_is_not_copied(self, data, logistic):
        X, y = data
        explainer = CaseExplainer(X, y, model=logistic, n_jobs=1)
        distances, indices = explainer_module._kneighbors(explainer.nn_index, X[:1], 3)
        assert indices.shape == (1, 3)


class TestTreeLeafStreaming:
    def test_small_chunks_give_identical_results(self, data, tree, monkeypatch):
        X, y = data
        queries = X[:40] + 0.01
        for overflow in ("truncate", "nearest"):
            explainer = CaseExplainer(
                X, y, similarity=TreeLeaf(tree, overflow=overflow), metric="manhattan"
            )
            expected = explainer.explain_batch(queries, predictions=y[:40])
            # Force three-row distance blocks.
            monkeypatch.setattr(
                explainer_module, "_TREE_LEAF_CHUNK_ELEMENTS", 3 * len(X)
            )
            chunked = explainer.explain_batch(queries, predictions=y[:40])
            monkeypatch.undo()
            for a, b in zip(expected, chunked):
                assert [n.index for n in a.neighbors] == [n.index for n in b.neighbors]
                assert [n.distance for n in a.neighbors] == [
                    n.distance for n in b.neighbors
                ]

    def test_distance_blocks_are_bounded(self, data, tree, monkeypatch):
        X, y = data
        explainer = CaseExplainer(X, y, similarity=TreeLeaf(tree))
        monkeypatch.setattr(explainer_module, "_TREE_LEAF_CHUNK_ELEMENTS", 5 * len(X))
        blocks = list(explainer._strategy._distance_rows(explainer, X[:23]))
        assert [start for start, _ in blocks] == [0, 5, 10, 15, 20]
        assert max(len(block) for _, block in blocks) == 5


class TestPlot:
    def test_saving_closes_the_figure(self, data, tmp_path):
        X, y = data
        explanation = CaseExplainer(X, y).explain_instance(X[0], predicted_class=0)
        before = len(plt.get_fignums())
        for i in range(3):
            explanation.plot(save_path=str(tmp_path / f"plot{i}.png"))
        assert len(plt.get_fignums()) == before
        assert (tmp_path / "plot2.png").stat().st_size > 0

    def test_show_distances_controls_the_legend(self, data, tmp_path, monkeypatch):
        X, y = data
        explanation = CaseExplainer(X, y).explain_instance(X[0], predicted_class=0)
        labels = {}

        def capture(figure, *args, **kwargs):
            _, text = figure.axes[0].get_legend_handles_labels()
            labels["text"] = text

        monkeypatch.setattr(plt, "savefig", lambda *a, **k: capture(plt.gcf()))
        explanation.plot(save_path=str(tmp_path / "a.png"), show_distances=True)
        assert all("(d=" in text for text in labels["text"][1:])
        explanation.plot(save_path=str(tmp_path / "b.png"), show_distances=False)
        assert not any("(d=" in text for text in labels["text"])


@pytest.fixture(scope="module")
def frame(data):
    X, y = data
    columns = [f"col_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


@pytest.fixture(scope="module")
def frame_model(frame):
    X, y = frame
    return LogisticRegression(max_iter=500).fit(X, y)


def _same_results(a, b):
    return [
        (e.predicted_class, e.correspondence, [n.index for n in e.neighbors]) for e in a
    ] == [
        (e.predicted_class, e.correspondence, [n.index for n in e.neighbors]) for e in b
    ]


class TestDataFrameInputs:
    """Models fitted on a DataFrame keep receiving one; queries align by name."""

    def test_name_based_pipeline_works_as_model(self, frame):
        X, y = frame
        selected = list(X.columns[:4])
        pipe = make_pipeline(
            ColumnTransformer([("sel", StandardScaler(), selected)]),
            LogisticRegression(max_iter=500),
        ).fit(X, y)
        explainer = CaseExplainer(X, y, model=pipe)
        row = explainer.explain_instance(X.iloc[5])
        batch = explainer.explain_batch(X.iloc[:5])
        assert row.predicted_class == pipe.predict(X.iloc[[5]])[0]
        assert [e.predicted_class for e in batch] == list(pipe.predict(X.iloc[:5]))

    def test_frame_fitted_model_does_not_warn(self, frame):
        X, y = frame
        forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = CaseExplainer(X, y, model=forest)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            explainer.explain_instance(X.iloc[0])
            explainer.explain_instance(X.iloc[0].to_numpy())
            explainer.explain_batch(X.iloc[:4].to_numpy())

    def test_reordered_query_columns_match_original_order(self, frame, frame_model):
        X, y = frame
        explainer = CaseExplainer(X, y, model=frame_model)
        reversed_cols = list(X.columns)[::-1]
        single = explainer.explain_instance(X.iloc[[3]])
        reordered = explainer.explain_instance(X.iloc[[3]][reversed_cols])
        assert _same_results([single], [reordered])
        batch = explainer.explain_batch(X.iloc[:8])
        batch_reordered = explainer.explain_batch(X.iloc[:8][reversed_cols])
        assert _same_results(batch, batch_reordered)

    def test_series_row_aligns_by_label(self, frame, frame_model):
        X, y = frame
        explainer = CaseExplainer(X, y, model=frame_model)
        row = X.iloc[3]
        shuffled = row.iloc[::-1]
        assert _same_results(
            [explainer.explain_instance(row)], [explainer.explain_instance(shuffled)]
        )

    def test_missing_and_unexpected_columns_raise(self, frame, frame_model):
        X, y = frame
        explainer = CaseExplainer(X, y, model=frame_model)
        dropped = X.iloc[:3].drop(columns="col_0")
        with pytest.raises(ValueError, match="missing \\['col_0'\\]"):
            explainer.explain_batch(dropped)
        renamed = X.iloc[:3].rename(columns={"col_0": "other"})
        with pytest.raises(ValueError, match="unexpected \\['other'\\]"):
            explainer.explain_batch(renamed)
        with pytest.raises(ValueError, match="test_sample columns"):
            explainer.explain_instance(X.iloc[[0]].rename(columns={"col_1": "z"}))

    def test_range_index_columns_are_positional(self, frame, frame_model):
        X, y = frame
        explainer = CaseExplainer(X, y, model=frame_model)
        plain = pd.DataFrame(X.iloc[:3].to_numpy())
        assert _same_results(
            explainer.explain_batch(plain), explainer.explain_batch(X.iloc[:3])
        )

    def test_array_trained_explainer_takes_frames_positionally(
        self, data, frame, logistic
    ):
        X_arr, y_arr = data
        X, _ = frame
        explainer = CaseExplainer(X_arr, y_arr, model=logistic)
        reversed_cols = list(X.columns)[::-1]
        # Without training columns there is nothing to align to; order is the caller's.
        assert _same_results(
            explainer.explain_batch(X.iloc[:3]),
            explainer.explain_batch(X.iloc[:3].to_numpy()),
        )
        assert not _same_results(
            explainer.explain_batch(X.iloc[:3]),
            explainer.explain_batch(X.iloc[:3][reversed_cols]),
        )

    def test_input_transform_fitted_on_frame(self, frame):
        X, y = frame
        scaler = StandardScaler().fit(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp = MLPClassifier(
                hidden_layer_sizes=(8,), max_iter=300, random_state=0
            ).fit(scaler.transform(X), y)
        for weighting in ("none", "predicted_class"):
            explainer = CaseExplainer(
                X,
                y,
                similarity=HiddenActivations(
                    mlp, input_transform=scaler, output_weighting=weighting
                ),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                single = explainer.explain_instance(X.iloc[2])
                batch = explainer.explain_batch(X.iloc[2:3])
            assert _same_results([single], batch)

    def test_tree_spaces_with_frame_fitted_models(self, frame):
        X, y = frame
        tree = DecisionTreeClassifier(random_state=0).fit(X, y)
        forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        for strategy in (TreeLeaf(tree), ForestProximity(forest)):
            explainer = CaseExplainer(X, y, similarity=strategy)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                single = explainer.explain_instance(X.iloc[1])
                batch = explainer.explain_batch(X.iloc[1:2].to_numpy())
            assert _same_results([single], batch)

    def test_custom_extractor_receives_arrays(self, frame, frame_model):
        X, y = frame
        seen = []

        def fn(model, X_query):
            seen.append(type(X_query))
            return np.asarray(X_query)[:, :2]

        explainer = CaseExplainer(
            X,
            y,
            similarity=CustomActivations(frame_model, CallableActivationExtractor(fn)),
        )
        explainer.explain_instance(X.iloc[0])
        assert seen and all(kind is np.ndarray for kind in seen)

    def test_training_frame_is_not_mutated(self, frame, frame_model):
        X, y = frame
        before = X.copy()
        CaseExplainer(X, y, model=frame_model).explain_batch(
            X.iloc[:3][list(X.columns)[::-1]]
        )
        pd.testing.assert_frame_equal(X, before)
