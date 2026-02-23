"""Tests for case_explainer.explanation module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from case_explainer.explanation import Neighbor, Explanation


# --- Helpers ---

def _make_neighbor(index=0, distance=0.1, label=1, n_features=4, metadata=None):
    return Neighbor(
        index=index,
        distance=distance,
        label=label,
        features=np.array([1.0] * n_features),
        metadata=metadata,
    )


def _make_explanation(
    n_neighbors=3,
    predicted_class=1,
    true_class=0,
    correspondence=0.85,
    interp="high",
    class_names=None,
    feature_names=None,
    test_index=42,
):
    neighbors = [
        _make_neighbor(index=i, distance=0.1 * (i + 1), label=(1 if i < 2 else 0))
        for i in range(n_neighbors)
    ]
    return Explanation(
        test_sample=np.array([1.0, 2.0, 3.0, 4.0]),
        test_index=test_index,
        neighbors=neighbors,
        predicted_class=predicted_class,
        true_class=true_class,
        correspondence=correspondence,
        correspondence_interpretation=interp,
        feature_names=feature_names,
        class_names=class_names,
    )


# --- Neighbor ---

class TestNeighbor:
    def test_repr_no_metadata(self):
        n = _make_neighbor()
        r = repr(n)
        assert "index=0" in r
        assert "0.1000" in r
        assert "label=1" in r

    def test_repr_with_metadata(self):
        n = _make_neighbor(metadata={"circuit": "AES"})
        r = repr(n)
        assert "circuit" in r
        assert "AES" in r

    def test_default_metadata_empty(self):
        n = _make_neighbor()
        assert n.metadata == {}

    def test_stores_features(self):
        n = _make_neighbor(n_features=3)
        assert n.features.shape == (3,)


# --- Explanation init ---

class TestExplanationInit:
    def test_default_feature_names(self):
        exp = _make_explanation(feature_names=None)
        assert exp.feature_names == ["feature_0", "feature_1", "feature_2", "feature_3"]

    def test_custom_feature_names(self):
        exp = _make_explanation(feature_names=["a", "b", "c", "d"])
        assert exp.feature_names == ["a", "b", "c", "d"]

    def test_class_names_dict(self):
        exp = _make_explanation(class_names={0: "benign", 1: "trojan"})
        assert exp.class_names == {0: "benign", 1: "trojan"}

    def test_class_names_list_converted(self):
        exp = _make_explanation(class_names=["benign", "trojan"])
        assert exp.class_names == {0: "benign", 1: "trojan"}

    def test_class_names_default_empty(self):
        exp = _make_explanation(class_names=None)
        assert exp.class_names == {}


# --- get_predicted_class_name / get_true_class_name ---

class TestClassNames:
    def test_predicted_class_name_with_mapping(self):
        exp = _make_explanation(predicted_class=1, class_names={0: "A", 1: "B"})
        assert exp.get_predicted_class_name() == "B"

    def test_predicted_class_name_fallback(self):
        exp = _make_explanation(predicted_class=1, class_names=None)
        assert exp.get_predicted_class_name() == "1"

    def test_true_class_name_with_mapping(self):
        exp = _make_explanation(true_class=0, class_names={0: "A", 1: "B"})
        assert exp.get_true_class_name() == "A"

    def test_true_class_name_none(self):
        exp = _make_explanation(true_class=None)
        assert exp.get_true_class_name() is None

    def test_true_class_name_fallback(self):
        exp = _make_explanation(true_class=0, class_names=None)
        assert exp.get_true_class_name() == "0"


# --- is_correct ---

class TestIsCorrect:
    def test_correct_prediction(self):
        exp = _make_explanation(predicted_class=1, true_class=1)
        assert exp.is_correct() is True

    def test_incorrect_prediction(self):
        exp = _make_explanation(predicted_class=1, true_class=0)
        assert exp.is_correct() is False

    def test_no_true_class(self):
        exp = _make_explanation(true_class=None)
        assert exp.is_correct() is None


# --- summary ---

class TestSummary:
    def test_summary_contains_header(self):
        exp = _make_explanation()
        s = exp.summary()
        assert "CASE-BASED EXPLANATION" in s

    def test_summary_contains_correspondence(self):
        exp = _make_explanation(correspondence=0.85, interp="high")
        s = exp.summary()
        assert "85.00%" in s
        assert "high" in s

    def test_summary_shows_test_index(self):
        exp = _make_explanation(test_index=42)
        s = exp.summary()
        assert "42" in s

    def test_summary_no_test_index(self):
        exp = _make_explanation(test_index=None)
        s = exp.summary()
        assert "Test sample index" not in s

    def test_summary_no_true_class(self):
        exp = _make_explanation(true_class=None)
        s = exp.summary()
        assert "True class" not in s

    def test_summary_correct_mark(self):
        exp = _make_explanation(predicted_class=1, true_class=1)
        s = exp.summary()
        assert "[OK]" in s

    def test_summary_incorrect_mark(self):
        exp = _make_explanation(predicted_class=1, true_class=0)
        s = exp.summary()
        assert "[X]" in s

    def test_summary_neighbor_metadata(self):
        exp = _make_explanation(n_neighbors=1)
        exp.neighbors[0].metadata = {"circuit": "AES"}
        s = exp.summary()
        assert "circuit" in s
        assert "AES" in s

    def test_summary_class_names(self):
        exp = _make_explanation(class_names={0: "benign", 1: "trojan"})
        s = exp.summary()
        assert "trojan" in s


# --- to_dict ---

class TestToDict:
    def test_keys_present(self):
        exp = _make_explanation()
        d = exp.to_dict()
        expected_keys = {
            "test_index", "test_sample", "predicted_class",
            "predicted_class_name", "true_class", "true_class_name",
            "is_correct", "correspondence", "correspondence_interpretation",
            "neighbors", "feature_names",
        }
        assert expected_keys == set(d.keys())

    def test_test_sample_is_list(self):
        d = _make_explanation().to_dict()
        assert isinstance(d["test_sample"], list)

    def test_neighbors_serialized(self):
        exp = _make_explanation(n_neighbors=2)
        d = exp.to_dict()
        assert len(d["neighbors"]) == 2
        n = d["neighbors"][0]
        assert "index" in n
        assert "distance" in n
        assert "features" in n
        assert isinstance(n["features"], list)

    def test_correspondence_is_float(self):
        d = _make_explanation().to_dict()
        assert isinstance(d["correspondence"], float)


# --- plot ---

class TestPlot:
    @patch("case_explainer.explanation.plt")
    def test_bar_plot(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.cm.Blues.return_value = np.array([[0, 0, 0, 1]])
        exp = _make_explanation(n_neighbors=1)
        exp.plot(plot_type="bar")
        mock_plt.subplots.assert_called_once()

    @patch("case_explainer.explanation.plt")
    def test_radar_falls_back_to_bar(self, mock_plt, capsys):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.cm.Blues.return_value = np.array([[0, 0, 0, 1]])
        exp = _make_explanation(n_neighbors=1)
        exp.plot(plot_type="radar")
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out

    @patch("case_explainer.explanation.plt")
    def test_parallel_falls_back_to_bar(self, mock_plt, capsys):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.cm.Blues.return_value = np.array([[0, 0, 0, 1]])
        exp = _make_explanation(n_neighbors=1)
        exp.plot(plot_type="parallel")
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out

    def test_unknown_plot_type_raises(self):
        exp = _make_explanation()
        with pytest.raises(ValueError, match="Unknown plot type"):
            exp.plot(plot_type="scatter")

    @patch("case_explainer.explanation.plt")
    def test_save_path(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_plt.cm.Blues.return_value = np.array([[0, 0, 0, 1]])
        exp = _make_explanation(n_neighbors=1)
        exp.plot(plot_type="bar", save_path="/tmp/test.png")
        mock_plt.savefig.assert_called_once()


# --- __repr__ ---

class TestExplanationRepr:
    def test_repr_contains_fields(self):
        exp = _make_explanation(predicted_class=1, correspondence=0.9)
        r = repr(exp)
        assert "test_index=42" in r
        assert "90.00%" in r
        assert "neighbors=3" in r

    def test_repr_with_class_names(self):
        exp = _make_explanation(predicted_class=1, class_names={0: "A", 1: "B"})
        r = repr(exp)
        assert "predicted=B" in r
