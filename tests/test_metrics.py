"""Tests for case_explainer.metrics module."""

import numpy as np
import pytest
from case_explainer.metrics import (
    euclidean_distance,
    compute_correspondence,
    compute_all_distances,
)


# --- euclidean_distance ---

class TestEuclideanDistance:
    def test_identical_points(self):
        p = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(p, p) == 0.0

    def test_known_value(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        assert euclidean_distance(p1, p2) == pytest.approx(5.0)

    def test_single_dimension(self):
        p1 = np.array([2.0])
        p2 = np.array([5.0])
        assert euclidean_distance(p1, p2) == pytest.approx(3.0)

    def test_symmetry(self):
        p1 = np.array([1.0, 2.0])
        p2 = np.array([4.0, 6.0])
        assert euclidean_distance(p1, p2) == pytest.approx(euclidean_distance(p2, p1))


# --- compute_correspondence ---

class TestComputeCorrespondence:
    def test_empty_neighbors(self):
        corr, interp = compute_correspondence([], predicted_class=0)
        assert corr == 0.0
        assert interp == "undefined"

    def test_all_same_class_weighted(self):
        """All neighbors match predicted class -> correspondence = 1.0."""
        neighbors = [(0, 0.1, 1), (1, 0.2, 1), (2, 0.3, 1)]
        corr, interp = compute_correspondence(neighbors, predicted_class=1)
        assert corr == pytest.approx(1.0)
        assert interp == "high"

    def test_no_matching_class_weighted(self):
        """No neighbors match predicted class -> correspondence = 0.0."""
        neighbors = [(0, 0.1, 0), (1, 0.2, 0)]
        corr, interp = compute_correspondence(neighbors, predicted_class=1)
        assert corr == pytest.approx(0.0)
        assert interp == "low"

    def test_mixed_classes_weighted(self):
        """Mix of classes with distance weighting."""
        neighbors = [(0, 0.0, 1), (1, 0.0, 0), (2, 0.0, 1)]
        corr, _ = compute_correspondence(neighbors, predicted_class=1, distance_weighted=True)
        # equal distances -> 2/3 match predicted
        assert corr == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_unweighted_simple_voting(self):
        """Unweighted: simple count-based correspondence."""
        neighbors = [(0, 10.0, 1), (1, 0.01, 0), (2, 10.0, 1)]
        corr, _ = compute_correspondence(neighbors, predicted_class=1, distance_weighted=False)
        # 2 out of 3 match, regardless of distance
        assert corr == pytest.approx(2.0 / 3.0)

    def test_distance_weighting_favors_closer(self):
        """Closer neighbor of predicted class should boost correspondence."""
        # Neighbor of class 1 is very close, class 0 is far
        neighbors = [(0, 0.0, 1), (1, 100.0, 0)]
        corr_weighted, _ = compute_correspondence(neighbors, predicted_class=1, distance_weighted=True)
        corr_unweighted, _ = compute_correspondence(neighbors, predicted_class=1, distance_weighted=False)
        # Weighted should be higher than 50% since class 1 is closer
        assert corr_weighted > corr_unweighted

    def test_class_weights(self):
        """Class weights should affect correspondence."""
        neighbors = [(0, 0.0, 0), (1, 0.0, 1)]
        # Without class weights: 50/50
        corr_no_wt, _ = compute_correspondence(neighbors, predicted_class=1)
        # With class 1 weighted 10x
        corr_wt, _ = compute_correspondence(
            neighbors, predicted_class=1, class_weights={0: 1.0, 1: 10.0}
        )
        assert corr_wt > corr_no_wt

    def test_class_weights_unweighted(self):
        """Class weights with unweighted distance."""
        neighbors = [(0, 0.5, 0), (1, 0.5, 1)]
        corr, _ = compute_correspondence(
            neighbors, predicted_class=1, distance_weighted=False,
            class_weights={0: 1.0, 1: 3.0}
        )
        # class 1 weight=3, class 0 weight=1 -> 3/(3+1) = 0.75
        assert corr == pytest.approx(0.75)

    def test_interpretation_high(self):
        neighbors = [(0, 0.0, 1)] * 5
        _, interp = compute_correspondence(neighbors, predicted_class=1)
        assert interp == "high"

    def test_interpretation_medium(self):
        # 3 match, 1 doesn't -> ~75% with equal distances
        neighbors = [(0, 0.0, 1), (1, 0.0, 1), (2, 0.0, 1), (3, 0.0, 0)]
        corr, interp = compute_correspondence(neighbors, predicted_class=1)
        assert interp == "medium"

    def test_interpretation_low(self):
        # 1 match, 4 don't -> ~20%
        neighbors = [(0, 0.0, 1), (1, 0.0, 0), (2, 0.0, 0), (3, 0.0, 0), (4, 0.0, 0)]
        _, interp = compute_correspondence(neighbors, predicted_class=1)
        assert interp == "low"


class TestCorrespondenceNumericalAccuracy:
    """Verify exact numerical values from the correspondence formula.

    Formula: weight = 1 / (distance + 1)^3
    Correspondence = weight(predicted_class) / sum(weight(all_classes))
    """

    def test_exact_two_neighbors_different_distances(self):
        """Hand-computed: one class-1 at dist=0.5, one class-0 at dist=2.0."""
        neighbors = [(0, 0.5, 1), (1, 2.0, 0)]
        # w(class 1) = 1 / (0.5 + 1)^3 = 1 / 1.5^3 = 1 / 3.375
        # w(class 0) = 1 / (2.0 + 1)^3 = 1 / 3.0^3 = 1 / 27.0
        w1 = 1.0 / (1.5 ** 3)  # 0.296296...
        w0 = 1.0 / (3.0 ** 3)  # 0.037037...
        expected = w1 / (w1 + w0)  # 0.888888...
        corr, interp = compute_correspondence(neighbors, predicted_class=1)
        assert corr == pytest.approx(expected, abs=1e-10)
        assert interp == "high"

    def test_exact_zero_distance(self):
        """Distance=0 -> weight = 1/(0+1)^3 = 1.0 (no divide-by-zero)."""
        neighbors = [(0, 0.0, 1), (1, 1.0, 0)]
        # w(class 1) = 1 / 1^3 = 1.0
        # w(class 0) = 1 / 2^3 = 0.125
        expected = 1.0 / (1.0 + 0.125)
        corr, _ = compute_correspondence(neighbors, predicted_class=1)
        assert corr == pytest.approx(expected, abs=1e-10)

    def test_exact_three_neighbors_with_class_weights(self):
        """Hand-computed with class_weights={0: 1.0, 1: 2.0}."""
        neighbors = [(0, 0.0, 1), (1, 0.0, 0), (2, 1.0, 1)]
        # w(class 1, idx 0) = 2.0 / (0+1)^3 = 2.0
        # w(class 0, idx 1) = 1.0 / (0+1)^3 = 1.0
        # w(class 1, idx 2) = 2.0 / (1+1)^3 = 2.0/8 = 0.25
        w1_total = 2.0 + 0.25  # 2.25
        w0_total = 1.0
        expected = w1_total / (w1_total + w0_total)  # 2.25 / 3.25
        corr, _ = compute_correspondence(
            neighbors, predicted_class=1,
            class_weights={0: 1.0, 1: 2.0}
        )
        assert corr == pytest.approx(expected, abs=1e-10)

    def test_exact_unweighted_with_class_weights(self):
        """Unweighted distance with class weights: pure vote weighting."""
        neighbors = [(0, 99.0, 1), (1, 0.01, 0), (2, 50.0, 1)]
        # Unweighted ignores distance. class 1 count=2*weight, class 0 count=1*weight
        # With weights {0: 1.0, 1: 4.0}: class 1 = 4.0+4.0=8.0, class 0 = 1.0
        expected = 8.0 / 9.0
        corr, _ = compute_correspondence(
            neighbors, predicted_class=1, distance_weighted=False,
            class_weights={0: 1.0, 1: 4.0}
        )
        assert corr == pytest.approx(expected, abs=1e-10)


# --- compute_all_distances ---

class TestComputeAllDistances:
    def test_distances_shape(self):
        point = np.array([0.0, 0.0])
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        dists = compute_all_distances(point, data)
        assert dists.shape == (3,)

    def test_distances_values(self):
        point = np.array([0.0, 0.0])
        data = np.array([[3.0, 4.0], [0.0, 0.0]])
        dists = compute_all_distances(point, data)
        assert dists[0] == pytest.approx(5.0)
        assert dists[1] == pytest.approx(0.0)

    def test_single_point(self):
        point = np.array([1.0, 2.0])
        data = np.array([[1.0, 2.0]])
        dists = compute_all_distances(point, data)
        assert dists[0] == pytest.approx(0.0)
