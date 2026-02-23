"""Tests for case_explainer.indexing module."""

import numpy as np
import pytest
from case_explainer.indexing import (
    BruteForceIndex,
    KDTreeIndex,
    BallTreeIndex,
    create_index,
)


@pytest.fixture
def small_data():
    """5 points in 2-D with predictable nearest neighbors."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [10.0, 11.0],
    ])


# --- BruteForceIndex ---

class TestBruteForceIndex:
    def test_build_and_query(self, small_data):
        idx = BruteForceIndex()
        idx.build(small_data)
        dists, inds = idx.query(np.array([0.0, 0.0]), k=2)
        assert len(dists) == 2
        assert inds[0] == 0  # itself is the closest
        assert dists[0] == pytest.approx(0.0)

    def test_k_equals_n(self, small_data):
        idx = BruteForceIndex()
        idx.build(small_data)
        dists, inds = idx.query(np.array([0.0, 0.0]), k=5)
        assert len(dists) == 5
        # Distances should be sorted ascending
        assert all(dists[i] <= dists[i + 1] for i in range(4))


# --- KDTreeIndex ---

class TestKDTreeIndex:
    def test_build_and_query(self, small_data):
        idx = KDTreeIndex(leaf_size=2)
        idx.build(small_data)
        dists, inds = idx.query(np.array([0.0, 0.0]), k=3)
        assert len(dists) == 3
        assert 0 in inds

    def test_nearest_is_correct(self, small_data):
        idx = KDTreeIndex()
        idx.build(small_data)
        dists, inds = idx.query(np.array([10.0, 10.5]), k=1)
        # Closest to (10, 10.5) should be (10, 10) at index 3
        assert inds[0] == 3


# --- BallTreeIndex ---

class TestBallTreeIndex:
    def test_build_and_query(self, small_data):
        idx = BallTreeIndex(leaf_size=2)
        idx.build(small_data)
        dists, inds = idx.query(np.array([0.0, 0.0]), k=2)
        assert len(dists) == 2
        assert inds[0] == 0

    def test_nearest_is_correct(self, small_data):
        idx = BallTreeIndex()
        idx.build(small_data)
        dists, inds = idx.query(np.array([10.0, 10.5]), k=1)
        assert inds[0] == 3


# --- create_index factory ---

class TestCreateIndex:
    def test_brute(self):
        idx = create_index('brute')
        assert isinstance(idx, BruteForceIndex)

    def test_kd_tree(self):
        idx = create_index('kd_tree')
        assert isinstance(idx, KDTreeIndex)

    def test_ball_tree(self):
        idx = create_index('ball_tree')
        assert isinstance(idx, BallTreeIndex)

    def test_kd_tree_kwargs(self):
        idx = create_index('kd_tree', leaf_size=10)
        assert idx.leaf_size == 10

    def test_ball_tree_kwargs(self):
        idx = create_index('ball_tree', leaf_size=15)
        assert idx.leaf_size == 15

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown index method"):
            create_index('magic_tree')


# --- Consistency across strategies ---

class TestConsistency:
    def test_all_strategies_find_same_nearest(self, small_data):
        query_point = np.array([0.5, 0.5])
        k = 3
        results = {}
        for name, cls in [('brute', BruteForceIndex), ('kd', KDTreeIndex), ('ball', BallTreeIndex)]:
            idx = cls()
            idx.build(small_data)
            dists, inds = idx.query(query_point, k=k)
            results[name] = set(inds)
        # All strategies should find the same k neighbors
        assert results['brute'] == results['kd'] == results['ball']
