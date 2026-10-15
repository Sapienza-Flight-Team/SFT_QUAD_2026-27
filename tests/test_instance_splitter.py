"""Unit tests for the decision-system instance splitter and Decision.subset.

The instance splitter separates the accumulated detections of a single class
into spatially distinct instances, by a pure distance threshold, *before*
clustering. This is what allows e.g. the 4 perimeter no-number GCPs (all
emitted by YOLO under the same class id) to yield 4 separate centroids
instead of a single fused one.

Run with:  PYTHONPATH=src python -m pytest tests/test_instance_splitter.py
"""

import logging

import numpy as np
import pytest

from stages.decision_system.default import DecisionSystem
from stages.decision_system.default.decision import Decision


LOGGER = logging.getLogger("test_instance_splitter")
DECISION_CONFIG = {"horizon": 0.99, "filter": 0.6}


def _make_decision(points) -> Decision:
    """Build a Decision pre-loaded with `points` (identity covariances)."""
    decision = Decision(LOGGER, DECISION_CONFIG)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] > 0:
        covs = np.tile(np.eye(2), (points.shape[0], 1, 1))
        decision.insert(points, covs)
    return decision


def _four_corner_points(spread: float, jitter: float, per_corner: int = 5):
    """`per_corner` noisy detections around each of 4 corners `spread` apart."""
    rng = np.random.default_rng(42)
    corners = np.array(
        [[0.0, 0.0], [spread, 0.0], [0.0, spread], [spread, spread]]
    )
    pts = [c + rng.normal(0.0, jitter, size=(per_corner, 2)) for c in corners]
    return np.vstack(pts)


# --------------------------------------------------------------------------
# _split_instances_by_distance
# --------------------------------------------------------------------------


def test_splitter_separates_four_instances():
    """4 spatial groups 50 m apart -> 4 sub-decisions."""
    points = _four_corner_points(spread=50.0, jitter=2.0)
    subs = DecisionSystem._split_instances_by_distance(
        _make_decision(points), min_distance_m=20.0
    )
    assert len(subs) == 4
    assert sum(s.data.points.shape[0] for s in subs) == points.shape[0]
    assert all(s.data.points.shape[0] == 5 for s in subs)


def test_splitter_keeps_single_instance_when_all_close():
    """All detections within the threshold -> original returned unchanged."""
    rng = np.random.default_rng(0)
    points = rng.normal(0.0, 1.0, size=(20, 2))
    decision = _make_decision(points)
    subs = DecisionSystem._split_instances_by_distance(
        decision, min_distance_m=20.0
    )
    assert len(subs) == 1
    assert subs[0] is decision


def test_splitter_single_detection():
    """A lone detection still works (graceful) -> 1 sub-decision."""
    decision = _make_decision(np.array([[3.0, 7.0]]))
    subs = DecisionSystem._split_instances_by_distance(
        decision, min_distance_m=15.0
    )
    assert len(subs) == 1
    assert subs[0] is decision
    assert subs[0].data.points.shape[0] == 1


def test_splitter_no_detections():
    """Empty decision -> single-element list, no crash."""
    decision = _make_decision(np.empty((0, 2)))
    subs = DecisionSystem._split_instances_by_distance(
        decision, min_distance_m=15.0
    )
    assert len(subs) == 1
    assert subs[0] is decision


def test_splitter_partial_detection_graceful():
    """Only 2 of the 4 expected instances observed -> 2 sub-decisions."""
    points = np.vstack(
        [
            np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]]),
            np.array([[80.0, 80.0], [81.0, 79.0]]),
        ]
    )
    subs = DecisionSystem._split_instances_by_distance(
        _make_decision(points), min_distance_m=15.0
    )
    assert len(subs) == 2
    assert sorted(s.data.points.shape[0] for s in subs) == [2, 3]


def test_splitter_chain_transitivity():
    """Chained points (each within threshold of the next) stay one instance."""
    points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    subs = DecisionSystem._split_instances_by_distance(
        _make_decision(points), min_distance_m=15.0
    )
    assert len(subs) == 1


def test_splitter_threshold_is_respected():
    """Two groups exactly straddling the threshold are kept separate."""
    points = np.array([[0.0, 0.0], [0.0, 1.0], [40.0, 0.0], [40.0, 1.0]])
    subs = DecisionSystem._split_instances_by_distance(
        _make_decision(points), min_distance_m=15.0
    )
    assert len(subs) == 2


# --------------------------------------------------------------------------
# Decision.subset
# --------------------------------------------------------------------------


def test_subset_selects_masked_points():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    decision = _make_decision(points)
    mask = np.array([True, False, True, False])
    sub = decision.subset(mask)

    assert sub.data.points.shape[0] == 2
    np.testing.assert_allclose(sub.data.points, points[mask])
    # original is untouched
    assert decision.data.points.shape[0] == 4
    # subset starts with no clustering feedback
    assert sub.feedback is None


def test_subset_covariances_follow_mask():
    points = np.array([[0.0, 0.0], [5.0, 5.0]])
    sub = _make_decision(points).subset(np.array([False, True]))
    assert sub.data.covs.shape == (1, 2, 2)
    np.testing.assert_allclose(sub.data.points, np.array([[5.0, 5.0]]))


# --------------------------------------------------------------------------
# end-to-end: split + clustering produces one centroid per instance
# --------------------------------------------------------------------------


def test_split_then_fit_yields_one_centroid_per_instance():
    """Each spatially separated instance, clustered on its own, returns a
    centroid close to its true position."""
    spread, jitter = 50.0, 2.0
    points = _four_corner_points(spread=spread, jitter=jitter, per_corner=8)
    subs = DecisionSystem._split_instances_by_distance(
        _make_decision(points), min_distance_m=20.0
    )
    assert len(subs) == 4

    centroids = []
    for sub in subs:
        cls_id, result = DecisionSystem._fit_decision(0, sub, n_clusters=1)
        assert result.shape[0] == 1
        centroids.append(result[0])
    centroids = np.array(centroids)

    expected = np.array(
        [[0.0, 0.0], [spread, 0.0], [0.0, spread], [spread, spread]]
    )
    # every expected corner is matched by exactly one centroid within a
    # few metres (order is not guaranteed)
    for corner in expected:
        assert np.min(np.linalg.norm(centroids - corner, axis=1)) < 5.0
