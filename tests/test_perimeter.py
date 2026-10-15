"""Unit tests for the PERIMETER corner localizer (decision.perimeter).

The PERIMETER mission must turn the accumulated detections of the identical
no-number GCPs (one classifier class) into the 3-4 corner coordinates that
delimit the survey area, while rejecting scattered false positives and never
fabricating a corner the detector did not actually see.

These tests reuse the scenario from the brainstorming simulation
(decision/brainstorming/target_localization_sim.py): Gaussian detection clouds
around each corner plus uniform false positives.

Run with:  PYTHONPATH=src python -m pytest tests/test_perimeter.py
"""

import numpy as np
import pytest

from stages.decision_system.default.decision.perimeter import localize_corners


CFG = {
    "method": "dbscan",
    "eps_m": 4.0,
    "min_samples": 4,
    "centroid": "weighted_mean",
    "keep_extra_frac": 0.5,
}

SQUARE = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]])


def _identity_covs(n: int) -> np.ndarray:
    return np.tile(np.eye(2), (n, 1, 1))


def _make_detections(corners, per_corner=15, std=1.5, n_fp=80, seed=42, area=100.0):
    """Detection clouds around each corner + uniform false positives.

    Returns ``(points (N,2), covs (N,2,2))`` with identity covariances.
    """
    rng = np.random.default_rng(seed)
    clouds = [rng.normal(c, std, size=(per_corner, 2)) for c in corners]
    pts = np.vstack(clouds) if clouds else np.empty((0, 2))
    if n_fp:
        fps = rng.uniform(0.0, area, size=(n_fp, 2))
        pts = np.vstack([pts, fps]) if pts.shape[0] else fps
    rng.shuffle(pts)
    return pts, _identity_covs(pts.shape[0])


def _assert_matches_corners(found, expected, tol=3.0):
    """Every expected corner is matched by some found corner within `tol`."""
    for corner in expected:
        assert np.min(np.linalg.norm(found - corner, axis=1)) < tol, (
            f"corner {corner} not matched by {found}"
        )


# --------------------------------------------------------------------------
# known count
# --------------------------------------------------------------------------


def test_known_four_all_present():
    """4 corners present + 80 FPs, spec=4 -> 4 corners, FPs rejected."""
    pts, covs = _make_detections(SQUARE)
    corners = localize_corners(pts, covs, 4, CFG)
    assert corners.shape == (4, 2)
    _assert_matches_corners(corners, SQUARE)


def test_known_four_with_one_missing_does_not_fabricate():
    """Only 3 corners detected, spec=4 -> returns 3, never invents a 4th."""
    pts, covs = _make_detections(SQUARE[:3])
    corners = localize_corners(pts, covs, 4, CFG)
    assert corners.shape == (3, 2)
    _assert_matches_corners(corners, SQUARE[:3])


def test_known_three_triangle():
    """A genuine 3-GCP (triangle) layout, spec=3 -> exactly 3 corners."""
    triangle = np.array([[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]])
    pts, covs = _make_detections(triangle)
    corners = localize_corners(pts, covs, 3, CFG)
    assert corners.shape == (3, 2)
    _assert_matches_corners(corners, triangle)


# --------------------------------------------------------------------------
# unknown count [min, max] -> runtime 3-vs-4 decision
# --------------------------------------------------------------------------


def test_unknown_range_all_four_present_returns_four():
    pts, covs = _make_detections(SQUARE)
    corners = localize_corners(pts, covs, [3, 4], CFG)
    assert corners.shape == (4, 2)
    _assert_matches_corners(corners, SQUARE)


def test_unknown_range_three_present_returns_three():
    """Only 3 real corners + FPs; the unknown-count path must settle on 3."""
    pts, covs = _make_detections(SQUARE[:3])
    corners = localize_corners(pts, covs, [3, 4], CFG)
    assert corners.shape == (3, 2)
    _assert_matches_corners(corners, SQUARE[:3])


def test_unknown_range_rejects_interior_false_cluster():
    """A 4th dense cluster *inside* the triangle is an FP, not a corner.

    Even though it clears the support gate, it does not expand the convex hull
    of the 3 real corners, so it must be rejected -> 3 corners.
    """
    triangle = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]])
    rng = np.random.default_rng(7)
    real = [rng.normal(c, 1.5, size=(20, 2)) for c in triangle]
    interior = rng.normal([50.0, 30.0], 1.5, size=(10, 2))  # well inside the hull
    pts = np.vstack(real + [interior])
    corners = localize_corners(pts, _identity_covs(pts.shape[0]), [3, 4], CFG)
    assert corners.shape == (3, 2)
    _assert_matches_corners(corners, triangle)


def test_unknown_range_accepts_weaker_but_geometric_fourth():
    """A 4th corner with lower (but sufficient) support that extends the hull
    is kept."""
    rng = np.random.default_rng(3)
    strong = [rng.normal(c, 1.5, size=(20, 2)) for c in SQUARE[:3]]
    weak = rng.normal(SQUARE[3], 1.5, size=(10, 2))  # support 10 >= 0.5*20
    pts = np.vstack(strong + [weak])
    corners = localize_corners(pts, _identity_covs(pts.shape[0]), [3, 4], CFG)
    assert corners.shape == (4, 2)
    _assert_matches_corners(corners, SQUARE)


# --------------------------------------------------------------------------
# robustness: low support, false positives, empties, determinism
# --------------------------------------------------------------------------


def test_low_support_corner_not_lost():
    """A corner seen only twice survives (min_samples=2), unlike the sim's
    min_samples=4 which would discard it."""
    cfg = {**CFG, "min_samples": 2}
    rng = np.random.default_rng(1)
    clouds = [rng.normal(c, 1.0, size=(15, 2)) for c in SQUARE[:3]]
    sparse = rng.normal(SQUARE[3], 1.0, size=(2, 2))  # only 2 detections
    pts = np.vstack(clouds + [sparse])
    corners = localize_corners(pts, _identity_covs(pts.shape[0]), 4, cfg)
    assert corners.shape == (4, 2)
    _assert_matches_corners(corners, SQUARE, tol=3.0)


def test_only_false_positives_returns_empty():
    """Pure scattered noise with no real cluster -> no corners (no crash)."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.0, 100.0, size=(60, 2))
    corners = localize_corners(pts, _identity_covs(pts.shape[0]), [3, 4], CFG)
    assert corners.shape[0] == 0


def test_empty_input_returns_empty():
    corners = localize_corners(
        np.empty((0, 2)), np.empty((0, 2, 2)), 4, CFG
    )
    assert corners.shape == (0, 2)


def test_median_centroid_mode():
    """`centroid: median` works and ignores covariances."""
    cfg = {**CFG, "centroid": "median"}
    pts, _ = _make_detections(SQUARE)
    corners = localize_corners(pts, None, 4, cfg)
    assert corners.shape == (4, 2)
    _assert_matches_corners(corners, SQUARE)


def test_deterministic():
    """Identical inputs yield byte-identical outputs across runs."""
    pts, covs = _make_detections(SQUARE)
    a = localize_corners(pts, covs, [3, 4], CFG)
    b = localize_corners(pts, covs, [3, 4], CFG)
    np.testing.assert_array_equal(a, b)
