"""Robust corner localization for the PERIMETER mission.

The PERIMETER mission detects 3 or 4 identical ground-control points (the same
classifier class) that delimit the area containing the real, scored targets.
Because the detector cannot tell the GCPs apart, the only discriminator is their
geographic location: each GCP shows up as a spatially separated cluster of
geolocated detections, surrounded by scattered false positives.

`localize_corners` turns the accumulated detections of one class into the corner
coordinates:

1. cluster the points with DBSCAN/HDBSCAN, discarding isolated false positives as
   noise;
2. estimate each cluster's centre robustly (precision-weighted mean using the
   geoloc covariances, or spatial median);
3. rank clusters by support and select the expected number of corners.

The expected count may be *known* (an int, e.g. ``4``) or *unknown* (a
``[min, max]`` range, e.g. ``[3, 4]``); in the unknown case the top end is decided
at runtime by a support threshold plus a convex-hull-expansion test, so a false
positive is never promoted to an extra corner and a missing corner is never
fabricated.

This module is intentionally standalone and side-effect free so it can be unit
tested in isolation; it does not touch the existing ``CentroidAnalyzer`` /
``Decision`` clustering used by the other missions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import typing as np_typing
from sklearn.cluster import DBSCAN

try:  # HDBSCAN is available in scikit-learn >= 1.3; guard just in case
    from sklearn.cluster import HDBSCAN
except Exception:  # pragma: no cover - environment without HDBSCAN
    HDBSCAN = None  # type: ignore[assignment]

logger = logging.getLogger("decision_system")

CountSpec = Union[int, Sequence[int]]

DEFAULTS: Dict[str, Any] = {
    "method": "dbscan",  # "dbscan" | "hdbscan"
    "eps_m": 4.0,  # DBSCAN neighbourhood radius in metres
    "min_samples": 2,  # require corroboration from >= 2 frames
    "centroid": "weighted_mean",  # "weighted_mean" | "median"
    "keep_extra_frac": 0.5,  # unknown-count: support gate for an extra corner
}


def _parse_spec(spec: CountSpec) -> Tuple[int, int]:
    """Return ``(k_min, k_max)`` from an int or a ``[min, max]`` range."""
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("clusters spec list must not be empty")
        if len(spec) == 1:
            k = int(spec[0])
            return k, k
        k_min, k_max = int(spec[0]), int(spec[1])
        return (k_min, k_max) if k_min <= k_max else (k_max, k_min)
    k = int(spec)
    return k, k


def _cluster_labels(
    points: np_typing.NDArray[np.float64], cfg: Dict[str, Any]
) -> np_typing.NDArray[np.int64]:
    """Assign each point a cluster label; ``-1`` marks noise / false positives."""
    min_samples = int(cfg["min_samples"])
    method = str(cfg["method"]).lower()
    if method == "hdbscan":
        if HDBSCAN is None:
            raise RuntimeError(
                "method 'hdbscan' requested but sklearn.cluster.HDBSCAN is "
                "unavailable; use method 'dbscan'"
            )
        model = HDBSCAN(min_cluster_size=max(2, min_samples))
        return model.fit_predict(points).astype(np.int64)
    eps = float(cfg["eps_m"])
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points).astype(np.int64)


def _cluster_centroid(
    pts: np_typing.NDArray[np.float64],
    covs_inv: Optional[np_typing.NDArray[np.float64]],
    mode: str,
) -> np_typing.NDArray[np.float64]:
    """Robust centre of one cluster.

    ``weighted_mean`` solves ``(sum W_i) x = sum W_i p_i`` with ``W_i`` the
    inverse geoloc covariances, so more certain detections dominate. Falls back
    to the per-axis spatial median when covariances are missing or the system is
    singular.
    """
    if mode != "median" and covs_inv is not None:
        A = covs_inv.sum(axis=0)  # (2, 2)
        b = np.einsum("nij,nj->i", covs_inv, pts)  # (2,)
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning("Perimeter: singular weighted-mean system; using median")
    return np.median(pts, axis=0)


def _expands_hull(
    selected: np_typing.NDArray[np.float64], cand: np_typing.NDArray[np.float64]
) -> bool:
    """True if ``cand`` lies outside the convex hull of ``selected``.

    With fewer than 3 corners any new point necessarily extends the polygon. A
    genuine extra corner sits outside the hull of the others (GCPs are the
    extreme points of the delimited area); a false positive falls inside it and
    leaves the hull area unchanged.
    """
    if selected.shape[0] < 3:
        return True
    try:
        from scipy.spatial import ConvexHull

        base_area = ConvexHull(selected).volume  # 2D: .volume == area
        aug_area = ConvexHull(np.vstack([selected, cand])).volume
        return aug_area > base_area * (1.0 + 1e-6)
    except Exception:  # pragma: no cover - degenerate (collinear) inputs
        return True


def _select(
    centroids: np_typing.NDArray[np.float64],
    supports: np_typing.NDArray[np.float64],
    k_min: int,
    k_max: int,
    cfg: Dict[str, Any],
) -> np_typing.NDArray[np.float64]:
    """Pick corners from support-ranked candidates.

    Known count (``k_min == k_max``): the top ``k_min``. Unknown count: the top
    ``k_min`` always, then each further candidate up to ``k_max`` only while it
    clears the support gate *and* expands the hull.
    """
    n = centroids.shape[0]
    if n <= k_min:
        return centroids  # caller warns when this is below k_min
    if k_max == k_min:
        return centroids[:k_min]

    keep_frac = float(cfg["keep_extra_frac"])
    base_support = float(np.median(supports[:k_min]))
    selected: List[np_typing.NDArray[np.float64]] = list(centroids[:k_min])
    for i in range(k_min, min(k_max, n)):
        support_ok = supports[i] >= keep_frac * base_support
        if support_ok and _expands_hull(np.asarray(selected), centroids[i]):
            selected.append(centroids[i])
        else:
            break  # candidates are support-ranked: once one fails, the rest do too
    return np.asarray(selected, dtype=np.float64)


def localize_corners(
    points: np_typing.ArrayLike,
    covs: Optional[np_typing.ArrayLike],
    spec: CountSpec,
    cfg: Optional[Dict[str, Any]] = None,
) -> np_typing.NDArray[np.float64]:
    """Estimate GCP corner coordinates from accumulated detections.

    Args:
        points: ``(N, 2)`` detection coordinates in local ENU metres.
        covs: ``(N, 2, 2)`` geoloc covariances, or ``None`` to force the median.
        spec: expected corner count, ``int`` (known) or ``[min, max]`` (unknown).
        cfg: optional overrides for :data:`DEFAULTS`.

    Returns:
        ``(k, 2)`` corner coordinates in ENU metres, ranked by support. May
        contain fewer than ``k_min`` rows (with a warning) when the detector
        missed corners; never more than ``k_max``; never fabricates a corner.
    """
    merged = {**DEFAULTS, **(cfg or {})}
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    k_min, k_max = _parse_spec(spec)

    if pts.shape[0] == 0:
        logger.warning("Perimeter: no detections received; returning no corners")
        return np.zeros((0, 2), dtype=np.float64)

    covs_inv: Optional[np_typing.NDArray[np.float64]] = None
    if merged["centroid"] != "median" and covs is not None:
        covs_arr = np.asarray(covs, dtype=np.float64)
        if covs_arr.shape == (pts.shape[0], 2, 2):
            try:
                covs_inv = np.linalg.inv(covs_arr)
            except np.linalg.LinAlgError:
                logger.warning("Perimeter: non-invertible covariances; using median")
        else:
            logger.warning(
                "Perimeter: covariances shape %s != %s; using median",
                covs_arr.shape,
                (pts.shape[0], 2, 2),
            )

    labels = _cluster_labels(pts, merged)

    centroids: List[np_typing.NDArray[np.float64]] = []
    supports: List[int] = []
    for lbl in sorted(set(int(x) for x in labels)):
        if lbl == -1:
            continue  # isolated false positives
        mask = labels == lbl
        ci = covs_inv[mask] if covs_inv is not None else None
        centroids.append(_cluster_centroid(pts[mask], ci, str(merged["centroid"])))
        supports.append(int(mask.sum()))

    if not centroids:
        logger.warning(
            "Perimeter: clustering found only noise (%d detections); no corners",
            pts.shape[0],
        )
        return np.zeros((0, 2), dtype=np.float64)

    order = np.argsort(supports)[::-1]  # most-supported first
    centroids_arr = np.asarray(centroids, dtype=np.float64)[order]
    supports_arr = np.asarray(supports, dtype=np.float64)[order]

    selected = _select(centroids_arr, supports_arr, k_min, k_max, merged)

    if selected.shape[0] < k_min:
        logger.warning(
            "Perimeter: found %d corner(s), fewer than the expected minimum %d; "
            "returning what was found without fabricating corners",
            selected.shape[0],
            k_min,
        )
    else:
        logger.debug(
            "Perimeter: selected %d corner(s) from %d candidate cluster(s)",
            selected.shape[0],
            centroids_arr.shape[0],
        )
    return selected
