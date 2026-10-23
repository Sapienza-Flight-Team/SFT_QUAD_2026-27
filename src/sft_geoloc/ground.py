"""Forward intersections for broadcast ``(..., 3)`` rays of arbitrary length.

Return ``(points, valid)``; invalid/behind hits are NaN. Zero-distance hits count.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import check


def _rays(
    origins: ArrayLike, directions: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    arrays = [np.asarray(values, dtype=np.float64) for values in (origins, directions)]
    check.ray_shape(arrays)
    origin, direction = np.broadcast_arrays(*arrays)
    finite = np.isfinite(origin).all(axis=-1) & np.isfinite(direction).all(axis=-1)
    # Scale first so tiny/large nonzero directions can be normalized safely.
    scale = np.max(np.abs(direction), axis=-1, keepdims=True)
    valid = finite & (scale[..., 0] > 0)
    unit = np.zeros_like(direction)
    np.divide(direction, scale, out=unit, where=valid[..., None])
    norm = np.linalg.norm(unit, axis=-1, keepdims=True)
    np.divide(unit, norm, out=unit, where=valid[..., None])
    return np.where(finite[..., None], origin, 0.0), unit, valid


def intersect_sphere(
    origins: ArrayLike, directions: ArrayLike, radius_m: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Nearest forward ECEF intersection with an origin-centered sphere.

    Stable quadratic roots preserve accuracy near the surface and horizon.
    Tangencies within floating-point roundoff are accepted.
    """
    check.radius(radius_m)
    origin, unit, valid = _rays(origins, directions)
    radius = float(radius_m)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        along = np.einsum("...i,...i->...", origin, unit)
        # Perpendicular distance avoids cancellation near the horizon.
        perpendicular = np.hypot.reduce(np.cross(origin, unit), axis=-1)
        gap = radius - perpendicular
        tolerance = 8 * np.finfo(np.float64).eps * radius
        intersects = gap >= -tolerance
        root = np.sqrt(np.maximum(gap, 0.0) * (radius + perpendicular))
        q = -along - np.copysign(root, along)
        origin_radius = np.hypot.reduce(origin, axis=-1)
        constant = (origin_radius - radius) * (origin_radius + radius)
        other = np.zeros_like(q)
        np.divide(constant, q, out=other, where=q != 0)
        distance = np.minimum(
            np.where(q >= 0, q, np.inf), np.where(other >= 0, other, np.inf)
        )
        valid &= intersects & np.isfinite(distance)
        points = origin + distance[..., None] * unit
    valid &= np.isfinite(points).all(axis=-1)
    return np.where(valid[..., None], points, np.nan), valid


def intersect_plane(
    origins: ArrayLike, directions: ArrayLike, down_m: ArrayLike = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Intersect NED rays with broadcast ``down = down_m``; parallel is invalid."""
    origin, unit, valid = _rays(origins, directions)
    down = np.asarray(down_m, dtype=np.float64)
    check.finite(down, name="down_m")
    shape = np.broadcast_shapes(origin.shape[:-1], down.shape)
    origin = np.broadcast_to(origin, shape + (3,))
    unit = np.broadcast_to(unit, shape + (3,))
    valid = np.broadcast_to(valid, shape) & (unit[..., 2] != 0)
    distance = np.full(shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(down - origin[..., 2], unit[..., 2], out=distance, where=valid)
        valid &= (distance >= 0) & np.isfinite(distance)
        points = origin + distance[..., None] * unit
    valid &= np.isfinite(points).all(axis=-1)
    return np.where(valid[..., None], points, np.nan), valid
