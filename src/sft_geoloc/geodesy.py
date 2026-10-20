"""Spherical GPS (latitude, longitude degrees), ECEF and north/east metres."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Geod, Transformer
from scipy.spatial.transform import Rotation

from . import check


def _vectors(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    check.vector_shape(array, size=size, name=name)
    check.finite(array, name=name)
    return array


def _latlon(values: ArrayLike) -> NDArray[np.float64]:
    array = _vectors(values, 2, "latlon_deg")
    check.latitude(array)
    return array


def _proj_values(values: NDArray[np.float64]) -> float | NDArray[np.float64]:
    # Avoid PROJ's deprecated NumPy singleton-array-to-scalar conversion.
    return float(values[0]) if values.size == 1 else values


@lru_cache(maxsize=16)
def _sphere_tools(radius: float) -> tuple[Transformer, Transformer, Geod]:
    geographic = CRS.from_proj4(f"+proj=longlat +R={radius} +type=crs")
    geocentric = CRS.from_proj4(f"+proj=geocent +R={radius} +units=m +type=crs")
    return (
        Transformer.from_crs(geographic, geocentric, always_xy=True),
        Transformer.from_crs(geocentric, geographic, always_xy=True),
        Geod(a=radius, b=radius),
    )


@dataclass(frozen=True)
class SphericalEarth:
    """Cache PROJ transforms and geodesics for one positive sphere radius."""

    radius_m: float = 6_371_008.8
    geod: Geod = field(init=False, repr=False, compare=False)
    _to_ecef: Transformer = field(init=False, repr=False, compare=False)
    _from_ecef: Transformer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        check.radius_scalar(self.radius_m)
        radius = float(self.radius_m)
        check.radius(radius)
        forward, inverse, geod = _sphere_tools(radius)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "_to_ecef", forward)
        object.__setattr__(self, "_from_ecef", inverse)
        object.__setattr__(self, "geod", geod)

    def geodetic_to_ecef(
        self, latlon_deg: ArrayLike, altitude_m: ArrayLike = 0.0
    ) -> NDArray[np.float64]:
        """Convert GPS ``(..., 2)`` and radial height ``(...)`` to ECEF XYZ."""
        latlon = _latlon(latlon_deg)
        altitude = np.asarray(altitude_m, dtype=np.float64)
        check.finite(altitude, name="altitude_m")
        shape = np.broadcast_shapes(latlon.shape[:-1], altitude.shape)
        if 0 in shape:
            return np.empty(shape + (3,), dtype=np.float64)

        coordinates = np.broadcast_to(latlon, shape + (2,)).reshape(-1, 2)
        heights = np.broadcast_to(altitude, shape).reshape(-1)
        xyz = self._to_ecef.transform(
            _proj_values(coordinates[:, 1]),
            _proj_values(coordinates[:, 0]),
            _proj_values(heights),
            errcheck=True,
        )
        return np.stack(xyz, axis=-1).reshape(shape + (3,))

    def ecef_to_geodetic(self, xyz: ArrayLike) -> NDArray[np.float64]:
        """Convert ``(..., 3)`` XYZ to latitude/longitude/height, deg/deg/m."""
        points = _vectors(xyz, 3, "xyz")
        if points.size == 0:
            return np.empty_like(points)
        flat = points.reshape(-1, 3)
        lon, lat, altitude = self._from_ecef.transform(
            _proj_values(flat[:, 0]),
            _proj_values(flat[:, 1]),
            _proj_values(flat[:, 2]),
            errcheck=True,
        )
        return np.stack((lat, lon, altitude), axis=-1).reshape(points.shape)


@dataclass(frozen=True, eq=False)
class LocalMap:
    """Spherical azimuthal-equidistant map using bearing FROM the origin.

    Origins and points broadcast. Inversion is unique for distances < pi*radius.
    """

    origin_latlon_deg: ArrayLike
    radius_m: float = 6_371_008.8
    earth: SphericalEarth = field(init=False, repr=False)

    def __post_init__(self) -> None:
        origin = _latlon(self.origin_latlon_deg).copy()
        origin.setflags(write=False)
        object.__setattr__(self, "origin_latlon_deg", origin)
        object.__setattr__(self, "earth", SphericalEarth(self.radius_m))

    def gps_to_ne(self, latlon_deg: ArrayLike) -> NDArray[np.float64]:
        """Map ``(..., 2)`` latitude/longitude to ``(..., 2)`` north/east."""
        target, origin = np.broadcast_arrays(
            _latlon(latlon_deg), self.origin_latlon_deg
        )
        if target.size == 0:
            return np.empty_like(target)
        points, centers = target.reshape(-1, 2), origin.reshape(-1, 2)
        bearing, _, distance = self.earth.geod.inv(
            _proj_values(centers[:, 1]),
            _proj_values(centers[:, 0]),
            _proj_values(points[:, 1]),
            _proj_values(points[:, 0]),
        )
        angle = np.deg2rad(bearing)
        return np.stack(
            (distance * np.cos(angle), distance * np.sin(angle)), axis=-1
        ).reshape(target.shape)

    def ne_to_gps(self, ne: ArrayLike) -> NDArray[np.float64]:
        """Map ``(..., 2)`` north/east metres to latitude/longitude degrees."""
        offsets, origin = np.broadcast_arrays(
            _vectors(ne, 2, "ne"), self.origin_latlon_deg
        )
        if offsets.size == 0:
            return np.empty_like(offsets)
        points, centers = offsets.reshape(-1, 2), origin.reshape(-1, 2)
        distance = np.hypot(points[:, 0], points[:, 1])
        bearing = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
        lon, lat, _ = self.earth.geod.fwd(
            _proj_values(centers[:, 1]),
            _proj_values(centers[:, 0]),
            _proj_values(bearing),
            _proj_values(distance),
        )
        return np.stack((lat, lon), axis=-1).reshape(offsets.shape)


def ned_to_ecef(latlon_deg: ArrayLike) -> Rotation:
    """Map NED to ECEF with rotation shape ``latlon_deg.shape[:-1]``.

    At lat=lon=0, north/east/down map to ECEF +Z/+Y/-X.
    """
    latlon = _latlon(latlon_deg)
    angles = np.stack((latlon[..., 1], -90.0 - latlon[..., 0]), axis=-1)
    return Rotation.from_euler("ZY", angles, degrees=True)
