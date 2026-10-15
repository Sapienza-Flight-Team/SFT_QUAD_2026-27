from sft_math.geo.polar import GPS
from sft_math.geo.class_template import GPSBase
from sft_math.typing import Float, Array
from sft_math.geo.constants import EARTH_RADIUS
import numpy as np
from typing import Tuple
from numpy import typing as np_typing


class GeoGPS(GPSBase["GPS", "GeoGPS"]):
    """Class to represent a GPS coordinate in the Earth

    Parameters:
        gps (GPS): GPS coordinate
        radius (Array): Earth radius
        shape (Tuple[int, ...]): shape of the GPS coordinate
    """

    class Movement(GPSBase.Movement["GeoGPS"]):

        yaw: Array
        distance: Array
        shape: Tuple[int, ...]

        __constants__ = [
            (lambda x: np.rad2deg(x), "yaw", "°"),
            (lambda x: x, "distance", "m"),
            (lambda x: x, "shape", ""),
        ]

        def __init__(
            self,
            yaw: np_typing.ArrayLike,
            distance: np_typing.ArrayLike,
            is_deg: bool = True,
        ):
            """Initializes the GPS movement

            Args:
                yaw (np_typing.ArrayLike): yaw angle
                distance (np_typing.ArrayLike): distance
                is_deg (bool, optional): True if yaw is in degrees. Defaults to True.

            Raises:
                ValueError: yaw and distance must be broadcastable
            """
            self.yaw = np.atleast_1d(np.array(yaw, dtype=Float))
            self.distance = np.atleast_1d(np.array(distance, dtype=Float))
            try:
                self.shape = np.broadcast_shapes(self.yaw.shape, self.distance.shape)
                self.yaw = np.broadcast_to(self.yaw, self.shape)
                self.distance = np.broadcast_to(self.distance, self.shape)
            except ValueError:
                raise ValueError("yaw and distance must be broadcastable")
            if is_deg:
                self.yaw = np.deg2rad(self.yaw)
            # normalise yaw
            self.yaw = np.where(self.distance < 0, -self.yaw, self.yaw)
            self.distance = np.abs(self.distance)
            self.yaw = (np.mod(self.yaw + np.pi, 2 * np.pi) - np.pi).astype(Float)

        def __neg__(self) -> "GeoGPS.Movement":
            return GeoGPS.Movement(self.yaw + np.pi, self.distance, is_deg=False)

        def __pos__(self) -> "GeoGPS.Movement":
            return GeoGPS.Movement(self.yaw, self.distance, is_deg=False)

        def __mul__(self, scalar: np_typing.ArrayLike) -> "GeoGPS.Movement":
            return GeoGPS.Movement(
                self.yaw, self.distance * np.array(scalar, dtype=Float), is_deg=False
            )

        def __rmul__(self, scalar: np_typing.ArrayLike) -> "GeoGPS.Movement":
            return GeoGPS.Movement(
                self.yaw, self.distance * np.array(scalar, dtype=Float), is_deg=False
            )

        def __truediv__(self, scalar: np_typing.ArrayLike) -> "GeoGPS.Movement":
            return GeoGPS.Movement(
                self.yaw, self.distance / np.array(scalar, dtype=Float), is_deg=False
            )

        def to_vector(self) -> "GeoGPS.Vector":
            return GeoGPS.Vector(
                self.distance * np.cos(self.yaw), self.distance * np.sin(self.yaw)
            )

        def __getitem__(self, key: Tuple) -> "GeoGPS.Movement":
            return GeoGPS.Movement(self.yaw[key], self.distance[key], is_deg=False)

        def reshape(self, shape: Tuple[int, ...]) -> "GeoGPS.Movement":
            return GeoGPS.Movement(
                self.yaw.reshape(shape), self.distance.reshape(shape), is_deg=False
            )

        def broadcast_to(self, shape: Tuple[int, ...]) -> "GeoGPS.Movement":
            return GeoGPS.Movement(
                np.broadcast_to(self.yaw, shape),
                np.broadcast_to(self.distance, shape),
                is_deg=False,
            )

    class Vector(GPSBase.Vector["GeoGPS"]):

        nord: Array
        east: Array
        shape: Tuple[int, ...]

        __constants__ = [
            (lambda x: x, "nord", "m"),
            (lambda x: x, "east", "m"),
            (lambda x: x, "shape", ""),
        ]

        def __init__(self, nord: np_typing.ArrayLike, east: np_typing.ArrayLike):
            self.nord = np.atleast_1d(np.array(nord, dtype=Float))
            self.east = np.atleast_1d(np.array(east, dtype=Float))
            try:
                self.shape = np.broadcast_shapes(self.nord.shape, self.east.shape)
                self.nord = np.broadcast_to(self.nord, self.shape)
                self.east = np.broadcast_to(self.east, self.shape)
            except ValueError:
                raise ValueError("nord and east must be broadcastable")

        def to_movement(self) -> "GeoGPS.Movement":
            return GeoGPS.Movement(
                np.arctan2(self.east, self.nord),
                np.sqrt(self.nord**2 + self.east**2),
                is_deg=False,
            )

        def __neg__(self) -> "GeoGPS.Vector":
            return GeoGPS.Vector(-self.nord, -self.east)

        def __pos__(self) -> "GeoGPS.Vector":
            return GeoGPS.Vector(self.nord, self.east)

        def __add__(self, other: "GeoGPS.Vector") -> "GeoGPS.Vector":
            return GeoGPS.Vector(self.nord + other.nord, self.east + other.east)

        def __sub__(self, other: "GeoGPS.Vector") -> "GeoGPS.Vector":
            return GeoGPS.Vector(self.nord - other.nord, self.east - other.east)

        def __mul__(self, scalar: np_typing.ArrayLike) -> "GeoGPS.Vector":
            return GeoGPS.Vector(
                self.nord * np.array(scalar, dtype=Float),
                self.east * np.array(scalar, dtype=Float),
            )

        def __rmul__(self, scalar: np_typing.ArrayLike) -> "GeoGPS.Vector":
            return GeoGPS.Vector(
                self.nord * np.array(scalar, dtype=Float),
                self.east * np.array(scalar, dtype=Float),
            )

        def __truediv__(self, scalar: np_typing.ArrayLike) -> "GeoGPS.Vector":
            return GeoGPS.Vector(
                self.nord / np.array(scalar, dtype=Float),
                self.east / np.array(scalar, dtype=Float),
            )

        def norm(self) -> Array:
            return np.array(np.sqrt(self.nord**2 + self.east**2), dtype=Float)

        def angle(self) -> Array:
            return np.array(np.arctan2(self.east, self.nord), dtype=Float)

        def __getitem__(self, key: Tuple) -> "GeoGPS.Vector":
            return GeoGPS.Vector(self.nord[key], self.east[key])

        def reshape(self, shape: Tuple[int, ...]) -> "GeoGPS.Vector":
            return GeoGPS.Vector(self.nord.reshape(shape), self.east.reshape(shape))

        def broadcast_to(self, shape: Tuple[int, ...]) -> "GeoGPS.Vector":
            return GeoGPS.Vector(
                np.broadcast_to(self.nord, shape), np.broadcast_to(self.east, shape)
            )

    gps: GPS
    radius: Array
    shape: Tuple[int, ...]

    __constants__ = [
        (lambda x: x, "gps", ""),
        (lambda x: x, "radius", "m"),
        (lambda x: x, "shape", ""),
    ]

    def __init__(self, gps: GPS, radius: np_typing.ArrayLike = EARTH_RADIUS):
        """Initializes the GeoGPS coordinate

        Args:
            gps (GPS): GPS coordinate
            radius (np_typing.ArrayLike, optional): Earth radius. Defaults to EARTH_RADIUS.

        Raises:
            ValueError: gps and radius must be broadcastable
            ValueError: radius must be positive
        """
        radius = np.atleast_1d(np.array(radius, dtype=Float))
        try:
            self.shape = np.broadcast(gps.lat, gps.lon, radius).shape
            gps = gps.broadcast_to(self.shape)
            radius = np.broadcast_to(radius, self.shape)
        except ValueError:
            raise ValueError("gps and radius must be broadcastable")
        if np.any(radius <= 0):
            raise ValueError("radius must be positive")
        self.radius = radius
        self.gps = gps
        self.shape = gps.shape

    def __sub__(self, other: "GPS") -> "GeoGPS.Movement":
        movement = self.gps - other
        return GeoGPS.Movement(movement.yaw, movement.alpha * self.radius, is_deg=False)

    def __add__(self, movement: "GeoGPS.Movement") -> "GPS":
        gps_move = GPS.Movement(
            movement.yaw, movement.distance / self.radius, is_deg=False
        )
        return self.gps + gps_move

    def asvector(self) -> "GeoGPS.Vector":
        polar_asvector = self.gps.asvector()
        return GeoGPS.Vector(
            self.radius * polar_asvector.dlat, self.radius * polar_asvector.dlon
        )

    def ned2ecef(self) -> Array:
        return self.gps.ned2ecef()

    def ecef2ned(self) -> Array:
        return self.gps.ecef2ned()

    def vec(self) -> Array:
        return (self.gps.vec() * self.radius[..., None]).astype(Float)

    def __getitem__(self, key: Tuple) -> "GeoGPS":
        return GeoGPS(self.gps[key], self.radius[key])

    def reshape(self, shape: Tuple[int, ...]) -> "GeoGPS":
        return GeoGPS(self.gps.reshape(shape), np.reshape(self.radius, shape))

    def broadcast_to(self, shape: Tuple[int, ...]) -> "GeoGPS":
        return GeoGPS(self.gps.broadcast_to(shape), np.broadcast_to(self.radius, shape))
