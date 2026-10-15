from sft_math.typing import Array, Float
from sft_math.geo.class_template import GPSBase
from typing import Tuple
import numpy as np
from numpy import typing as np_typing


class GPS(GPSBase["GPS", "GPS"]):
    """Class for representing a GPS coordinate
    It is not necessary to know the radius of the earth to represent a GPS coordinate.

    Parameters:
        lat: latitude in radians
        lon: longitude in radians
        shape: shape of the data in this class
    """

    class Movement(GPSBase.Movement["GPS"]):
        """Movement is a class that represents movement in GPS coordinates.

        parameters:
            yaw: direction of the movement
                The angle of the plane with the circumference of the movement respecting the plane north-east of the origin.
                It is a value between 0 and 2*pi.
            alpha: angle of the movement
                The angle of the two vectors respecting the center of the polar system.
                It is a value between 0 and pi.
            shape: the shape of the data in this class
        """

        yaw: Array  # with shape shape
        alpha: Array  # with shape shape
        shape: Tuple[int, ...]

        __constants__ = [
            (lambda x: np.rad2deg(x), "yaw", "°"),
            (lambda x: np.rad2deg(x), "alpha", "°"),
            (lambda x: x, "shape", ""),
        ]

        def __init__(
            self,
            yaw: np_typing.ArrayLike,
            alpha: np_typing.ArrayLike,
            is_deg: bool = True,
        ):
            """Initializes a GPS movement

            Args:
                yaw (np_typing.ArrayLike): direction of the movement
                    the angle of the plane with the circumference of the movement respect the plane nord-east of the origin
                alpha (np_typing.ArrayLike): angle of the movement
                    the angle of the two vectors respect the center of the polar system
                is_deg (bool, optional): if the angles are in degrees. Defaults to True.

            Raises:
                ValueError: if yaw and alpha are not broadcastable
            """
            self.yaw = np.atleast_1d(np.array(yaw, dtype=Float))
            self.alpha = np.atleast_1d(np.array(alpha, dtype=Float))
            try:
                self.shape = np.broadcast(self.yaw, self.alpha).shape
                self.yaw = np.broadcast_to(self.yaw, self.shape)
                self.alpha = np.broadcast_to(self.alpha, self.shape)
            except ValueError:
                raise ValueError("yaw and alpha must be broadcastable")
            if is_deg:
                self.yaw = np.deg2rad(self.yaw)
                self.alpha = np.deg2rad(self.alpha)
            # normalize the angles
            counter = np.mod(np.floor_divide(self.alpha + np.pi, 2 * np.pi), 2)
            self.alpha = (self.alpha * (1 - counter * 2)).astype(Float)
            self.yaw = (self.yaw * (1 - counter * 2)).astype(Float)
            self.alpha = (np.mod(self.alpha + np.pi, (2 * np.pi)) - np.pi).astype(Float)
            self.yaw = (np.mod(self.yaw + np.pi, 2 * np.pi) - np.pi).astype(Float)

        def __neg__(self) -> "GPS.Movement":
            return GPS.Movement(self.yaw + np.pi, self.alpha, is_deg=False)

        def __pos__(self) -> "GPS.Movement":
            return GPS.Movement(self.yaw, self.alpha, is_deg=False)

        def __mul__(self, coef: np_typing.ArrayLike) -> "GPS.Movement":
            return GPS.Movement(
                self.yaw, self.alpha * np.array(coef, dtype=Float), is_deg=False
            )

        def __rmul__(self, coef: np_typing.ArrayLike) -> "GPS.Movement":
            return GPS.Movement(
                self.yaw, self.alpha * np.array(coef, dtype=Float), is_deg=False
            )

        def __truediv__(self, coef: np_typing.ArrayLike) -> "GPS.Movement":
            return GPS.Movement(
                self.yaw, self.alpha / np.array(coef, dtype=Float), is_deg=False
            )

        def to_vector(self) -> "GPS.Vector":
            gps = GPS(0, 0) + self
            return GPS.Vector(gps.lat, gps.lon, is_deg=False)

        def __getitem__(self, key: Tuple) -> "GPS.Movement":
            return GPS.Movement(self.yaw[key], self.alpha[key], is_deg=False)

        def reshape(self, shape: Tuple[int, ...]) -> "GPS.Movement":
            return GPS.Movement(
                np.reshape(self.yaw, shape), np.reshape(self.alpha, shape), is_deg=False
            )

        def broadcast_to(self, shape: Tuple[int, ...]) -> "GPS.Movement":
            return GPS.Movement(
                np.broadcast_to(self.yaw, shape),
                np.broadcast_to(self.alpha, shape),
                is_deg=False,
            )

        def moveaxis(
            self, source: Tuple[int, ...], destination: Tuple[int, ...]
        ) -> "GPS.Movement":
            return GPS.Movement(
                np.moveaxis(self.yaw, source, destination),
                np.moveaxis(self.alpha, source, destination),
                is_deg=False,
            )

    class Vector(GPSBase.Vector["GPS"]):
        """Class to represent a movement in a manifold
        Its peculiarity is that the movement is operable with other movements in the same manifold

        Parameters:
            dlat: variation of latitude movement in radians
            dlon: variation of longitude movement in radians
            shape: shape of the data in this class
        """

        dlat: Array  # with shape shape
        dlon: Array  # with shape shape
        shape: Tuple[int, ...]

        __constants__ = [
            (lambda x: np.rad2deg(x), "dlat", "°"),
            (lambda x: np.rad2deg(x), "dlon", "°"),
            (lambda x: x, "shape", ""),
        ]

        def __init__(
            self,
            dlat: np_typing.ArrayLike,
            dlon: np_typing.ArrayLike,
            is_deg: bool = True,
        ):
            """Initializes a movement in a manifold

            Args:
                dlat (np_typing.ArrayLike): Variation of latitude movement in radians
                dlon (np_typing.ArrayLike): Variation of longitude movement in radians
                is_deg (bool, optional): if the angles are in degrees. Defaults to True.

            Raises:
                ValueError: if lat and lon are not broadcastable
            """
            self.dlat = np.atleast_1d(np.array(dlat, dtype=Float))
            self.dlon = np.atleast_1d(np.array(dlon, dtype=Float))
            try:
                self.shape = np.broadcast(self.dlat, self.dlon).shape
                self.dlat = np.broadcast_to(self.dlat, self.shape)
                self.dlon = np.broadcast_to(self.dlon, self.shape)
            except ValueError:
                raise ValueError("lat and lon must be broadcastable")
            if is_deg:
                self.dlat = np.deg2rad(self.dlat)
                self.dlon = np.deg2rad(self.dlon)
            if np.abs(self.dlat) >= np.pi or np.abs(self.dlon) >= 2 * np.pi:
                import warnings

                warnings.warn("The movement is greater than the Earth's surface")

        def to_movement(self) -> "GPS.Movement":
            return GPS(self.dlat, self.dlon, is_deg=False) - GPS(0, 0, is_deg=False)

        def __neg__(self) -> "GPS.Vector":
            return GPS.Vector(-self.dlat, -self.dlon, is_deg=False)

        def __pos__(self) -> "GPS.Vector":
            return GPS.Vector(self.dlat, self.dlon, is_deg=False)

        def __add__(self, other: "GPS.Vector") -> "GPS.Vector":
            return GPS.Vector(
                self.dlat + other.dlat, self.dlon + other.dlon, is_deg=False
            )

        def __sub__(self, other: "GPS.Vector") -> "GPS.Vector":
            return GPS.Vector(
                self.dlat - other.dlat, self.dlon - other.dlon, is_deg=False
            )

        def __mul__(self, scalar: np_typing.ArrayLike) -> "GPS.Vector":
            return GPS.Vector(
                self.dlat * np.array(scalar, dtype=Float),
                self.dlon * np.array(scalar, dtype=Float),
                is_deg=False,
            )

        def __rmul__(self, scalar: np_typing.ArrayLike) -> "GPS.Vector":
            return GPS.Vector(
                self.dlat * np.array(scalar, dtype=Float),
                self.dlon * np.array(scalar, dtype=Float),
                is_deg=False,
            )

        def __truediv__(self, scalar: np_typing.ArrayLike) -> "GPS.Vector":
            return GPS.Vector(
                self.dlat / np.array(scalar, dtype=Float),
                self.dlon / np.array(scalar, dtype=Float),
                is_deg=False,
            )

        def norm(self) -> Array:
            return np.array(np.sqrt(self.dlat**2 + self.dlon**2), dtype=Float)

        def angle(self) -> Array:
            return np.array(np.arctan2(self.dlon, self.dlat), dtype=Float)

        def __getitem__(self, key: Tuple) -> "GPS.Vector":
            return GPS.Vector(self.dlat[key], self.dlon[key], is_deg=False)

        def reshape(self, shape: Tuple[int, ...]) -> "GPS.Vector":
            return GPS.Vector(
                np.reshape(self.dlat, shape), np.reshape(self.dlon, shape), is_deg=False
            )

        def broadcast_to(self, shape: Tuple[int, ...]) -> "GPS.Vector":
            return GPS.Vector(
                np.broadcast_to(self.dlat, shape),
                np.broadcast_to(self.dlon, shape),
                is_deg=False,
            )

        def moveaxis(
            self, source: Tuple[int, ...], destination: Tuple[int, ...]
        ) -> "GPS.Movement":
            return GPS.Movement(
                np.moveaxis(self.dlat, source, destination),
                np.moveaxis(self.dlon, source, destination),
                is_deg=False,
            )

        def to_numpy(self) -> Array:
            return np.array([self.dlat, self.dlon], dtype=Float)

    lat: Array
    lon: Array
    shape: Tuple[int, ...]

    __constants__ = [
        (lambda x: np.rad2deg(x), "lat", "°"),
        (lambda x: np.rad2deg(x), "lon", "°"),
        (lambda x: x, "shape", ""),
    ]

    def __init__(
        self, lat: np_typing.ArrayLike, lon: np_typing.ArrayLike, is_deg: bool = True
    ):
        """Initializes a GPS coordinate

        Args:
            lat (np_typing.ArrayLike): latitude in radians
            lon (np_typing.ArrayLike): longitude in radians
            is_deg (bool, optional): if the angles are in degrees. Defaults to True.

        Raises:
            ValueError: if lat and lon are not broadcastable
        """
        self.lat = np.atleast_1d(np.array(lat, dtype=Float))
        self.lon = np.atleast_1d(np.array(lon, dtype=Float))
        try:
            self.shape = np.broadcast(self.lat, self.lon).shape
            self.lat = np.broadcast_to(self.lat, self.shape)
            self.lon = np.broadcast_to(self.lon, self.shape)
        except ValueError:
            raise ValueError("lat and lon must be broadcastable")
        if is_deg:
            self.lat = np.deg2rad(self.lat)
            self.lon = np.deg2rad(self.lon)
        # normalize the angles
        counter = np.mod(np.floor_divide(self.lat + np.pi / 2, np.pi), 2)
        self.lat = (self.lat * (1 - counter * 2)).astype(Float)
        self.lon = (self.lon * (1 - counter * 2)).astype(Float)
        self.lat = (np.mod(self.lat + np.pi / 2, np.pi) - np.pi / 2).astype(Float)
        self.lon = (np.mod(self.lon + np.pi, 2 * np.pi) - np.pi).astype(Float)

    def __sub__(self, other: "GPS") -> Movement:
        """Computes the movement from one GPS coordinate to another

        Args:
            other (GPS): other GPS coordinate

        Returns:
            Movement: movement from self to other

        Details:
        The movement is computed as follows:
        1. Compute the vector in ECEF coordinates from self to other
        2. Write this vector respect the NED coordinates of self
        3. yaw is the angle of the vector in the plane NE
        4. alpha is the angle btween the vectors self and other
        """
        dest_vec = self.vec()
        origin_ned2ecef = other.ned2ecef()
        diff = (
            origin_ned2ecef[..., :, 2] + dest_vec
        )  # origin_down + dest_vec (= dest_vec - origin_vec)
        diff_nord = np.einsum(
            "...k, ...k -> ...", diff, origin_ned2ecef[..., :, 0]
        )  # fourier coefficient of diff over nord vec in origin
        diff_east = np.einsum(
            "...k, ...k -> ...", diff, origin_ned2ecef[..., :, 1]
        )  # fourier coefficient of diff over east vec in origin
        return GPS.Movement(
            np.arctan2(diff_east, diff_nord),
            np.arccos(
                np.clip(
                    np.einsum(
                        "...k, ...k -> ...",
                        -origin_ned2ecef[..., :, 2],
                        dest_vec,
                    ),
                    -1.0,
                    1.0,
                ),
            ),
            is_deg=False,
        )

    def __add__(self, movement: Movement) -> "GPS":
        """Computes the GPS coordinate after a movement

        Args:
            movement (Movement): movement to apply

        Returns:
            GPS: GPS coordinate after the movement

        Details:
        The movement is computed as follows:
        1. Compute the direction of the movement in ECEF coordinates,
            this vector is inside the plane with the circumference of the movement over the surface of the Earth
        2. Compute the destination of the movement in ECEF coordinates
        3. Compute the GPS coordinate of the destination
        """
        ned2ecef = self.ned2ecef()
        direction = (
            ned2ecef[..., :, 0] * np.cos(movement.yaw)[..., None]
            + ned2ecef[..., :, 1] * np.sin(movement.yaw)[..., None]
        )
        destination = (
            np.cos(movement.alpha)[..., None] * (-ned2ecef[..., :, 2])
            + np.sin(movement.alpha)[..., None] * direction
        )

        return GPS(
            np.arcsin(np.clip(destination[..., 2], -1.0, 1.0)),
            np.arctan2(destination[..., 1], destination[..., 0]),
            is_deg=False,
        )

    def asvector(self) -> "GPS.Vector":
        return GPS.Vector(self.lat, self.lon, is_deg=False)

    def vec(self) -> Array:
        """Converts the GPS coordinates to a unit vector, the direction of the GPS coordinates in the ECEF coordinates

        Returns:
            Array: unit vector with shape (..., 3) where:
                - [..., 0] is the x component
                - [..., 1] is the y component
                - [..., 2] is the z component
        """
        coslat = np.cos(self.lat)
        return np.stack(
            [coslat * np.cos(self.lon), coslat * np.sin(self.lon), np.sin(self.lat)],
            axis=-1,
            dtype=Float,
        )

    def ned2ecef(self) -> Array:
        """Converts the the local NED coordinates to ECEF coordinates

        Returns:
            - Array: Transformation matrix with shape (..., 3, 3) where:
                - [..., :, 0] is the Nord vector in ECEF coordinates
                - [..., :, 1] is the East vector in ECEF coordinates
                - [..., :, 2] is the Down vector in ECEF coordinates

        Details:
        The transformation matrix is computed as follows:
        1. Compute the gps vector
        2. Compute the East vector, we know that it is parallel to the plane xy and orthogonal to the gps vector
        3. Compute the Nord vector, we know that it is orthogonal to the gps vector and the East vector (cross product)
        """
        ecef_vec = self.vec()
        ecef_east = np.stack(
            [-ecef_vec[..., 1], ecef_vec[..., 0], np.zeros(ecef_vec.shape[:-1])],
            axis=-1,
        )
        np.errstate(divide="ignore", invalid="ignore")
        ecef_east_norm = np.linalg.norm(ecef_east, axis=-1, keepdims=True)
        ecef_east = np.where(
            ecef_east_norm == 0,
            np.tile(
                np.array([0.0, 1.0, 0.0], dtype=Float), ecef_east.shape[:-1] + (1,)
            ),
            ecef_east / ecef_east_norm,
        )
        np.errstate(divide="warn", invalid="warn")
        ecef_nord = np.cross(ecef_vec, ecef_east, axis=-1)
        return np.stack((ecef_nord, ecef_east, -ecef_vec), axis=-1, dtype=Float)

    def ecef2ned(self) -> Array:
        return np.moveaxis(self.ned2ecef(), -1, -2)

    def __getitem__(self, key: Tuple) -> "GPS":
        return GPS(self.lat[key], self.lon[key], is_deg=False)

    def reshape(self, shape: Tuple[int, ...]) -> "GPS":
        return GPS(
            np.reshape(self.lat, shape), np.reshape(self.lon, shape), is_deg=False
        )

    def broadcast_to(self, shape: Tuple[int, ...]) -> "GPS":
        return GPS(
            np.broadcast_to(self.lat, shape),
            np.broadcast_to(self.lon, shape),
            is_deg=False,
        )
