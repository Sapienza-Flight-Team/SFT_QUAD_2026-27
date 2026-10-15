from sft_math.cam.affine import Affine, Point
from sft_math.cam import Camera
from sft_math.geo.polar import utils as polar_utils
from sft_math.geo.polar import GPS
from sft_math.geo import GeoGPS
from sft_math.class_templates import ClassBase
import numpy as np
from typing import Tuple


class GeoLoc(ClassBase):

    camera: Camera
    configuration: Camera.Configuration
    gps: GeoGPS
    shape: Tuple[int, ...]

    __constants__ = [
        (lambda x: x, "camera", ""),
        (lambda x: x, "configuration", ""),
        (lambda x: x, "gps", ""),
        (lambda x: x, "shape", ""),
    ]

    def __init__(
        self, camera: Camera, configuration: Camera.Configuration, gps: GeoGPS
    ):
        try:
            self.shape = np.broadcast_shapes(
                camera.shape, configuration.shape, gps.shape
            )
        except:
            raise ValueError(
                "The shapes of camera, configuration and gps must be broadcastable"
            )
        self.camera = camera
        self.configuration = configuration
        self.gps = gps

    def to_gps(self, px_point: Point) -> GeoGPS:
        try:
            np.broadcast_shapes(px_point.shape, self.shape)
        except ValueError:
            raise ValueError("The shapes of px_point and geoloc must be broadcastable")

        # traduco px_point in una retta in coordinate ned usando camera e configuration
        nedSpace = self.camera.px2ned(px_point, self.configuration)

        # calcolo l'intersezione con la sfera terrestre
        height = -self.configuration.position.vector[..., 2]

        if np.any(height < 0):
            raise ValueError("The camera is below the ground")

        cosalpha = np.abs(nedSpace.orthCoos2globalCoos_matrix[..., 2, 0])
        sinalpha = np.abs(1 - cosalpha**2) ** 0.5
        fact1 = self.gps.radius - (height + self.gps.radius) * sinalpha
        if np.any(fact1 < 0):
            raise RuntimeError("The straight does not intersect the sphere")
        fact2 = self.gps.radius + (height + self.gps.radius) * sinalpha

        distance = (height + self.gps.radius) * cosalpha - fact1**0.5 * fact2**0.5
        distance = np.where(
            nedSpace.orthCoos2globalCoos_matrix[..., 2, 0] < 0,
            -np.abs(distance),
            np.abs(distance),
        )

        groundvec = nedSpace(distance[..., None])
        groundvec = np.einsum("...ik,...k->...i", self.gps.gps.ned2ecef(), groundvec)
        groundvec += self.gps.gps.vec() * self.gps.radius[..., None]

        return GeoGPS(polar_utils.from_vec(groundvec), self.gps.radius)

    def from_gps(self, gps_point: GPS) -> Point:
        # calcolo in coordinate ned la posizione del punto target
        target = gps_point.vec() * self.gps.radius[..., None]
        ground = self.gps.vec()
        groundvec = target - ground
        groundvec = np.einsum("...ik,...k->...i", self.gps.ecef2ned(), groundvec)
        # definisco la retta che passa per il punto target e la camera
        direction = groundvec - self.configuration.position.vector
        straight = Affine(self.configuration.position.vector, direction[..., :, None])
        # calcolo il punto in coordinate pixel
        px_point = self.camera.ned2px(straight, self.configuration)
        return px_point

    def __getitem__(self, key: Tuple) -> "GeoLoc":
        return GeoLoc(self.camera[key], self.configuration[key], self.gps[key])

    def reshape(self, shape: Tuple[int, ...]) -> "GeoLoc":
        return GeoLoc(
            self.camera.reshape(shape),
            self.configuration.reshape(shape),
            self.gps.reshape(shape),
        )

    def broadcast_to(self, shape: Tuple[int, ...]) -> "GeoLoc":
        return GeoLoc(
            self.camera.broadcast_to(shape),
            self.configuration.broadcast_to(shape),
            self.gps.broadcast_to(shape),
        )
