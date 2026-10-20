"""Camera/GPS projection on a sphere; see README for coordinate conventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import RigidTransform

from . import check
from .camera import CameraModel, body_to_ned
from .geodesy import LocalMap, SphericalEarth, ned_to_ecef
from .ground import intersect_plane, intersect_sphere


@dataclass(frozen=True, eq=False)
class Geolocator:
    """Broadcast pixel/GPS projections with a fixed north/east mission map."""

    camera: CameraModel
    mission_origin_latlon_deg: ArrayLike
    radius_m: float = 6_371_008.8
    earth: SphericalEarth = field(init=False, repr=False)
    mission_map: LocalMap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mission_map = LocalMap(self.mission_origin_latlon_deg, self.radius_m)
        object.__setattr__(
            self, "mission_origin_latlon_deg", mission_map.origin_latlon_deg
        )
        object.__setattr__(self, "earth", mission_map.earth)
        object.__setattr__(self, "mission_map", mission_map)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Geolocator:
        """Read the full pipeline config; accept an optional ``earth_radius_m``."""
        geoloc = config["decision_system"]["geoloc"]
        return cls(
            CameraModel.from_config(geoloc, config.get("camera", {})),
            geoloc["manifold_center"],
            radius_m=geoloc.get("earth_radius_m", 6_371_008.8),
        )

    @staticmethod
    def _altitude(altitude_m: ArrayLike) -> NDArray[np.float64]:
        altitude = np.asarray(altitude_m, dtype=np.float64)
        check.surface_altitude(altitude)
        return altitude

    def _body_to_ecef(
        self,
        observer_latlon_deg: ArrayLike,
        altitude_m: ArrayLike,
        rpy_deg: ArrayLike,
    ) -> RigidTransform:
        """Return the aircraft pose; camera mounting stays in ``CameraModel``."""
        position = self.earth.geodetic_to_ecef(
            observer_latlon_deg, self._altitude(altitude_m)
        )
        orientation = ned_to_ecef(observer_latlon_deg) * body_to_ned(rpy_deg)
        return RigidTransform.from_components(position, orientation)

    def pixels_to_gps(
        self,
        pixels_px: ArrayLike,
        observer_latlon_deg: ArrayLike,
        altitude_m: ArrayLike,
        rpy_deg: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return ``(latlon_deg, valid)``; missed forward rays produce NaNs."""
        pose = self._body_to_ecef(observer_latlon_deg, altitude_m, rpy_deg)
        rays_ecef = pose.rotation.apply(self.camera.pixel_to_body_rays(pixels_px))
        hits, valid = intersect_sphere(pose.translation, rays_ecef, self.radius_m)
        # Keep invalid geometry out of PROJ while preserving the public mask.
        safe_hits = np.where(valid[..., None], hits, [self.radius_m, 0.0, 0.0])
        latlon = self.earth.ecef_to_geodetic(safe_hits)[..., :2]
        return np.where(valid[..., None], latlon, np.nan), valid

    def pixels_to_ne(
        self,
        pixels_px: ArrayLike,
        observer_latlon_deg: ArrayLike,
        altitude_m: ArrayLike,
        rpy_deg: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Project pixels into the fixed mission map, returning metres and mask."""
        latlon, valid = self.pixels_to_gps(
            pixels_px, observer_latlon_deg, altitude_m, rpy_deg
        )
        safe_latlon = np.where(valid[..., None], latlon, self.mission_origin_latlon_deg)
        ne = self.mission_map.gps_to_ne(safe_latlon)
        return np.where(valid[..., None], ne, np.nan), valid

    def gps_to_pixels(
        self,
        target_latlon_deg: ArrayLike,
        observer_latlon_deg: ArrayLike,
        altitude_m: ArrayLike,
        rpy_deg: ArrayLike,
        target_altitude_m: ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        """Project targets; return NaNs when behind the camera or Earth."""
        pose = self._body_to_ecef(observer_latlon_deg, altitude_m, rpy_deg)
        target = self.earth.geodetic_to_ecef(
            target_latlon_deg, self._altitude(target_altitude_m)
        )
        origin = pose.translation
        vector_ecef = target - origin
        pixels = self.camera.body_to_pixels(pose.apply(target, inverse=True))
        hit, intersects = intersect_sphere(origin, vector_ecef, self.radius_m)
        distance = np.linalg.norm(vector_ecef, axis=-1)
        hit_distance = np.linalg.norm(hit - origin, axis=-1)
        occluded = intersects & (hit_distance < distance - 1e-5)
        return np.where(occluded[..., None], np.nan, pixels)

    def pixels_to_ground(
        self, pixels_px: ArrayLike, altitude_m: ArrayLike, rpy_deg: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return ``(NED, valid)`` on z=0, with the camera at [0, 0, -altitude]."""
        altitude = self._altitude(altitude_m)
        origin = np.zeros(altitude.shape + (3,), dtype=np.float64)
        origin[..., 2] = -altitude
        return intersect_plane(origin, self.camera.rays_ned(pixels_px, rpy_deg))

    def ground_homography(
        self, altitude_m: float, rpy_deg: ArrayLike, meters_per_pixel: float
    ) -> tuple[NDArray[np.float64], tuple[int, int]]:
        """Return ``(H, (width, height))`` for an east-right, north-up ground view.

        Requires an undistorted image, positive altitude, and all corners on ground.
        """
        check.meters_per_pixel(meters_per_pixel)
        check.homography_pose(rpy_deg, altitude_m)
        check.homography_altitude(altitude_m)
        distortion = self.camera.distortion_coefficients
        check.homography_distortion(distortion)
        width, height = self.camera.image_size_px
        check.homography_image_size(width, height)
        corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float64,
        )
        ground, valid = self.pixels_to_ground(corners, altitude_m, rpy_deg)
        check.homography_corners(valid)
        xy = np.column_stack((ground[:, 1], -ground[:, 0]))
        destination = (xy - xy.min(axis=0)) / meters_per_pixel
        output_size = tuple(int(np.ceil(v)) + 1 for v in destination.max(axis=0))
        matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32), destination.astype(np.float32)
        )
        return np.asarray(matrix, dtype=np.float64), (output_size[0], output_size[1])
