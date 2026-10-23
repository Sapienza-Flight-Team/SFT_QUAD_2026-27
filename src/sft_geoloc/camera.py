"""OpenCV camera projection and SciPy attitude; conventions are in README.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from . import check




@dataclass(frozen=True, eq=False)
class CameraModel:
    """Fixed calibration: zero-skew OpenCV intrinsics and ``(width, height)``.

    Mounting must be orthogonal; reflections are supported. Distortion uses
    OpenCV's standard coefficient order. Arrays are copied and read-only.
    """

    intrinsic_matrix: ArrayLike
    image_size_px: tuple[int, int]
    camera_to_body: ArrayLike = field(default_factory=lambda: np.eye(3))
    distortion_coefficients: ArrayLike | None = None

    def __post_init__(self) -> None:
        intrinsic = np.array(self.intrinsic_matrix, dtype=np.float64, copy=True)
        check.intrinsic_matrix(intrinsic)
        size = np.asarray(self.image_size_px, dtype=np.float64)
        check.image_size(size)
        mounting = np.array(self.camera_to_body, dtype=np.float64, copy=True)
        check.camera_to_body(mounting)
        distortion = None
        if self.distortion_coefficients is not None:
            distortion = np.array(
                self.distortion_coefficients, dtype=np.float64, copy=True
            ).reshape(-1)
            check.distortion_coefficients(distortion)
            distortion.setflags(write=False)
        intrinsic.setflags(write=False)
        mounting.setflags(write=False)
        object.__setattr__(self, "intrinsic_matrix", intrinsic)
        object.__setattr__(self, "image_size_px", tuple(int(value) for value in size))
        object.__setattr__(self, "camera_to_body", mounting)
        object.__setattr__(self, "distortion_coefficients", distortion)

    @classmethod
    def from_config(
        cls,
        geoloc_config: Mapping[str, Any],
        camera_config: Mapping[str, Any] | None = None,
    ) -> CameraModel:
        """Read legacy config; geolocation takes precedence over camera settings.

        Focal length is mm, pixel size is micrometres, shape is [width, height].
        Optical centre defaults to half-size; mounting correction acts on the left.
        """
        camera_config = camera_config or {}

        def setting(*keys: str, default: Any = None) -> Any:
            for config in (geoloc_config, camera_config):
                for key in keys:
                    if config.get(key) is not None:
                        return config[key]
            return default

        focal_length = setting("focal_length", "focal_length_mm")
        pixel_size = setting("pixel_size")
        shape = setting("image_shape")
        matrix = setting("px2body_matrix")
        check.camera_config(focal_length, pixel_size, shape, matrix)
        focal_length, pixel_size = float(focal_length), float(pixel_size)
        check.focal_length_pixel_size(focal_length, pixel_size)
        size = np.asarray(shape, dtype=np.float64)
        check.image_shape(size)
        center = np.asarray(
            setting("optical_center", "optic_center", default=size / 2),
            dtype=np.float64,
        )
        check.optical_center(center)
        px2body = np.asarray(matrix, dtype=np.float64)
        check.px2body_matrix(px2body)
        angle = float(setting("px2body_error_angle_deg", default=0.0))
        check.finite(angle, name="px2body_error_angle_deg")
        mounting = np.eye(3)
        mounting[:2, :2] = px2body
        mounting = Rotation.from_euler("z", angle, degrees=True).as_matrix() @ mounting
        focal_px = focal_length * 1000 / pixel_size
        intrinsic = [[focal_px, 0, center[0]], [0, focal_px, center[1]], [0, 0, 1]]
        return cls(intrinsic, tuple(size), mounting, setting("distortion_coefficients"))

    @classmethod
    def from_parameters(
        cls,
        focal_length_mm: float,
        pixel_size_um: float,
        image_size_px: tuple[int, int],
        optical_center_px: ArrayLike,
        px2body_matrix: ArrayLike,
    ) -> CameraModel:
        """Build from physical calibration and an already corrected mounting."""
        return cls.from_config(
            {
                "focal_length": focal_length_mm,
                "pixel_size": pixel_size_um,
                "image_shape": image_size_px,
                "optical_center": optical_center_px,
                "px2body_matrix": px2body_matrix,
            }
        )

    def pixel_to_body_rays(self, pixels_px: ArrayLike) -> NDArray[np.float64]:
        """Return ``(..., 3)`` unit rays for ``(..., 2)`` pixels; invalids are NaN."""
        pixels = _vectors(pixels_px, 2, "pixels_px")
        flat = pixels.reshape(-1, 2)
        result = np.full((len(flat), 3), np.nan, dtype=np.float64)
        valid = np.isfinite(flat).all(axis=-1)
        if np.any(valid):
            selected = np.ascontiguousarray(flat[valid].reshape(-1, 1, 2))
            intrinsic = np.asarray(self.intrinsic_matrix)
            if self.distortion_coefficients is None:
                normalized = cv2.undistortPoints(selected, intrinsic, np.zeros(4))
            else:
                # More accurate than OpenCV's default five-iteration inversion.
                normalized = cv2.undistortPointsIter(
                    selected,
                    intrinsic,
                    np.asarray(self.distortion_coefficients),
                    np.eye(3),
                    np.eye(3),
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
                )
            camera_rays = np.column_stack(
                (normalized.reshape(-1, 2), np.ones(np.count_nonzero(valid)))
            )
            body_rays = camera_rays @ np.asarray(self.camera_to_body).T
            result[valid] = body_rays / np.linalg.norm(
                body_rays, axis=-1, keepdims=True
            )
        return result.reshape(pixels.shape[:-1] + (3,))

    def body_to_pixels(self, points_body: ArrayLike) -> NDArray[np.float64]:
        """Project camera-relative ``(..., 3)`` body points to ``(..., 2)`` pixels.

        Nonfinite/on/behind-camera points return NaN; image bounds are not clipped.
        """
        points = _vectors(points_body, 3, "points_body")
        camera_points = points.reshape(-1, 3) @ np.asarray(self.camera_to_body)
        result = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
        valid = np.isfinite(camera_points).all(axis=-1) & (camera_points[:, 2] > 0)
        if np.any(valid):
            projected, _ = cv2.projectPoints(
                np.ascontiguousarray(camera_points[valid]),
                np.zeros(3),
                np.zeros(3),
                np.asarray(self.intrinsic_matrix),
                (
                    np.zeros(4)
                    if self.distortion_coefficients is None
                    else np.asarray(self.distortion_coefficients)
                ),
            )
            result[valid] = projected.reshape(-1, 2)
        return result.reshape(points.shape[:-1] + (2,))

    def rays_ned(self, pixels_px: ArrayLike, rpy_deg: ArrayLike) -> NDArray[np.float64]:
        """Return NED rays, broadcasting leading pixel/attitude dimensions."""
        return body_to_ned(rpy_deg).apply(self.pixel_to_body_rays(pixels_px))

def _vectors(values: ArrayLike, length: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    check.vector_shape(array, size=length, name=name)
    return array


def body_to_ned(rpy_deg: ArrayLike) -> Rotation:
    """Convert ``(..., 3)`` roll/pitch/yaw degrees using extrinsic ``xyz``."""
    angles = _vectors(rpy_deg, 3, "rpy_deg")
    check.finite(angles, name="rpy_deg")
    return Rotation.from_euler("xyz", angles, degrees=True)
