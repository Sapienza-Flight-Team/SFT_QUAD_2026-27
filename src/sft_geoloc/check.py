"""Input validation shared by the geolocation modules.

Checks return None on success and raise ValueError with a descriptive message on
failure. Call them directly so validation also runs under Python's -O/-OO modes.
Shape checks expect arrays already converted by the calling module.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vector_shape(array: NDArray[np.float64], *, size: int, name: str) -> None:
    """Check the trailing dimension of a vector or batch of vectors."""
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size})")


def finite(values: ArrayLike, *, name: str) -> None:
    """Reject NaN and infinity in a scalar or array."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")


def finite_nonnegative(value: float, *, name: str) -> None:
    """Check a scalar error bound or other nonnegative quantity."""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def intrinsic_matrix(intrinsic: NDArray[np.float64]) -> None:
    """Check finite 3x3 intrinsics, positive focal lengths, and pinhole form."""
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic_matrix must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Camera focal lengths in pixels must be positive")
    expected = np.array(
        [
            [intrinsic[0, 0], 0, intrinsic[0, 2]],
            [0, intrinsic[1, 1], intrinsic[1, 2]],
            [0, 0, 1],
        ]
    )
    if not np.allclose(intrinsic, expected, atol=1e-12, rtol=0):
        raise ValueError("intrinsic_matrix must have OpenCV zero-skew pinhole form")


def image_size(size: NDArray[np.float64]) -> None:
    """Check positive, finite, whole-pixel width and height."""
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("image_size_px must contain positive finite width and height")
    if np.any(size != np.floor(size)):
        raise ValueError("image_size_px width and height must be whole pixels")


def camera_to_body(mounting: NDArray[np.float64]) -> None:
    """Check an orthogonal mounting matrix; reflections are allowed."""
    if (
        mounting.shape != (3, 3)
        or not np.all(np.isfinite(mounting))
        or not np.allclose(mounting.T @ mounting, np.eye(3), atol=1e-9, rtol=0)
    ):
        raise ValueError("camera_to_body must be an orthogonal finite 3x3 matrix")


def distortion_coefficients(distortion: NDArray[np.float64]) -> None:
    """Check the coefficient count and finiteness for OpenCV distortion."""
    if distortion.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
        raise ValueError(
            "distortion_coefficients must have 4, 5, 8, 12, or 14 finite values"
        )


def camera_config(
    focal_length: Any, pixel_size: Any, image_shape: Any, px2body_matrix: Any
) -> None:
    """Check required calibration settings before converting their values."""
    if any(
        value is None
        for value in (focal_length, pixel_size, image_shape, px2body_matrix)
    ):
        raise ValueError(
            "Config requires focal_length, pixel_size, image_shape, and px2body_matrix"
        )


def focal_length_pixel_size(focal_length: float, pixel_size: float) -> None:
    """Check the physical calibration values used to calculate focal pixels."""
    if (
        not np.isfinite([focal_length, pixel_size]).all()
        or min(focal_length, pixel_size) <= 0
    ):
        raise ValueError("focal_length and pixel_size must be positive and finite")


def image_shape(size: NDArray[np.float64]) -> None:
    """Check the config shape before calculating the default optical center."""
    if size.shape != (2,):
        raise ValueError("image_shape must be [width, height]")


def optical_center(center: NDArray[np.float64]) -> None:
    """Check the two finite pixel coordinates of the optical center."""
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("optical_center must contain two finite pixel coordinates")


def px2body_matrix(matrix: NDArray[np.float64]) -> None:
    """Check the config mounting shape before embedding it in a 3x3 matrix."""
    if matrix.shape != (2, 2):
        raise ValueError("px2body_matrix must be a 2x2 matrix")


def latitude(latlon: NDArray[np.float64]) -> None:
    """Check latitude bounds after vector shape and finiteness validation."""
    if np.any(np.abs(latlon[..., 0]) > 90):
        raise ValueError("latitude must be between -90 and 90 degrees")


def radius_scalar(radius: ArrayLike) -> None:
    """Check radius dimensionality before converting it to a float."""
    if np.ndim(radius) != 0:
        raise ValueError("radius_m must be a finite positive scalar")


def radius(radius_m: float) -> None:
    """Check that the sphere radius is a finite positive scalar."""
    if np.ndim(radius_m) != 0 or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be a finite positive scalar")


def surface_altitude(altitude: NDArray[np.float64]) -> None:
    """Check finite heights on or above the model surface."""
    if not np.all(np.isfinite(altitude)) or np.any(altitude < 0):
        raise ValueError("altitude_m must be finite and above the model surface")


def meters_per_pixel(value: float) -> None:
    """Check the positive finite scale of the output ground image."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError("meters_per_pixel must be finite and positive")


def homography_pose(rpy: ArrayLike, altitude: float) -> None:
    """Require one attitude and one altitude for a ground homography."""
    if np.shape(rpy) != (3,) or np.ndim(altitude) != 0:
        raise ValueError("ground_homography requires a single camera pose")


def homography_altitude(altitude: float) -> None:
    """Require a positive finite height after validating pose dimensionality."""
    if not np.isfinite(altitude) or altitude <= 0:
        raise ValueError("ground_homography requires positive finite altitude_m")


def homography_distortion(distortion: ArrayLike | None) -> None:
    """Require absent or zero lens distortion for a ground homography."""
    if distortion is not None and np.any(np.asarray(distortion) != 0):
        raise ValueError(
            "undistort the image first, then use a CameraModel with "
            "distortion_coefficients=None for its homography"
        )


def homography_image_size(width: int, height: int) -> None:
    """Require distinct corner pixels on both image axes."""
    if min(width, height) < 2:
        raise ValueError("homography requires at least two pixels on each axis")


def homography_corners(valid: NDArray[np.bool_]) -> None:
    """Require all image corners to intersect the forward ground plane."""
    if not np.all(valid):
        raise ValueError("every image corner must intersect the forward ground plane")


def ray_shape(arrays: Sequence[NDArray[np.float64]]) -> None:
    """Check the trailing XYZ dimensions of ray origins and directions."""
    if any(array.ndim == 0 or array.shape[-1] != 3 for array in arrays):
        raise ValueError("origins and directions must have shape (..., 3)")


def pixels_shape(pixels: NDArray[np.float64]) -> None:
    """Require a batch of N two-dimensional pixels for uncertainty estimation."""
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_px must have shape (N, 2)")


def altitude_samples(altitude: NDArray[np.float64]) -> None:
    """Require a one-dimensional array containing at least two samples."""
    if altitude.ndim != 1 or len(altitude) < 2:
        raise ValueError("altitude_m must contain at least two samples, shape (S,)")


def sample_shapes(
    positions: NDArray[np.float64], rpy: NDArray[np.float64], count: int
) -> None:
    """Check that position and attitude batches match the altitude sample count."""
    if positions.shape != (count, 2) or rpy.shape != (count, 3):
        raise ValueError("sample positions/attitudes must have shapes (S, 2)/(S, 3)")


def finite_samples(*arrays: NDArray[np.float64]) -> None:
    """Require finite pixels and telemetry throughout the sample batches."""
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("samples and pixels must be finite")


def sample_count(samples: int) -> None:
    """Require an integer sample count of at least two; reject booleans."""
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, np.integer))
        or samples < 2
    ):
        raise ValueError("samples must be an integer >= 2")


def confidence_percent(percent: float) -> None:
    """Check that confidence is strictly between zero and one hundred percent."""
    if not 0 < percent < 100:
        raise ValueError("confidence_percent must be between 0 and 100")


def estimate_pose(observer: ArrayLike, rpy: ArrayLike) -> None:
    """Require one observer position and attitude for an exposure estimate."""
    if np.shape(observer) != (2,) or np.shape(rpy) != (3,):
        raise ValueError("estimate requires one observer (2,) and attitude (3,)")


def estimate_altitude(altitude: float) -> None:
    """Require one finite nonnegative height for an exposure estimate."""
    if np.ndim(altitude) != 0 or not np.isfinite(altitude) or altitude < 0:
        raise ValueError("altitude_m must be a finite nonnegative scalar")


def positive_scalar(value: float, *, name: str) -> None:
    """Require a finite positive scalar, including physical lengths and scales."""
    if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")


def positive_integer(value: int, *, name: str) -> None:
    """Require a positive integer; booleans do not represent counts."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def clustering_data(
    points: NDArray[np.float64], covariance: NDArray[np.float64]
) -> None:
    """Require finite (N,D) points and symmetric positive-definite (N,D,D) errors."""
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError("points must have shape (N, D), with D >= 1")
    count, dimensions = points.shape
    if covariance.shape != (count, dimensions, dimensions):
        raise ValueError("covariances must have shape (N, D, D), matching points")
    finite(points, name="points")
    finite(covariance, name="covariances")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError("covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise ValueError("covariances must be positive definite")


def clustering_dimensions(actual: int, expected: int) -> None:
    """Require matching coordinate dimensions when appending detections."""
    if actual != expected:
        raise ValueError("appended data must have the same coordinate dimensions")


def centroid_options(horizon: float, density_filter: float) -> None:
    """Validate the Mahalanobis confidence and relative population cutoff."""
    if not np.isfinite(horizon) or not 0 <= horizon <= 1:
        raise ValueError("horizon must be between 0 and 1")
    if not np.isfinite(density_filter) or not 0 <= density_filter < 1:
        raise ValueError("filter must be between 0 (inclusive) and 1 (exclusive)")


def centroid_feedback(feedback: NDArray[np.float64], dimensions: int) -> None:
    """Validate previously selected centroids before reusing them as seeds."""
    if feedback.ndim != 2 or feedback.shape[1] != dimensions:
        raise ValueError("feedback must have shape (N, D), matching the data")
    finite(feedback, name="feedback")


def clustering_plot(dimensions: int, confidence: float) -> None:
    """Require two-dimensional data and a finite ellipse confidence in (0,1)."""
    if dimensions != 2:
        raise ValueError("clustering plots require two-dimensional points")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("covs_p must be between 0 and 1")


def survey_camera(params: Any) -> None:
    """Validate physical calibration, positive flight height, and overlap fractions."""
    for name in (
        "sensor_width_mm",
        "sensor_height_mm",
        "image_width_px",
        "image_height_px",
        "focal_length_mm",
        "altitude_m",
    ):
        positive_scalar(getattr(params, name), name=name)
    for name in ("image_width_px", "image_height_px"):
        value = getattr(params, name)
        if value != int(value):
            raise ValueError(f"{name} must be a whole pixel count")
    for name in ("frontal_overlap", "side_overlap"):
        value = getattr(params, name)
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be between 0 (inclusive) and 1 (exclusive)")
    if not isinstance(params.landscape, (bool, np.bool_)):
        raise ValueError("landscape must be a boolean")


def survey_vertices(vertices: NDArray[np.float64]) -> None:
    """Require at least three finite latitude/longitude vertices in degrees."""
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("polygon must have shape (N, 2), with at least three vertices")
    finite(vertices, name="polygon")
    latitude(vertices)


def survey_polygon(polygon: Any) -> None:
    """Reject empty, zero-area, or self-intersecting projected survey boundaries."""
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("survey polygon must be simple and have positive area")


def waypoint_position(latlon: NDArray[np.float64], altitude: float) -> None:
    """Validate one geographic waypoint and nonnegative relative altitude."""
    if latlon.shape != (2,):
        raise ValueError("waypoint must contain one latitude/longitude pair")
    finite(latlon, name="waypoint")
    latitude(latlon)
    estimate_altitude(altitude)


def mounting_required(matrix: Any) -> None:
    """Keep missing configuration distinct from malformed calibration."""
    if matrix is None:
        raise KeyError("px2body_matrix must be defined in YAML config")


def finite_scalar(value: float, *, name: str) -> None:
    """Require a finite scalar angle or other unrestricted scalar quantity."""
    if np.ndim(value) != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
