import math
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from sft_math.cam import Camera
from sft_math.cam.affine import Affine
from sft_math.cam.px2body_matrix import resolve_px2body_matrix


@dataclass
class HomographyComputation:
    matrix: np.ndarray
    inverse_matrix: np.ndarray
    output_size: tuple[int, int]
    meters_per_pixel: float | None


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _float_from_row(
    row: dict[str, Any],
    names: Iterable[str],
    *,
    default: float | None = None,
) -> float:
    value = _first_present(row, names)
    if value is None:
        if default is None:
            raise ValueError(f"Missing metadata column: one of {', '.join(names)}")
        return default
    return float(value)


def _numpy_shape_to_width_height(shape: Iterable[int]) -> tuple[int, int]:
    dims = tuple(int(v) for v in shape)
    if len(dims) < 2:
        raise ValueError(f"image shape must have at least 2 dimensions, got {shape}")
    # Numpy/OpenCV arrays are HxW[xC]; Camera/OpenCV dsize use WxH.
    return dims[1], dims[0]


def _numpy_shape_to_height_width(shape: Iterable[int]) -> tuple[int, int]:
    dims = tuple(int(v) for v in shape)
    if len(dims) < 2:
        raise ValueError(f"image shape must have at least 2 dimensions, got {shape}")
    return dims[0], dims[1]


def build_camera(config: dict[str, Any], image_width: int, image_height: int) -> Camera:
    geoloc_config = config.get("decision_system", {}).get("geoloc", {})
    camera_config = config.get("camera", {})
    focal_length = float(
        geoloc_config.get("focal_length")
        or camera_config.get("focal_length")
        or camera_config.get("focal_length_mm")
        or 16.0
    )
    pixel_size = float(
        geoloc_config.get("pixel_size", camera_config.get("pixel_size", 2.74))
    )
    px2body_matrix = resolve_px2body_matrix(geoloc_config, camera_config)
    optical_center = (
        geoloc_config.get("optical_center")
        or geoloc_config.get("optic_center")
        or camera_config.get("optical_center")
        or camera_config.get("optic_center")
    )
    if optical_center is None:
        optical_center = (image_width / 2, image_height / 2)

    return Camera(
        focal_length,
        pixel_size,
        (image_width, image_height),
        optical_center,
        px2body_matrix,
    )


def metadata_angles(
    row: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, float, float, bool]:
    """Return roll, pitch, yaw in degrees."""
    _ = config
    roll = _float_from_row(row, ("roll", "phi"), default=0.0)
    pitch = _float_from_row(row, ("pitch", "theta"), default=0.0)
    yaw = _float_from_row(row, ("yaw", "psi"))
    return roll, pitch, yaw, True


def _bayer_code(pixel_format: str) -> int:
    fmt = (pixel_format or "").lower()
    if "bayerrg" in fmt:
        return cv2.COLOR_BayerRG2RGB
    if "bayerbg" in fmt:
        return cv2.COLOR_BayerBG2RGB
    if "bayergr" in fmt:
        return cv2.COLOR_BayerGR2RGB
    if "bayergb" in fmt:
        return cv2.COLOR_BayerGB2RGB
    return cv2.COLOR_BayerBG2RGB


def _normalize_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - minimum) * (255.0 / (maximum - minimum))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def image_array_to_uint8_rgb(array: np.ndarray, pixel_format: str = "") -> np.ndarray:
    if array.ndim == 2:
        array = _normalize_uint8(array)
        return cv2.cvtColor(array, _bayer_code(pixel_format))
    if array.ndim == 3 and array.shape[2] >= 3:
        return _normalize_uint8(array[:, :, :3])
    raise ValueError(f"Unsupported image array shape: {array.shape}")


def image_corners(image_width: int, image_height: int) -> np.ndarray:
    right = max(image_width - 1, 0)
    bottom = max(image_height - 1, 0)
    return np.array(
        [[0.0, 0.0], [right, 0.0], [right, bottom], [0.0, bottom]],
        dtype=np.float32,
    )


def estimate_meters_per_pixel(
    source_corners_px: np.ndarray,
    flat_corners_m: np.ndarray,
) -> float:
    source_edges = (
        np.linalg.norm(source_corners_px[1] - source_corners_px[0]),
        np.linalg.norm(source_corners_px[2] - source_corners_px[3]),
        np.linalg.norm(source_corners_px[3] - source_corners_px[0]),
        np.linalg.norm(source_corners_px[2] - source_corners_px[1]),
    )
    flat_edges = (
        np.linalg.norm(flat_corners_m[1] - flat_corners_m[0]),
        np.linalg.norm(flat_corners_m[2] - flat_corners_m[3]),
        np.linalg.norm(flat_corners_m[3] - flat_corners_m[0]),
        np.linalg.norm(flat_corners_m[2] - flat_corners_m[1]),
    )
    candidates = [
        flat_edge / source_edge
        for flat_edge, source_edge in zip(flat_edges, source_edges)
        if flat_edge > 0 and source_edge > 0 and np.isfinite(flat_edge)
    ]
    if not candidates:
        raise ValueError("Could not estimate meters per pixel.")
    return float(np.median(candidates))


def homography_output_size_for(
    extent_m: np.ndarray,
    meters_per_pixel: float,
    *,
    padding_px: int = 0,
) -> tuple[int, int]:
    width = max(1, int(math.ceil(extent_m[0] / meters_per_pixel)) + 2 * padding_px)
    height = max(1, int(math.ceil(extent_m[1] / meters_per_pixel)) + 2 * padding_px)
    return width, height


def flat_world_corners_m(
    row: dict[str, Any],
    config: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    altitude_offset_m: float = 0.0,
) -> np.ndarray:
    altitude = _float_from_row(
        row,
        ("h", "alt_rel", "altitude", "altitude_rel", "relative_altitude", "alt"),
    ) + float(altitude_offset_m)
    if altitude <= 0:
        raise ValueError(f"altitude must be positive, got {altitude}")

    roll, pitch, yaw, is_deg = metadata_angles(
        row,
        config,
    )
    camera = build_camera(config, image_width, image_height)
    configuration = Camera.Configuration(
        Affine((0.0, 0.0, -altitude), ((), (), ())),
        roll,
        pitch,
        yaw,
        is_deg=is_deg,
    )

    src_corners = image_corners(image_width, image_height)
    corner_points = Affine(src_corners, np.empty((2, 0)))
    rays = camera.px2ned(corner_points, configuration)
    origins = np.asarray(rays.vector, dtype=np.float64)
    directions = np.asarray(
        rays.orthCoos2globalCoos_matrix[..., :, 0],
        dtype=np.float64,
    )
    if np.any(np.abs(directions[:, 2]) < 1e-9):
        raise ValueError("At least one image corner ray is parallel to the ground.")

    distance_along_ray = -origins[:, 2] / directions[:, 2]
    ground_ned = origins + directions * distance_along_ray[:, None]
    if not np.all(np.isfinite(ground_ned)):
        raise ValueError("Computed non-finite flat-world corners.")

    north = ground_ned[:, 0]
    east = ground_ned[:, 1]
    return np.column_stack([east, -north])


def compute_homography(
    row: dict[str, Any],
    config: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    output_size: tuple[int, int] | None = None,
    meters_per_pixel: float | None = None,
    padding_px: int = 0,
    max_output_side: int = 0,
    altitude_offset_m: float = 0.0,
    augment: Any = None,
    heading_conversion_mode: Any = None,
) -> HomographyComputation:
    _ = (augment, heading_conversion_mode)
    src_corners = image_corners(image_width, image_height)
    flat_corners = flat_world_corners_m(
        row,
        config,
        image_width,
        image_height,
        altitude_offset_m=altitude_offset_m,
    )
    min_xy = flat_corners.min(axis=0)
    max_xy = flat_corners.max(axis=0)
    extent_m = max_xy - min_xy
    if np.any(extent_m <= 1e-9):
        raise ValueError("Flat-world footprint has zero area.")

    padding_px = max(0, int(padding_px))
    if output_size is not None:
        width, height = output_size
        available_width = max(1, width - 2 * padding_px)
        available_height = max(1, height - 2 * padding_px)
        pixels_per_meter = min(
            available_width / extent_m[0],
            available_height / extent_m[1],
        )
        if not np.isfinite(pixels_per_meter) or pixels_per_meter <= 0:
            raise ValueError("Invalid homography scale from configured output size.")
        meters_per_pixel = float(1.0 / pixels_per_meter)
        used_size = extent_m * pixels_per_meter
        offset = np.array(
            [
                (width - used_size[0]) / 2.0,
                (height - used_size[1]) / 2.0,
            ],
            dtype=np.float64,
        )
        destination = (flat_corners - min_xy) * pixels_per_meter + offset
    else:
        if meters_per_pixel is None:
            meters_per_pixel = estimate_meters_per_pixel(src_corners, flat_corners)
        if not np.isfinite(meters_per_pixel) or meters_per_pixel <= 0:
            raise ValueError(f"Invalid meters_per_pixel: {meters_per_pixel}")

        width, height = homography_output_size_for(
            extent_m,
            meters_per_pixel,
            padding_px=padding_px,
        )
        if max_output_side > 0 and max(width, height) > max_output_side:
            available_side = max(1, int(max_output_side) - 2 * padding_px)
            meters_per_pixel = float(max(extent_m) / available_side)
            width, height = homography_output_size_for(
                extent_m,
                meters_per_pixel,
                padding_px=padding_px,
            )
        output_size = (width, height)
        destination = (flat_corners - min_xy) / meters_per_pixel
        destination += padding_px

    matrix = cv2.getPerspectiveTransform(
        src_corners.astype(np.float32),
        destination.astype(np.float32),
    )
    return HomographyComputation(
        matrix=matrix,
        inverse_matrix=np.linalg.inv(matrix),
        output_size=output_size,
        meters_per_pixel=float(meters_per_pixel),
    )


def warp_image_to_north_up(
    image: np.ndarray,
    row: dict[str, Any],
    config: dict[str, Any],
    **homography_kwargs: Any,
) -> tuple[np.ndarray, HomographyComputation]:
    image_width, image_height = _numpy_shape_to_width_height(image.shape)
    computation = compute_homography(row, config, image_width, image_height, **homography_kwargs)
    warped = cv2.warpPerspective(
        image,
        computation.matrix.astype(np.float64),
        computation.output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, computation


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix.astype(np.float64)).reshape(-1, 2)


def inverse_transform_points(
    warped_points_xy: np.ndarray,
    computation: HomographyComputation,
) -> np.ndarray:
    return transform_points(warped_points_xy, computation.inverse_matrix)


def bbox_xyxy_to_corners(bbox_xyxy: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def corners_to_bbox_xyxy(corners_xy: np.ndarray) -> list[float]:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(-1, 2)
    return [
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    ]


def clip_bbox_to_shape(bbox: Iterable[float], image_shape: tuple[int, ...]) -> list[float]:
    image_height, image_width = _numpy_shape_to_height_width(image_shape)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [
        min(max(x1, 0.0), float(image_width)),
        min(max(y1, 0.0), float(image_height)),
        min(max(x2, 0.0), float(image_width)),
        min(max(y2, 0.0), float(image_height)),
    ]


def yolo_bbox_to_xyxy(
    bbox_yolo: tuple[int, float, float, float, float],
    image_shape: tuple[int, ...],
) -> list[float]:
    _, x_center, y_center, width_norm, height_norm = bbox_yolo
    image_height, image_width = _numpy_shape_to_height_width(image_shape)

    width = float(width_norm) * image_width
    height = float(height_norm) * image_height
    center_x = float(x_center) * image_width
    center_y = float(y_center) * image_height

    return [
        max(0.0, center_x - width / 2.0),
        max(0.0, center_y - height / 2.0),
        min(float(image_width), center_x + width / 2.0),
        min(float(image_height), center_y + height / 2.0),
    ]


def parse_output_size(value: Any) -> tuple[int, int] | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower().replace(",", "x")
        parts = [part.strip() for part in cleaned.split("x") if part.strip()]
        if len(parts) != 2:
            return None
        try:
            width = int(float(parts[0]))
            height = int(float(parts[1]))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            if len(value) >= 3:
                width, height = _numpy_shape_to_width_height(value)
            else:
                width = int(float(value[0]))
                height = int(float(value[1]))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
