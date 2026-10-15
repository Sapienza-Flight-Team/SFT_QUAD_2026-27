import math
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif
from stuff_i_moved_away.inference.find_bullseye import (
    find_bullseye_in_image,
    load_config as load_bullseye_config,
)
from stages.inference.default.utils import bbox_utils
from sft_math.cam import Camera
from sft_math.cam.affine import Affine
from sft_math.cam.px2body_matrix import resolve_px2body_matrix


@dataclass
class HomographyComputation:
    matrix: np.ndarray
    inverse_matrix: np.ndarray
    output_size: tuple[int, int]
    meters_per_pixel: float | None
    source_corners_px: np.ndarray
    destination_corners_px: np.ndarray
    flat_world_corners_m: np.ndarray


def _first_present(row: Mapping[str, Any], names: Iterable[str]) -> Any | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _float_from_row(
    row: Mapping[str, Any],
    names: Iterable[str],
    *,
    default: float | None = None,
) -> float:
    value = _first_present(row, names)
    if value is None:
        if default is None:
            raise ValueError(f"Missing metadata value: one of {', '.join(names)}")
        return default
    return float(value)


def _numpy_shape_to_height_width(shape: Iterable[int]) -> tuple[int, int]:
    dims = tuple(int(v) for v in shape)
    if len(dims) < 2:
        raise ValueError(f"image shape must have at least 2 dimensions, got {shape}")
    return dims[0], dims[1]


def _numpy_shape_to_width_height(shape: Iterable[int]) -> tuple[int, int]:
    height, width = _numpy_shape_to_height_width(shape)
    # Numpy/OpenCV arrays are HxW[xC], while Camera/OpenCV dsize use WxH.
    return width, height


def metadata_to_homography_row(metadata: PhotoExif | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(metadata, Mapping):
        return dict(metadata)

    row: dict[str, Any] = {
        "filename": getattr(metadata, "filename", ""),
        "pixel_format": getattr(metadata, "pixel_format", ""),
    }
    pix_data = getattr(metadata, "pix_data", None)
    if pix_data is not None:
        for name in (
            "lat",
            "lon",
            "alt_rel",
            "alt_msl",
            "roll",
            "pitch",
            "yaw",
            "vx",
            "vy",
            "vz",
            "vground",
        ):
            if hasattr(pix_data, name):
                row[name] = getattr(pix_data, name)
    return row


def _configured_image_size(config: Mapping[str, Any]) -> tuple[float, float] | None:
    geoloc_config = config.get("decision_system", {}).get("geoloc", {})
    camera_config = config.get("camera", {})
    configured = (
        geoloc_config.get("image_shape")
        or camera_config.get("resolution")
        or camera_config.get("photo_shape")
    )
    if configured is None or len(configured) < 2:
        return None
    if isinstance(configured, Mapping):
        width = configured.get("width")
        height = configured.get("height")
        if width is None or height is None:
            return None
        return float(width), float(height)
    if len(configured) >= 3:
        width, height = _numpy_shape_to_width_height(configured)
        return float(width), float(height)
    # Project config image sizes are stored as [width, height].
    return float(configured[0]), float(configured[1])


def _configured_optical_center(
    config: Mapping[str, Any],
    image_width: int,
    image_height: int,
    original_size: tuple[float, float] | None,
) -> tuple[float, float]:
    geoloc_config = config.get("decision_system", {}).get("geoloc", {})
    camera_config = config.get("camera", {})
    optical_center = (
        geoloc_config.get("optical_center")
        or geoloc_config.get("optic_center")
        or camera_config.get("optical_center")
        or camera_config.get("optic_center")
    )
    if optical_center is None:
        return image_width / 2.0, image_height / 2.0

    center_x = float(optical_center[0])
    center_y = float(optical_center[1])
    if original_size is not None:
        original_width, original_height = original_size
        if original_width > 0 and original_height > 0:
            center_x *= image_width / original_width
            center_y *= image_height / original_height
    return center_x, center_y


def build_camera(config: Mapping[str, Any], image_width: int, image_height: int) -> Camera:
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

    original_size = _configured_image_size(config)
    if original_size is not None and image_width > 0 and image_height > 0:
        original_width, original_height = original_size
        scale_x = original_width / image_width
        scale_y = original_height / image_height
        pixel_size *= float((scale_x + scale_y) / 2.0)

    return Camera(
        focal_length,
        pixel_size,
        (image_width, image_height),
        _configured_optical_center(config, image_width, image_height, original_size),
        resolve_px2body_matrix(geoloc_config, camera_config),
    )


def metadata_angles(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[float, float, float, bool]:
    _ = config
    roll = _float_from_row(row, ("roll", "phi"), default=0.0)
    pitch = _float_from_row(row, ("pitch", "theta"), default=0.0)
    yaw = _float_from_row(row, ("yaw", "psi"))
    return roll, pitch, yaw, True


def image_corners(image_width: int, image_height: int) -> np.ndarray:
    right = max(image_width - 1, 0)
    bottom = max(image_height - 1, 0)
    return np.array(
        [[0.0, 0.0], [right, 0.0], [right, bottom], [0.0, bottom]],
        dtype=np.float32,
    )


def flat_world_corners_m(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
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

    source_corners = image_corners(image_width, image_height)
    rays = camera.px2ned(Affine(source_corners, np.empty((2, 0))), configuration)
    origins = np.asarray(rays.vector, dtype=np.float64)
    directions = np.asarray(
        rays.orthCoos2globalCoos_matrix[..., :, 0],
        dtype=np.float64,
    )
    if np.any(np.abs(directions[:, 2]) < 1e-9):
        raise ValueError("at least one image corner ray is parallel to the ground")

    distance_along_ray = -origins[:, 2] / directions[:, 2]
    ground_ned = origins + directions * distance_along_ray[:, None]
    if not np.all(np.isfinite(ground_ned)):
        raise ValueError("computed non-finite flat-world corners")

    north = ground_ned[:, 0]
    east = ground_ned[:, 1]
    return np.column_stack([east, -north])


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
        raise ValueError("could not estimate meters per pixel")
    return float(np.median(candidates))


def homography_output_size_for(
    extent_m: np.ndarray,
    meters_per_pixel: float,
    *,
    padding_px: int = 0,
) -> tuple[int, int]:
    return (
        max(1, int(math.ceil(extent_m[0] / meters_per_pixel)) + 2 * padding_px),
        max(1, int(math.ceil(extent_m[1] / meters_per_pixel)) + 2 * padding_px),
    )


def compute_homography(
    row: Mapping[str, Any] | PhotoExif,
    config: Mapping[str, Any],
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
    metadata_row = metadata_to_homography_row(row)
    source_corners = image_corners(image_width, image_height)
    flat_corners = flat_world_corners_m(
        metadata_row,
        config,
        image_width,
        image_height,
        altitude_offset_m=altitude_offset_m,
    )
    min_xy = flat_corners.min(axis=0)
    max_xy = flat_corners.max(axis=0)
    extent_m = max_xy - min_xy
    if np.any(extent_m <= 1e-9):
        raise ValueError("flat-world footprint has zero area")

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
            raise ValueError("invalid homography scale from configured output size")
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
            meters_per_pixel = estimate_meters_per_pixel(source_corners, flat_corners)
        if not np.isfinite(meters_per_pixel) or meters_per_pixel <= 0:
            raise ValueError(f"invalid meters_per_pixel: {meters_per_pixel}")

        output_size = homography_output_size_for(
            extent_m,
            meters_per_pixel,
            padding_px=padding_px,
        )
        if max_output_side > 0 and max(output_size) > max_output_side:
            available_side = max(1, int(max_output_side) - 2 * padding_px)
            meters_per_pixel = float(max(extent_m) / available_side)
            output_size = homography_output_size_for(
                extent_m,
                meters_per_pixel,
                padding_px=padding_px,
            )
        destination = (flat_corners - min_xy) / meters_per_pixel
        destination += padding_px

    matrix = cv2.getPerspectiveTransform(
        source_corners.astype(np.float32),
        destination.astype(np.float32),
    )
    return HomographyComputation(
        matrix=matrix,
        inverse_matrix=np.linalg.inv(matrix),
        output_size=output_size,
        meters_per_pixel=float(meters_per_pixel),
        source_corners_px=source_corners,
        destination_corners_px=destination.astype(np.float64),
        flat_world_corners_m=flat_corners,
    )


def warp_image_to_north_up(
    image: np.ndarray,
    row: Mapping[str, Any] | PhotoExif,
    config: Mapping[str, Any],
    **homography_kwargs: Any,
) -> tuple[np.ndarray, HomographyComputation]:
    image_width, image_height = _numpy_shape_to_width_height(image.shape)
    computation = compute_homography(
        row,
        config,
        image_width,
        image_height,
        **homography_kwargs,
    )
    warped = cv2.warpPerspective(
        image,
        computation.matrix.astype(np.float64),
        computation.output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, computation


def transform_points(
    points_xy: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
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


def _clip_bbox_to_shape(bbox: Iterable[float], image_shape: tuple[int, ...]) -> list[float]:
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


def _bayer_code(pixel_format: str) -> int:
    fmt = (pixel_format or "").lower()
    if "bayerrg" in fmt:
        return cv2.COLOR_BayerRG2RGB
    if "bayergr" in fmt:
        return cv2.COLOR_BayerGR2RGB
    if "bayergb" in fmt:
        return cv2.COLOR_BayerGB2RGB
    return cv2.COLOR_BayerBG2RGB


def ensure_rgb_image(image: np.ndarray, pixel_format: str = "") -> np.ndarray:
    if image.ndim == 3:
        return image
    if image.ndim != 2:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return cv2.cvtColor(image, _bayer_code(pixel_format))


def detect_bullseye_homography_cv(
    image: np.ndarray,
    metadata: PhotoExif | Mapping[str, Any],
    config: Mapping[str, Any],
    bullseye_config: Mapping[str, Any],
    *,
    output_dir: str | None = None,
    debug: bool = False,
    output_size: tuple[int, int] | None = None,
    meters_per_pixel: float | None = None,
    padding_px: int = 0,
    max_output_side: int = 0,
    altitude_offset_m: float = 0.0,
    heading_conversion_mode: Any = None,
) -> tuple[list[dict[str, Any]], np.ndarray, HomographyComputation]:
    _ = heading_conversion_mode
    metadata_row = metadata_to_homography_row(metadata)
    image_rgb = ensure_rgb_image(image, str(metadata_row.get("pixel_format", "")))
    warped_rgb, computation = warp_image_to_north_up(
        image_rgb,
        metadata_row,
        config,
        output_size=output_size,
        meters_per_pixel=meters_per_pixel,
        padding_px=padding_px,
        max_output_side=max_output_side,
        altitude_offset_m=altitude_offset_m,
    )

    result = find_bullseye_in_image(
        image=warped_rgb,
        config=dict(bullseye_config),
        output_dir=output_dir,
        debug=debug,
    )
    if result is None:
        return [], warped_rgb, computation

    bbox_yolo = result.get("bbox_yolo")
    if bbox_yolo is None or len(bbox_yolo) != 5:
        raise ValueError("Bullseye detector returned an invalid YOLO bbox")

    warped_bbox = yolo_bbox_to_xyxy(tuple(bbox_yolo), warped_rgb.shape)
    warped_corners = bbox_xyxy_to_corners(warped_bbox)
    source_corners = inverse_transform_points(warped_corners, computation)
    source_bbox = _clip_bbox_to_shape(corners_to_bbox_xyxy(source_corners), image_rgb.shape)

    warped_center = np.asarray(
        result.get(
            "center_xy_px",
            [
                (warped_bbox[0] + warped_bbox[2]) / 2.0,
                (warped_bbox[1] + warped_bbox[3]) / 2.0,
            ],
        ),
        dtype=np.float32,
    ).reshape(1, 2)
    source_center = inverse_transform_points(warped_center, computation).reshape(2)

    return [
        {
            "class": int(bbox_yolo[0]),
            "conf": float(result.get("score", 1.0)),
            "bbox": source_bbox,
            "center_xy_px": [float(source_center[0]), float(source_center[1])],
            "polygon_px": source_corners.astype(float).tolist(),
            "warped_bbox": [float(v) for v in warped_bbox],
            "warped_center_xy_px": [float(v) for v in warped_center.reshape(2)],
            "homography_output_size": list(computation.output_size),
            "homography_meters_per_pixel": computation.meters_per_pixel,
        }
    ], warped_rgb, computation


def scale_homography_detections_to_full_image(
    detections: list[dict[str, Any]],
    inference_shape: tuple[int, ...] | None,
    full_image_shape: tuple[int, ...] | None,
) -> list[dict[str, Any]]:
    if inference_shape is None or full_image_shape is None:
        return detections
    inference_height, inference_width = _numpy_shape_to_height_width(inference_shape)
    full_height, full_width = _numpy_shape_to_height_width(full_image_shape)
    inference_height = float(inference_height)
    inference_width = float(inference_width)
    full_height = float(full_height)
    full_width = float(full_width)
    if inference_height <= 0 or inference_width <= 0:
        return detections

    scale_x = full_width / inference_width
    scale_y = full_height / inference_height
    scaled: list[dict[str, Any]] = []
    for detection in detections:
        item = dict(detection)
        if "bbox" in item:
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [
                float(x1) * scale_x,
                float(y1) * scale_y,
                float(x2) * scale_x,
                float(y2) * scale_y,
            ]
        if "center_xy_px" in item:
            cx, cy = item["center_xy_px"]
            item["center_xy_px"] = [float(cx) * scale_x, float(cy) * scale_y]
        if "polygon_px" in item:
            item["polygon_px"] = [
                [float(x) * scale_x, float(y) * scale_y]
                for x, y in item["polygon_px"]
            ]
        scaled.append(item)
    return scaled


class BullseyeHomoCVStage(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        ack_q=None,
        stage_name="bullseye_homo_cv",
        idx=3,
    ):
        super().__init__(
            configs=configs,
            logger=logger,
            in_q=in_q,
            out_q=out_q,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            ack_q=ack_q,
            stage_name=stage_name,
            idx=idx,
        )
        self.stopped.clear()

        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        homography_cfg = self.configs.config_dict.get("homography_stage", {})
        self.bullseye_config_path = inference_cfg.get("bullseye_config")
        self.bullseye_output_dir = inference_cfg.get("bullseye_output_dir")
        self.bullseye_debug = bool(inference_cfg.get("bullseye_debug", False))
        self.bullseye_config = self._load_bullseye_config()
        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution",
            [2656, 1504],
        )

        self.homography_output_size = _parse_output_size(homography_cfg.get("output_size"))
        meters_per_pixel = homography_cfg.get("meters_per_pixel")
        self.homography_meters_per_pixel = (
            float(meters_per_pixel) if meters_per_pixel not in (None, "") else None
        )
        self.homography_padding_px = max(0, int(homography_cfg.get("padding_px", 0)))
        self.homography_max_output_side = max(
            0,
            int(homography_cfg.get("max_output_side", 0)),
        )
        self.altitude_offset_m = float(homography_cfg.get("altitude_offset_m", 0.0))

    def _load_bullseye_config(self) -> dict[str, Any]:
        return load_bullseye_config(self.bullseye_config_path)

    def _bullseye_enabled(self) -> bool:
        return self.current_task == "BULLSEYE"

    def _process_metadata(self, metadata: PhotoExif) -> None:
        t_start = time.perf_counter()
        with self.consume_shm_array(metadata) as in_array:
            image = np.asarray(in_array)
            image_shape = image.shape
            detections, _, computation = detect_bullseye_homography_cv(
                image,
                metadata,
                self.configs.config_dict,
                self.bullseye_config,
                output_dir=getattr(self, "bullseye_output_dir", None),
                debug=bool(getattr(self, "bullseye_debug", False)),
                output_size=self.homography_output_size,
                meters_per_pixel=self.homography_meters_per_pixel,
                padding_px=self.homography_padding_px,
                max_output_side=self.homography_max_output_side,
                altitude_offset_m=self.altitude_offset_m,
            )

        full_image_shape = getattr(metadata, "full_image_shape", None) or image_shape
        full_image_detections = scale_homography_detections_to_full_image(
            detections,
            image_shape,
            full_image_shape,
        )
        metadata.bboxes = full_image_detections
        metadata.homography_matrix = computation.matrix.tolist()
        metadata.homography_inverse_matrix = computation.inverse_matrix.tolist()
        metadata.homography_output_size = computation.output_size
        metadata.shm_name = None

        bbox_utils.save_full_image_bboxes(
            metadata,
            full_image_detections,
            image_shape,
            full_image_shape,
            self.target_resolution,
            producer=self._stage_name,
            mission=self.current_task,
            bbox_logging_config=self.configs.config_dict.get("bbox_logging", {}),
            logger=self.logger,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.logger.info(
            f"BullseyeHomoCV processed {metadata.filename}: "
            f"detections={len(full_image_detections)}, elapsed_ms={elapsed_ms:.1f}, "
            f"homography_output={computation.output_size}"
        )

    def mission(self) -> bool:
        while not self.stopped.is_set():
            self.check_inbound_command()
            try:
                metadata: Any = self.receive_from_queue(timeout=1)
            except Empty:
                time.sleep(0.01)
                continue
            except Exception as exc:
                self.logger.error(f"BullseyeHomoCVStage queue error: {exc}")
                time.sleep(0.01)
                continue

            if metadata is None:
                time.sleep(0.01)
                continue
            if not isinstance(metadata, PhotoExif):
                self.send_to_queues(metadata)
                continue
            if not self._bullseye_enabled():
                self.fast_ack(metadata)
                continue

            try:
                self._process_metadata(metadata)
                self.send_to_queues(metadata)
            except FileNotFoundError:
                self.logger.warning(
                    f"BullseyeHomoCVStage: shared memory {metadata.shm_name} not found"
                )
                self.fast_ack(metadata)
            except Exception as exc:
                self.logger.error(f"BullseyeHomoCVStage error on {metadata.filename}: {exc}")
                self.fast_ack(metadata)

        return True

    def update(self) -> None:
        super().update()
        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        homography_cfg = self.configs.config_dict.get("homography_stage", {})
        self.bullseye_config_path = inference_cfg.get(
            "bullseye_config",
            self.bullseye_config_path,
        )
        self.bullseye_output_dir = inference_cfg.get(
            "bullseye_output_dir",
            self.bullseye_output_dir,
        )
        self.bullseye_debug = bool(
            inference_cfg.get("bullseye_debug", self.bullseye_debug)
        )
        self.bullseye_config = self._load_bullseye_config()
        self.homography_output_size = _parse_output_size(
            homography_cfg.get("output_size")
        )
        meters_per_pixel = homography_cfg.get("meters_per_pixel")
        self.homography_meters_per_pixel = (
            float(meters_per_pixel) if meters_per_pixel not in (None, "") else None
        )
        self.homography_padding_px = max(0, int(homography_cfg.get("padding_px", 0)))
        self.homography_max_output_side = max(
            0,
            int(homography_cfg.get("max_output_side", 0)),
        )
        self.altitude_offset_m = float(
            homography_cfg.get("altitude_offset_m", self.altitude_offset_m)
        )

    def stop(self) -> None:
        self.logger.info("Stopping BullseyeHomoCVStage")
        self.stopped.set()

    def get(self, data: Any) -> None:
        raise NotImplementedError()

    def put(self, data: Any) -> None:
        raise NotImplementedError()

    def _dump_data(self) -> None:
        pass

    def ff15(self) -> None:
        pass

    def glhf(self) -> None:
        pass

    def bullseye_stop(self) -> None:
        pass

    def perimeter_stop(self) -> None:
        pass

    def numbers_stop(self) -> None:
        pass

    def error_handler(self) -> None:
        pass


def _parse_output_size(value: Any) -> tuple[int, int] | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower().replace(",", "x")
        parts = [part.strip() for part in cleaned.split("x") if part.strip()]
        if len(parts) != 2:
            return None
        return max(1, int(float(parts[0]))), max(1, int(float(parts[1])))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if len(value) >= 3:
            width, height = _numpy_shape_to_width_height(value)
            return max(1, width), max(1, height)
        return max(1, int(float(value[0]))), max(1, int(float(value[1])))
    return None
