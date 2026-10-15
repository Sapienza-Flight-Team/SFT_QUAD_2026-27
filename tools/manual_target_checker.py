import argparse
import csv
import json
import datetime as dt
import logging
import math
import os
import sqlite3
import statistics as st
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "outs"
for import_path in (REPO_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "manual_target_checker_mpl"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import matplotlib

    matplotlib.use("TkAgg")  # use a non-blocking backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    plt = None
    Poly3DCollection = None

try:
    import cv2
    from common.homography import (
        compute_homography as _compute_homography_fn,
    )
except ImportError:
    cv2 = None

try:
    import pygame
except ImportError:
    pygame = None

from common.config import SFTConfig
from stages.decision_system.default.decision import Decision
from sft_math import GeoLoc
from sft_math.cam import Camera
from sft_math.cam.affine import Affine, Point
from sft_math.cam.px2body_matrix import resolve_px2body_matrix
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS

VALID_INPUTS = set("0123456789b")
ZOOM_MIN = 1.0
ZOOM_MAX = 12.0
ZOOM_STEP = 1.2
MANUAL_COLUMNS = [
    "manual_point_index",
    "manual_input",
    "manual_target_x_px",
    "manual_target_y_px",
    "manual_target_lat",
    "manual_target_lon",
    "manual_geoloc_error",
    "manual_saved",
    "manual_saved_at",
]
# The manual checker labels numbered GCP targets with their painted digit
# (classes 0-9). The bullseye has no digit, so it gets a dedicated class id
# placed just past the digit range to avoid colliding with number_2 etc.
BULLSEYE_CLASS_ID = 10
DEFAULT_MANUAL_LABEL_CLASS_MAP = {"bullseye": BULLSEYE_CLASS_ID, "b": BULLSEYE_CLASS_ID}
DEFAULT_MANUAL_DIGIT_TARGET_NAMES = {i: f"number_{i}" for i in range(10)}
DEFAULT_MANUAL_NON_DIGIT_TARGET_NAMES = {BULLSEYE_CLASS_ID: "BULLSEYE"}
DEFAULT_IMAGE_W = 5320
DEFAULT_IMAGE_H = 3032
DEG2M_LAT = 111_320.0

BBOX_DB_FILENAME = "bboxes.sqlite3"
PRODUCER_FILTER_MODES = (
    "all",
    "inference_engine",
    "bullseye_cv_engine",
)
PRODUCER_COLORS: dict[str, tuple[int, int, int]] = {
    "inference_engine": (0, 230, 118),
    "bullseye_cv_engine": (135, 206, 235),
}
PRODUCER_LABEL_BG: dict[str, tuple[int, int, int]] = {
    "inference_engine": (0, 80, 40),
    "bullseye_cv_engine": (30, 70, 110),
}
DEFAULT_PRODUCER_COLOR = (0, 230, 118)
DEFAULT_PRODUCER_LABEL_BG = (0, 80, 40)

FIG_BG = "#12161c"
AXES_BG = "#1a2030"
FG = "#c9d1d9"
GRID = "#2a3344"
TRAJECTORY_LINE = "#8b949e"
CURRENT_POS_COLOR = "#ff4444"
FOV_LINE_COLOR = "#00d4ff"
FOV_FILL_COLOR = (0.0, 0.83, 1.0, 0.10)
DETECTION_COLOR = "#00e676"
DETECTION_CUR_COLOR = "#ff9f0a"
MANUAL_POINT_COLOR = "#ff4c4c"


@dataclass
class Annotation:
    label: str = ""
    x_px: float | None = None
    y_px: float | None = None
    lat: float | None = None
    lon: float | None = None
    geoloc_error: str = ""
    saved: bool = False
    saved_at: str = ""
    is_auto_center: bool = False


@dataclass
class LoadedImage:
    path: Path
    surface: Any
    width: int
    height: int
    rgb: np.ndarray | None = None


@dataclass
class GeolocatorContext:
    geoloc: GeoLoc

    def to_gps_arrays(
        self, points_px: Iterable[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(list(points_px), dtype=np.float64)
        if points.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        points = points.reshape(-1, 2)
        point: Point = Affine(
            points,
            np.zeros((points.shape[0], 2, 0), dtype=np.float64),
        )
        gps_points = self.geoloc.to_gps(point)
        lats = np.rad2deg(gps_points.gps.lat).reshape(-1).astype(np.float64)
        lons = np.rad2deg(gps_points.gps.lon).reshape(-1).astype(np.float64)
        return lats, lons


@dataclass
class Button:
    name: str
    rect: Any
    label: str


@dataclass
class FlightDroneState:
    image_index: int
    metadata_index: int
    filename: str
    lat: float
    lon: float
    alt_rel: float
    roll: float
    pitch: float
    yaw: float
    timestamp: str | None
    row: dict[str, str]
    east_m: float = 0.0
    north_m: float = 0.0


def _first_present(row: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _float_from_row(
    row: dict[str, str],
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


def _resolve_image_path(filename: str, metadata_path: Path) -> Path:
    image_path = Path(filename).expanduser()
    candidates = []
    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        candidates.extend(
            [
                Path.cwd() / image_path,
                metadata_path.parent / image_path,
                REPO_ROOT / image_path,
            ]
        )
    # Fallback: look for the bare filename in the metadata's own directory.
    # This handles renamed subfolders — e.g. the CSV was written with path
    # "./data/outs/20260526-133058/img.npy" but the folder was later renamed
    # to "90m-12ms".  As long as the files themselves weren't moved, just
    # searching by basename in the metadata directory resolves them.
    candidates.append(metadata_path.parent / image_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else image_path


def _open_bbox_db(metadata_path: Path) -> sqlite3.Connection | None:
    db_path = metadata_path.parent / BBOX_DB_FILENAME
    if not db_path.exists():
        return None
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _iter_bbox_rows(
    conn: sqlite3.Connection,
    image_filename: str,
    producer_filter: str,
) -> list[sqlite3.Row]:
    base_sql = (
        "SELECT image_filename, producer, inference_image_shape_json, "
        "full_image_shape_json, boxes_json FROM bbox_records "
        "WHERE image_filename = ?"
    )
    params: list[Any] = [image_filename]
    if producer_filter != "all":
        base_sql += " AND producer = ?"
        params.append(producer_filter)
    rows = conn.execute(base_sql, params).fetchall()
    if rows:
        return rows

    basename = Path(image_filename).name
    fallback_sql = (
        "SELECT image_filename, producer, inference_image_shape_json, "
        "full_image_shape_json, boxes_json FROM bbox_records "
        "WHERE image_filename LIKE ?"
    )
    fallback_params: list[Any] = [f"%/{basename}"]
    if producer_filter != "all":
        fallback_sql += " AND producer = ?"
        fallback_params.append(producer_filter)
    return conn.execute(fallback_sql, fallback_params).fetchall()


def _decode_box_row(row: sqlite3.Row) -> dict[str, Any]:
    boxes_raw = row["boxes_json"] or "[]"
    try:
        boxes = json.loads(boxes_raw)
        if not isinstance(boxes, list):
            boxes = []
    except json.JSONDecodeError:
        boxes = []

    def _decode_shape(value: str | None) -> dict[str, int] | None:
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    return {
        "producer": row["producer"],
        "image_filename": row["image_filename"],
        "boxes": boxes,
        "inference_shape": _decode_shape(row["inference_image_shape_json"]),
        "full_shape": _decode_shape(row["full_image_shape_json"]),
    }


def load_metadata(metadata_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{metadata_path} does not contain a CSV header")
        if "filename" not in reader.fieldnames:
            raise ValueError(f"{metadata_path} must contain a filename column")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{metadata_path} contains no image rows")
    return rows, list(reader.fieldnames)


def _output_path_for(metadata_path: Path) -> Path:
    return metadata_path.with_name(f"{metadata_path.stem}_manual.csv")


def _default_metadata_path() -> Path:
    if not DEFAULT_METADATA_DIR.exists():
        raise ValueError(f"Metadata directory does not exist: {DEFAULT_METADATA_DIR}")

    candidates = []
    for item in sorted(DEFAULT_METADATA_DIR.iterdir()):
        if item.is_dir():
            csvs = sorted(
                p for p in item.glob("metadata*.csv") if not p.stem.endswith("_manual")
            )
            if csvs:
                candidates.append((item, csvs))

    if not candidates:
        root_csvs = sorted(
            p
            for p in DEFAULT_METADATA_DIR.glob("metadata*.csv")
            if not p.stem.endswith("_manual")
        )
        if root_csvs:
            return root_csvs[0].resolve()
        raise ValueError(
            f"No metadata*.csv files found in subfolders or directly in {DEFAULT_METADATA_DIR}"
        )

    print("Found the following subfolders with metadata in data/outs:")
    for idx, (subfolder, csvs) in enumerate(candidates, start=1):
        csv_names = ", ".join(c.name for c in csvs)
        print(f"  {idx}) {subfolder.name} (contains: {csv_names})")

    while True:
        try:
            choice = input(
                f"Select a subfolder [1-{len(candidates)}] (default 1): "
            ).strip()
            if not choice:
                selected_idx = 0
                break
            val = int(choice)
            if 1 <= val <= len(candidates):
                selected_idx = val - 1
                break
            print(f"Please enter a number between 1 and {len(candidates)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)

    selected_subfolder, selected_csvs = candidates[selected_idx]
    return selected_csvs[0].resolve()


def _discover_all_metadata_paths() -> list[Path]:
    if not DEFAULT_METADATA_DIR.exists():
        raise ValueError(f"Metadata directory does not exist: {DEFAULT_METADATA_DIR}")
    metadata_paths = sorted(
        p.resolve()
        for p in DEFAULT_METADATA_DIR.rglob("metadata*.csv")
        if p.is_file() and not p.stem.endswith("_manual")
    )
    if not metadata_paths:
        raise ValueError(f"No metadata*.csv files found under {DEFAULT_METADATA_DIR}")
    return metadata_paths


def _load_existing_annotations(
    output_path: Path, rows: list[dict[str, str]]
) -> list[list[Annotation]]:
    annotations = [[Annotation()] for _ in rows]
    if not output_path.exists():
        return annotations

    with output_path.open("r", newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))

    existing_by_filename: dict[str, list[dict[str, str]]] = {}
    for existing_row in existing_rows:
        filename = existing_row.get("filename", "")
        if filename:
            existing_by_filename.setdefault(filename, []).append(existing_row)

    for idx, row in enumerate(rows):
        existing_points = existing_by_filename.get(row.get("filename", ""), [])
        loaded_points = []
        for existing in existing_points:
            if not _has_manual_annotation(existing):
                continue
            loaded_points.append(
                Annotation(
                    label=existing.get("manual_input", ""),
                    x_px=_optional_float(existing.get("manual_target_x_px")),
                    y_px=_optional_float(existing.get("manual_target_y_px")),
                    lat=_optional_float(existing.get("manual_target_lat")),
                    lon=_optional_float(existing.get("manual_target_lon")),
                    geoloc_error=existing.get("manual_geoloc_error", ""),
                    saved=existing.get("manual_saved", "").lower() == "true",
                    saved_at=existing.get("manual_saved_at", ""),
                )
            )
        if not loaded_points:
            continue
        annotations[idx] = loaded_points
    return annotations


def _has_manual_annotation(row: dict[str, str]) -> bool:
    if row.get("manual_saved", "").lower() == "true":
        return True
    return any(
        row.get(column, "")
        for column in (
            "manual_input",
            "manual_target_x_px",
            "manual_target_y_px",
            "manual_target_lat",
            "manual_target_lon",
        )
    )


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _format_optional(value: float | None, precision: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def _parse_class_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _manual_label_to_class_id(label: str, label_map: dict[str, int]) -> int | None:
    normalized = (label or "").strip().lower()
    if not normalized:
        return None
    if normalized.isdigit():
        return int(normalized)
    if "_" in normalized:
        class_prefix = normalized.split("_", 1)[0]
        if class_prefix.isdigit():
            return int(class_prefix)
    return label_map.get(normalized)


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annotation_is_empty(annotation: Annotation) -> bool:
    return (
        not annotation.saved
        and not annotation.label
        and annotation.x_px is None
        and annotation.y_px is None
        and annotation.lat is None
        and annotation.lon is None
    )


def write_output(
    output_path: Path,
    rows: list[dict[str, str]],
    input_fieldnames: list[str],
    annotations: list[list[Annotation]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = input_fieldnames + [
        column for column in MANUAL_COLUMNS if column not in input_fieldnames
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, row_annotations in zip(rows, annotations):
            wrote_saved_point = False
            for point_index, annotation in enumerate(row_annotations, start=1):
                if not annotation.saved:
                    continue
                writer.writerow(_manual_output_row(row, annotation, point_index))
                wrote_saved_point = True
            if not wrote_saved_point:
                writer.writerow(_manual_output_row(row, None, None))


def _manual_output_row(
    row: dict[str, str], annotation: Annotation | None, point_index: int | None
) -> dict[str, str]:
    out_row = dict(row)
    if annotation is None:
        out_row.update(
            {
                "manual_point_index": "",
                "manual_input": "",
                "manual_target_x_px": "",
                "manual_target_y_px": "",
                "manual_target_lat": "",
                "manual_target_lon": "",
                "manual_geoloc_error": "",
                "manual_saved": "",
                "manual_saved_at": "",
            }
        )
        return out_row

    out_row.update(
        {
            "manual_point_index": str(point_index or ""),
            "manual_input": annotation.label,
            "manual_target_x_px": _format_optional(annotation.x_px, 2),
            "manual_target_y_px": _format_optional(annotation.y_px, 2),
            "manual_target_lat": _format_optional(annotation.lat, 10),
            "manual_target_lon": _format_optional(annotation.lon, 10),
            "manual_geoloc_error": annotation.geoloc_error,
            "manual_saved": "true",
            "manual_saved_at": annotation.saved_at,
        }
    )
    return out_row


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return SFTConfig(str(config_path), setup_logger=False).config_dict


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


def _metadata_angles(
    row: dict[str, str],
    config: dict[str, Any],
) -> tuple[float, float, float, bool]:
    _ = config
    roll = _float_from_row(row, ("roll", "phi"), default=0.0)
    pitch = _float_from_row(row, ("pitch", "theta"), default=0.0)
    yaw = _float_from_row(row, ("yaw", "psi"))
    return roll, pitch, yaw, True


def geolocate_point(
    row: dict[str, str],
    x_px: float,
    y_px: float,
    image_width: int,
    image_height: int,
    config: dict[str, Any],
    *,
    altitude_offset_m: float = 0.0,
) -> tuple[float | None, float | None, str]:
    try:
        context = build_geolocator_context(
            row,
            image_width,
            image_height,
            config,
            altitude_offset_m=altitude_offset_m,
        )
        lats, lons = context.to_gps_arrays(((x_px, y_px),))
        target_lat = float(lats[0])
        target_lon = float(lons[0])
        return target_lat, target_lon, ""
    except Exception as exc:
        return None, None, str(exc)


def build_geolocator_context(
    row: dict[str, str],
    image_width: int,
    image_height: int,
    config: dict[str, Any],
    *,
    altitude_offset_m: float = 0.0,
) -> GeolocatorContext:
    lat = _float_from_row(row, ("lat", "latitude"))
    lon = _float_from_row(row, ("lon", "longitude"))
    altitude = _float_from_row(
        row, ("alt_rel", "h", "altitude", "alt_msl", "alt")
    ) + float(altitude_offset_m)
    roll, pitch, yaw, is_deg = _metadata_angles(
        row,
        config,
    )
    camera = build_camera(config, image_width, image_height)
    configuration = camera.Configuration(
        Affine(((0.0, 0.0, -altitude),), ((), (), ())),
        roll,
        pitch,
        yaw,
        is_deg=is_deg,
    )
    return GeolocatorContext(GeoLoc(camera, configuration, GeoGPS(GPS(lat, lon))))


def real_target_location_from_config(
    config: dict[str, Any],
) -> tuple[float, float] | None:
    geoloc_config = config.get("decision_system", {}).get("geoloc", {})
    value = geoloc_config.get("real_target_location")
    if value in (None, ""):
        return None

    if isinstance(value, dict):
        lat = value.get("lat", value.get("latitude"))
        lon = value.get("lon", value.get("longitude"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        lat, lon = value[0], value[1]
    else:
        raise ValueError("real_target_location must be [lat, lon] or {lat, lon}")

    if lat in (None, "") or lon in (None, ""):
        raise ValueError("real_target_location must include both lat and lon")
    return float(lat), float(lon)


def gps_distance_m(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
) -> float:
    reference = GeoGPS(GPS(lat_a, lon_a))
    target = GeoGPS(GPS(lat_b, lon_b))
    return float(reference.map(target).norm().reshape(-1)[0])


def _bayer_code(pixel_format: str) -> int | None:
    if cv2 is None:
        return None
    return cv2.COLOR_BayerBG2RGB


def _image_array_to_uint8_rgb(array: np.ndarray, pixel_format: str = "") -> np.ndarray:
    if array.ndim == 2:
        array = _normalize_uint8(array)
        bayer_code = _bayer_code(pixel_format)
        if bayer_code is not None:
            return cv2.cvtColor(array, bayer_code)
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] >= 3:
        return _normalize_uint8(array[:, :, :3])
    raise ValueError(f"Unsupported image array shape: {array.shape}")


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


def _numpy_shape_to_width_height(shape: Iterable[int]) -> tuple[int, int]:
    dims = tuple(int(v) for v in shape)
    if len(dims) < 2:
        raise ValueError(f"image shape must have at least 2 dimensions, got {shape}")
    # Numpy/OpenCV arrays are HxW[xC], while Camera/OpenCV dsize use WxH.
    return dims[1], dims[0]


def load_image(image_path: Path, pixel_format: str = "") -> LoadedImage:
    rgb: np.ndarray | None = None
    if image_path.suffix.lower() == ".npy":
        image = np.load(image_path)
        rgb = _image_array_to_uint8_rgb(image, pixel_format)
        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    else:
        surface = pygame.image.load(str(image_path)).convert()
    width, height = surface.get_size()
    return LoadedImage(image_path, surface, width, height, rgb)


def draw_button(screen: Any, font: Any, button: Button, enabled: bool = True) -> None:
    bg = (52, 62, 74) if enabled else (34, 40, 48)
    border = (110, 125, 145) if enabled else (65, 74, 84)
    text_color = (245, 248, 252) if enabled else (145, 153, 163)
    pygame.draw.rect(screen, bg, button.rect, border_radius=6)
    pygame.draw.rect(screen, border, button.rect, width=1, border_radius=6)
    text = font.render(button.label, True, text_color)
    screen.blit(text, text.get_rect(center=button.rect.center))


def draw_text(
    screen: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int]
) -> None:
    screen.blit(font.render(text, True, color), pos)


def fit_rect(
    image_width: int,
    image_height: int,
    area: Any,
) -> tuple[Any, float]:
    scale = min(area.width / image_width, area.height / image_height)
    scale = max(scale, 0.01)
    display_width = max(1, int(image_width * scale))
    display_height = max(1, int(image_height * scale))
    rect = pygame.Rect(0, 0, display_width, display_height)
    rect.center = area.center
    return rect, scale


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


class Flight3DPlot:
    """Lazy 3D flight companion window for ManualTargetChecker.

    The expensive parts are intentionally demand-driven:
    - no figure is created until the operator opens the 3D window;
    - bbox GPS points are reused from ManualTargetChecker._all_bbox_gps;
    - FOV footprints are computed one frame at a time and cached.
    """

    def __init__(self, checker: Any) -> None:
        self.checker = checker
        self.index = 0
        self.drone_states: list[FlightDroneState] = []
        self._state_by_image_index: dict[int, FlightDroneState] = {}
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self._state_key: tuple[Any, ...] | None = None
        self._footprint_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._detection_cache_key: tuple[Any, ...] | None = None
        self._detection_points: list[dict[str, Any]] = []
        self._dynamic_artists: list[Any] = []

        self.fig: Any | None = None
        self.ax3d: Any | None = None
        self.metadata_fig: Any | None = None
        self.metadata_ax: Any | None = None

    @property
    def is_open(self) -> bool:
        if plt is None or self.fig is None:
            return False
        try:
            return bool(plt.fignum_exists(self.fig.number))
        except Exception:
            return False

    def open(self) -> bool:
        if plt is None or Poly3DCollection is None:
            self.checker.status = "Matplotlib is required for the 3D flight map"
            return False

        changed = self._ensure_states()
        if not self.drone_states:
            self.checker.status = "No usable drone states for 3D flight map"
            return False

        if not self.is_open:
            self._build_figure()
            plt.show(block=False)
        if not self._metadata_is_open:
            self._open_metadata_window()
        self.refresh(rebuild_static=changed, index=self.checker.index)
        self.checker.status = "3D flight map open"
        return True

    def open_metadata(self) -> bool:
        if plt is None:
            self.checker.status = "Matplotlib is required for flight metadata"
            return False

        changed = self._ensure_states()
        if not self._metadata_is_open:
            self._open_metadata_window()
        self.refresh(rebuild_static=changed, index=self.checker.index)
        return True

    def close(self) -> None:
        if self.fig is not None and plt is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
        if self.metadata_fig is not None and plt is not None:
            try:
                plt.close(self.metadata_fig)
            except Exception:
                pass
        self._on_close(None)
        self._on_metadata_close(None)

    def flush_events(self) -> None:
        fig_open = self.is_open
        metadata_open = self._metadata_is_open
        if not fig_open and not metadata_open:
            return
        for fig in (self.fig if fig_open else None, self.metadata_fig):
            if fig is None:
                continue
            try:
                fig.canvas.flush_events()
            except Exception:
                pass

    def refresh(
        self, *, rebuild_static: bool = False, index: int | None = None
    ) -> None:
        fig_open = self.is_open
        metadata_open = self._metadata_is_open
        if not fig_open and not metadata_open:
            return

        states_changed = self._ensure_states()
        if not self.drone_states and not metadata_open:
            return

        if index is None:
            index = self.checker.index
        index = max(0, min(int(index), len(self.checker.rows) - 1))
        self.index = index

        if fig_open:
            if states_changed or rebuild_static:
                self._draw_static_3d()

            self._clear_dynamic()
            state = self._state_by_image_index.get(index)
            if state is not None:
                self._draw_dynamic_3d(state)
        if metadata_open:
            self._draw_metadata(index)

        for fig in (self.fig if fig_open else None, self.metadata_fig):
            if fig is None:
                continue
            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                pass

    def invalidate(self) -> None:
        self._state_key = None
        self._detection_cache_key = None
        self._footprint_cache.clear()

    def _on_close(self, _event: Any) -> None:
        self.fig = None
        self.ax3d = None
        self._dynamic_artists = []

    @property
    def _metadata_is_open(self) -> bool:
        if plt is None or self.metadata_fig is None:
            return False
        try:
            return bool(plt.fignum_exists(self.metadata_fig.number))
        except Exception:
            return False

    def _on_metadata_close(self, _event: Any) -> None:
        self.metadata_fig = None
        self.metadata_ax = None

    def _open_metadata_window(self) -> None:
        if plt is None:
            return
        self.metadata_fig, self.metadata_ax = plt.subplots(
            figsize=(4.8, 3.6), num="Flight Metadata"
        )
        self.metadata_fig.patch.set_facecolor(FIG_BG)
        self.metadata_ax.set_facecolor(AXES_BG)
        self.metadata_ax.axis("off")
        self.metadata_fig.canvas.manager.set_window_title("Flight Metadata")
        self.metadata_fig.canvas.mpl_connect("close_event", self._on_metadata_close)
        plt.show(block=False)

    def _state_signature(self) -> tuple[Any, ...]:
        return (
            len(self.checker.rows),
            self.checker.metadata_offset,
            round(float(self.checker.altitude_offset_m), 6),
            round(float(self.checker._px2body_error_angle_deg()), 6),
        )

    def _ensure_states(self) -> bool:
        key = self._state_signature()
        if self._state_key == key and self.drone_states:
            return False

        states: list[FlightDroneState] = []
        for image_index, image_row in enumerate(self.checker.rows):
            metadata_index = self.checker._metadata_index_for_image(image_index)
            metadata_row = self.checker.rows[metadata_index]
            try:
                lat = _float_from_row(metadata_row, ("lat", "latitude"))
                lon = _float_from_row(metadata_row, ("lon", "longitude"))
                alt = _float_from_row(
                    metadata_row, ("alt_rel", "h", "altitude", "alt_msl", "alt")
                ) + float(self.checker.altitude_offset_m)
                roll, pitch, yaw, _ = _metadata_angles(
                    metadata_row,
                    self.checker.config,
                )
            except (TypeError, ValueError, KeyError):
                continue

            image_filename = image_row.get("filename") or metadata_row.get(
                "filename", ""
            )
            states.append(
                FlightDroneState(
                    image_index=image_index,
                    metadata_index=metadata_index,
                    filename=Path(image_filename).name,
                    lat=lat,
                    lon=lon,
                    alt_rel=alt,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    timestamp=_first_present(metadata_row, ("timestamp",)),
                    row=metadata_row,
                )
            )

        self.drone_states = states
        self._state_by_image_index = {state.image_index: state for state in states}
        self._state_key = key
        self._footprint_cache.clear()
        self._detection_cache_key = None

        if states:
            self.ref_lat = float(np.mean([state.lat for state in states]))
            self.ref_lon = float(np.mean([state.lon for state in states]))
            for state in states:
                state.east_m, state.north_m = self._gps_to_metric(state.lat, state.lon)
        else:
            self.ref_lat = 0.0
            self.ref_lon = 0.0
        return True

    def _gps_to_metric(self, lat: float, lon: float) -> tuple[float, float]:
        deg2m_lon = DEG2M_LAT * math.cos(math.radians(self.ref_lat))
        east_m = (lon - self.ref_lon) * deg2m_lon
        north_m = (lat - self.ref_lat) * DEG2M_LAT
        return east_m, north_m

    def _image_size_for_index(self, image_index: int) -> tuple[int, int]:
        loaded = self.checker.loaded_image
        if image_index == self.checker.index and loaded is not None:
            return loaded.width, loaded.height

        res = self.checker.config.get("camera", {}).get("resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            try:
                return int(float(res[0])), int(float(res[1]))
            except (TypeError, ValueError):
                pass
        return DEFAULT_IMAGE_W, DEFAULT_IMAGE_H

    def _metric_detection_points(self) -> list[dict[str, Any]]:
        source = self.checker._all_bbox_gps
        key = (
            id(source),
            len(source),
            round(self.ref_lat, 9),
            round(self.ref_lon, 9),
        )
        if self._detection_cache_key == key:
            return self._detection_points

        points: list[dict[str, Any]] = []
        for det in source:
            lat = _coerce_optional_float(det.get("lat"))
            lon = _coerce_optional_float(det.get("lon"))
            if lat is None or lon is None:
                continue
            east_m, north_m = self._gps_to_metric(lat, lon)
            enriched = dict(det)
            enriched["east_m"] = east_m
            enriched["north_m"] = north_m
            points.append(enriched)

        self._detection_cache_key = key
        self._detection_points = points
        return points

    def _manual_annotation_points(self) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for image_index, annotations in enumerate(self.checker.annotations):
            image_filename = self.checker.rows[image_index].get("filename", "")
            for annotation in annotations:
                if (
                    not annotation.saved
                    or annotation.lat is None
                    or annotation.lon is None
                ):
                    continue
                east_m, north_m = self._gps_to_metric(annotation.lat, annotation.lon)
                points.append(
                    {
                        "east_m": east_m,
                        "north_m": north_m,
                        "label": annotation.label,
                        "filename": image_filename,
                    }
                )
        return points

    def _compute_fov_footprint(self, state: FlightDroneState) -> np.ndarray:
        image_w, image_h = self._image_size_for_index(state.image_index)
        key = (
            state.image_index,
            state.metadata_index,
            int(image_w),
            int(image_h),
            round(float(self.checker.altitude_offset_m), 6),
            round(float(self.checker._px2body_error_angle_deg()), 6),
        )
        cached = self._footprint_cache.get(key)
        if cached is not None:
            return cached

        corners_px = [
            (0.0, 0.0),
            (float(image_w), 0.0),
            (float(image_w), float(image_h)),
            (0.0, float(image_h)),
        ]
        out: list[tuple[float, float]] = []
        for lat, lon, error in self.checker._geolocate_points_for_image(
            state.image_index,
            corners_px,
            image_w,
            image_h,
        ):
            if error or lat is None or lon is None:
                continue
            out.append(self._gps_to_metric(lat, lon))

        footprint = np.array(out, dtype=float) if out else np.empty((0, 2))
        self._footprint_cache[key] = footprint
        return footprint

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(9, 8), num="Manual Target Flight 3D")
        self.ax3d = self.fig.add_axes([0.06, 0.06, 0.90, 0.88], projection="3d")
        self._style_axes()
        self._draw_static_3d()

        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _style_axes(self) -> None:
        if self.fig is None or self.ax3d is None:
            return
        self.fig.patch.set_facecolor(FIG_BG)
        self.ax3d.set_facecolor(AXES_BG)
        self.ax3d.tick_params(colors=FG)
        self.ax3d.set_xlabel("East (m)", color=FG)
        self.ax3d.set_ylabel("North (m)", color=FG)
        self.ax3d.set_zlabel("Altitude (m)", color=FG)
        self.ax3d.xaxis.label.set_color(FG)
        self.ax3d.yaxis.label.set_color(FG)
        self.ax3d.zaxis.label.set_color(FG)
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.set_pane_color((0.1, 0.12, 0.16, 1.0))

    def _draw_static_3d(self) -> None:
        if self.ax3d is None:
            return
        self._clear_dynamic()
        self.ax3d.cla()
        self._style_axes()

        states = self.drone_states
        if not states:
            self.ax3d.text2D(
                0.5,
                0.5,
                "No usable drone states",
                color=FG,
                ha="center",
                va="center",
                transform=self.ax3d.transAxes,
            )
            return

        es = [state.east_m for state in states]
        ns = [state.north_m for state in states]
        alts = [state.alt_rel for state in states]
        self.ax3d.plot(es, ns, alts, color=TRAJECTORY_LINE, lw=1.0, alpha=0.65)
        self.ax3d.scatter(
            es,
            ns,
            alts,
            c=range(len(es)),
            cmap="plasma",
            s=8,
            depthshade=False,
        )

        detections = self._metric_detection_points()
        if detections:
            self.ax3d.scatter(
                [det["east_m"] for det in detections],
                [det["north_m"] for det in detections],
                [0.0] * len(detections),
                color=DETECTION_COLOR,
                marker="D",
                s=10,
                depthshade=False,
                label="Model bboxes",
            )

        manual_points = self._manual_annotation_points()
        if manual_points:
            self.ax3d.scatter(
                [point["east_m"] for point in manual_points],
                [point["north_m"] for point in manual_points],
                [0.0] * len(manual_points),
                color=MANUAL_POINT_COLOR,
                marker="^",
                s=45,
                depthshade=False,
                label="Manual saved",
            )

        all_e = list(es) + [det["east_m"] for det in detections]
        all_n = list(ns) + [det["north_m"] for det in detections]
        all_z = list(alts) + [0.0]
        all_e.extend(point["east_m"] for point in manual_points)
        all_n.extend(point["north_m"] for point in manual_points)
        all_z.extend(0.0 for _ in manual_points)
        self._set_axis_limits(all_e, all_n, all_z)
        if detections or manual_points:
            self.ax3d.legend(
                loc="upper left",
                fontsize=8,
                facecolor=AXES_BG,
                edgecolor="#3a4a5e",
                labelcolor=FG,
            )
        self.ax3d.grid(True, color=GRID, linewidth=0.5)
        self.ax3d.set_title("Flight trajectory and geolocated targets", color=FG)

    def _set_axis_limits(
        self, east_values: list[float], north_values: list[float], z_values: list[float]
    ) -> None:
        if self.ax3d is None:
            return

        def _pad(
            values: list[float], *, minimum_span: float = 1.0
        ) -> tuple[float, float]:
            lo = min(values)
            hi = max(values)
            span = max(hi - lo, minimum_span)
            margin = span * 0.08
            return lo - margin, hi + margin

        xlo, xhi = _pad(east_values)
        ylo, yhi = _pad(north_values)
        zlo, zhi = _pad(z_values + [0.0])
        if zlo > 0:
            zlo = 0.0
        self.ax3d.set_xlim(xlo, xhi)
        self.ax3d.set_ylim(ylo, yhi)
        self.ax3d.set_zlim(zlo, zhi)
        self.ax3d.set_autoscale_on(False)
        self._set_box_aspect()

    def _set_box_aspect(self) -> None:
        if self.ax3d is None:
            return
        xlo, xhi = self.ax3d.get_xlim()
        ylo, yhi = self.ax3d.get_ylim()
        zlo, zhi = self.ax3d.get_zlim()
        self.ax3d.set_box_aspect(
            (
                max(xhi - xlo, 1.0),
                max(yhi - ylo, 1.0),
                max(zhi - zlo, 1.0),
            )
        )

    def _expand_limits_for_points(
        self,
        east_values: Iterable[float],
        north_values: Iterable[float],
        z_values: Iterable[float],
    ) -> None:
        if self.ax3d is None:
            return
        east = list(east_values)
        north = list(north_values)
        z = list(z_values)
        if not east or not north or not z:
            return
        xlo, xhi = self.ax3d.get_xlim()
        ylo, yhi = self.ax3d.get_ylim()
        zlo, zhi = self.ax3d.get_zlim()
        new_xlo = min(xlo, min(east))
        new_xhi = max(xhi, max(east))
        new_ylo = min(ylo, min(north))
        new_yhi = max(yhi, max(north))
        new_zlo = min(zlo, min(z))
        new_zhi = max(zhi, max(z))
        if (new_xlo, new_xhi, new_ylo, new_yhi, new_zlo, new_zhi) == (
            xlo,
            xhi,
            ylo,
            yhi,
            zlo,
            zhi,
        ):
            return
        self._set_axis_limits(
            [new_xlo, new_xhi],
            [new_ylo, new_yhi],
            [new_zlo, new_zhi],
        )

    def _clear_dynamic(self) -> None:
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._dynamic_artists = []

    def _draw_dynamic_3d(self, state: FlightDroneState) -> None:
        if self.ax3d is None:
            return
        marker = self.ax3d.scatter(
            [state.east_m],
            [state.north_m],
            [state.alt_rel],
            color=CURRENT_POS_COLOR,
            s=130,
            marker="*",
            depthshade=False,
        )
        self._dynamic_artists.append(marker)

        footprint = self._compute_fov_footprint(state)
        if len(footprint) >= 1:
            self._expand_limits_for_points(
                list(footprint[:, 0]) + [state.east_m],
                list(footprint[:, 1]) + [state.north_m],
                [0.0] * len(footprint) + [state.alt_rel],
            )
            for east, north in footprint:
                line = self.ax3d.plot(
                    [state.east_m, east],
                    [state.north_m, north],
                    [state.alt_rel, 0.0],
                    color=FOV_LINE_COLOR,
                    lw=1.2,
                )[0]
                self._dynamic_artists.append(line)
            if len(footprint) >= 2:
                loop = np.vstack([footprint, footprint[0]])
                base = self.ax3d.plot(
                    loop[:, 0],
                    loop[:, 1],
                    [0.0] * len(loop),
                    color=FOV_LINE_COLOR,
                    lw=1.2,
                )[0]
                self._dynamic_artists.append(base)
            if len(footprint) >= 3:
                verts = [[(east, north, 0.0) for east, north in footprint]]
                poly = Poly3DCollection(
                    verts, facecolor=FOV_FILL_COLOR, edgecolor="none"
                )
                self.ax3d.add_collection3d(poly)
                self._dynamic_artists.append(poly)

        current_filename = self.checker.rows[state.image_index].get("filename", "")
        detections = [
            det
            for det in self._metric_detection_points()
            if det.get("filename") == current_filename
        ]
        if detections:
            current_dets = self.ax3d.scatter(
                [det["east_m"] for det in detections],
                [det["north_m"] for det in detections],
                [0.0] * len(detections),
                color=DETECTION_CUR_COLOR,
                marker="D",
                s=42,
                depthshade=False,
            )
            self._dynamic_artists.append(current_dets)

        if state.image_index == self.checker.index:
            annotation = self.checker._current_annotation()
            if (
                not annotation.saved
                and annotation.lat is not None
                and annotation.lon is not None
            ):
                east_m, north_m = self._gps_to_metric(annotation.lat, annotation.lon)
                current_manual = self.ax3d.scatter(
                    [east_m],
                    [north_m],
                    [0.0],
                    color="#e040fb",
                    marker="^",
                    s=65,
                    depthshade=False,
                )
                self._dynamic_artists.append(current_manual)

    def _draw_metadata(self, image_index: int) -> None:
        if not self._metadata_is_open or self.metadata_ax is None:
            return
        ax = self.metadata_ax
        ax.cla()
        ax.set_facecolor(AXES_BG)
        ax.axis("off")

        state = self._state_by_image_index.get(image_index)
        if state is None:
            ax.text(
                0.5,
                0.5,
                "No metadata for current frame",
                color=FG,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        def _fmt(value: Any, precision: int = 3) -> str:
            if value in (None, ""):
                return "n/a"
            try:
                return f"{float(value):.{precision}f}"
            except (TypeError, ValueError):
                return str(value)

        image_row = self.checker.rows[state.image_index]
        rows = [
            ("image index", f"{state.image_index} / {len(self.checker.rows) - 1}"),
            ("metadata index", str(state.metadata_index)),
            (
                "file",
                truncate_middle(
                    Path(image_row.get("filename", state.filename)).name, 36
                ),
            ),
            ("timestamp", state.timestamp or "n/a"),
            ("height", f"{state.alt_rel:.3f} m"),
            ("lat", f"{state.lat:.8f}"),
            ("lon", f"{state.lon:.8f}"),
            ("roll", f"{state.roll:.3f} deg"),
            ("pitch", f"{state.pitch:.3f} deg"),
            ("yaw", f"{state.yaw:.3f} deg"),
        ]

        if abs(float(self.checker.altitude_offset_m)) > 1e-6:
            rows.append(("alt offset", f"{self.checker.altitude_offset_m:+.3f} m"))
        rows.append(
            (
                "px2body err",
                f"{self.checker._px2body_error_angle_deg():.3f} deg",
            )
        )

        ax.text(
            0.04,
            0.94,
            "Flight Metadata",
            color="#e0e8f4",
            fontsize=13,
            fontweight="bold",
            va="top",
            transform=ax.transAxes,
        )

        y = 0.82
        row_step = min(0.07, 0.78 / max(len(rows), 1))
        for key, value in rows:
            ax.text(
                0.05,
                y,
                f"{key}:",
                color="#8fa0b8",
                fontsize=10,
                va="top",
                transform=ax.transAxes,
            )
            ax.text(
                0.40,
                y,
                value,
                color=FG,
                fontsize=10,
                va="top",
                transform=ax.transAxes,
            )
            y -= row_step


class ManualTargetChecker:
    def __init__(
        self,
        metadata_path: Path,
        output_path: Path,
        config_path: Path,
        window_size: tuple[int, int],
        off_by_n: int = 0,
        metadata_shift: int = 0,
        homography_preview_scale: float | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.rows, self.input_fieldnames = load_metadata(metadata_path)
        self.annotations = _load_existing_annotations(output_path, self.rows)
        self.active_point_indices = [0 for _ in self.rows]
        self.config = load_config(config_path)
        checker_cfg = self.config.get("manual_target_checker", {})
        self.window_size = window_size
        self.index = 0
        requested_shift = int(metadata_shift)
        self.metadata_offset_max = max(0, int(off_by_n), abs(requested_shift))
        self.metadata_offset = min(
            max(requested_shift, -self.metadata_offset_max),
            self.metadata_offset_max,
        )
        self.loaded_image: LoadedImage | None = None
        self.scaled_surface: Any | None = None
        self.scaled_surface_key: tuple[Any, ...] | None = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_panning = False
        self.last_pan_pos: tuple[int, int] | None = None
        self.status = ""
        self._keys_down: set[int] = set()
        self.buttons: dict[str, Button] = {}
        self.frame_jump_active = False
        self.frame_jump_text = ""
        self.frame_jump_rect: Any | None = None
        self.loaded_bboxes: list[dict[str, Any]] = []
        self.show_model_bboxes = True
        # Producer filter state
        self.producer_filter_index = 0
        self.producer_filter: str = PRODUCER_FILTER_MODES[self.producer_filter_index]
        # Bbox SQLite connection
        self._bbox_db_path = metadata_path.parent / BBOX_DB_FILENAME
        self._bbox_conn: sqlite3.Connection | None = _open_bbox_db(metadata_path)
        # GPS plot state
        self._gps_fig: Any | None = None
        self._gps_ax: Any | None = None
        self._flight3d_plot: Flight3DPlot | None = None
        # Homography preview state
        self._homography_fig: Any | None = None
        self._homography_ax: Any | None = None
        self._homography_error: str = ""
        self._homography_enabled = cv2 is not None
        self._all_bbox_gps: list[dict[str, Any]] = []  # precomputed once at startup
        self._decision_report_path = self.output_path.with_name(
            f"{self.output_path.stem}_decision.json"
        )
        # Debounce repeated click actions (some touchpads/mice emit duplicate
        # press events that can trigger accidental multi-step navigation).
        self._last_click_action_ms: dict[str, int] = {}
        self._image_cache: "OrderedDict[tuple[str, str], LoadedImage]" = OrderedDict()
        self._geolocator_cache: "OrderedDict[tuple[Any, ...], GeolocatorContext]" = (
            OrderedDict()
        )
        self._image_area_cache: tuple[tuple[int, int], Any] | None = None
        self._label_surface_cache: dict[tuple[int, str, tuple[int, int, int]], Any] = {}

        homography_cfg = self.config.get("homography_stage", {})
        self._homography_output_size = self._parse_output_size(
            homography_cfg.get("output_size")
        )
        meters_per_pixel = homography_cfg.get("meters_per_pixel")
        self._homography_meters_per_pixel: float | None = None
        if meters_per_pixel not in (None, ""):
            try:
                meters_per_pixel_float = float(meters_per_pixel)
                if meters_per_pixel_float > 0:
                    self._homography_meters_per_pixel = meters_per_pixel_float
            except (TypeError, ValueError):
                self._homography_meters_per_pixel = None
        self._homography_padding_px = max(0, int(homography_cfg.get("padding_px", 0)))
        self._homography_max_output_side = max(
            0, int(homography_cfg.get("max_output_side", 0))
        )
        preview_scale_value = (
            checker_cfg.get("homography_preview_scale", 0.5)
            if homography_preview_scale is None
            else homography_preview_scale
        )
        try:
            self._homography_preview_scale = float(preview_scale_value)
        except (TypeError, ValueError):
            self._homography_preview_scale = 0.5
        self._homography_preview_scale = min(
            max(self._homography_preview_scale, 0.05),
            1.0,
        )
        if not self._homography_enabled:
            self._homography_error = "opencv-python is required for homography preview"

        # Picker state
        self.picker_active = False
        self.picker_cycle_delta = 0
        self.picker_buttons: dict[str, Button] = {}
        self.available_classes: list[str] = [str(i) for i in range(10)] + ["BULLSEYE"]
        self.altitude_offset_m = 0.0
        self.image_cache_size = int(checker_cfg.get("image_cache_size", 3))
        if self.image_cache_size < 1:
            self.image_cache_size = 1
        self.geolocator_cache_size = int(checker_cfg.get("geolocator_cache_size", 64))
        if self.geolocator_cache_size < 1:
            self.geolocator_cache_size = 1
        min_alt = float(checker_cfg.get("altitude_slider_min_m", -30.0))
        max_alt = float(checker_cfg.get("altitude_slider_max_m", 30.0))
        if min_alt >= max_alt:
            min_alt, max_alt = -30.0, 30.0
        self.altitude_offset_min_m = min_alt
        self.altitude_offset_max_m = max_alt

    def _metadata_index_for_image(self, image_index: int) -> int:
        if not self.rows:
            return 0
        shifted = int(image_index) + int(self.metadata_offset)
        return min(max(shifted, 0), len(self.rows) - 1)

    def _metadata_row_for_image(self, image_index: int) -> dict[str, str]:
        return self.rows[self._metadata_index_for_image(image_index)]

    def _metadata_row_for_current_image(self) -> dict[str, str]:
        return self._metadata_row_for_image(self.index)

    def _clear_geolocator_cache(self) -> None:
        self._geolocator_cache.clear()

    def _geolocator_context_for_image(
        self,
        image_index: int,
        image_width: int,
        image_height: int,
    ) -> GeolocatorContext:
        metadata_index = self._metadata_index_for_image(image_index)
        key = (
            metadata_index,
            int(image_width),
            int(image_height),
            round(float(self.altitude_offset_m), 6),
            round(float(self._px2body_error_angle_deg()), 6),
        )
        cached = self._geolocator_cache.get(key)
        if cached is not None:
            self._geolocator_cache.move_to_end(key)
            return cached

        context = build_geolocator_context(
            self.rows[metadata_index],
            int(image_width),
            int(image_height),
            self.config,
            altitude_offset_m=self.altitude_offset_m,
        )
        self._geolocator_cache[key] = context
        while len(self._geolocator_cache) > self.geolocator_cache_size:
            self._geolocator_cache.popitem(last=False)
        return context

    def _geolocate_points_for_image(
        self,
        image_index: int,
        points_px: Iterable[tuple[float, float]],
        image_width: int,
        image_height: int,
    ) -> list[tuple[float | None, float | None, str]]:
        points = list(points_px)
        if not points:
            return []

        try:
            context = self._geolocator_context_for_image(
                image_index,
                image_width,
                image_height,
            )
        except Exception as exc:
            error = str(exc)
            return [(None, None, error) for _ in points]

        try:
            lats, lons = context.to_gps_arrays(points)
            return [(float(lat), float(lon), "") for lat, lon in zip(lats, lons)]
        except Exception as batch_exc:
            results: list[tuple[float | None, float | None, str]] = []
            for point in points:
                try:
                    lats, lons = context.to_gps_arrays((point,))
                    results.append((float(lats[0]), float(lons[0]), ""))
                except Exception:
                    results.append((None, None, str(batch_exc)))
            return results

    def _geolocate_point_for_image(
        self,
        image_index: int,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float | None, float | None, str]:
        return self._geolocate_points_for_image(
            image_index,
            ((x_px, y_px),),
            image_width,
            image_height,
        )[0]

    def _metadata_offset_label(self) -> str:
        if self.metadata_offset_max <= 0:
            return "meta shift: 0 (disabled)"
        metadata_index = self._metadata_index_for_image(self.index)
        clamped = metadata_index != (self.index + self.metadata_offset)
        clamped_text = " clamped" if clamped else ""
        return (
            f"meta shift: {self.metadata_offset:+d}/{self.metadata_offset_max}"
            f" -> metadata index {metadata_index}{clamped_text}"
        )

    def _set_metadata_offset(self, offset: int) -> None:
        if self.metadata_offset_max <= 0:
            self.metadata_offset = 0
            self.status = "Metadata shift disabled; set --off-by-n > 0 to enable."
            return
        clamped = min(
            max(int(offset), -self.metadata_offset_max), self.metadata_offset_max
        )
        if clamped == self.metadata_offset:
            return
        self.metadata_offset = clamped
        self._clear_geolocator_cache()
        self._recompute_current_annotation_geoloc(include_saved=False)
        self._recompute_all_saved_points_geoloc(
            write_file=True,
            refresh_views=False,
            recompute_bboxes=False,
            update_status=False,
        )
        self._precompute_all_bbox_gps()
        self._refresh_gps_plot()
        self._refresh_homography_plot()
        self._refresh_flight_3d_plot(rebuild_static=True)
        self.status = self._metadata_offset_label()

    def _shift_metadata_offset(self, delta: int) -> None:
        self._set_metadata_offset(self.metadata_offset + int(delta))

    # ------------------------------------------------------------------
    # GPS Plot helpers
    # ------------------------------------------------------------------

    def _geolocate_fov_corners(
        self,
        image_index: int,
        image_width: int,
        image_height: int,
    ) -> list[tuple[float, float]]:
        """Return GPS (lat, lon) for the 4 image corners (camera FOV on ground).

        Corners are ordered: top-left, top-right, bottom-right, bottom-left.
        Any corner that fails geolocation is silently skipped.
        """
        corners_px = [
            (0.0, 0.0),
            (float(image_width), 0.0),
            (float(image_width), float(image_height)),
            (0.0, float(image_height)),
        ]
        result: list[tuple[float, float]] = []
        for lat, lon, error in self._geolocate_points_for_image(
            image_index,
            corners_px,
            image_width,
            image_height,
        ):
            if error or lat is None or lon is None:
                continue
            result.append((lat, lon))
        return result

    def _precompute_all_bbox_gps(self) -> None:
        """Single SQL pass over bbox_records, geolocate each bbox center."""
        self._all_bbox_gps = []
        if self._bbox_conn is None:
            return

        filename_to_index: dict[str, int] = {}
        for image_index, row in enumerate(self.rows):
            fname = row.get("filename", "")
            if fname:
                filename_to_index[fname] = image_index
        basename_to_index: dict[str, int] = {}
        for fname, image_index in filename_to_index.items():
            basename_to_index.setdefault(Path(fname).name, image_index)

        sql = (
            "SELECT image_filename, producer, inference_image_shape_json, "
            "full_image_shape_json, boxes_json FROM bbox_records"
        )
        params: list[Any] = []
        if self.producer_filter != "all":
            sql += " WHERE producer = ?"
            params.append(self.producer_filter)

        try:
            db_rows = self._bbox_conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            print(
                f"Error reading bbox_records from {self._bbox_db_path}: {e}",
                file=sys.stderr,
            )
            return

        for db_row in db_rows:
            image_index = filename_to_index.get(db_row["image_filename"])
            if image_index is None:
                image_index = basename_to_index.get(Path(db_row["image_filename"]).name)
            if image_index is None:
                continue
            image_row = self.rows[image_index]

            decoded = _decode_box_row(db_row)
            full_shape = decoded["full_shape"] or {}
            inf_shape = decoded["inference_shape"] or {}
            img_w = full_shape.get("width") or inf_shape.get("width")
            img_h = full_shape.get("height") or inf_shape.get("height")
            if img_w is None or img_h is None:
                if self.loaded_image is not None and image_index == self.index:
                    img_w = self.loaded_image.width
                    img_h = self.loaded_image.height
                else:
                    continue

            try:
                image_width = int(float(img_w))
                image_height = int(float(img_h))
            except (TypeError, ValueError):
                continue

            centers: list[tuple[float, float]] = []
            center_boxes: list[dict[str, Any]] = []
            for box in decoded["boxes"]:
                if not isinstance(box, dict):
                    continue
                bbox = box.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                xmin, ymin, xmax, ymax = bbox
                centers.append(((xmin + xmax) / 2, (ymin + ymax) / 2))
                center_boxes.append(box)

            for box, (lat, lon, error) in zip(
                center_boxes,
                self._geolocate_points_for_image(
                    image_index,
                    centers,
                    image_width,
                    image_height,
                ),
            ):
                if error or lat is None or lon is None:
                    continue
                self._all_bbox_gps.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "class": box.get("class", ""),
                        "conf": box.get("conf"),
                        "filename": image_row.get("filename", ""),
                        "producer": decoded["producer"],
                    }
                )

    def _open_gps_plot(self) -> None:
        if plt is None:
            return
        plt.ion()
        self._gps_fig, self._gps_ax = plt.subplots(
            figsize=(7, 7), num="GPS Bounding Box Map"
        )
        self._gps_fig.patch.set_facecolor("#12161c")
        self._gps_ax.set_facecolor("#1a2030")
        self._gps_fig.canvas.manager.set_window_title("GPS Bounding Box Map")
        self._refresh_gps_plot()
        plt.show(block=False)
        self._gps_fig.canvas.flush_events()

    def _open_homography_plot(self) -> None:
        if plt is None:
            return
        plt.ion()
        self._homography_fig, self._homography_ax = plt.subplots(
            figsize=(9, 5), num="Homographied Image"
        )
        self._homography_fig.patch.set_facecolor("#12161c")
        self._homography_ax.set_facecolor("#1a2030")
        self._homography_fig.canvas.manager.set_window_title("Homographied Image")
        self._refresh_homography_plot()
        plt.show(block=False)
        self._homography_fig.canvas.flush_events()

    def _ensure_flight_plot_controller(self) -> Flight3DPlot:
        if self._flight3d_plot is None:
            self._flight3d_plot = Flight3DPlot(self)
        return self._flight3d_plot

    def _open_flight_3d_plot(self) -> None:
        self._ensure_flight_plot_controller().open()

    def _open_flight_metadata_window(self) -> None:
        self._ensure_flight_plot_controller().open_metadata()

    def _refresh_flight_3d_plot(self, *, rebuild_static: bool = False) -> None:
        if self._flight3d_plot is None:
            return
        self._flight3d_plot.refresh(rebuild_static=rebuild_static)

    @staticmethod
    def _parse_output_size(value: Any) -> tuple[int, int] | None:
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

    def _compute_homography(
        self,
        row: dict[str, str],
        image_width: int,
        image_height: int,
    ) -> "HomographyComputation":
        if cv2 is None:
            raise RuntimeError("opencv-python is required for homography preview")
        return _compute_homography_fn(
            row,
            self.config,
            image_width,
            image_height,
            output_size=self._homography_output_size,
            meters_per_pixel=self._homography_meters_per_pixel,
            padding_px=self._homography_padding_px,
            max_output_side=self._homography_max_output_side,
            altitude_offset_m=self.altitude_offset_m,
        )

    def _loaded_image_rgb(self) -> np.ndarray | None:
        if self.loaded_image is None:
            return None
        if self.loaded_image.rgb is not None:
            return self.loaded_image.rgb
        # pygame surfarray exposes width-major arrays, swap axes to HxWxC.
        rgb = pygame.surfarray.array3d(self.loaded_image.surface)
        self.loaded_image.rgb = np.swapaxes(rgb, 0, 1)
        return self.loaded_image.rgb

    def _refresh_homography_plot(self) -> None:
        if plt is None or self._homography_ax is None or self._homography_fig is None:
            return
        ax = self._homography_ax
        ax.cla()
        ax.set_facecolor("#1a2030")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a4a5e")
        ax.tick_params(colors="#8fa0b8")
        ax.set_xticks([])
        ax.set_yticks([])

        if self.loaded_image is None:
            ax.text(
                0.5,
                0.5,
                "No image loaded",
                color="#c0d0e8",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        elif not self._homography_enabled:
            reason = self._homography_error or "homography unavailable"
            ax.text(
                0.5,
                0.5,
                f"Homography unavailable:\n{reason}",
                color="#ffb68a",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            metadata = self._metadata_row_for_current_image()
            rgb = self._loaded_image_rgb()
            if rgb is None:
                ax.text(
                    0.5,
                    0.5,
                    "Could not read image pixels",
                    color="#ffb68a",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                try:
                    image_width, image_height = _numpy_shape_to_width_height(rgb.shape)
                    comp = self._compute_homography(
                        metadata,
                        image_width,
                        image_height,
                    )
                    matrix = comp.matrix.astype(np.float64)
                    preview_output_size = comp.output_size
                    preview_scale = self._homography_preview_scale
                    if preview_scale < 0.999:
                        preview_output_size = (
                            max(1, int(round(comp.output_size[0] * preview_scale))),
                            max(1, int(round(comp.output_size[1] * preview_scale))),
                        )
                        matrix = matrix.copy()
                        matrix[0, :] *= preview_scale
                        matrix[1, :] *= preview_scale
                    warped_rgb = cv2.warpPerspective(
                        rgb,
                        matrix,
                        preview_output_size,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    ax.imshow(warped_rgb)
                    meters_per_pixel = comp.meters_per_pixel
                    mpp_text = (
                        "n/a"
                        if meters_per_pixel is None
                        else f"{meters_per_pixel:.4f} m/px"
                    )
                    size_text = f"{comp.output_size[0]}x{comp.output_size[1]}"
                    if preview_output_size != comp.output_size:
                        size_text += (
                            f", preview {preview_output_size[0]}x"
                            f"{preview_output_size[1]}"
                        )
                        if meters_per_pixel is not None:
                            mpp_text = (
                                f"{meters_per_pixel:.4f} m/source px, "
                                f"{meters_per_pixel / preview_scale:.4f} m/preview px"
                            )
                    ax.set_title(
                        (
                            f"Homographied view  —  index {self.index}/{len(self.rows) - 1}"
                            f"  ({size_text}, {mpp_text})"
                        ),
                        color="#e0e8f4",
                        fontsize=10,
                        pad=8,
                    )
                except (ValueError, RuntimeError) as exc:
                    ax.text(
                        0.5,
                        0.5,
                        f"Could not compute homography:\n{exc}",
                        color="#ffb68a",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                except Exception as exc:
                    ax.text(
                        0.5,
                        0.5,
                        f"Homography error:\n{exc}",
                        color="#ffb68a",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

        try:
            self._homography_fig.canvas.draw_idle()
            self._homography_fig.canvas.flush_events()
        except Exception:
            pass

    def _refresh_gps_plot(self) -> None:
        if plt is None or self._gps_ax is None or self._gps_fig is None:
            return
        ax = self._gps_ax

        # Collect the data we need first --------------------------------
        current_image_row = self.rows[self.index]
        current_metadata_row = self._metadata_row_for_current_image()
        current_filename = current_image_row.get("filename", "")
        current_pts = [
            p for p in self._all_bbox_gps if p["filename"] == current_filename
        ]

        saved_current_lats: list[float] = []
        saved_current_lons: list[float] = []
        saved_current_labels: list[str] = []
        saved_other_lats: list[float] = []
        saved_other_lons: list[float] = []
        saved_other_labels: list[str] = []
        center_lats: list[float] = []
        center_lons: list[float] = []
        for row, row_annotations in zip(self.rows, self.annotations):
            row_filename = row.get("filename", "")
            is_current_image = row_filename == current_filename
            for ann in row_annotations:
                if ann.is_auto_center and ann.lat is not None and ann.lon is not None:
                    center_lats.append(ann.lat)
                    center_lons.append(ann.lon)
                if ann.saved and ann.lat is not None and ann.lon is not None:
                    if is_current_image:
                        saved_current_lats.append(ann.lat)
                        saved_current_lons.append(ann.lon)
                        saved_current_labels.append(ann.label)
                    else:
                        saved_other_lats.append(ann.lat)
                        saved_other_lons.append(ann.lon)
                        saved_other_labels.append(ann.label)

        ann = self._current_annotation()

        # ------------------------------------------------------------------
        # Compute reference point for metric projection.
        # All plotted points are converted to metre offsets from this
        # centroid so axis ticks show small, readable numbers instead of
        # raw degree values that trigger matplotlib's offset/scientific
        # notation when the spread is sub-metre.
        # ------------------------------------------------------------------
        all_lats: list[float] = (
            [p["lat"] for p in self._all_bbox_gps]
            + center_lats
            + saved_current_lats
            + saved_other_lats
            + ([ann.lat] if ann.lat is not None and not ann.saved else [])
        )
        all_lons: list[float] = (
            [p["lon"] for p in self._all_bbox_gps]
            + center_lons
            + saved_current_lons
            + saved_other_lons
            + ([ann.lon] if ann.lon is not None and not ann.saved else [])
        )

        if all_lats:
            ref_lat = sum(all_lats) / len(all_lats)
            ref_lon = sum(all_lons) / len(all_lons)
        else:
            try:
                ref_lat = float(current_metadata_row.get("lat") or 0)
                ref_lon = float(current_metadata_row.get("lon") or 0)
            except (ValueError, TypeError):
                ref_lat, ref_lon = 0.0, 0.0

        _DEG2M_LAT = 111_320.0
        _deg2m_lon = 111_320.0 * math.cos(math.radians(ref_lat))

        def _mx(lons: list[float]) -> list[float]:
            return [(v - ref_lon) * _deg2m_lon for v in lons]

        def _my(lats: list[float]) -> list[float]:
            return [(v - ref_lat) * _DEG2M_LAT for v in lats]

        # ------------------------------------------------------------------
        # Draw
        # ------------------------------------------------------------------
        ax.cla()
        ax.set_facecolor("#1a2030")
        ax.tick_params(colors="#8fa0b8")
        ax.xaxis.label.set_color("#8fa0b8")
        ax.yaxis.label.set_color("#8fa0b8")
        ax.title.set_color("#e0e8f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a4a5e")

        # All model bbox GPS points
        if self._all_bbox_gps:
            b_lats = [p["lat"] for p in self._all_bbox_gps]
            b_lons = [p["lon"] for p in self._all_bbox_gps]
            ax.scatter(
                _mx(b_lons),
                _my(b_lats),
                s=48,
                c="#00e676",
                marker="D",
                alpha=0.82,
                linewidths=0.5,
                edgecolors="#00a050",
                zorder=3,
                label="Model bbox center",
            )

        # Current image bbox points highlighted
        if current_pts:
            c_lats = [p["lat"] for p in current_pts]
            c_lons = [p["lon"] for p in current_pts]
            ax.scatter(
                _mx(c_lons),
                _my(c_lats),
                s=120,
                c="#ff9f0a",
                marker="D",
                alpha=1.0,
                linewidths=1.0,
                edgecolors="#ff6000",
                zorder=5,
                label="Current image bboxes",
            )

        # Auto-center point for each image (all frames)
        if center_lats:
            ax.scatter(
                _mx(center_lons),
                _my(center_lats),
                s=42,
                c="#55d6ff",
                marker="o",
                alpha=0.72,
                linewidths=0.7,
                edgecolors="#1d7fa0",
                zorder=4,
                label="Image centers",
            )

        # Saved manual annotations from other images
        if saved_other_lats:
            ax.scatter(
                _mx(saved_other_lons),
                _my(saved_other_lats),
                s=90,
                c="#ff4c4c",
                marker="^",
                alpha=0.95,
                linewidths=0.8,
                edgecolors="#cc0000",
                zorder=6,
                label="Manual annotation (other images)",
            )
            for lat, lon, lbl in zip(
                saved_other_lats, saved_other_lons, saved_other_labels
            ):
                ax.annotate(
                    str(lbl),
                    xy=((lon - ref_lon) * _deg2m_lon, (lat - ref_lat) * _DEG2M_LAT),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#ffdddd",
                )

        # Saved manual annotations from current image
        if saved_current_lats:
            ax.scatter(
                _mx(saved_current_lons),
                _my(saved_current_lats),
                s=96,
                c="#32cd32",
                marker="^",
                alpha=1.0,
                linewidths=0.9,
                edgecolors="#0d7f0d",
                zorder=7,
                label="Manual annotation (current image)",
            )
            for lat, lon, lbl in zip(
                saved_current_lats, saved_current_lons, saved_current_labels
            ):
                ax.annotate(
                    str(lbl),
                    xy=((lon - ref_lon) * _deg2m_lon, (lat - ref_lat) * _DEG2M_LAT),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#d7ffd7",
                )

        # Current unsaved annotation preview
        if ann.lat is not None and ann.lon is not None and not ann.saved:
            ax.scatter(
                _mx([ann.lon]),
                _my([ann.lat]),
                s=100,
                c="#e040fb",
                marker="^",
                alpha=0.9,
                linewidths=0.8,
                edgecolors="#a000d0",
                zorder=7,
                label="Current (unsaved)",
            )

        # Camera FOV corners for the current image
        if self.loaded_image is not None:
            fov_corners = self._geolocate_fov_corners(
                self.index,
                self.loaded_image.width,
                self.loaded_image.height,
            )
            if fov_corners:
                fov_lats = [c[0] for c in fov_corners]
                fov_lons = [c[1] for c in fov_corners]
                ax.scatter(
                    _mx(fov_lons),
                    _my(fov_lats),
                    s=36,
                    c="#888888",
                    marker="s",
                    alpha=0.85,
                    linewidths=0.6,
                    edgecolors="#555555",
                    zorder=4,
                    label="Camera FOV corners",
                )
                # Draw outline connecting the corners
                if len(fov_corners) >= 3:
                    poly_lons = _mx(fov_lons + [fov_lons[0]])
                    poly_lats = _my(fov_lats + [fov_lats[0]])
                    ax.plot(
                        poly_lons,
                        poly_lats,
                        color="#888888",
                        linewidth=1.0,
                        alpha=0.6,
                        linestyle="--",
                        zorder=4,
                    )

        # Reference coordinates embedded in axis labels so absolute GPS is
        # always readable even though ticks show metre offsets.
        ax.set_xlabel(f"East offset (m)  [ref lon {ref_lon:.6f}°]", fontsize=9)
        ax.set_ylabel(f"North offset (m)  [ref lat {ref_lat:.6f}°]", fontsize=9)
        ax.set_title(
            f"GPS Map  —  index {self.index}/{len(self.rows) - 1}: "
            f"{Path(current_filename).name}",
            fontsize=10,
            pad=8,
        )
        # Disable scientific / offset notation — ticks are now small metres
        ax.ticklabel_format(style="plain", useOffset=False, axis="both")
        ax.legend(
            loc="upper right",
            fontsize=8,
            facecolor="#1a2030",
            edgecolor="#3a4a5e",
            labelcolor="#c0d0e8",
        )
        ax.grid(True, color="#2a3a50", linewidth=0.5, linestyle="--")
        # Equal aspect: 1 m north == 1 m east visually
        ax.set_aspect("equal", adjustable="datalim")
        try:
            self._gps_fig.canvas.draw_idle()
            self._gps_fig.canvas.flush_events()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _manual_label_class_map(self) -> dict[str, int]:
        label_map = dict(DEFAULT_MANUAL_LABEL_CLASS_MAP)
        configured = (
            self.config.get("manual_target_checker", {}).get("label_to_class", {}) or {}
        )
        if isinstance(configured, dict):
            for key, value in configured.items():
                cls_id = _parse_class_id(value)
                if cls_id is None:
                    continue
                label_map[str(key).strip().lower()] = cls_id
        return label_map

    def _target_name_for_class(
        self, cls_id: int | None, *, manual_digit_context: bool = False
    ) -> str:
        if cls_id is None:
            return "unknown"
        if manual_digit_context:
            if cls_id in DEFAULT_MANUAL_DIGIT_TARGET_NAMES:
                return DEFAULT_MANUAL_DIGIT_TARGET_NAMES[cls_id]
            if cls_id in DEFAULT_MANUAL_NON_DIGIT_TARGET_NAMES:
                return DEFAULT_MANUAL_NON_DIGIT_TARGET_NAMES[cls_id]
        targets = self.config.get("decision_system", {}).get("targets", {})
        target_cfg = targets.get(cls_id) or targets.get(str(cls_id)) or {}
        if isinstance(target_cfg, dict):
            target_name = target_cfg.get("name")
            if target_name:
                return str(target_name)
        return f"class_{cls_id}"

    def _collect_decision_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        label_map = self._manual_label_class_map()

        for bbox in self._all_bbox_gps:
            lat = _coerce_optional_float(bbox.get("lat"))
            lon = _coerce_optional_float(bbox.get("lon"))
            if lat is None or lon is None:
                continue
            candidates.append(
                {
                    "source": "bbox",
                    "lat": lat,
                    "lon": lon,
                    "class_id": _parse_class_id(bbox.get("class")),
                    "conf": _coerce_optional_float(bbox.get("conf")),
                }
            )

        for row_annotations in self.annotations:
            for annotation in row_annotations:
                if annotation.lat is None or annotation.lon is None:
                    continue
                # Auto-generated center points are useful visual references, but
                # they should not influence the final target estimate unless the
                # operator explicitly saves them as manual annotations.
                if annotation.is_auto_center and not annotation.saved:
                    continue
                candidates.append(
                    {
                        "source": "manual_saved" if annotation.saved else "manual",
                        "lat": float(annotation.lat),
                        "lon": float(annotation.lon),
                        "class_id": _manual_label_to_class_id(
                            annotation.label, label_map
                        ),
                        "conf": 1.0,
                    }
                )

        return candidates

    def _estimate_target_from_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[float, float, str, float]:
        if not candidates:
            raise ValueError("No candidates available for target decision")

        lats = np.array(
            [candidate["lat"] for candidate in candidates], dtype=np.float64
        )
        lons = np.array(
            [candidate["lon"] for candidate in candidates], dtype=np.float64
        )

        geoloc_cfg = self.config.get("decision_system", {}).get("geoloc", {})
        manifold_center = geoloc_cfg.get("manifold_center")
        ref_lat: float
        ref_lon: float
        if (
            isinstance(manifold_center, (list, tuple))
            and len(manifold_center) >= 2
            and manifold_center[0] is not None
            and manifold_center[1] is not None
        ):
            ref_lat = float(manifold_center[0])
            ref_lon = float(manifold_center[1])
        else:
            ref_lat = float(np.mean(lats))
            ref_lon = float(np.mean(lons))

        reference_gps = GeoGPS(GPS(ref_lat, ref_lon))
        ne_vec = reference_gps.map(GeoGPS(GPS(lats, lons)))
        points = np.column_stack(
            [
                np.asarray(ne_vec.nord, dtype=np.float64),
                np.asarray(ne_vec.east, dtype=np.float64),
            ]
        )

        checker_cfg = self.config.get("manual_target_checker", {})
        manual_sigma_m = float(checker_cfg.get("manual_sigma_m", 2.0))
        bbox_sigma_m = float(checker_cfg.get("bbox_sigma_m", 8.0))
        conf_gain = float(checker_cfg.get("bbox_confidence_gain", 0.5))

        covariances = np.zeros((len(candidates), 2, 2), dtype=np.float64)
        for index, candidate in enumerate(candidates):
            source = str(candidate["source"])
            if source.startswith("manual"):
                sigma = manual_sigma_m
            else:
                conf = candidate.get("conf")
                if conf is None:
                    sigma = bbox_sigma_m
                else:
                    conf_value = min(max(float(conf), 0.05), 1.0)
                    sigma = bbox_sigma_m * (1.0 - conf_gain * conf_value)
            sigma = max(sigma, 0.1)
            covariances[index] = np.eye(2, dtype=np.float64) * (sigma**2)

        method = "single_candidate"
        if points.shape[0] == 1:
            centroid = points[0]
        else:
            decision_cfg = self.config.get("decision_system", {}).get("decision", {})
            decision = Decision(
                logging.getLogger("manual_target_checker.decision"),
                {
                    "horizon": float(decision_cfg.get("horizon", 0.99)),
                    "filter": float(decision_cfg.get("filter", 0.6)),
                },
            )
            decision.insert(points, covariances)
            try:
                centroids = decision.fit(n_clusters=1)
                if centroids.shape[0] == 0:
                    raise RuntimeError("Decision returned no centroids")
                centroid = centroids[0]
                method = "decision_clustering"
            except Exception as exc:
                weights = 1.0 / np.maximum(covariances[:, 0, 0], 1e-9)
                weights /= np.sum(weights)
                centroid = np.sum(points * weights[:, None], axis=0)
                method = f"weighted_mean_fallback:{exc.__class__.__name__}"

        spread_m = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
        final_gps = reference_gps.unmap(
            GeoGPS.Vector(
                np.array([centroid[0]], dtype=np.float64),
                np.array([centroid[1]], dtype=np.float64),
            )
        )
        lat = float(np.rad2deg(final_gps.lat).reshape(-1)[0])
        lon = float(np.rad2deg(final_gps.lon).reshape(-1)[0])
        return lat, lon, method, spread_m

    def _run_final_decision_if_needed(
        self, *, verbose: bool = True
    ) -> dict[str, Any] | None:
        candidates = self._collect_decision_candidates()
        if not candidates:
            if verbose:
                print("No bbox/manual points found; skipping final target decision.")
            return None

        bbox_count = sum(1 for candidate in candidates if candidate["source"] == "bbox")
        manual_count = sum(
            1
            for candidate in candidates
            if str(candidate["source"]).startswith("manual")
        )

        manual_labeled_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("source")).startswith("manual")
            and candidate.get("class_id") is not None
        ]

        # Build one cluster group per class.  When the operator has manually
        # labelled points, every distinct manual class is clustered separately
        # so the decision system emits a refined prediction for each target
        # (mirroring the real decision_system, which keys its output by class).
        # With no manual labels we fall back to the dominant class across all
        # (bbox) candidates, preserving the legacy single-target behaviour.
        class_groups: "OrderedDict[int | None, list[dict[str, Any]]]" = OrderedDict()
        if manual_labeled_candidates:
            selection_mode = "per_manual_class"
            manual_digit_context = True
            for candidate in sorted(
                manual_labeled_candidates, key=lambda c: c["class_id"]
            ):
                class_groups.setdefault(candidate["class_id"], []).append(candidate)
        else:
            manual_digit_context = False
            class_counts = Counter(
                candidate["class_id"]
                for candidate in candidates
                if candidate.get("class_id") is not None
            )
            if class_counts:
                selection_mode = "dominant_class_fallback"
                dominant_class = class_counts.most_common(1)[0][0]
                class_groups[dominant_class] = [
                    candidate
                    for candidate in candidates
                    if candidate.get("class_id") in (None, dominant_class)
                ]
            else:
                selection_mode = "all_candidates_fallback"
                class_groups[None] = list(candidates)

        real_target_position: tuple[float, float] | None = None
        real_target_config_error = ""
        try:
            real_target_position = real_target_location_from_config(self.config)
        except Exception as exc:
            real_target_config_error = str(exc)

        targets: list[dict[str, Any]] = []
        for class_id, group in class_groups.items():
            try:
                lat, lon, method, spread_m = self._estimate_target_from_candidates(
                    group
                )
            except Exception as exc:
                if verbose:
                    print(f"Final target decision failed for class {class_id}: {exc}")
                continue

            entry: dict[str, Any] = {
                "manual_target_id": class_id,
                "target_name": self._target_name_for_class(
                    class_id, manual_digit_context=manual_digit_context
                ),
                "candidate_count_used": len(group),
                "decision_method": method,
                "spread_m": spread_m,
                "target_position": {"lat": lat, "lon": lon},
            }
            if real_target_position is not None:
                real_lat, real_lon = real_target_position
                entry["real_target_position"] = {"lat": real_lat, "lon": real_lon}
                entry["target_error_m"] = gps_distance_m(lat, lon, real_lat, real_lon)
            targets.append(entry)

        if not targets:
            if verbose:
                print("Final target decision produced no targets.")
            return None

        report = {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "metadata_csv": str(self.metadata_path),
            "manual_output_csv": str(self.output_path),
            "target_id_namespace": "manual_target_checker",
            "selection_mode": selection_mode,
            "candidate_count_total": len(candidates),
            "bbox_candidate_count": bbox_count,
            "manual_candidate_count": manual_count,
            "targets": targets,
        }
        if real_target_config_error:
            report["real_target_config_error"] = real_target_config_error

        self._decision_report_path.parent.mkdir(parents=True, exist_ok=True)
        with self._decision_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if verbose:
            print("\nFinal target decision")
            print(
                f"  Selection mode: {selection_mode} - {len(targets)} target(s) "
                f"(bbox={bbox_count}, manual={manual_count})"
            )
            for entry in targets:
                class_id = entry["manual_target_id"]
                class_text = "n/a" if class_id is None else str(class_id)
                pos = entry["target_position"]
                print(
                    f"  - {entry['target_name']} (class {class_text}): "
                    f"lat={pos['lat']:.10f}, lon={pos['lon']:.10f} "
                    f"[{entry['decision_method']}, "
                    f"used={entry['candidate_count_used']}, "
                    f"spread={entry['spread_m']:.3f} m]"
                )
                if "target_error_m" in entry:
                    print(
                        f"      error to real target: "
                        f"{entry['target_error_m']:.3f} m"
                    )
            if real_target_config_error:
                print(f"  Real target config error: {real_target_config_error}")
            print(f"  Saved report: {self._decision_report_path}")

        csv_path = self.output_path.with_name("SFT_localization_report.csv")
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "latitude", "longitude"])
                for entry in targets:
                    class_id = entry["manual_target_id"]
                    if class_id is None:
                        continue
                    pos = entry["target_position"]
                    # The bullseye has no painted digit; report it by name
                    # ("BULLSEYE") instead of its internal numeric class id.
                    report_id = (
                        "BULLSEYE" if class_id == BULLSEYE_CLASS_ID else class_id
                    )
                    writer.writerow([report_id, pos["lat"], pos["lon"]])
            if verbose:
                print(f"  Saved CSV report: {csv_path}")
        except Exception as exc:
            if verbose:
                print(f"  Failed to save CSV report: {exc}")

        return report

    def run(self) -> None:
        if pygame is None:
            raise ValueError(
                "pygame is required for this tool. Install it in the project "
                "environment, for example: pdm add pygame"
            )

        pygame.init()
        pygame.key.set_repeat()
        pygame.display.set_caption("Manual Target Checker")
        screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        font = pygame.font.SysFont("Arial", 18)
        small_font = pygame.font.SysFont("Arial", 15)
        clock = pygame.time.Clock()

        try:
            print("Precomputing GPS coordinates for all bounding boxes…")
            self._precompute_all_bbox_gps()
            print(f"  → {len(self._all_bbox_gps)} bbox GPS points computed.")
            total, updated, failed = self._recompute_all_saved_points_geoloc(
                write_file=True,
                refresh_views=False,
                recompute_bboxes=False,
                update_status=False,
            )
            if total:
                print(
                    "Recomputed manual saved points from original pixels: "
                    f"{updated}/{total} updated, {failed} failed. "
                    f"Wrote {self.output_path.name}."
                )
            self._load_current_image()
            self._open_gps_plot()
            self._open_homography_plot()
            self._open_flight_metadata_window()
            self._open_flight_3d_plot()
            running = True
            frame = 0
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.window_size = (event.w, event.h)
                        self._constrain_pan()
                        screen = pygame.display.set_mode(
                            self.window_size, pygame.RESIZABLE
                        )
                    elif event.type == pygame.KEYDOWN:
                        if event.key in self._keys_down:
                            continue
                        self._keys_down.add(event.key)
                        running = self._handle_key(
                            event.key,
                            event.unicode,
                            is_repeat=bool(getattr(event, "repeat", 0)),
                        )
                    elif event.type == pygame.KEYUP:
                        self._keys_down.discard(event.key)
                    elif (
                        hasattr(pygame, "WINDOWFOCUSLOST")
                        and event.type == pygame.WINDOWFOCUSLOST
                    ):
                        self._keys_down.clear()
                    elif event.type == pygame.MOUSEWHEEL:
                        self._zoom_at(pygame.mouse.get_pos(), event.y)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self._handle_click(event.pos)
                        elif event.button == 3:
                            self._start_pan(event.pos)
                        elif event.button == 4:
                            self._zoom_at(event.pos, 1)
                        elif event.button == 5:
                            self._zoom_at(event.pos, -1)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 3:
                            self._stop_pan()
                    elif event.type == pygame.MOUSEMOTION:
                        self._handle_pan_motion(event.pos)

                self._draw(screen, font, small_font)
                pygame.display.flip()
                # Pump matplotlib events every ~10 frames to keep it responsive
                frame += 1
                if frame % 10 == 0 and plt is not None:
                    if self._gps_fig is not None:
                        try:
                            self._gps_fig.canvas.flush_events()
                        except Exception:
                            pass
                    if self._homography_fig is not None:
                        try:
                            self._homography_fig.canvas.flush_events()
                        except Exception:
                            pass
                    if self._flight3d_plot is not None:
                        self._flight3d_plot.flush_events()
                clock.tick(30)
        finally:
            if plt is not None:
                try:
                    plt.close("all")
                except Exception:
                    pass
            pygame.quit()
            self._run_final_decision_if_needed()
            self._close_bbox_db()

    def _close_bbox_db(self) -> None:
        if self._bbox_conn is not None:
            try:
                self._bbox_conn.close()
            except sqlite3.Error:
                pass
            self._bbox_conn = None

    def _ensure_center_point_for_current_image(self) -> None:
        loaded = self.loaded_image
        if loaded is None:
            return

        points = self._current_points()
        if any(annotation.is_auto_center for annotation in points):
            return

        center_x = (loaded.width - 1) / 2.0
        center_y = (loaded.height - 1) / 2.0
        lat, lon, error = self._geolocate_point_for_image(
            self.index,
            center_x,
            center_y,
            loaded.width,
            loaded.height,
        )
        center_annotation = Annotation(
            x_px=center_x,
            y_px=center_y,
            lat=lat,
            lon=lon,
            geoloc_error=error,
            is_auto_center=True,
        )

        if len(points) == 1 and _annotation_is_empty(points[0]):
            points[0] = center_annotation
            self.active_point_indices[self.index] = 0
            return
        points.append(center_annotation)

    def _load_current_image(self) -> None:
        row = self.rows[self.index]
        image_path = _resolve_image_path(row["filename"], self.metadata_path)
        pixel_format = row.get("pixel_format", "")
        self._reset_view()
        try:
            cache_key = (str(image_path.resolve()), pixel_format)
            cached = self._image_cache.get(cache_key)
            if cached is None:
                cached = load_image(image_path, pixel_format)
                self._image_cache[cache_key] = cached
                if len(self._image_cache) > self.image_cache_size:
                    self._image_cache.popitem(last=False)
            else:
                self._image_cache.move_to_end(cache_key)

            self.loaded_image = cached
            self.scaled_surface = None
            self.scaled_surface_key = None
            self.loaded_bboxes = self._load_bboxes()
            self._ensure_center_point_for_current_image()
            self._recompute_current_annotation_geoloc(include_saved=True)
            self.status = f"Loaded {image_path.name}"
        except Exception as exc:
            self.loaded_image = None
            self.scaled_surface = None
            self.scaled_surface_key = None
            self.loaded_bboxes = []
            self.status = f"Could not load {image_path}: {exc}"
        self._refresh_homography_plot()

    def _load_bboxes(self) -> list[dict[str, Any]]:
        if self._bbox_conn is None:
            return []
        image_filename = self.rows[self.index].get("filename", "")
        if not image_filename:
            return []
        try:
            rows = _iter_bbox_rows(
                self._bbox_conn, image_filename, self.producer_filter
            )
        except sqlite3.Error as e:
            print(
                f"Error querying bounding boxes from {self._bbox_db_path}: {e}",
                file=sys.stderr,
            )
            return []

        collected: list[dict[str, Any]] = []
        for row in rows:
            decoded = _decode_box_row(row)
            producer = decoded["producer"]
            for box in decoded["boxes"]:
                if not isinstance(box, dict):
                    continue
                enriched = dict(box)
                enriched["_producer"] = producer
                collected.append(enriched)
        return collected

    def _handle_key(self, key: int, unicode: str, *, is_repeat: bool = False) -> bool:
        if self.picker_active:
            if key == pygame.K_ESCAPE:
                self.picker_active = False
                self.status = "Save cancelled"
            return True

        if self.frame_jump_active:
            self._handle_frame_jump_key(key, unicode)
            return True

        # Ignore keyboard auto-repeat for navigation/toggle keys so one
        # physical press does not occasionally advance multiple images.
        if is_repeat and (
            key
            in (
                pygame.K_RIGHT,
                pygame.K_LEFT,
                pygame.K_UP,
                pygame.K_DOWN,
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_MINUS,
                pygame.K_KP_MINUS,
                pygame.K_EQUALS,
                pygame.K_KP_PLUS,
            )
            or unicode.lower() in {"m", "p", "h", "j", "u", "r"}
            or unicode in {"[", "]", "-", "+", "="}
        ):
            return True

        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_RIGHT:
            self._go(1)
            return True
        if key == pygame.K_LEFT:
            self._go(-1)
            return True
        if key == pygame.K_UP:
            self._save_and_cycle_point(-1)
            return True
        if key == pygame.K_DOWN:
            self._save_and_cycle_point(1)
            return True
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._save_current()
            return True

        if unicode.lower() == "m":
            self.show_model_bboxes = not self.show_model_bboxes
            self.status = f"Model bounding boxes {'shown' if self.show_model_bboxes else 'hidden'}"
            self._refresh_gps_plot()
            self._refresh_flight_3d_plot()
            return True

        if unicode.lower() == "p":
            self._cycle_producer_filter()
            return True
        if key == pygame.K_LEFTBRACKET:
            self._shift_metadata_offset(-1)
            return True
        if key == pygame.K_RIGHTBRACKET:
            self._shift_metadata_offset(1)
            return True
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS) or unicode in {"-", "_"}:
            self._shift_altitude_offset(-1.0)
            return True
        if key in (pygame.K_EQUALS, pygame.K_KP_PLUS) or unicode in {"+", "="}:
            self._shift_altitude_offset(1.0)
            return True
        if unicode.lower() == "u":
            self._set_metadata_offset(0)
            return True
        if unicode.lower() == "r":
            self._recompute_all_saved_points_geoloc()
            return True

        value = unicode.lower()
        if value in VALID_INPUTS:
            annotation = self._current_annotation()
            annotation.label = value
            annotation.saved = False
            self.status = (
                f"Input for point {self._current_point_number()} set to {value}"
            )
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.picker_active:
            for name, button in self.picker_buttons.items():
                if button.rect.collidepoint(pos):
                    if not self._allow_click_action(f"picker:{name}"):
                        return
                    if name == "cancel":
                        self.picker_active = False
                        self.status = "Save cancelled"
                    else:
                        self._execute_save(name)
                    return
            return

        if self.frame_jump_rect is not None and self.frame_jump_rect.collidepoint(pos):
            self.frame_jump_active = True
            self.frame_jump_text = ""
            self.status = "Type a metadata index and press Enter"
            return
        self.frame_jump_active = False

        for name, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                min_interval_ms = (
                    80
                    if name
                    in {
                        "altitude_down",
                        "altitude_up",
                        "error_angle_down",
                        "error_angle_up",
                    }
                    else 180
                )
                if not self._allow_click_action(
                    f"button:{name}",
                    min_interval_ms=min_interval_ms,
                ):
                    return
                if name == "prev":
                    self._go(-1)
                elif name == "next":
                    self._go(1)
                elif name == "save":
                    self._save_current()
                elif name == "add_point":
                    self._add_point()
                elif name == "prev_point":
                    self._cycle_point(-1)
                elif name == "next_point":
                    self._cycle_point(1)
                elif name == "altitude_down":
                    self._shift_altitude_offset(-1.0)
                elif name == "altitude_up":
                    self._shift_altitude_offset(1.0)
                elif name == "error_angle_down":
                    self._shift_px2body_error_angle(-1.0)
                elif name == "error_angle_up":
                    self._shift_px2body_error_angle(1.0)
                return

        loaded = self.loaded_image
        if loaded is None:
            return

        image_point = self._screen_to_image(pos)
        if image_point is None:
            return

        x_px, y_px = image_point

        lat, lon, error = self._geolocate_point_for_image(
            self.index,
            x_px,
            y_px,
            loaded.width,
            loaded.height,
        )
        annotation = self._current_annotation()
        annotation.x_px = x_px
        annotation.y_px = y_px
        annotation.lat = lat
        annotation.lon = lon
        annotation.geoloc_error = error
        annotation.saved = False
        if error:
            self.status = (
                f"Point {self._current_point_number()} set, GPS failed: {error}"
            )
        else:
            self.status = (
                f"Point {self._current_point_number()} set: "
                f"px=({x_px:.1f}, {y_px:.1f})"
            )
        self._refresh_gps_plot()
        self._refresh_flight_3d_plot()

    def _handle_frame_jump_key(self, key: int, unicode: str) -> None:
        if key == pygame.K_ESCAPE:
            self.frame_jump_active = False
            self.frame_jump_text = ""
            self.status = "Frame jump cancelled"
            return
        if key == pygame.K_BACKSPACE:
            self.frame_jump_text = self.frame_jump_text[:-1]
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._submit_frame_jump()
            return
        if unicode.isdigit():
            max_digits = max(1, len(str(len(self.rows))))
            if len(self.frame_jump_text) < max_digits:
                self.frame_jump_text += unicode

    def _submit_frame_jump(self) -> None:
        value = self.frame_jump_text.strip()
        if not value:
            self.frame_jump_active = False
            self.status = "Frame jump cancelled"
            return
        requested = int(value)
        clamped_index = max(0, min(requested, len(self.rows) - 1))
        self.frame_jump_active = False
        self.frame_jump_text = ""
        self._go_to_index(clamped_index)
        self.status = f"Jumped to metadata index {clamped_index}/{len(self.rows) - 1}"

    def _allow_click_action(self, action: str, *, min_interval_ms: int = 180) -> bool:
        now_ms = int(pygame.time.get_ticks())
        last_ms = self._last_click_action_ms.get(action)
        if last_ms is not None and (now_ms - last_ms) < min_interval_ms:
            return False
        self._last_click_action_ms[action] = now_ms
        return True

    def _geoloc_config(self) -> dict[str, Any]:
        decision_cfg = self.config.setdefault("decision_system", {})
        return decision_cfg.setdefault("geoloc", {})

    def _px2body_error_angle_deg(self) -> float:
        value = (
            self.config.get("decision_system", {})
            .get("geoloc", {})
            .get("px2body_error_angle_deg", 0.0)
        )
        if value in (None, ""):
            return 0.0
        return float(value)

    def _set_px2body_error_angle_deg(
        self,
        value: float,
        *,
        recompute_all_bboxes: bool,
        refresh_views: bool = True,
    ) -> None:
        current = self._px2body_error_angle_deg()
        new_value = float(value)
        if abs(new_value - current) < 1e-9:
            return
        self._geoloc_config()["px2body_error_angle_deg"] = new_value
        self._clear_geolocator_cache()
        if self._flight3d_plot is not None:
            self._flight3d_plot.invalidate()
        self._recompute_current_annotation_geoloc(include_saved=True)
        if recompute_all_bboxes:
            self._precompute_all_bbox_gps()
        if refresh_views:
            self._refresh_gps_plot()
            self._refresh_homography_plot()
            self._refresh_flight_3d_plot(rebuild_static=True)
        self.status = f"Px2body error angle: {new_value:.2f} deg"

    def _shift_px2body_error_angle(self, delta_deg: float) -> None:
        previous = self._px2body_error_angle_deg()
        self._set_px2body_error_angle_deg(
            previous + float(delta_deg),
            recompute_all_bboxes=False,
            refresh_views=False,
        )
        if abs(previous - self._px2body_error_angle_deg()) < 1e-9:
            return
        self._recompute_all_saved_points_geoloc(
            write_file=True,
            refresh_views=False,
            recompute_bboxes=False,
            update_status=False,
        )
        self._precompute_all_bbox_gps()
        self._refresh_gps_plot()
        self._refresh_homography_plot()
        self._refresh_flight_3d_plot(rebuild_static=True)
        self.status = f"Px2body error angle: {self._px2body_error_angle_deg():.2f} deg"

    def _clamp_altitude_offset(self, value: float) -> float:
        return min(
            max(float(value), self.altitude_offset_min_m),
            self.altitude_offset_max_m,
        )

    def _set_altitude_offset(
        self,
        value: float,
        *,
        recompute_all_bboxes: bool,
        refresh_views: bool = True,
    ) -> None:
        clamped = self._clamp_altitude_offset(value)
        if abs(clamped - self.altitude_offset_m) < 1e-6:
            return
        self.altitude_offset_m = clamped
        self._clear_geolocator_cache()
        self._recompute_current_annotation_geoloc(include_saved=True)
        if recompute_all_bboxes:
            self._precompute_all_bbox_gps()
        if refresh_views:
            self._refresh_gps_plot()
            self._refresh_homography_plot()
            self._refresh_flight_3d_plot(rebuild_static=True)
        self.status = f"Altitude offset: {self.altitude_offset_m:+.2f} m"

    def _shift_altitude_offset(self, delta_m: float) -> None:
        previous = self.altitude_offset_m
        self._set_altitude_offset(
            self.altitude_offset_m + float(delta_m),
            recompute_all_bboxes=False,
            refresh_views=False,
        )
        if abs(previous - self.altitude_offset_m) < 1e-6:
            self.status = (
                f"Altitude offset already at limit: {self.altitude_offset_m:+.2f} m"
            )
            return
        self._recompute_all_saved_points_geoloc(
            write_file=True,
            refresh_views=False,
            recompute_bboxes=False,
            update_status=False,
        )
        self._precompute_all_bbox_gps()
        self._refresh_gps_plot()
        self._refresh_homography_plot()
        self._refresh_flight_3d_plot(rebuild_static=True)

    def _go(self, delta: int) -> None:
        if not self.rows:
            return
        self._go_to_index((self.index + delta) % len(self.rows), prefetch_delta=delta)

    def _go_to_index(
        self,
        image_index: int,
        *,
        prefetch_delta: int | None = None,
    ) -> None:
        if not self.rows:
            return
        self.index = max(0, min(int(image_index), len(self.rows) - 1))
        self._load_current_image()
        self._refresh_gps_plot()
        self._refresh_flight_3d_plot()
        if prefetch_delta is not None:
            self._prefetch_image(self.index + prefetch_delta)

    def _prefetch_image(self, index: int) -> None:
        if not self.rows:
            return
        row = self.rows[index % len(self.rows)]
        image_filename = row.get("filename")
        if not image_filename:
            return
        image_path = _resolve_image_path(image_filename, self.metadata_path)
        pixel_format = row.get("pixel_format", "")
        cache_key = (str(image_path.resolve()), pixel_format)
        if cache_key in self._image_cache:
            self._image_cache.move_to_end(cache_key)
            return
        try:
            self._image_cache[cache_key] = load_image(image_path, pixel_format)
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        except Exception:
            # Best effort prefetch only; a missing/invalid neighbor file should
            # never disrupt current UI navigation.
            return

    def _save_current(self) -> bool:
        annotation = self._current_annotation()
        if annotation.x_px is None or annotation.y_px is None:
            self.status = "Click a target point before saving"
            return False

        if annotation.label:
            self.picker_cycle_delta = 0
            self._execute_save(annotation.label)
            return True

        self.picker_active = True
        self.picker_cycle_delta = 0
        return False

    def _execute_save(self, label: str) -> None:
        annotation = self._current_annotation()
        annotation.label = label
        annotation.saved = True
        annotation.saved_at = dt.datetime.now().isoformat(timespec="seconds")
        write_output(
            self.output_path,
            self.rows,
            self.input_fieldnames,
            self.annotations,
        )
        saved_point = self._current_point_number()
        self.status = (
            f"Saved point {saved_point} for index {self.index} "
            f"to {self.output_path}"
        )
        self._refresh_gps_plot()
        self._refresh_flight_3d_plot(rebuild_static=True)
        self.picker_active = False

        if self.picker_cycle_delta != 0:
            self._cycle_point(self.picker_cycle_delta)
            self.status = (
                f"Saved point {saved_point}; selected point "
                f"{self._current_point_number()} of {len(self._current_points())}"
            )

    def _save_and_cycle_point(self, delta: int) -> None:
        annotation = self._current_annotation()
        if annotation.x_px is None or annotation.y_px is None:
            self.status = "Click a target point before saving"
            return
        if annotation.label:
            self.picker_cycle_delta = delta
            self._execute_save(annotation.label)
            return
        self.picker_active = True
        self.picker_cycle_delta = delta

    def _current_points(self) -> list[Annotation]:
        points = self.annotations[self.index]
        if not points:
            points.append(Annotation())
            self.active_point_indices[self.index] = 0
        return points

    def _current_point_index(self) -> int:
        points = self._current_points()
        current_index = self.active_point_indices[self.index]
        current_index = min(max(current_index, 0), len(points) - 1)
        self.active_point_indices[self.index] = current_index
        return current_index

    def _current_point_number(self) -> int:
        return self._current_point_index() + 1

    def _current_annotation(self) -> Annotation:
        return self._current_points()[self._current_point_index()]

    def _add_point(self) -> None:
        points = self._current_points()
        points.append(Annotation())
        self.active_point_indices[self.index] = len(points) - 1
        self.status = f"Added point {len(points)}"

    def _cycle_point(self, delta: int) -> None:
        points = self._current_points()
        self.active_point_indices[self.index] = (
            self._current_point_index() + delta
        ) % len(points)
        self.status = f"Selected point {self._current_point_number()} of {len(points)}"

    def _cycle_producer_filter(self) -> None:
        self.producer_filter_index = (self.producer_filter_index + 1) % len(
            PRODUCER_FILTER_MODES
        )
        self.producer_filter = PRODUCER_FILTER_MODES[self.producer_filter_index]
        self.loaded_bboxes = self._load_bboxes()
        self._precompute_all_bbox_gps()
        self._refresh_gps_plot()
        self._refresh_flight_3d_plot(rebuild_static=True)
        self.status = f"Producer filter: {self.producer_filter}"

    def _recompute_all_saved_points_geoloc(
        self,
        *,
        write_file: bool = True,
        refresh_views: bool = True,
        recompute_bboxes: bool = True,
        update_status: bool = True,
    ) -> tuple[int, int, int]:
        total_saved = 0
        updated = 0
        failed = 0

        for image_index, image_row in enumerate(self.rows):
            points = self.annotations[image_index]
            saved_points = [
                annotation
                for annotation in points
                if annotation.saved
                and annotation.x_px is not None
                and annotation.y_px is not None
            ]
            if not saved_points:
                continue

            total_saved += len(saved_points)
            image_filename = image_row.get("filename", "")
            if not image_filename:
                for annotation in saved_points:
                    annotation.geoloc_error = "missing filename in metadata row"
                failed += len(saved_points)
                continue

            image_path = _resolve_image_path(image_filename, self.metadata_path)
            pixel_format = image_row.get("pixel_format", "")
            cache_key = (str(image_path.resolve()), pixel_format)
            cached = self._image_cache.get(cache_key)
            if cached is None:
                try:
                    cached = load_image(image_path, pixel_format)
                    self._image_cache[cache_key] = cached
                    while len(self._image_cache) > self.image_cache_size:
                        self._image_cache.popitem(last=False)
                except Exception as exc:
                    load_error = f"image load failed: {exc}"
                    for annotation in saved_points:
                        annotation.geoloc_error = load_error
                    failed += len(saved_points)
                    continue
            else:
                self._image_cache.move_to_end(cache_key)

            point_results = self._geolocate_points_for_image(
                image_index,
                (
                    (annotation.x_px, annotation.y_px)
                    for annotation in saved_points
                    if annotation.x_px is not None and annotation.y_px is not None
                ),
                cached.width,
                cached.height,
            )
            for annotation, (lat, lon, error) in zip(saved_points, point_results):
                if error or lat is None or lon is None:
                    annotation.geoloc_error = error or "unknown geolocation failure"
                    failed += 1
                else:
                    annotation.lat = lat
                    annotation.lon = lon
                    annotation.geoloc_error = ""
                    updated += 1

        if total_saved == 0:
            if update_status:
                self.status = "No saved points found to recompute."
            return total_saved, updated, failed

        if write_file:
            write_output(
                self.output_path,
                self.rows,
                self.input_fieldnames,
                self.annotations,
            )
        if recompute_bboxes:
            self._precompute_all_bbox_gps()
        if refresh_views:
            self._refresh_gps_plot()
            self._refresh_homography_plot()
            self._refresh_flight_3d_plot(rebuild_static=True)
        if update_status:
            write_text = f" Wrote {self.output_path.name}" if write_file else ""
            self.status = (
                f"Recomputed saved points: {updated}/{total_saved} updated, "
                f"{failed} failed.{write_text}"
            )
        return total_saved, updated, failed

    def _recompute_current_annotation_geoloc(
        self, *, include_saved: bool = False
    ) -> None:
        if self.loaded_image is None:
            return
        annotations = [
            annotation
            for annotation in self._current_points()
            if annotation.x_px is not None
            and annotation.y_px is not None
            and (include_saved or not annotation.saved)
        ]
        point_results = self._geolocate_points_for_image(
            self.index,
            (
                (annotation.x_px, annotation.y_px)
                for annotation in annotations
                if annotation.x_px is not None and annotation.y_px is not None
            ),
            self.loaded_image.width,
            self.loaded_image.height,
        )
        for annotation, (lat, lon, error) in zip(annotations, point_results):
            if annotation.x_px is None or annotation.y_px is None:
                continue
            annotation.lat = lat
            annotation.lon = lon
            annotation.geoloc_error = error

    def _reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_panning = False
        self.last_pan_pos = None
        self.scaled_surface = None
        self.scaled_surface_key = None

    def _image_area(self) -> Any:
        if (
            self._image_area_cache is not None
            and self._image_area_cache[0] == self.window_size
        ):
            return self._image_area_cache[1]
        width, height = self.window_size
        rect = pygame.Rect(18, 128, max(1, width - 36), max(1, height - 146))
        self._image_area_cache = (self.window_size, rect)
        return rect

    def _current_image_rect(self) -> tuple[Any, float]:
        if self.loaded_image is None:
            return pygame.Rect(0, 0, 0, 0), 1.0
        image_area = self._image_area()
        _, base_scale = fit_rect(
            self.loaded_image.width, self.loaded_image.height, image_area
        )
        scale = base_scale * self.zoom
        display_width = max(1, int(round(self.loaded_image.width * scale)))
        display_height = max(1, int(round(self.loaded_image.height * scale)))
        rect = pygame.Rect(0, 0, display_width, display_height)
        rect.center = (
            int(round(image_area.centerx + self.pan_x)),
            int(round(image_area.centery + self.pan_y)),
        )
        return rect, scale

    def _screen_to_image(self, pos: tuple[int, int]) -> tuple[float, float] | None:
        loaded = self.loaded_image
        if loaded is None or not self._image_area().collidepoint(pos):
            return None

        image_rect, scale = self._current_image_rect()
        if not image_rect.collidepoint(pos):
            return None

        x_px = (pos[0] - image_rect.x) / scale
        y_px = (pos[1] - image_rect.y) / scale
        if not (0 <= x_px < loaded.width and 0 <= y_px < loaded.height):
            return None
        return x_px, y_px

    def _constrain_pan(self) -> None:
        loaded = self.loaded_image
        if loaded is None:
            self.pan_x = 0.0
            self.pan_y = 0.0
            return

        image_area = self._image_area()
        _, base_scale = fit_rect(loaded.width, loaded.height, image_area)
        display_width = loaded.width * base_scale * self.zoom
        display_height = loaded.height * base_scale * self.zoom

        if display_width <= image_area.width:
            self.pan_x = 0.0
        else:
            max_pan_x = (display_width - image_area.width) / 2
            self.pan_x = min(max(self.pan_x, -max_pan_x), max_pan_x)

        if display_height <= image_area.height:
            self.pan_y = 0.0
        else:
            max_pan_y = (display_height - image_area.height) / 2
            self.pan_y = min(max(self.pan_y, -max_pan_y), max_pan_y)

    def _zoom_at(self, pos: tuple[int, int], wheel_delta: int) -> None:
        loaded = self.loaded_image
        if loaded is None or wheel_delta == 0:
            return
        image_area = self._image_area()
        if not image_area.collidepoint(pos):
            return

        old_rect, old_scale = self._current_image_rect()
        focus = self._screen_to_image(pos)
        if focus is None:
            focus = (
                min(
                    max((image_area.centerx - old_rect.x) / old_scale, 0.0),
                    loaded.width,
                ),
                min(
                    max((image_area.centery - old_rect.y) / old_scale, 0.0),
                    loaded.height,
                ),
            )

        old_zoom = self.zoom
        self.zoom = min(max(self.zoom * (ZOOM_STEP**wheel_delta), ZOOM_MIN), ZOOM_MAX)
        if self.zoom == old_zoom:
            return

        _, base_scale = fit_rect(loaded.width, loaded.height, image_area)
        new_scale = base_scale * self.zoom
        self.pan_x = pos[0] - focus[0] * new_scale + loaded.width * new_scale / 2
        self.pan_x -= image_area.centerx
        self.pan_y = pos[1] - focus[1] * new_scale + loaded.height * new_scale / 2
        self.pan_y -= image_area.centery
        self._constrain_pan()
        self.status = f"Zoom {self.zoom:.2f}x"

    def _start_pan(self, pos: tuple[int, int]) -> None:
        if self.loaded_image is None or not self._image_area().collidepoint(pos):
            return
        self.is_panning = True
        self.last_pan_pos = pos

    def _stop_pan(self) -> None:
        self.is_panning = False
        self.last_pan_pos = None

    def _handle_pan_motion(self, pos: tuple[int, int]) -> None:
        if not self.is_panning or self.last_pan_pos is None:
            return
        dx = pos[0] - self.last_pan_pos[0]
        dy = pos[1] - self.last_pan_pos[1]
        self.last_pan_pos = pos
        self.pan_x += dx
        self.pan_y += dy
        self._constrain_pan()

    def _draw(self, screen: Any, font: Any, small_font: Any) -> None:
        screen.fill((18, 22, 27))
        self._draw_toolbar(screen, font, small_font)
        loaded = self.loaded_image
        if loaded is None:
            draw_text(screen, font, self.status, (24, 140), (255, 175, 175))
            return

        image_area = self._image_area()
        image_rect, scale = self._current_image_rect()
        dest_rect = image_area.clip(image_rect)
        if dest_rect.width > 0 and dest_rect.height > 0:
            scaled = self._scaled_visible_image(loaded, image_rect, scale, dest_rect)
            screen.blit(scaled, dest_rect)
        pygame.draw.rect(screen, (85, 95, 108), image_area, width=1)

        old_clip = screen.get_clip()
        screen.set_clip(image_area)

        # Draw model-generated bounding boxes
        if self.show_model_bboxes:
            for box in self.loaded_bboxes:
                bbox = box.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                xmin, ymin, xmax, ymax = bbox
                s_xmin = image_rect.x + int(xmin * scale)
                s_ymin = image_rect.y + int(ymin * scale)
                s_xmax = image_rect.x + int(xmax * scale)
                s_ymax = image_rect.y + int(ymax * scale)
                s_width = s_xmax - s_xmin
                s_height = s_ymax - s_ymin
                if (
                    s_width <= 0
                    or s_height <= 0
                    or s_xmax < image_area.left
                    or s_xmin > image_area.right
                    or s_ymax < image_area.top
                    or s_ymin > image_area.bottom
                ):
                    continue

                producer = box.get("_producer")
                outline = PRODUCER_COLORS.get(producer, DEFAULT_PRODUCER_COLOR)
                label_bg = PRODUCER_LABEL_BG.get(producer, DEFAULT_PRODUCER_LABEL_BG)

                # Draw bounding box outline and diagonals
                pygame.draw.rect(
                    screen, outline, (s_xmin, s_ymin, s_width, s_height), width=2
                )
                pygame.draw.line(screen, outline, (s_xmin, s_ymin), (s_xmax, s_ymax), 1)
                pygame.draw.line(screen, outline, (s_xmin, s_ymax), (s_xmax, s_ymin), 1)

                # Draw class and confidence label
                class_val = box.get("class", "")
                conf_val = box.get("conf")
                cnn_label = box.get("cnn_label")
                label_text = f"cls: {class_val}"
                if conf_val is not None:
                    label_text += f" ({conf_val:.2f})"
                if cnn_label is not None:
                    label_text += f" -> {cnn_label}"

                text_surface = self._render_cached_text(
                    small_font,
                    label_text,
                    (255, 255, 255),
                )
                # Put label inside or above the box
                label_y = s_ymin - 20
                if label_y < image_area.y:
                    label_y = s_ymin + 4
                text_rect = text_surface.get_rect(topleft=(s_xmin + 4, label_y))
                bg_rect = text_rect.inflate(8, 4)
                pygame.draw.rect(screen, label_bg, bg_rect, border_radius=3)
                screen.blit(text_surface, text_rect)

        active_point_number = self._current_point_number()
        for point_index, annotation in enumerate(self._current_points(), start=1):
            if annotation.x_px is None or annotation.y_px is None:
                continue
            sx = image_rect.x + int(annotation.x_px * scale)
            sy = image_rect.y + int(annotation.y_px * scale)
            active = point_index == active_point_number
            color = (255, 64, 64) if active else (255, 210, 72)
            radius = 8 if active else 6
            pygame.draw.circle(screen, color, (sx, sy), radius, width=2)
            pygame.draw.line(screen, color, (sx - 12, sy), (sx + 12, sy), 2)
            pygame.draw.line(screen, color, (sx, sy - 12), (sx, sy + 12), 2)
            marker_label = str(point_index)
            if annotation.label:
                marker_label = f"{point_index}:{annotation.label}"
            elif annotation.is_auto_center:
                marker_label = f"{point_index}:C"
            marker_text = self._render_cached_text(small_font, marker_label, color)
            screen.blit(marker_text, (sx + 10, sy + 8))
        screen.set_clip(old_clip)

        if self.picker_active:
            self._draw_picker(screen, font)

    def _draw_picker(self, screen: Any, font: Any) -> None:
        width, height = self.window_size
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        box_width = 400
        box_height = 60 + len(self.available_classes) * 40 + 60
        box_rect = pygame.Rect(
            (width - box_width) // 2, (height - box_height) // 2, box_width, box_height
        )
        pygame.draw.rect(screen, (34, 40, 48), box_rect, border_radius=8)
        pygame.draw.rect(screen, (85, 95, 108), box_rect, width=1, border_radius=8)

        title = font.render("Select Target Class", True, (245, 248, 252))
        screen.blit(title, title.get_rect(center=(box_rect.centerx, box_rect.top + 30)))

        self.picker_buttons.clear()
        start_y = box_rect.top + 60
        for i, class_name in enumerate(self.available_classes):
            btn_rect = pygame.Rect(
                box_rect.left + 20, start_y + i * 40, box_width - 40, 32
            )
            btn = Button(class_name, btn_rect, class_name)
            self.picker_buttons[class_name] = btn
            draw_button(screen, font, btn)

        cancel_rect = pygame.Rect(
            box_rect.left + 20,
            start_y + len(self.available_classes) * 40 + 10,
            box_width - 40,
            32,
        )
        cancel_btn = Button("cancel", cancel_rect, "Cancel")
        self.picker_buttons["cancel"] = cancel_btn
        draw_button(screen, font, cancel_btn)

    def _scaled_visible_image(
        self, loaded: LoadedImage, image_rect: Any, scale: float, dest_rect: Any
    ) -> Any:
        src_left = min(
            max(0, int(math.floor((dest_rect.left - image_rect.left) / scale))),
            loaded.width - 1,
        )
        src_top = min(
            max(0, int(math.floor((dest_rect.top - image_rect.top) / scale))),
            loaded.height - 1,
        )
        src_right = min(
            loaded.width,
            max(
                src_left + 1,
                int(math.ceil((dest_rect.right - image_rect.left) / scale)),
            ),
        )
        src_bottom = min(
            loaded.height,
            max(
                src_top + 1,
                int(math.ceil((dest_rect.bottom - image_rect.top) / scale)),
            ),
        )
        src_rect = pygame.Rect(
            src_left,
            src_top,
            max(1, src_right - src_left),
            max(1, src_bottom - src_top),
        )
        key = (
            loaded.path,
            src_rect.x,
            src_rect.y,
            src_rect.width,
            src_rect.height,
            dest_rect.width,
            dest_rect.height,
        )
        if self.scaled_surface_key != key or self.scaled_surface is None:
            source = loaded.surface.subsurface(src_rect)
            self.scaled_surface = pygame.transform.smoothscale(source, dest_rect.size)
            self.scaled_surface_key = key
        return self.scaled_surface

    def _ensure_toolbar_buttons(self) -> None:
        if self.buttons:
            return
        self.buttons = {
            "prev": Button("prev", pygame.Rect(18, 18, 82, 38), "Prev"),
            "next": Button("next", pygame.Rect(108, 18, 76, 38), "Next"),
            "save": Button("save", pygame.Rect(192, 18, 68, 38), "Save"),
            "add_point": Button(
                "add_point", pygame.Rect(268, 18, 108, 38), "Add Point"
            ),
            "prev_point": Button("prev_point", pygame.Rect(384, 18, 44, 38), "<"),
            "next_point": Button("next_point", pygame.Rect(436, 18, 44, 38), ">"),
            "altitude_down": Button(
                "altitude_down", pygame.Rect(18, 70, 54, 28), "-1m"
            ),
            "altitude_up": Button("altitude_up", pygame.Rect(80, 70, 54, 28), "+1m"),
            "error_angle_down": Button(
                "error_angle_down", pygame.Rect(146, 70, 58, 28), "Err -1"
            ),
            "error_angle_up": Button(
                "error_angle_up", pygame.Rect(212, 70, 58, 28), "Err +1"
            ),
        }

    def _render_cached_text(
        self,
        font: Any,
        text: str,
        color: tuple[int, int, int],
    ) -> Any:
        key = (id(font), text, color)
        surface = self._label_surface_cache.get(key)
        if surface is not None:
            return surface
        if len(self._label_surface_cache) > 512:
            self._label_surface_cache.clear()
        surface = font.render(text, True, color)
        self._label_surface_cache[key] = surface
        return surface

    def _draw_toolbar(self, screen: Any, font: Any, small_font: Any) -> None:
        width, _ = self.window_size
        toolbar = pygame.Rect(0, 0, width, 116)
        pygame.draw.rect(screen, (28, 34, 42), toolbar)
        pygame.draw.line(screen, (70, 80, 94), (0, 115), (width, 115))

        self._ensure_toolbar_buttons()
        for button in self.buttons.values():
            draw_button(screen, font, button)

        annotation = self._current_annotation()
        point_count = len(self._current_points())
        point_number = self._current_point_number()
        image_row = self.rows[self.index]
        filename = truncate_middle(image_row.get("filename", ""), 58)
        saved = "saved" if annotation.saved else "not saved"
        point = "point: none"
        if annotation.x_px is not None and annotation.y_px is not None:
            point = f"point: ({annotation.x_px:.1f}, {annotation.y_px:.1f}) px"
        gps = "gps: none"
        if annotation.lat is not None and annotation.lon is not None:
            gps = f"gps: {annotation.lat:.7f}, {annotation.lon:.7f}"
        zoom = f"zoom: {self.zoom:.2f}x"

        draw_text(
            screen,
            font,
            f"{self.index}/{len(self.rows) - 1}  {filename}",
            (500, 14),
            (242, 246, 250),
        )
        draw_text(
            screen,
            small_font,
            (
                f"point: {point_number}/{point_count}   "
                f"input: {annotation.label or ('center' if annotation.is_auto_center else '-')}   "
                f"{point}   {gps}   {zoom}   {saved}   {self._metadata_offset_label()}"
            ),
            (500, 44),
            (190, 202, 216),
        )
        self._draw_frame_jump_input(screen, small_font)
        self._draw_altitude_stepper(screen, small_font)
        color = (255, 185, 118) if "failed" in self.status.lower() else (164, 214, 174)
        draw_text(screen, small_font, self.status, (500, 92), color)
        gps_state = (
            "[GPS map open]" if plt is not None else "[install matplotlib for GPS map]"
        )
        homography_state: str
        if not self._homography_enabled:
            homography_state = "[homography unavailable]"
        elif plt is None:
            homography_state = "[homography disabled: no matplotlib]"
        else:
            homography_state = "[homography window open]"
        gps_hint = (
            f"[-/=] alt ±1m  [m] bboxes  [p] producer: {self.producer_filter}  [h] mag->true  [j] true->mag  [[] dec  []] inc  [u] shift=0  [r] recalc saved"
            + f"  {gps_state}  {homography_state}"
        )
        draw_text(screen, small_font, gps_hint, (width - 520, 92), (145, 153, 163))

    def _draw_frame_jump_input(self, screen: Any, font: Any) -> None:
        label_pos = (282, 76)
        self.frame_jump_rect = pygame.Rect(334, 70, 76, 28)

        draw_text(screen, font, "Index", label_pos, (190, 202, 216))
        bg = (42, 50, 61) if self.frame_jump_active else (34, 40, 48)
        border = (116, 166, 235) if self.frame_jump_active else (85, 95, 108)
        text_color = (245, 248, 252) if self.frame_jump_active else (204, 214, 226)
        pygame.draw.rect(screen, bg, self.frame_jump_rect, border_radius=4)
        pygame.draw.rect(
            screen,
            border,
            self.frame_jump_rect,
            width=1,
            border_radius=4,
        )

        text = self.frame_jump_text if self.frame_jump_active else str(self.index)
        value_surface = self._render_cached_text(font, text, text_color)
        value_rect = value_surface.get_rect(
            midright=(self.frame_jump_rect.right - 8, self.frame_jump_rect.centery)
        )
        screen.blit(value_surface, value_rect)

        if self.frame_jump_active:
            caret_x = max(
                self.frame_jump_rect.left + 8,
                min(value_rect.right + 3, self.frame_jump_rect.right - 8),
            )
            pygame.draw.line(
                screen,
                text_color,
                (caret_x, self.frame_jump_rect.top + 7),
                (caret_x, self.frame_jump_rect.bottom - 7),
                1,
            )

    def _draw_altitude_stepper(self, screen: Any, font: Any) -> None:
        label_x = 422
        label_y = 76

        effective_alt_text = "n/a"
        try:
            row = self._metadata_row_for_current_image()
            base_alt = _float_from_row(
                row, ("alt_rel", "h", "altitude", "alt_msl", "alt")
            )
            effective_alt_text = f"{base_alt + self.altitude_offset_m:.2f} m"
        except Exception:
            pass
        draw_text(
            screen,
            font,
            (
                f"Alt offset: {self.altitude_offset_m:+.2f} m"
                f"  effective: {effective_alt_text}"
                f"  err angle: {self._px2body_error_angle_deg():.2f} deg"
                f"  range: {self.altitude_offset_min_m:+.0f}.."
                f"{self.altitude_offset_max_m:+.0f} m"
            ),
            (label_x, label_y),
            (190, 202, 216),
        )


def _spread_stats(spreads: list[float]) -> dict[str, float | int]:
    if not spreads:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "q3": 0.0,
        }
    if len(spreads) == 1:
        value = float(spreads[0])
        return {
            "count": 1,
            "mean": value,
            "median": value,
            "std": 0.0,
            "min": value,
            "max": value,
            "q1": value,
            "q3": value,
        }
    quartiles = st.quantiles(spreads, n=4, method="inclusive")
    return {
        "count": len(spreads),
        "mean": float(st.mean(spreads)),
        "median": float(st.median(spreads)),
        "std": float(st.pstdev(spreads)),
        "min": float(min(spreads)),
        "max": float(max(spreads)),
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
    }


def run_all_metadata_simulation(
    config_path: Path,
    window_size: tuple[int, int],
    off_by_n: int = 0,
    metadata_shift: int = 0,
) -> None:
    metadata_paths = _discover_all_metadata_paths()
    print(
        f"Running batch decision simulation on {len(metadata_paths)} metadata files..."
    )

    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, metadata_path in enumerate(metadata_paths, start=1):
        output_path = _output_path_for(metadata_path)
        print(f"[{index}/{len(metadata_paths)}] {metadata_path}")
        checker: ManualTargetChecker | None = None
        try:
            checker = ManualTargetChecker(
                metadata_path=metadata_path,
                output_path=output_path,
                config_path=config_path,
                window_size=window_size,
                off_by_n=off_by_n,
                metadata_shift=metadata_shift,
            )
            checker._precompute_all_bbox_gps()
            checker._recompute_all_saved_points_geoloc(
                write_file=False,
                refresh_views=False,
                recompute_bboxes=False,
                update_status=False,
            )
            report = checker._run_final_decision_if_needed(verbose=False)
            if report is None:
                print("  -> skipped (no candidates)")
                continue
            reports.append(report)
            target_entries = report.get("targets", [])
            used = sum(int(t.get("candidate_count_used", 0)) for t in target_entries)
            mean_spread = (
                st.mean(float(t["spread_m"]) for t in target_entries)
                if target_entries
                else 0.0
            )
            print(
                f"  -> targets={len(target_entries)}, "
                f"mean_spread_m={mean_spread:.6f}, candidates={used}"
            )
        except Exception as exc:
            failures.append({"metadata_csv": str(metadata_path), "error": str(exc)})
            print(f"  -> failed: {exc}")
        finally:
            if checker is not None:
                checker._close_bbox_db()

    spreads = [
        float(target["spread_m"])
        for report in reports
        for target in report.get("targets", [])
    ]
    stats = _spread_stats(spreads)
    print("\nOverall error summary (spread_m, meters)")
    print(f"  Processed: {len(reports)} / {len(metadata_paths)}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  IQR: [{stats['q1']:.6f}, {stats['q3']:.6f}]")

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "metadata_count": len(metadata_paths),
        "processed_count": len(reports),
        "failed_count": len(failures),
        "spread_m_stats": stats,
        "reports": reports,
        "failures": failures,
    }
    summary_path = DEFAULT_METADATA_DIR / "manual_decision_all_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved batch summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually label targets in images listed by a metadata CSV."
    )
    parser.add_argument(
        "-a",
        "--metadata",
        help=(
            "Input metadata CSV containing at least a filename column. "
            "Use 'all' to run every metadata*.csv in data/outs. "
            "Defaults to the first metadata*.csv in data/outs."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output metadata CSV. Defaults to <metadata-stem>_manual.csv.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "sft_config.yaml"),
        help="SFT config path used for camera/geoloc parameters.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run a headless simulation across all metadata*.csv files under "
            "data/outs and print overall spread/error statistics."
        ),
    )
    parser.add_argument(
        "--window-size",
        default="1280x900",
        help="Initial Pygame window size, for example 1280x900.",
    )
    parser.add_argument(
        "--off-by-n",
        type=int,
        default=0,
        help=(
            "Enable metadata shift controls in the range [-n, +n]. "
            "Use [ and ] to change shift, u to reset to 0. "
            "If --metadata-shift is outside this range, the range is auto-expanded."
        ),
    )
    parser.add_argument(
        "--metadata-shift",
        type=int,
        default=0,
        help=(
            "Initial metadata shift applied at startup "
            "(metadata_index = image_index + shift)."
        ),
    )
    parser.add_argument(
        "--homography-preview-scale",
        type=float,
        default=None,
        help=(
            "Scale for the rendered homography preview. Defaults to 0.5, "
            "or manual_target_checker.homography_preview_scale in the config."
        ),
    )
    return parser.parse_args()


def _parse_window_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", maxsplit=1)
        return max(640, int(width)), max(480, int(height))
    except Exception as exc:
        raise argparse.ArgumentTypeError("window size must look like 1280x900") from exc


def main() -> None:
    args = parse_args()
    try:
        metadata_arg_is_all = (
            bool(args.metadata) and args.metadata.strip().lower() == "all"
        )
        run_all_mode = args.all or metadata_arg_is_all
        config_path = Path(args.config).expanduser().resolve()
        window_size = _parse_window_size(args.window_size)
        off_by_n = max(0, int(args.off_by_n))
        metadata_shift = int(args.metadata_shift)

        if run_all_mode:
            if args.output:
                raise ValueError("--output cannot be used with --all")
            run_all_metadata_simulation(
                config_path=config_path,
                window_size=window_size,
                off_by_n=off_by_n,
                metadata_shift=metadata_shift,
            )
            return

        metadata_path = (
            Path(args.metadata).expanduser().resolve()
            if args.metadata
            else _default_metadata_path()
        )
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else _output_path_for(metadata_path)
        )
        checker = ManualTargetChecker(
            metadata_path=metadata_path,
            output_path=output_path,
            config_path=config_path,
            window_size=window_size,
            off_by_n=off_by_n,
            metadata_shift=metadata_shift,
            homography_preview_scale=args.homography_preview_scale,
        )
        checker.run()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"manual_target_checker: {exc}") from None


if __name__ == "__main__":
    main()
