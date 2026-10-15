import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "outs"
for import_path in (REPO_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    pygame = None

from common.config import SFTConfig
from sft_math.cam.px2body_matrix import resolve_px2body_matrix
from sft_math.cam import Camera
from sft_math.cam.affine import Affine

ROTATED_COLUMNS = [
    "homography_filename",
    "homography_status",
    "homography_width_px",
    "homography_height_px",
    "homography_meters_per_pixel",
    "homography_matrix",
]


@dataclass
class Button:
    name: str
    rect: Any
    label: str


@dataclass
class HomographyResult:
    matrix: np.ndarray
    source_corners_px: np.ndarray
    flat_corners_m: np.ndarray
    destination_corners_px: np.ndarray
    output_size: tuple[int, int]
    meters_per_pixel: float


@dataclass
class ProcessedImage:
    row_index: int
    image_path: Path
    original_rgb: np.ndarray | None = None
    warped_rgb: np.ndarray | None = None
    output_path: Path | None = None
    homography: HomographyResult | None = None
    status: str = ""


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

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else image_path


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


def _default_metadata_path() -> Path:
    metadata_paths = sorted(
        path
        for path in DEFAULT_METADATA_DIR.glob("metadata*.csv")
        if not path.stem.endswith("_manual") and not path.stem.endswith("_homographied")
    )
    if not metadata_paths:
        raise ValueError(f"No metadata*.csv files found in {DEFAULT_METADATA_DIR}")
    return metadata_paths[0].resolve()


def _output_dir_for(metadata_path: Path) -> Path:
    return metadata_path.with_name(f"{metadata_path.stem}_homographied")


def _output_metadata_path_for(metadata_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{metadata_path.stem}_homographied.csv"


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return SFTConfig(str(config_path), setup_logger=False).config_dict


def build_camera(config: dict[str, Any], image_width: int, image_height: int) -> Camera:
    geoloc_config = config.get("decision_system", {}).get("geoloc", {})
    camera_config = config.get("camera", {})

    focal_length = geoloc_config.get("focal_length")
    if focal_length is None:
        focal_length = camera_config.get("focal_length")
    if focal_length is None:
        focal_length = camera_config.get("focal_length_mm", 16.0)
    focal_length = float(focal_length)

    pixel_size = geoloc_config.get("pixel_size")
    if pixel_size is None:
        pixel_size = camera_config.get("pixel_size")
    if pixel_size is None:
        sensor_width = camera_config.get("sensor_width_mm")
        sensor_height = camera_config.get("sensor_height_mm")
        resolution = camera_config.get("resolution", (image_width, image_height))
        if sensor_width is not None and sensor_height is not None:
            width_um = float(sensor_width) * 1000.0 / float(resolution[0])
            height_um = float(sensor_height) * 1000.0 / float(resolution[1])
            pixel_size = (width_um + height_um) / 2.0
        else:
            pixel_size = 2.74
    pixel_size = float(pixel_size)

    px2body_matrix = resolve_px2body_matrix(geoloc_config, camera_config)

    optical_center = geoloc_config.get("optical_center")
    if optical_center is None:
        optical_center = geoloc_config.get("optic_center")
    if optical_center is None:
        optical_center = camera_config.get("optical_center")
    if optical_center is None:
        optical_center = camera_config.get("optic_center")
    if optical_center is None:
        optical_center = (image_width / 2, image_height / 2)

    return Camera(
        focal_length,
        pixel_size,
        (image_width, image_height),
        optical_center,
        px2body_matrix,
    )


def _metadata_angles(row: dict[str, str]) -> tuple[float, float, float]:
    roll = _float_from_row(row, ("roll", "phi"), default=0.0)
    pitch = _float_from_row(row, ("pitch", "theta"), default=0.0)

    yaw_value = _first_present(row, ("yaw", "psi"))
    yaw_is_degrees = False
    if yaw_value is None:
        yaw = _float_from_row(row, ("heading",), default=0.0)
        yaw_is_degrees = True
    else:
        yaw = float(yaw_value)

    values = (roll, pitch, yaw)
    values_are_degrees = yaw_is_degrees or any(
        abs(value) > 2 * math.pi for value in values
    )
    if values_are_degrees:
        values = tuple(math.radians(value) for value in values)
    return values[0], values[1], values[2]


def _altitude_from_row(row: dict[str, str]) -> float:
    return _float_from_row(
        row,
        ("h", "alt_rel", "altitude", "altitude_rel", "relative_altitude", "alt"),
    )


def _bayer_code(pixel_format: str) -> int | None:
    if cv2 is None:
        return None

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


def _image_array_to_uint8_rgb(array: np.ndarray, pixel_format: str = "") -> np.ndarray:
    if array.ndim == 2:
        array = _normalize_uint8(array)
        bayer_code = _bayer_code(pixel_format)
        if bayer_code is not None:
            return cv2.cvtColor(array, bayer_code)
        return np.repeat(array[:, :, None], 3, axis=2)

    if array.ndim == 3 and array.shape[2] >= 3:
        array = _normalize_uint8(array[:, :, :3])
        return array

    raise ValueError(f"Unsupported image array shape: {array.shape}")


def load_rgb_image(image_path: Path, pixel_format: str = "") -> np.ndarray:
    if cv2 is None:
        raise ValueError("opencv-python is required for image_rotator")

    if image_path.suffix.lower() == ".npy":
        return _image_array_to_uint8_rgb(np.load(image_path), pixel_format)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image {image_path}")
    image = _normalize_uint8(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image array shape: {image.shape}")


def save_rgb_image(output_path: Path, rgb: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".npy":
        np.save(output_path, rgb, allow_pickle=False)
        return

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), bgr):
        raise ValueError(f"Could not write image {output_path}")


def image_corners(width: int, height: int) -> np.ndarray:
    right = max(width - 1, 0)
    bottom = max(height - 1, 0)
    return np.array(
        [[0.0, 0.0], [right, 0.0], [right, bottom], [0.0, bottom]],
        dtype=np.float32,
    )


def flat_world_corners_m(
    row: dict[str, str],
    image_width: int,
    image_height: int,
    config: dict[str, Any],
) -> np.ndarray:
    altitude = _altitude_from_row(row)
    if altitude <= 0:
        raise ValueError(f"Altitude must be positive, got {altitude}")

    roll, pitch, yaw = _metadata_angles(row)
    camera = build_camera(config, image_width, image_height)
    configuration = Camera.Configuration(
        Affine((0.0, 0.0, -altitude), ((), (), ())),
        roll,
        pitch,
        yaw,
        is_deg=False,
    )
    corners = image_corners(image_width, image_height)
    corner_points = Affine(corners, np.empty((2, 0)))
    rays = camera.px2ned(corner_points, configuration)

    origins = np.asarray(rays.vector, dtype=np.float64)
    directions = np.asarray(
        rays.orthCoos2globalCoos_matrix[..., :, 0], dtype=np.float64
    )
    if np.any(np.abs(directions[:, 2]) < 1e-9):
        raise ValueError("At least one image corner ray is parallel to the ground")

    distance_along_ray = -origins[:, 2] / directions[:, 2]
    ground_ned = origins + directions * distance_along_ray[:, None]
    if not np.all(np.isfinite(ground_ned)):
        raise ValueError("Computed non-finite flat-world image corners")

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
        raise ValueError("Could not estimate meters per pixel")
    return float(np.median(candidates))


def _output_size_for(
    extent_m: np.ndarray, meters_per_pixel: float, padding_px: int
) -> tuple[int, int]:
    return (
        max(1, int(math.ceil(extent_m[0] / meters_per_pixel)) + 2 * padding_px),
        max(1, int(math.ceil(extent_m[1] / meters_per_pixel)) + 2 * padding_px),
    )


def build_flat_world_homography(
    row: dict[str, str],
    image_width: int,
    image_height: int,
    config: dict[str, Any],
    *,
    meters_per_pixel: float | None,
    max_output_side: int,
    padding_px: int,
) -> HomographyResult:
    if cv2 is None:
        raise ValueError("opencv-python is required for image_rotator")

    source_corners = image_corners(image_width, image_height)
    flat_corners = flat_world_corners_m(row, image_width, image_height, config)

    if meters_per_pixel is None:
        meters_per_pixel = estimate_meters_per_pixel(source_corners, flat_corners)
    if meters_per_pixel <= 0 or not np.isfinite(meters_per_pixel):
        raise ValueError(f"Invalid meters per pixel: {meters_per_pixel}")

    min_xy = flat_corners.min(axis=0)
    max_xy = flat_corners.max(axis=0)
    extent_m = max_xy - min_xy

    output_width, output_height = _output_size_for(
        extent_m, meters_per_pixel, padding_px
    )
    if max_output_side > 0 and max(output_width, output_height) > max_output_side:
        available_side = max(1, max_output_side - 2 * padding_px)
        meters_per_pixel = max(extent_m) / available_side
        output_width, output_height = _output_size_for(
            extent_m, meters_per_pixel, padding_px
        )

    destination_corners = (flat_corners - min_xy) / meters_per_pixel
    destination_corners += padding_px
    homography = cv2.getPerspectiveTransform(
        source_corners.astype(np.float32),
        destination_corners.astype(np.float32),
    )

    return HomographyResult(
        matrix=homography,
        source_corners_px=source_corners,
        flat_corners_m=flat_corners,
        destination_corners_px=destination_corners,
        output_size=(output_width, output_height),
        meters_per_pixel=meters_per_pixel,
    )


def warp_to_flat_world(rgb: np.ndarray, homography: HomographyResult) -> np.ndarray:
    width, height = homography.output_size
    return cv2.warpPerspective(
        rgb,
        homography.matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _matrix_to_csv(matrix: np.ndarray) -> str:
    return ";".join(f"{value:.12g}" for value in matrix.reshape(-1))


def write_output_metadata(
    output_path: Path,
    rows: list[dict[str, str]],
    input_fieldnames: list[str],
    rotated_rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = input_fieldnames + [
        column for column in ROTATED_COLUMNS if column not in input_fieldnames
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, rotated in zip(rows, rotated_rows):
            out_row = dict(row)
            out_row.update(rotated)
            writer.writerow(out_row)


def _surface_from_rgb(rgb: np.ndarray) -> Any:
    return pygame.surfarray.make_surface(np.ascontiguousarray(np.swapaxes(rgb, 0, 1)))


def fit_rect(source_width: int, source_height: int, area: Any) -> tuple[Any, float]:
    scale = min(area.width / source_width, area.height / source_height)
    scale = max(scale, 0.01)
    display_width = max(1, int(source_width * scale))
    display_height = max(1, int(source_height * scale))
    rect = pygame.Rect(0, 0, display_width, display_height)
    rect.center = area.center
    return rect, scale


def draw_button(screen: Any, font: Any, button: Button) -> None:
    bg = (52, 62, 74)
    border = (110, 125, 145)
    text_color = (245, 248, 252)
    pygame.draw.rect(screen, bg, button.rect, border_radius=6)
    pygame.draw.rect(screen, border, button.rect, width=1, border_radius=6)
    text = font.render(button.label, True, text_color)
    screen.blit(text, text.get_rect(center=button.rect.center))


def draw_text(
    screen: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int]
) -> None:
    screen.blit(font.render(text, True, color), pos)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


class ImageRotator:
    def __init__(
        self,
        metadata_path: Path,
        output_dir: Path,
        output_metadata_path: Path,
        config_path: Path,
        window_size: tuple[int, int],
        output_format: str,
        meters_per_pixel: float | None,
        max_output_side: int,
        padding_px: int,
        limit: int | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.output_metadata_path = output_metadata_path
        self.rows, self.input_fieldnames = load_metadata(metadata_path)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.config = load_config(config_path)
        self.window_size = window_size
        self.output_format = output_format
        self.meters_per_pixel = meters_per_pixel
        self.max_output_side = max_output_side
        self.padding_px = padding_px
        self.index = 0
        self.current: ProcessedImage | None = None
        self.buttons: dict[str, Button] = {}
        self.rotated_rows = [
            {column: "" for column in ROTATED_COLUMNS} for _ in self.rows
        ]

    def process_all(self) -> None:
        for index in range(len(self.rows)):
            self._process_index(index)

    def run(self) -> None:
        if pygame is None:
            raise ValueError(
                "pygame is required for the preview window. Install it in the "
                "project environment, for example: pdm add pygame"
            )

        pygame.init()
        pygame.key.set_repeat(280, 95)
        pygame.display.set_caption("Image Rotator")
        screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        font = pygame.font.SysFont("Arial", 18)
        small_font = pygame.font.SysFont("Arial", 15)
        clock = pygame.time.Clock()

        try:
            self._load_current()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.window_size = (event.w, event.h)
                        screen = pygame.display.set_mode(
                            self.window_size, pygame.RESIZABLE
                        )
                    elif event.type == pygame.KEYDOWN:
                        running = self._handle_key(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)

                self._draw(screen, font, small_font)
                pygame.display.flip()
                clock.tick(30)
        finally:
            pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key in (pygame.K_RIGHT, pygame.K_d, pygame.K_SPACE):
            self._go(1)
            return True
        if key in (pygame.K_LEFT, pygame.K_a):
            self._go(-1)
            return True
        if key == pygame.K_r:
            self._load_current()
            return True
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for name, button in self.buttons.items():
            if not button.rect.collidepoint(pos):
                continue
            if name == "prev":
                self._go(-1)
            elif name == "next":
                self._go(1)
            elif name == "reload":
                self._load_current()
            return

    def _go(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.rows)
        self._load_current()

    def _load_current(self) -> None:
        self.current = self._process_index(self.index)

    def _process_index(self, index: int) -> ProcessedImage:
        row = self.rows[index]
        image_path = _resolve_image_path(row["filename"], self.metadata_path)
        processed = ProcessedImage(row_index=index, image_path=image_path)
        try:
            original_rgb = load_rgb_image(image_path, row.get("pixel_format", ""))
            processed.original_rgb = original_rgb
        except Exception as exc:
            processed.status = f"Could not load {image_path}: {exc}"
            self._mark_row(index, status=processed.status)
            return processed

        height, width = original_rgb.shape[:2]
        try:
            homography = build_flat_world_homography(
                row,
                width,
                height,
                self.config,
                meters_per_pixel=self.meters_per_pixel,
                max_output_side=self.max_output_side,
                padding_px=self.padding_px,
            )
            warped_rgb = warp_to_flat_world(original_rgb, homography)
        except Exception as exc:
            processed.status = f"Could not homography {image_path.name}: {exc}"
            self._mark_row(index, status=processed.status)
            return processed

        output_path = self._output_path_for(index, image_path)
        try:
            save_rgb_image(output_path, warped_rgb)
        except Exception as exc:
            processed.status = f"Could not save {output_path}: {exc}"
            self._mark_row(index, status=processed.status)
            return processed

        processed.warped_rgb = warped_rgb
        processed.output_path = output_path
        processed.homography = homography
        processed.status = (
            f"Saved {output_path.name} "
            f"({homography.output_size[0]}x{homography.output_size[1]}, "
            f"{homography.meters_per_pixel:.4f} m/px)"
        )
        self._mark_row(
            index,
            filename=str(output_path),
            status="ok",
            width=homography.output_size[0],
            height=homography.output_size[1],
            meters_per_pixel=homography.meters_per_pixel,
            matrix=homography.matrix,
        )
        return processed

    def _output_path_for(self, index: int, image_path: Path) -> Path:
        suffix = ".npy" if self.output_format == "npy" else ".png"
        return self.output_dir / f"{index:05d}_{image_path.stem}_homographied{suffix}"

    def _mark_row(
        self,
        index: int,
        *,
        filename: str = "",
        status: str,
        width: int | None = None,
        height: int | None = None,
        meters_per_pixel: float | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        self.rotated_rows[index] = {
            "homography_filename": filename,
            "homography_status": status,
            "homography_width_px": str(width or ""),
            "homography_height_px": str(height or ""),
            "homography_meters_per_pixel": (
                f"{meters_per_pixel:.8f}" if meters_per_pixel is not None else ""
            ),
            "homography_matrix": _matrix_to_csv(matrix) if matrix is not None else "",
        }
        write_output_metadata(
            self.output_metadata_path,
            self.rows,
            self.input_fieldnames,
            self.rotated_rows,
        )

    def _draw(self, screen: Any, font: Any, small_font: Any) -> None:
        screen.fill((18, 22, 27))
        self._draw_toolbar(screen, font, small_font)
        self._draw_images(screen, font, small_font)

    def _draw_toolbar(self, screen: Any, font: Any, small_font: Any) -> None:
        width, _ = self.window_size
        toolbar = pygame.Rect(0, 0, width, 92)
        pygame.draw.rect(screen, (28, 34, 42), toolbar)
        pygame.draw.line(screen, (70, 80, 94), (0, 91), (width, 91))

        self.buttons = {
            "prev": Button("prev", pygame.Rect(18, 18, 82, 38), "Prev"),
            "next": Button("next", pygame.Rect(108, 18, 76, 38), "Next"),
            "reload": Button("reload", pygame.Rect(192, 18, 92, 38), "Reload"),
        }
        for button in self.buttons.values():
            draw_button(screen, font, button)

        current = self.current
        filename = ""
        status = ""
        if current is not None:
            filename = truncate_middle(str(current.image_path), 72)
            status = current.status

        draw_text(
            screen,
            font,
            f"{self.index + 1}/{len(self.rows)}  {filename}",
            (304, 14),
            (242, 246, 250),
        )
        color = (255, 185, 118) if "could not" in status.lower() else (164, 214, 174)
        draw_text(screen, small_font, status, (304, 44), color)
        draw_text(
            screen,
            small_font,
            f"metadata: {self.output_metadata_path}",
            (18, 68),
            (190, 202, 216),
        )

    def _draw_images(self, screen: Any, font: Any, small_font: Any) -> None:
        current = self.current
        if current is None:
            return

        width, height = self.window_size
        gap = 14
        top = 116
        pane_height = max(1, height - top - 18)
        pane_width = max(1, (width - 36 - gap) // 2)
        left_area = pygame.Rect(18, top, pane_width, pane_height)
        right_area = pygame.Rect(18 + pane_width + gap, top, pane_width, pane_height)

        self._draw_panel(
            screen,
            font,
            small_font,
            left_area,
            "Original",
            current.original_rgb,
        )
        self._draw_panel(
            screen,
            font,
            small_font,
            right_area,
            "Flat-world homography",
            current.warped_rgb,
        )

    def _draw_panel(
        self,
        screen: Any,
        font: Any,
        small_font: Any,
        area: Any,
        title: str,
        image: np.ndarray | None,
    ) -> None:
        pygame.draw.rect(screen, (85, 95, 108), area, width=1)
        draw_text(screen, font, title, (area.x, area.y - 26), (242, 246, 250))
        if image is None:
            draw_text(
                screen,
                small_font,
                "No image",
                (area.x + 12, area.y + 12),
                (255, 185, 118),
            )
            return

        image_height, image_width = image.shape[:2]
        rect, _ = fit_rect(image_width, image_height, area.inflate(-12, -12))
        surface = _surface_from_rgb(image)
        if rect.size != surface.get_size():
            surface = pygame.transform.smoothscale(surface, rect.size)
        screen.blit(surface, rect)


def process_all_without_window(rotator: ImageRotator) -> None:
    rotator.process_all()
    ok = sum(1 for row in rotator.rotated_rows if row["homography_status"] == "ok")
    print(
        f"Processed {len(rotator.rows)} rows, saved {ok} images to "
        f"{rotator.output_dir}"
    )
    print(f"Wrote metadata to {rotator.output_metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate a metadata CSV, project each image onto a flat-world ground "
            "plane from h/yaw/pitch/roll, save the warped image, and preview it."
        )
    )
    parser.add_argument(
        "-m",
        "--metadata",
        help=(
            "Input metadata CSV containing at least filename plus h/alt_rel, "
            "roll, pitch, and yaw. Defaults to the first metadata*.csv in data/outs."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory for homographied images. Defaults to <metadata-stem>_homographied.",
    )
    parser.add_argument(
        "--output-metadata",
        help="Output metadata CSV. Defaults inside the output directory.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "sft_config.yaml"),
        help="SFT config path used for camera intrinsics.",
    )
    parser.add_argument(
        "--window-size",
        default="1440x900",
        help="Initial Pygame window size, for example 1440x900.",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "npy"),
        default="png",
        help="Saved homographied image format.",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        help="Override output scale. Defaults to an estimate from the ground footprint.",
    )
    parser.add_argument(
        "--max-output-side",
        type=int,
        default=4096,
        help="Clamp the largest saved image side. Use 0 to disable.",
    )
    parser.add_argument(
        "--padding-px",
        type=int,
        default=16,
        help="Padding around the warped ground footprint.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only use the first N metadata rows.",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process every row immediately instead of opening the preview window.",
    )
    return parser.parse_args()


def _parse_window_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", maxsplit=1)
        return max(640, int(width)), max(480, int(height))
    except Exception as exc:
        raise argparse.ArgumentTypeError("window size must look like 1440x900") from exc


def main() -> None:
    args = parse_args()
    try:
        metadata_path = (
            Path(args.metadata).expanduser().resolve()
            if args.metadata
            else _default_metadata_path()
        )
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else _output_dir_for(metadata_path)
        )
        output_metadata_path = (
            Path(args.output_metadata).expanduser().resolve()
            if args.output_metadata
            else _output_metadata_path_for(metadata_path, output_dir)
        )
        rotator = ImageRotator(
            metadata_path=metadata_path,
            output_dir=output_dir,
            output_metadata_path=output_metadata_path,
            config_path=Path(args.config).expanduser().resolve(),
            window_size=_parse_window_size(args.window_size),
            output_format=args.output_format,
            meters_per_pixel=args.meters_per_pixel,
            max_output_side=args.max_output_side,
            padding_px=max(0, args.padding_px),
            limit=args.limit,
        )
        if args.process_all:
            process_all_without_window(rotator)
        else:
            rotator.run()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"image_rotator: {exc}") from None


if __name__ == "__main__":
    main()
