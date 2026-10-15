#!/usr/bin/env python3
"""
Standalone first-version red-filled circle detector.

By default this scans the example captures in data/outs, reads the matching
metadata CSV, demosaics BayerRG/BayerRG8 frames, detects circles by shape, and
then keeps only circles whose internal area is mostly red.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    pygame = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "outs"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".npy", ".png", ".webp"}
VIEWER_TOOLBAR_HEIGHT = 138


@dataclass(frozen=True)
class CircleColor:
    name: str
    confidence: float
    mean_rgb: tuple[int, int, int]
    median_rgb: tuple[int, int, int]
    mean_hsv: tuple[int, int, int]


@dataclass(frozen=True)
class CircleDetection:
    center_x: float
    center_y: float
    radius_px: float
    bbox_px: tuple[int, int, int, int]
    edge_support: float
    internal_color: CircleColor
    red_fill_ratio: float = 0.0
    circularity: float = 0.0
    source: str = "unknown"


@dataclass(frozen=True)
class Button:
    name: str
    rect: Any
    label: str


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_default_metadata(data_dir: Path) -> Path | None:
    metadata_files = sorted(
        path
        for path in data_dir.glob("metadata*.csv")
        if not path.name.startswith("._") and "manual" not in path.stem
    )
    if metadata_files:
        return metadata_files[0]
    metadata_files = sorted(
        path for path in data_dir.glob("metadata*.csv") if not path.name.startswith("._")
    )
    return metadata_files[0] if metadata_files else None


def load_metadata(metadata_path: Path | None) -> dict[str, dict[str, str]]:
    if metadata_path is None:
        return {}

    rows_by_name: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename", "")
            if not filename:
                continue
            rows_by_name[Path(filename).name] = row
    return rows_by_name


def iter_image_paths(path: Path, metadata: dict[str, dict[str, str]]) -> list[Path]:
    if path.is_file():
        return [path]

    if metadata:
        paths: list[Path] = []
        for filename in metadata:
            candidate = path / filename
            if candidate.exists():
                paths.append(candidate)
        return paths

    return sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file()
            and item.suffix.lower() in IMAGE_EXTENSIONS
            and not item.name.startswith("._")
        ),
        key=natural_sort_key,
    )


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.size == 0:
        return image.astype(np.uint8)

    image_float = image.astype(np.float32)
    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image_float - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def bayer_to_bgr(
    image: np.ndarray,
    pixel_format: str,
    bayer_mode: str = "metadata",
) -> np.ndarray:
    if bayer_mode == "manual":
        return cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)

    normalized_format = pixel_format.strip().upper()
    conversion_codes = {
        "BAYERRG": cv2.COLOR_BayerRG2BGR,
        "BAYERRG8": cv2.COLOR_BayerRG2BGR,
        "BAYERBG": cv2.COLOR_BayerBG2BGR,
        "BAYERBG8": cv2.COLOR_BayerBG2BGR,
        "BAYERGR": cv2.COLOR_BayerGR2BGR,
        "BAYERGR8": cv2.COLOR_BayerGR2BGR,
        "BAYERGB": cv2.COLOR_BayerGB2BGR,
        "BAYERGB8": cv2.COLOR_BayerGB2BGR,
    }
    conversion_code = conversion_codes.get(normalized_format)
    if conversion_code is None:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, conversion_code)


def load_image_bgr(
    image_path: Path,
    metadata: dict[str, str] | None = None,
    bayer_mode: str = "metadata",
) -> np.ndarray:
    metadata = metadata or {}
    pixel_format = metadata.get("pixel_format", "")

    if image_path.suffix.lower() == ".npy":
        image = np.load(image_path)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

    image = to_uint8(image)

    if image.ndim == 2:
        return bayer_to_bgr(image, pixel_format, bayer_mode=bayer_mode)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for {image_path}: {image.shape}")

    image_3ch = image[:, :, :3]
    if pixel_format.upper().startswith("RGB"):
        return cv2.cvtColor(image_3ch, cv2.COLOR_RGB2BGR)
    return image_3ch


def circle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        max(1, int(round(radius))),
        255,
        thickness=cv2.FILLED,
    )
    return mask


def mask_fill_ratio(mask: np.ndarray, center: tuple[float, float], radius: float) -> float:
    filled_circle = circle_mask(mask.shape[:2], center, radius)
    circle_pixels = int(np.count_nonzero(filled_circle))
    if circle_pixels == 0:
        return 0.0
    filled_pixels = int(np.count_nonzero(cv2.bitwise_and(mask, filled_circle)))
    return filled_pixels / circle_pixels


def red_like_mask(
    image_bgr: np.ndarray,
    hue_low: int,
    saturation_min: int,
    value_min: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (
        ((hue <= 12) | (hue >= int(hue_low)))
        & (saturation >= int(saturation_min))
        & (value >= int(value_min))
    ).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)


def classify_pixels(hsv_pixels: np.ndarray) -> tuple[str, float]:
    if hsv_pixels.size == 0:
        return "unknown", 0.0

    hue = hsv_pixels[:, 0]
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]
    labels = np.full(hue.shape, "unknown", dtype=object)

    labels[value < 45] = "black"
    labels[(value >= 45) & (saturation < 35) & (value > 185)] = "white"
    labels[(value >= 45) & (saturation < 35) & (value <= 185)] = "gray"

    colorful = (saturation >= 35) & (value >= 45)
    labels[colorful & ((hue <= 10) | (hue >= 170))] = "red"
    labels[colorful & (hue > 10) & (hue <= 24)] = "orange"
    labels[colorful & (hue > 24) & (hue <= 34)] = "yellow"
    labels[colorful & (hue > 34) & (hue <= 84)] = "green"
    labels[colorful & (hue > 84) & (hue <= 99)] = "cyan"
    labels[colorful & (hue > 99) & (hue <= 129)] = "blue"
    labels[colorful & (hue > 129) & (hue < 170)] = "purple"

    names, counts = np.unique(labels, return_counts=True)
    best_index = int(np.argmax(counts))
    return str(names[best_index]), float(counts[best_index] / labels.size)


def estimate_internal_color(
    image_bgr: np.ndarray,
    center: tuple[float, float],
    radius: float,
    inner_radius_ratio: float,
) -> CircleColor:
    mask = circle_mask(image_bgr.shape[:2], center, radius * inner_radius_ratio)
    pixels_bgr = image_bgr[mask > 0]
    if pixels_bgr.size == 0:
        return CircleColor(
            name="unknown",
            confidence=0.0,
            mean_rgb=(0, 0, 0),
            median_rgb=(0, 0, 0),
            mean_hsv=(0, 0, 0),
        )

    pixels_rgb = pixels_bgr[:, ::-1]
    pixels_hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(
        -1, 3
    )
    name, confidence = classify_pixels(pixels_hsv)
    mean_rgb = tuple(int(round(value)) for value in np.mean(pixels_rgb, axis=0))
    median_rgb = tuple(int(round(value)) for value in np.median(pixels_rgb, axis=0))
    mean_hsv = tuple(int(round(value)) for value in np.mean(pixels_hsv, axis=0))
    return CircleColor(
        name=name,
        confidence=confidence,
        mean_rgb=mean_rgb,
        median_rgb=median_rgb,
        mean_hsv=mean_hsv,
    )


def edge_support(
    edges: np.ndarray,
    center: tuple[float, float],
    radius: float,
    samples: int = 96,
) -> float:
    height, width = edges.shape[:2]
    center_x, center_y = center
    hits = 0
    valid_samples = 0

    for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
        point_x = int(round(center_x + np.cos(angle) * radius))
        point_y = int(round(center_y + np.sin(angle) * radius))
        if not (1 <= point_x < width - 1 and 1 <= point_y < height - 1):
            continue
        valid_samples += 1
        patch = edges[point_y - 1 : point_y + 2, point_x - 1 : point_x + 2]
        if np.any(patch > 0):
            hits += 1

    if valid_samples == 0:
        return 0.0
    return hits / valid_samples


def detect_circles(
    image_bgr: np.ndarray,
    min_radius: int = 8,
    max_radius: int = 0,
    min_dist: float = 30.0,
    hough_dp: float = 1.2,
    hough_param1: float = 100.0,
    hough_param2: float = 28.0,
    min_edge_support: float = 0.10,
    inner_radius_ratio: float = 0.65,
) -> list[CircleDetection]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 80, 160)

    raw_circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=hough_dp,
        minDist=min_dist,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if raw_circles is None:
        return []

    detections: list[CircleDetection] = []
    image_height, image_width = image_bgr.shape[:2]
    for raw_circle in np.round(raw_circles[0, :]).astype(int):
        center_x, center_y, radius = raw_circle.tolist()
        if radius <= 0:
            continue
        if (
            center_x - radius < 0
            or center_y - radius < 0
            or center_x + radius >= image_width
            or center_y + radius >= image_height
        ):
            continue

        support = edge_support(edges, (center_x, center_y), radius)
        if support < min_edge_support:
            continue

        bbox = (
            int(center_x - radius),
            int(center_y - radius),
            int(radius * 2),
            int(radius * 2),
        )
        color = estimate_internal_color(
            image_bgr,
            (float(center_x), float(center_y)),
            float(radius),
            inner_radius_ratio=inner_radius_ratio,
        )
        detections.append(
            CircleDetection(
                center_x=float(center_x),
                center_y=float(center_y),
                radius_px=float(radius),
                bbox_px=bbox,
                edge_support=float(support),
                internal_color=color,
                red_fill_ratio=float(color.confidence if color.name == "red" else 0.0),
                circularity=0.0,
                source="hough",
            )
        )

    return sorted(detections, key=lambda detection: detection.edge_support, reverse=True)


def detect_red_filled_circles(
    image_bgr: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_red_confidence: float,
    min_circularity: float,
    min_blob_fill_ratio: float,
    red_hue_low: int,
    red_saturation_min: int,
    red_value_min: int,
) -> list[CircleDetection]:
    mask = red_like_mask(
        image_bgr,
        hue_low=red_hue_low,
        saturation_min=red_saturation_min,
        value_min=red_value_min,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[CircleDetection] = []
    image_height, image_width = image_bgr.shape[:2]
    min_area = max(50.0, np.pi * float(min_radius) * float(min_radius) * 0.35)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius:
            continue
        if max_radius > 0 and radius > max_radius:
            continue
        if (
            center_x - radius < 0
            or center_y - radius < 0
            or center_x + radius >= image_width
            or center_y + radius >= image_height
        ):
            continue

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
        aspect_ratio = max(bbox_w, bbox_h) / max(1.0, min(bbox_w, bbox_h))
        if aspect_ratio > 1.25:
            continue

        enclosing_fill_ratio = float(area / max(1.0, np.pi * radius * radius))
        if enclosing_fill_ratio < min_blob_fill_ratio:
            continue

        red_fill_ratio = mask_fill_ratio(mask, (center_x, center_y), radius * 0.90)
        if red_fill_ratio < min_red_confidence:
            continue

        sampled_color = estimate_internal_color(
            image_bgr,
            (float(center_x), float(center_y)),
            float(radius),
            inner_radius_ratio=0.65,
        )
        internal_color = CircleColor(
            name="red",
            confidence=float(red_fill_ratio),
            mean_rgb=sampled_color.mean_rgb,
            median_rgb=sampled_color.median_rgb,
            mean_hsv=sampled_color.mean_hsv,
        )
        detections.append(
            CircleDetection(
                center_x=float(center_x),
                center_y=float(center_y),
                radius_px=float(radius),
                bbox_px=(int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
                edge_support=float(circularity),
                internal_color=internal_color,
                red_fill_ratio=float(red_fill_ratio),
                circularity=float(circularity),
                source="red_blob",
            )
        )

    return sorted(
        detections,
        key=lambda detection: (
            detection.red_fill_ratio,
            detection.circularity,
            detection.radius_px,
        ),
        reverse=True,
    )


def draw_detections(
    image_bgr: np.ndarray, detections: list[CircleDetection]
) -> np.ndarray:
    output = image_bgr.copy()
    for detection in detections:
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        radius = int(round(detection.radius_px))
        cv2.circle(output, center, radius, (0, 255, 0), 3)
        bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox_px
        cv2.rectangle(
            output,
            (bbox_x, bbox_y),
            (bbox_x + bbox_w, bbox_y + bbox_h),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            output,
            f"{detection.internal_color.name} {detection.edge_support:.2f}",
            (bbox_x, max(25, bbox_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return output


def bgr_to_surface(image_bgr: np.ndarray) -> Any:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.swapaxes(image_rgb, 0, 1))


def draw_text(
    screen: Any,
    font: Any,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (235, 240, 245),
) -> None:
    screen.blit(font.render(text, True, color), position)


def draw_button(screen: Any, font: Any, button: Button) -> None:
    pygame.draw.rect(screen, (52, 62, 74), button.rect, border_radius=6)
    pygame.draw.rect(screen, (110, 125, 145), button.rect, width=1, border_radius=6)
    label = font.render(button.label, True, (245, 248, 252))
    screen.blit(label, label.get_rect(center=button.rect.center))


def fit_image_rect(
    image_width: int,
    image_height: int,
    screen_width: int,
    screen_height: int,
) -> tuple[Any, float]:
    image_area = pygame.Rect(
        18,
        VIEWER_TOOLBAR_HEIGHT,
        max(1, screen_width - 36),
        max(1, screen_height - VIEWER_TOOLBAR_HEIGHT - 18),
    )
    scale = min(image_area.width / image_width, image_area.height / image_height)
    scale = max(scale, 0.01)
    display_width = max(1, int(round(image_width * scale)))
    display_height = max(1, int(round(image_height * scale)))
    rect = pygame.Rect(0, 0, display_width, display_height)
    rect.center = image_area.center
    return rect, scale


def draw_detection_overlay(
    screen: Any,
    font: Any,
    detections: list[CircleDetection],
    candidate_detections: list[CircleDetection],
    image_rect: Any,
    scale: float,
    show_candidates: bool,
) -> None:
    def draw_single_detection(
        detection: CircleDetection,
        circle_color: tuple[int, int, int],
        bbox_color: tuple[int, int, int],
        label_bg_color: tuple[int, int, int],
    ) -> None:
        center = (
            int(round(image_rect.x + detection.center_x * scale)),
            int(round(image_rect.y + detection.center_y * scale)),
        )
        radius = max(2, int(round(detection.radius_px * scale)))
        bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox_px
        screen_bbox = pygame.Rect(
            int(round(image_rect.x + bbox_x * scale)),
            int(round(image_rect.y + bbox_y * scale)),
            max(1, int(round(bbox_w * scale))),
            max(1, int(round(bbox_h * scale))),
        )
        pygame.draw.circle(screen, circle_color, center, radius, width=2)
        pygame.draw.rect(screen, bbox_color, screen_bbox, width=1)
        label = (
            f"{detection.internal_color.name} "
            f"{detection.internal_color.confidence:.2f}"
        )
        label_surface = font.render(label, True, (245, 248, 252))
        label_bg = label_surface.get_rect(
            topleft=(screen_bbox.x, max(image_rect.y, screen_bbox.y - 22))
        )
        pygame.draw.rect(screen, label_bg_color, label_bg.inflate(8, 4), border_radius=3)
        screen.blit(label_surface, label_bg)

    if show_candidates:
        kept_keys = {
            (
                round(detection.center_x),
                round(detection.center_y),
                round(detection.radius_px),
            )
            for detection in detections
        }
        for detection in candidate_detections:
            key = (
                round(detection.center_x),
                round(detection.center_y),
                round(detection.radius_px),
            )
            if key in kept_keys:
                continue
            draw_single_detection(
                detection,
                circle_color=(255, 170, 40),
                bbox_color=(255, 170, 40),
                label_bg_color=(92, 56, 12),
            )

    for detection in detections:
        draw_single_detection(
            detection,
            circle_color=(0, 255, 0),
            bbox_color=(80, 180, 255),
            label_bg_color=(20, 70, 35),
        )


def save_debug_image(
    image_path: Path,
    output_dir: Path,
    image_bgr: np.ndarray,
    detections: list[CircleDetection],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / f"{image_path.stem}_circles.jpg"
    cv2.imwrite(str(debug_path), draw_detections(image_bgr, detections))
    return debug_path


def parse_window_size(value: str) -> tuple[int, int]:
    cleaned = value.lower().replace(",", "x")
    parts = [part.strip() for part in cleaned.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window size must look like 1400x900")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("window width and height must be positive")
    return width, height


def compact_metadata(row: dict[str, str]) -> dict[str, str]:
    wanted_fields = (
        "filename",
        "pixel_format",
        "lat",
        "lon",
        "alt_msl",
        "alt_rel",
        "roll",
        "pitch",
        "yaw",
        "heading",
    )
    return {field: row[field] for field in wanted_fields if row.get(field)}


def collect_inputs(
    input_value: str,
    metadata_value: str | None,
) -> tuple[dict[str, dict[str, str]], list[Path]]:
    input_path = Path(input_value).expanduser()
    metadata_path = Path(metadata_value).expanduser() if metadata_value else None
    if metadata_path is None and input_path.is_dir():
        metadata_path = find_default_metadata(input_path)

    metadata_by_name = load_metadata(metadata_path)
    image_paths = iter_image_paths(input_path, metadata_by_name)
    return metadata_by_name, image_paths


def detect_image(
    image_path: Path,
    row: dict[str, str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[CircleDetection], list[CircleDetection], dict[str, Any]]:
    image_bgr = load_image_bgr(image_path, row, bayer_mode=args.bayer_mode)
    detections = detect_red_filled_circles(
        image_bgr,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_red_confidence=args.min_red_confidence,
        min_circularity=args.min_circularity,
        min_blob_fill_ratio=args.min_blob_fill_ratio,
        red_hue_low=args.red_hue_low,
        red_saturation_min=args.red_saturation_min,
        red_value_min=args.red_value_min,
    )

    hough_candidates: list[CircleDetection] = []
    if args.viewer or args.show_candidates:
        hough_candidates = detect_circles(
            image_bgr,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_dist=args.min_dist,
            hough_dp=args.hough_dp,
            hough_param1=args.hough_param1,
            hough_param2=args.hough_param2,
            min_edge_support=args.min_edge_support,
            inner_radius_ratio=args.inner_radius_ratio,
        )

    result = {
        "image": str(image_path),
        "metadata": compact_metadata(row),
        "hough_candidate_count": len(hough_candidates),
        "red_filled_count": len(detections),
        "detections": [asdict(detection) for detection in detections],
    }
    return image_bgr, detections, hough_candidates, result

def process_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    metadata_by_name, image_paths = collect_inputs(args.input, args.metadata)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if args.save_debug:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for image_path in image_paths:
        row = metadata_by_name.get(image_path.name, {})
        image_bgr, detections, _, result = detect_image(image_path, row, args)
        results.append(result)

        print(json.dumps(result))

        if args.save_debug:
            save_debug_image(image_path, args.output_dir, image_bgr, detections)

    return results


def run_viewer(args: argparse.Namespace) -> None:
    if pygame is None:
        raise RuntimeError(
            "pygame is required for --viewer. Run with the project environment, "
            "for example: pdm run python tools/red_circle_detector.py --viewer"
        )

    metadata_by_name, image_paths = collect_inputs(args.input, args.metadata)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError("No images found to display.")

    pygame.init()
    pygame.display.set_caption("Circle Color Detector")
    screen = pygame.display.set_mode(args.window_size, pygame.RESIZABLE)
    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 15)
    clock = pygame.time.Clock()

    index = 0
    cache: dict[
        int,
        tuple[
            np.ndarray,
            Any,
            list[CircleDetection],
            list[CircleDetection],
            dict[str, Any],
        ],
    ] = {}
    show_candidates = bool(args.show_candidates)
    status = "Right/Space: next  Left: prev  C: candidates  S: save  Q/Esc: quit"
    buttons: dict[str, Button] = {}

    def load_current() -> tuple[
        np.ndarray,
        Any,
        list[CircleDetection],
        list[CircleDetection],
        dict[str, Any],
    ]:
        nonlocal status
        if index in cache:
            return cache[index]
        image_path = image_paths[index]
        row = metadata_by_name.get(image_path.name, {})
        image_bgr, detections, candidate_detections, result = detect_image(
            image_path, row, args
        )
        surface = bgr_to_surface(image_bgr)
        cache[index] = (image_bgr, surface, detections, candidate_detections, result)
        if len(cache) > 3:
            oldest_key = next(iter(cache))
            if oldest_key != index:
                cache.pop(oldest_key, None)
        status = (
            f"Raw candidates: {len(candidate_detections)} | "
            f"red-filled kept: {len(detections)} in {image_path.name}"
        )
        return cache[index]

    def move(delta: int) -> None:
        nonlocal index
        index = (index + delta) % len(image_paths)
        load_current()

    running = True
    while running:
        image_bgr, surface, detections, candidate_detections, result = load_current()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_n):
                    move(1)
                elif event.key in (pygame.K_LEFT, pygame.K_BACKSPACE, pygame.K_p):
                    move(-1)
                elif event.key == pygame.K_s:
                    saved_path = save_debug_image(
                        image_paths[index], args.output_dir, image_bgr, detections
                    )
                    status = f"Saved {saved_path}"
                elif event.key == pygame.K_c:
                    show_candidates = not show_candidates
                    status = (
                        "Showing all circle candidates"
                        if show_candidates
                        else "Showing only red-filled circles"
                    )
                elif event.key == pygame.K_j:
                    print(json.dumps(result))
                    status = "Printed current detection JSON"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons.values():
                    if button.rect.collidepoint(event.pos):
                        if button.name == "prev":
                            move(-1)
                        elif button.name == "next":
                            move(1)
                        elif button.name == "save":
                            saved_path = save_debug_image(
                                image_paths[index],
                                args.output_dir,
                                image_bgr,
                                detections,
                            )
                            status = f"Saved {saved_path}"

        screen_width, screen_height = screen.get_size()
        screen.fill((18, 22, 28))
        toolbar = pygame.Rect(0, 0, screen_width, VIEWER_TOOLBAR_HEIGHT - 1)
        pygame.draw.rect(screen, (28, 34, 42), toolbar)
        pygame.draw.line(
            screen,
            (70, 80, 94),
            (0, VIEWER_TOOLBAR_HEIGHT - 1),
            (screen_width, VIEWER_TOOLBAR_HEIGHT - 1),
        )

        buttons = {
            "prev": Button("prev", pygame.Rect(18, 18, 82, 38), "Prev"),
            "next": Button("next", pygame.Rect(108, 18, 76, 38), "Next"),
            "save": Button("save", pygame.Rect(192, 18, 92, 38), "Save Img"),
        }
        for button in buttons.values():
            draw_button(screen, font, button)

        image_path = image_paths[index]
        draw_text(
            screen,
            font,
            f"{index + 1}/{len(image_paths)}  {image_path.name}",
            (302, 18),
        )
        draw_text(
            screen,
            small_font,
            status,
            (18, 68),
            (172, 203, 255),
        )
        draw_text(
            screen,
            small_font,
            "C toggles raw Hough candidates. Green=validated red circular blob. J prints red-only JSON.",
            (18, 96),
            (170, 178, 188),
        )

        shown_detections = detections[:4]
        summary = " | ".join(
            f"{idx + 1}: {det.internal_color.name} "
            f"r={det.radius_px:.0f} "
            f"shape={det.circularity:.2f}"
            for idx, det in enumerate(shown_detections)
        )
        if len(detections) > len(shown_detections):
            summary += f" | +{len(detections) - len(shown_detections)} more"
        if not summary:
            if candidate_detections:
                summary = (
                    f"Hough saw {len(candidate_detections)} candidate(s), "
                    "but none are validated red circular blobs."
                )
            else:
                summary = "No validated red circular blobs detected in this image."
        draw_text(screen, small_font, summary, (420, 68), (235, 240, 245))

        image_rect, scale = fit_image_rect(
            surface.get_width(), surface.get_height(), screen_width, screen_height
        )
        scaled_surface = pygame.transform.smoothscale(surface, image_rect.size)
        screen.blit(scaled_surface, image_rect)
        pygame.draw.rect(screen, (85, 95, 108), image_rect, width=1)
        draw_detection_overlay(
            screen,
            small_font,
            detections,
            candidate_detections,
            image_rect,
            scale,
            show_candidates,
        )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Output only validated red-filled circular blobs."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_DATA_DIR),
        help="Image file or directory to scan. Defaults to data/outs.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="CSV metadata file. Defaults to the first metadata*.csv in the input dir.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only scan N images.")
    parser.add_argument("--min-radius", type=int, default=8)
    parser.add_argument(
        "--max-radius",
        type=int,
        default=0,
        help="Maximum radius for HoughCircles. 0 means no maximum.",
    )
    parser.add_argument("--min-dist", type=float, default=30.0)
    parser.add_argument("--hough-dp", type=float, default=1.2)
    parser.add_argument("--hough-param1", type=float, default=100.0)
    parser.add_argument("--hough-param2", type=float, default=28.0)
    parser.add_argument("--min-edge-support", type=float, default=0.10)
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.78,
        help="Minimum contour circularity for the red blob validation stage.",
    )
    parser.add_argument(
        "--min-blob-fill-ratio",
        type=float,
        default=0.70,
        help="Minimum contour area divided by its enclosing circle area.",
    )
    parser.add_argument(
        "--red-hue-low",
        type=int,
        default=150,
        help="Lower HSV hue bound for red/magenta red-like pixels.",
    )
    parser.add_argument("--red-saturation-min", type=int, default=45)
    parser.add_argument("--red-value-min", type=int, default=45)
    parser.add_argument(
        "--bayer-mode",
        choices=("metadata", "manual"),
        default="metadata",
        help=(
            "metadata uses the CSV pixel_format, e.g. BayerRG8. "
            "manual reproduces manual_target_checker.py's hardcoded BayerBG view."
        ),
    )
    parser.add_argument(
        "--min-red-confidence",
        type=float,
        default=0.60,
        help="Minimum fraction of sampled internal pixels that must be red-like.",
    )
    parser.add_argument(
        "--inner-radius-ratio",
        type=float,
        default=0.65,
        help="Fraction of each detected circle used to sample internal color.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save annotated debug images.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open an interactive viewer to step through images and detections.",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="In viewer mode, start by showing all raw circle candidates in orange.",
    )
    parser.add_argument(
        "--window-size",
        type=parse_window_size,
        default=(1400, 900),
        help="Viewer window size, e.g. 1400x900.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "circle_debug",
        help="Directory for debug images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viewer:
        run_viewer(args)
        return

    results = process_images(args)
    detections_count = sum(len(result["detections"]) for result in results)
    print(
        f"Scanned {len(results)} image(s), found {detections_count} circle(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
