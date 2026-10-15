"""
find_bullseye.py

Recall-first bullseye detector based on classical computer vision only.
The target is assumed to be a single red concentric bullseye printed on a
light square support, observed from a drone with rotation and perspective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "bullseye_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config at {path}")
    for section in ("detector", "dataset", "benchmark"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing config section: {section}")
    return config


def detector_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["detector"]


def dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["dataset"]


def benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["benchmark"]


def _odd_kernel(size: int) -> int:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _image_scale(image_shape: tuple[int, int]) -> float:
    return float(min(image_shape[:2]))


def _red_mask(hsv: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    hsv_cfg = detector_config(config)["hsv"]
    morph_cfg = detector_config(config)["morphology"]

    mask1 = cv2.inRange(
        hsv,
        np.array([hsv_cfg["hue_low_1"], hsv_cfg["sat_min"], hsv_cfg["val_min"]]),
        np.array([hsv_cfg["hue_high_1"], 255, 255]),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array([hsv_cfg["hue_low_2"], hsv_cfg["sat_min"], hsv_cfg["val_min"]]),
        np.array([hsv_cfg["hue_high_2"], 255, 255]),
    )
    mask = cv2.bitwise_or(mask1, mask2)

    open_kernel = _odd_kernel(morph_cfg["open_kernel"])
    close_kernel = _odd_kernel(morph_cfg["close_kernel"])
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
    )
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def _support_mask(hsv: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    support_cfg = detector_config(config)["support"]
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, support_cfg["min_val"]]),
        np.array([179, support_cfg["max_sat"], 255]),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def _support_candidate_mask(hsv: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    support_cfg = detector_config(config)["support"]
    value = hsv[:, :, 2]
    top_hat_kernel = _odd_kernel(int(support_cfg["top_hat_kernel"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (top_hat_kernel, top_hat_kernel))
    local_bright = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(
        local_bright, int(support_cfg["local_threshold"]), 255, cv2.THRESH_BINARY
    )

    open_kernel = _odd_kernel(int(support_cfg["candidate_open_kernel"]))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = _odd_kernel(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def _dark_pattern_mask(
    gray: np.ndarray, search_mask: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    pattern_cfg = detector_config(config)["pattern"]
    block_size = _odd_kernel(pattern_cfg["adaptive_block_size"])
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(pattern_cfg["adaptive_c"]),
    )
    open_size = _odd_kernel(pattern_cfg["open_kernel"])
    close_size = _odd_kernel(pattern_cfg["close_kernel"])
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, open_kernel)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, close_kernel)
    return cv2.bitwise_and(adaptive, search_mask)


def extract_mask_components(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_cfg = detector_config(config)["candidates"]
    min_area_px = (
        image_shape[0] * image_shape[1] * float(candidate_cfg["min_blob_area_ratio"])
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        components.append(
            {
                "contour": contour,
                "area": area,
                "bbox": (x, y, w, h),
                "center": (x + w / 2.0, y + h / 2.0),
            }
        )
    return components


def extract_support_components(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_cfg = detector_config(config)["candidates"]
    support_cfg = detector_config(config)["support"]
    min_area_px = (
        image_shape[0]
        * image_shape[1]
        * float(candidate_cfg["min_candidate_area_ratio"])
        * 0.35
    )
    max_area_px = (
        image_shape[0]
        * image_shape[1]
        * float(support_cfg.get("max_component_area_ratio", 1.0))
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    components: list[dict[str, Any]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area_px = stats[label_idx]
        area = float(area_px)
        if area < min_area_px:
            continue
        if area > max_area_px:
            continue
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        if aspect_ratio > 1.35:
            continue
        fill_ratio = area / max(1.0, w * h)
        if fill_ratio < float(support_cfg.get("min_fill_ratio", 0.45)):
            continue
        components.append(
            {
                "area": area,
                "bbox": (int(x), int(y), int(w), int(h)),
                "center": (
                    float(centroids[label_idx][0]),
                    float(centroids[label_idx][1]),
                ),
            }
        )
    return components


def _components_mask(
    components: list[dict[str, Any]], image_shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for component in components:
        x, y, w, h = component["bbox"]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
    return mask


def _group_bbox(group: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    x1 = min(component["bbox"][0] for component in group)
    y1 = min(component["bbox"][1] for component in group)
    x2 = max(component["bbox"][0] + component["bbox"][2] for component in group)
    y2 = max(component["bbox"][1] + component["bbox"][3] for component in group)
    return x1, y1, x2 - x1, y2 - y1


def merge_components(
    components: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if not components:
        return []

    candidate_cfg = detector_config(config)["candidates"]
    merge_distance = _image_scale(image_shape) * float(
        candidate_cfg["merge_distance_ratio"]
    )
    groups: list[list[dict[str, Any]]] = []

    for component in sorted(components, key=lambda item: item["area"], reverse=True):
        assigned = False
        for group in groups:
            group_center = (
                float(np.mean([item["center"][0] for item in group])),
                float(np.mean([item["center"][1] for item in group])),
            )
            group_bbox = _group_bbox(group)
            group_radius = max(group_bbox[2], group_bbox[3]) / 2.0
            center_distance = float(
                np.hypot(
                    component["center"][0] - group_center[0],
                    component["center"][1] - group_center[1],
                )
            )
            if center_distance <= merge_distance + group_radius * 0.35:
                group.append(component)
                assigned = True
                break
        if not assigned:
            groups.append([component])

    return groups


def radial_consistency_score(
    mask: np.ndarray,
    center: tuple[float, float],
    radius: float,
    samples: int,
    min_ring_count: int,
) -> dict[str, float]:
    h, w = mask.shape[:2]
    cx, cy = center
    transition_scores: list[float] = []
    ring_scores: list[float] = []

    for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
        values: list[bool] = []
        for frac in np.linspace(0.08, 1.0, 28):
            x = int(round(cx + np.cos(angle) * radius * frac))
            y = int(round(cy + np.sin(angle) * radius * frac))
            if 0 <= x < w and 0 <= y < h:
                values.append(bool(mask[y, x] > 0))

        if len(values) < 6:
            continue

        compressed = [values[0]]
        for value in values[1:]:
            if value != compressed[-1]:
                compressed.append(value)

        red_runs = sum(1 for value in compressed if value)
        transitions = max(0, len(compressed) - 1)
        transition_scores.append(min(1.0, transitions / 8.0))
        ring_scores.append(min(1.0, red_runs / max(1, min_ring_count)))

    if not transition_scores:
        return {"radial_score": 0.0, "ring_score": 0.0}

    return {
        "radial_score": float(np.mean(transition_scores)),
        "ring_score": float(np.mean(ring_scores)),
    }


def annular_structure_score(
    mask: np.ndarray,
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    min_ring_count: int,
    angular_samples: int = 72,
    radial_steps: int = 18,
) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0

    cx, cy = center
    rx = max(1.0, w / 2.0)
    ry = max(1.0, h / 2.0)
    radial_means: list[float] = []

    for frac in np.linspace(0.12, 0.95, radial_steps):
        values: list[float] = []
        for angle in np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False):
            px = int(round(cx + np.cos(angle) * rx * frac))
            py = int(round(cy + np.sin(angle) * ry * frac))
            if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
                values.append(float(mask[py, px] > 0))
        if values:
            radial_means.append(float(np.mean(values)))

    if not radial_means:
        return 0.0

    band_clarity = float(np.mean([abs((2.0 * mean) - 1.0) for mean in radial_means]))

    discrete_bands: list[int] = []
    for mean in radial_means:
        if mean >= 0.72:
            discrete_bands.append(1)
        elif mean <= 0.18:
            discrete_bands.append(0)
        else:
            discrete_bands.append(-1)

    compressed: list[int] = []
    for value in discrete_bands:
        if value == -1:
            continue
        if not compressed or compressed[-1] != value:
            compressed.append(value)

    ring_runs = sum(1 for value in compressed if value == 1)
    run_score = min(1.0, ring_runs / max(1, min_ring_count))
    return 0.55 * band_clarity + 0.45 * run_score


def _support_shape_score(
    image_bgr: np.ndarray, bbox: tuple[int, int, int, int], support_mask: np.ndarray
) -> float:
    x, y, w, h = bbox
    roi = image_bgr[y : y + h, x : x + w]
    roi_mask = support_mask[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value = hsv_roi[:, :, 2]
    sat = hsv_roi[:, :, 1]

    bright_ratio = float(np.mean(value > 150))
    low_sat_ratio = float(np.mean(sat < 120))
    support_ratio = float(np.mean(roi_mask > 0))
    bbox_area = max(1, w * h)
    square_score = max(0.0, 1.0 - abs((w / max(1.0, h)) - 1.0))
    area_score = min(1.0, bbox_area / 40000.0)
    return max(
        0.0,
        min(
            1.0,
            0.25 * bright_ratio
            + 0.20 * low_sat_ratio
            + 0.35 * support_ratio
            + 0.10 * square_score
            + 0.10 * area_score,
        ),
    )


def _score_candidate_bbox(
    bbox: tuple[int, int, int, int],
    components: list[dict[str, Any]],
    image_bgr: np.ndarray,
    pattern_mask: np.ndarray,
    support_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    det_cfg = detector_config(config)
    candidate_cfg = det_cfg["candidates"]
    score_cfg = det_cfg["scoring"]
    image_shape = image_bgr.shape[:2]
    image_area = image_shape[0] * image_shape[1]

    x, y, w, h = bbox
    bbox_area_ratio = (w * h) / image_area
    if bbox_area_ratio < float(candidate_cfg["min_candidate_area_ratio"]):
        return None

    aspect_ratio = max(w, h) / max(1.0, min(w, h))
    if aspect_ratio > float(candidate_cfg["max_candidate_aspect_ratio"]):
        return None

    center = (x + w / 2.0, y + h / 2.0)
    radius = max(w, h) / 2.0
    radial = radial_consistency_score(
        pattern_mask,
        center,
        radius,
        int(det_cfg["radial_samples"]),
        int(det_cfg["min_ring_count"]),
    )
    annular_score = annular_structure_score(
        pattern_mask,
        center,
        (x, y, w, h),
        int(det_cfg["min_ring_count"]),
    )

    scale = _image_scale(image_shape)
    tolerance = max(1.0, scale * float(candidate_cfg["center_tolerance_ratio"]))
    distances = [
        float(
            np.hypot(
                component["center"][0] - center[0], component["center"][1] - center[1]
            )
        )
        for component in components
    ]
    mean_distance = float(np.mean(distances)) if distances else tolerance
    component_count_score = min(1.0, len(components) / 4.0)
    concentricity_score = (
        max(0.0, 1.0 - (mean_distance / tolerance)) * component_count_score
    )
    support_score = _support_shape_score(image_bgr, (x, y, w, h), support_mask)
    preferred_area_ratio = float(score_cfg["preferred_candidate_area_ratio"])
    area_score = min(1.0, bbox_area_ratio / max(preferred_area_ratio, 1e-6))
    radial_score = 0.5 * radial["radial_score"] + 0.5 * radial["ring_score"]

    score = (
        radial_score * float(score_cfg["radial_consistency_weight"])
        + annular_score * float(score_cfg.get("annular_structure_weight", 0.0))
        + concentricity_score * float(score_cfg["concentricity_weight"])
        + support_score * float(score_cfg["support_shape_weight"])
        + area_score * float(score_cfg["area_weight"])
    )

    return {
        "bbox": (x, y, w, h),
        "center": center,
        "score": float(score),
        "radial_score": radial["radial_score"],
        "ring_score": radial["ring_score"],
        "annular_score": annular_score,
        "concentricity_score": concentricity_score,
        "support_score": support_score,
        "component_count": len(components),
        "aspect_ratio": aspect_ratio,
        "area_score": area_score,
        "bbox_area_ratio": bbox_area_ratio,
    }


def _passes_source_specific_filters(
    candidate: dict[str, Any], source: str, config: dict[str, Any]
) -> bool:
    if source == "red":
        hsv_cfg = detector_config(config)["hsv"]
        min_annular_score = float(hsv_cfg.get("min_annular_score", 0.0))
        if float(candidate.get("annular_score", 0.0)) < min_annular_score:
            return False
        return True

    if source != "support":
        return True

    support_cfg = detector_config(config)["support"]
    max_aspect_ratio = float(support_cfg.get("max_candidate_aspect_ratio", 99.0))
    min_annular_score = float(support_cfg.get("min_annular_score", 0.0))

    if float(candidate["aspect_ratio"]) > max_aspect_ratio:
        return False
    if float(candidate.get("annular_score", 0.0)) < min_annular_score:
        return False
    return True


def score_candidate_group(
    group: list[dict[str, Any]],
    image_bgr: np.ndarray,
    pattern_mask: np.ndarray,
    support_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    x, y, w, h = _group_bbox(group)
    return _score_candidate_bbox(
        (x, y, w, h), group, image_bgr, pattern_mask, support_mask, config
    )


def _candidate_to_yolo(
    candidate: dict[str, Any],
    image_shape: tuple[int, int],
    class_id: int,
    margin_ratio: float,
) -> tuple[
    tuple[int, float, float, float, float], tuple[int, int], tuple[float, float]
]:
    image_h, image_w = image_shape[:2]
    x, y, w, h = candidate["bbox"]
    margin = int(round(max(w, h) * margin_ratio))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image_w - x, w + 2 * margin)
    h = min(image_h - y, h + 2 * margin)

    center_x = int(round(x + w / 2.0))
    center_y = int(round(y + h / 2.0))
    x_norm = (x + w / 2.0) / image_w
    y_norm = (y + h / 2.0) / image_h
    w_norm = w / image_w
    h_norm = h / image_h

    return (
        (class_id, float(x_norm), float(y_norm), float(w_norm), float(h_norm)),
        (center_x, center_y),
        (float(x_norm), float(y_norm)),
    )


def _draw_debug_image(
    image_bgr: np.ndarray, components: list[dict[str, Any]], best: dict[str, Any] | None
) -> np.ndarray:
    overlay = image_bgr.copy()

    for component in components:
        x, y, w, h = component["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 120, 0), 1)

    if best is not None:
        x, y, w, h = best["bbox_px"]
        center_x, center_y = best["center_xy_px"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawMarker(
            overlay, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 26, 2
        )
        cv2.putText(
            overlay,
            f"score={best['score']:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return overlay


def _save_debug_json(output_dir: str, stem: str, payload: dict[str, Any]) -> None:
    json_path = Path(output_dir) / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _maybe_save(
    image_path: str,
    debug_image: np.ndarray | None,
    result: dict[str, Any] | None,
    output_dir: str | None,
    debug_payload: dict[str, Any] | None = None,
    save_debug_images: bool = True,
) -> None:
    if output_dir is None:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    ext = Path(image_path).suffix
    txt_path = Path(output_dir) / f"{stem}.txt"
    labels = [result["bbox_yolo"]] if result is not None else []

    with open(txt_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(
                f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
            )

    if debug_payload is not None:
        _save_debug_json(output_dir, stem, debug_payload)

    if debug_image is not None and save_debug_images:
        out_image_path = Path(output_dir) / f"detected_{stem}{ext}"
        cv2.imwrite(str(out_image_path), debug_image)


def _find_bullseye_impl(
    image: np.ndarray | None,
    image_path: str,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any] | None:
    if image is None or image.size == 0:
        return None

    det_cfg = detector_config(config)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    red_mask = _red_mask(hsv, config)
    red_components = extract_mask_components(red_mask, image.shape[:2], config)
    red_groups = merge_components(red_components, image.shape[:2], config)
    support_mask = _support_mask(hsv, config)
    support_candidate_mask_raw = _support_candidate_mask(hsv, config)
    support_components = extract_support_components(
        support_candidate_mask_raw, image.shape[:2], config
    )
    support_candidate_mask = _components_mask(support_components, image.shape[:2])
    support_search_mask = _dilate_mask(
        support_candidate_mask,
        int(detector_config(config)["support"]["dilation_kernel"]),
    )
    dark_mask = _dark_pattern_mask(gray, support_search_mask, config)
    pattern_mask = cv2.bitwise_or(red_mask, dark_mask)
    components = extract_mask_components(pattern_mask, image.shape[:2], config)

    candidates: list[dict[str, Any]] = []
    for group in red_groups:
        candidate = score_candidate_group(
            group, image, pattern_mask, support_mask, config
        )
        if candidate is not None and _passes_source_specific_filters(
            candidate, "red", config
        ):
            candidate["source"] = "red"
            candidates.append(candidate)
    for support_component in support_components:
        bbox = support_component["bbox"]
        inner_components = [
            component
            for component in components
            if bbox[0] <= component["center"][0] <= bbox[0] + bbox[2]
            and bbox[1] <= component["center"][1] <= bbox[1] + bbox[3]
        ]
        candidate = _score_candidate_bbox(
            bbox, inner_components, image, pattern_mask, support_mask, config
        )
        if candidate is not None and _passes_source_specific_filters(
            candidate, "support", config
        ):
            candidate["source"] = "support"
            candidates.append(candidate)

    best_candidate = max(candidates, key=lambda item: item["score"], default=None)
    threshold = float(det_cfg["score_threshold"])

    result: dict[str, Any] | None = None
    if best_candidate is not None and best_candidate["score"] >= threshold:
        bbox_yolo, center_xy_px, center_xy_norm = _candidate_to_yolo(
            best_candidate,
            image.shape[:2],
            int(det_cfg["class_id"]),
            float(det_cfg["bbox_margin_ratio"]),
        )
        result = {
            "bbox_yolo": bbox_yolo,
            "center_xy_px": center_xy_px,
            "center_xy_norm": center_xy_norm,
            "score": float(best_candidate["score"]),
            "debug_info": {
                "radial_score": float(best_candidate["radial_score"]),
                "ring_score": float(best_candidate["ring_score"]),
                "annular_score": float(best_candidate.get("annular_score", 0.0)),
                "concentricity_score": float(best_candidate["concentricity_score"]),
                "support_score": float(best_candidate["support_score"]),
                "component_count": int(best_candidate["component_count"]),
                "aspect_ratio": float(best_candidate["aspect_ratio"]),
            },
            "bbox_px": best_candidate["bbox"],
        }

    debug_payload = None
    debug_image = None
    if debug or output_dir:
        debug_payload = {
            "image_path": str(image_path),
            "n_red_components": len(red_components),
            "n_red_groups": len(red_groups),
            "n_support_components": len(support_components),
            "n_pattern_components": len(components),
            "score_threshold": threshold,
            "selected": result,
            "candidates": [
                {
                    "bbox": list(candidate["bbox"]),
                    "center": [
                        float(candidate["center"][0]),
                        float(candidate["center"][1]),
                    ],
                    "source": candidate.get("source", "unknown"),
                    "score": float(candidate["score"]),
                    "radial_score": float(candidate["radial_score"]),
                    "ring_score": float(candidate["ring_score"]),
                    "annular_score": float(candidate.get("annular_score", 0.0)),
                    "concentricity_score": float(candidate["concentricity_score"]),
                    "support_score": float(candidate["support_score"]),
                    "component_count": int(candidate["component_count"]),
                }
                for candidate in candidates
            ],
        }
        debug_image = _draw_debug_image(image, components, result)

    _maybe_save(
        image_path=image_path,
        debug_image=debug_image,
        result=result,
        output_dir=output_dir,
        debug_payload=debug_payload if debug else None,
        save_debug_images=bool(benchmark_config(config).get("save_debug_images", True)),
    )
    return result


def find_bullseye_in_image(
    image: np.ndarray | None,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
    image_path: str = "memory_image.png",
) -> dict[str, Any] | None:
    return _find_bullseye_impl(
        image=image,
        image_path=image_path,
        config=config,
        output_dir=output_dir,
        debug=debug,
    )


def find_bullseye(
    image_path: str,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERR] Impossibile caricare: {image_path}")
        return None

    return _find_bullseye_impl(
        image=image,
        image_path=image_path,
        config=config,
        output_dir=output_dir,
        debug=debug,
    )


def load_bullseye_gt(label_path: Path, class_id: int) -> list[tuple[float, ...]]:
    gt_boxes: list[tuple[float, ...]] = []
    if label_path.exists() and label_path.stat().st_size > 0:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                values = tuple(float(value) for value in parts)
                if int(values[0]) == class_id:
                    gt_boxes.append(values)
    return gt_boxes


def _iou(
    box_a: tuple[float, ...], box_b: tuple[float, ...], img_w: int, img_h: int
) -> float:
    def to_xyxy(box: tuple[float, ...]) -> tuple[float, float, float, float]:
        center_x = box[1] * img_w
        center_y = box[2] * img_h
        width = box[3] * img_w
        height = box[4] * img_h
        return (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )

    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_on_dataset(
    images_dir: Path,
    labels_dir: Path,
    config: dict[str, Any],
    output_dir: str | None = None,
    iou_thresh: float = 0.5,
    debug: bool = False,
) -> dict[str, Any]:
    img_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    target_class = int(dataset_config(config)["bullseye_class_id"])
    tp = fp = fn = 0
    errors = 0
    false_positive_images: list[str] = []
    false_negative_images: list[str] = []

    for idx, img_path in enumerate(img_paths, start=1):
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_bullseye_gt(label_path, class_id=target_class)
        prediction = find_bullseye(
            str(img_path), config=config, output_dir=output_dir, debug=debug
        )
        preds = [prediction["bbox_yolo"]] if prediction is not None else []

        image = cv2.imread(str(img_path))
        if image is None:
            errors += 1
            continue
        img_h, img_w = image.shape[:2]

        matched_gt: set[int] = set()
        for pred in preds:
            matched = False
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if _iou(pred, gt_box, img_w, img_h) >= iou_thresh:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched = True
                    break
            if not matched:
                fp += 1
                false_positive_images.append(img_path.name)

        missing = len(gt_boxes) - len(matched_gt)
        if missing > 0:
            fn += missing
            false_negative_images.append(img_path.name)

        if idx % 100 == 0:
            print(f"  [{idx}/{len(img_paths)}] TP={tp} FP={fp} FN={fn}")

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "n_images": len(img_paths),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "errors": errors,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "false_positive_images": false_positive_images[:20],
        "false_negative_images": false_negative_images[:20],
    }


def _resolve_dataset_paths(
    config: dict[str, Any], data_root_override: str = "", split_override: str = ""
) -> tuple[Path, Path, str]:
    ds_cfg = dataset_config(config)
    data_root = Path(data_root_override or ds_cfg["data_root"]).expanduser()
    split = split_override or ds_cfg["split"]
    return data_root / split / "images", data_root / split / "labels", split


def infer_on_image_dir(
    image_dir: Path,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    img_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    detections = 0
    no_detections = 0
    errors = 0
    detected_images: list[str] = []

    for idx, img_path in enumerate(img_paths, start=1):
        result = find_bullseye(
            str(img_path), config=config, output_dir=output_dir, debug=debug
        )
        if result is not None:
            detections += 1
            detected_images.append(img_path.name)
        else:
            no_detections += 1

        if idx % 100 == 0:
            print(
                f"  [{idx}/{len(img_paths)}] detections={detections} no_detections={no_detections}"
            )

    return {
        "image_dir": str(image_dir),
        "n_images": len(img_paths),
        "detections": detections,
        "no_detections": no_detections,
        "errors": errors,
        "detected_images": detected_images[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rileva bullseye con CV classica recall-first."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path al file YAML di configurazione.",
    )
    parser.add_argument("--image", type=str, help="Singola immagine da analizzare.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Cartella piatta di immagini da analizzare senza GT.",
    )
    parser.add_argument(
        "--data_root", type=str, default="", help="Override della root dataset."
    )
    parser.add_argument(
        "--split", type=str, default="", help="Override dello split dataset."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory output per txt, overlay e debug JSON.",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=-1.0,
        help="Override della soglia IoU per evaluation.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Abilita debug JSON e overlay."
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Salta la valutazione GT ed esegue solo inferenza.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bench_cfg = benchmark_config(config)
    output_dir = args.out_dir or bench_cfg["debug_dir"]

    if args.image:
        result = find_bullseye(
            args.image, config=config, output_dir=output_dir, debug=args.debug
        )
        if result is None:
            print("Nessun bullseye trovato.")
            return
        label = result["bbox_yolo"]
        print("Bullseye trovato:")
        print(
            f"  bbox_yolo     : {label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}"
        )
        print(f"  center_xy_px  : {result['center_xy_px']}")
        print(f"  center_xy_norm: {result['center_xy_norm']}")
        print(f"  score         : {result['score']:.4f}")
        return

    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser()
        if not image_dir.exists():
            raise SystemExit(f"[ERR] Cartella non trovata: {image_dir}")

        summary = infer_on_image_dir(
            image_dir=image_dir,
            config=config,
            output_dir=output_dir,
            debug=args.debug,
        )
        print("\n" + "=" * 50)
        print(" RISULTATI DIRECTORY")
        print("=" * 50)
        print(f"  Cartella      : {summary['image_dir']}")
        print(f"  Immagini      : {summary['n_images']}")
        print(f"  Con detection : {summary['detections']}")
        print(f"  Senza target  : {summary['no_detections']}")
        print(f"  Errori        : {summary['errors']}")
        print("=" * 50)
        print(f"  Output → {output_dir}/")
        return

    images_dir, labels_dir, split = _resolve_dataset_paths(
        config, args.data_root, args.split
    )
    print(f"\n[*] Dataset : {images_dir}")
    print(f"[*] Split   : {split}")

    if not images_dir.exists():
        raise SystemExit(f"[ERR] Cartella non trovata: {images_dir}")

    if args.no_eval:
        total_found = 0
        img_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for idx, img_path in enumerate(img_paths, start=1):
            result = find_bullseye(
                str(img_path), config=config, output_dir=output_dir, debug=args.debug
            )
            if result is not None:
                total_found += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{len(img_paths)}] bullseye trovati: {total_found}")
        print(
            f"\nTotale immagini con detection: {total_found} su {len(img_paths)} immagini"
        )
        return

    iou_thresh = (
        args.iou_thresh if args.iou_thresh > 0 else float(bench_cfg["iou_threshold"])
    )
    metrics = evaluate_on_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        config=config,
        output_dir=output_dir,
        iou_thresh=iou_thresh,
        debug=args.debug,
    )
    print("\n" + "=" * 50)
    print(f" RISULTATI [{split}] — IoU≥{iou_thresh}")
    print("=" * 50)
    print(f"  Immagini     : {metrics['n_images']}")
    print(f"  TP / FP / FN : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
    print(f"  Precision    : {metrics['precision']} %")
    print(f"  Recall       : {metrics['recall']} %")
    print(f"  F1           : {metrics['f1']} %")
    print(f"  Errori       : {metrics['errors']}")
    print("=" * 50)
    print(f"  Output → {output_dir}/")


if __name__ == "__main__":
    main()
