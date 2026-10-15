"""
find_bw_targets.py

Recall-first classical CV detector for the small black-and-white square targets.
It is intentionally separate from the bullseye detector and is optimized to:
- find many small high-contrast B/W targets
- ignore large colored targets when possible
- tolerate partial visibility and mild occlusions
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from stuff_i_moved_away.inference.find_bullseye import (
        find_bullseye,
        load_config as load_bullseye_config,
    )
except ImportError:
    from stuff_i_moved_away.inference.find_bullseye import find_bullseye, load_config as load_bullseye_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "bw_targets_config.yaml"
DEFAULT_BULLSEYE_CONFIG_PATH = Path(__file__).resolve().parent / "bullseye_config.yaml"
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


def benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["benchmark"]


def _fast_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    fast_cfg = detector_config(config).get("fast_runtime", {})
    return {
        "enabled": bool(fast_cfg.get("enabled", True)),
        "enable_component_template": bool(
            fast_cfg.get("enable_component_template", True)
        ),
        "enable_self_template": bool(fast_cfg.get("enable_self_template", False)),
        "enable_template_rescue": bool(
            fast_cfg.get("enable_template_rescue", False)
        ),
        "min_base_candidates_for_skip_component_template": max(
            0, int(fast_cfg.get("min_base_candidates_for_skip_component_template", 1))
        ),
        "min_base_candidates_for_skip_template_rescue": max(
            0, int(fast_cfg.get("min_base_candidates_for_skip_template_rescue", 1))
        ),
        "enable_scene_anchor_for_fallback": bool(
            fast_cfg.get("enable_scene_anchor_for_fallback", False)
        ),
    }


def _odd_kernel(size: int) -> int:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _dark_mask(image_bgr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    det_cfg = detector_config(config)
    component_cfg = det_cfg["dark_component"]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mask = (
        (gray < int(det_cfg["dark_threshold"]))
        & (hsv[:, :, 1] < int(det_cfg["max_dark_saturation"]))
    ).astype(np.uint8) * 255

    open_kernel = _odd_kernel(int(component_cfg["open_kernel"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _support_anchor_candidates(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 145)).astype(np.uint8) * 255
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    image_h, image_w = image_bgr.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 1200.0 or area > 18000.0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        side = float(max(w, h))
        if side < 70.0:
            continue
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        if aspect_ratio > 1.22:
            continue

        border_margin = 20
        if (
            x <= border_margin
            or y <= border_margin
            or x + w >= image_w - border_margin
            or y + h >= image_h - border_margin
        ):
            continue

        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = area / hull_area
        if solidity < 0.72:
            continue

        perimeter = max(1.0, cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 8:
            continue

        anchors.append(
            {
                "center_x": float(x + w / 2.0),
                "center_y": float(y + h / 2.0),
                "side_px": side,
                "score": float(
                    min(1.0, area / 6000.0) * 0.45
                    + (1.0 - min(1.0, abs(aspect_ratio - 1.0) / 0.35)) * 0.30
                    + min(1.0, solidity / 0.9) * 0.25
                ),
            }
        )

    anchors.sort(key=lambda item: item["score"], reverse=True)
    return anchors


@lru_cache(maxsize=1)
def _load_bullseye_anchor_config() -> dict[str, Any] | None:
    if not DEFAULT_BULLSEYE_CONFIG_PATH.exists():
        return None
    try:
        return load_bullseye_config(str(DEFAULT_BULLSEYE_CONFIG_PATH))
    except Exception:
        return None


def _scene_anchor_candidates(
    image_path: str, image_bgr: np.ndarray
) -> list[dict[str, Any]]:
    anchors = _support_anchor_candidates(image_bgr)
    bullseye_cfg = _load_bullseye_anchor_config()
    if bullseye_cfg is not None:
        try:
            bullseye_result = find_bullseye(image_path, bullseye_cfg, debug=False)
        except Exception:
            bullseye_result = None
        if bullseye_result is not None:
            x, y, w, h = bullseye_result["bbox_px"]
            anchors.append(
                {
                    "center_x": float(bullseye_result["center_xy_px"][0]),
                    "center_y": float(bullseye_result["center_xy_px"][1]),
                    "side_px": float(max(w, h)),
                    "score": float(bullseye_result["score"]) + 0.5,
                }
            )

    deduped: list[dict[str, Any]] = []
    for anchor in sorted(anchors, key=lambda item: item["score"], reverse=True):
        too_close = False
        for kept in deduped:
            min_distance = 0.6 * max(anchor["side_px"], kept["side_px"])
            if (
                float(
                    np.hypot(
                        anchor["center_x"] - kept["center_x"],
                        anchor["center_y"] - kept["center_y"],
                    )
                )
                < min_distance
            ):
                too_close = True
                break
        if not too_close:
            deduped.append(anchor)
        if len(deduped) >= 3:
            break
    return deduped


def _extract_dark_components(
    mask: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    det_cfg = detector_config(config)
    component_cfg = det_cfg["dark_component"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(component_cfg["min_area_px"]) or area > float(
            component_cfg["max_area_px"]
        ):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        if aspect_ratio > float(component_cfg["max_aspect_ratio"]):
            continue

        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = area / hull_area
        if solidity < float(component_cfg["min_solidity"]):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if not (
            int(component_cfg["min_vertices"])
            <= len(approx)
            <= int(component_cfg["max_vertices"])
        ):
            continue

        components.append(
            {
                "bbox": (x, y, w, h),
                "center": (x + w / 2.0, y + h / 2.0),
                "area": area,
                "solidity": solidity,
            }
        )
    return components


def _extract_local_dark_components(
    image_bgr: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    det_cfg = detector_config(config)
    local_cfg = det_cfg.get("local_dark_component", {})
    if not bool(local_cfg.get("enabled", False)):
        return []

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blackhat_kernel = _odd_kernel(int(local_cfg.get("blackhat_kernel_px", 19)))
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (blackhat_kernel, blackhat_kernel)
        ),
    )
    mask = (
        (blackhat >= float(local_cfg.get("blackhat_threshold", 18.0)))
        & (hsv[:, :, 1] <= float(local_cfg.get("max_mask_saturation", 185.0)))
    ).astype(np.uint8) * 255

    open_kernel = _odd_kernel(int(local_cfg.get("open_kernel", 3)))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[dict[str, Any]] = []

    min_area = float(local_cfg.get("min_area_px", 8.0))
    max_area = float(local_cfg.get("max_area_px", 220.0))
    max_aspect_ratio = float(local_cfg.get("max_aspect_ratio", 2.8))
    min_solidity = float(local_cfg.get("min_solidity", 0.5))
    min_vertices = int(local_cfg.get("min_vertices", 3))
    max_vertices = int(local_cfg.get("max_vertices", 8))
    pad_ratio = float(local_cfg.get("roi_pad_ratio", 1.5))
    pad_px = int(local_cfg.get("roi_pad_px", 4))
    min_gap = float(local_cfg.get("min_intensity_gap", 18.0))
    min_inside_dark_ratio = float(local_cfg.get("min_inside_dark_ratio", 0.2))
    inside_dark_threshold = int(local_cfg.get("inside_dark_threshold", 150))
    min_blackhat_mean = float(local_cfg.get("min_blackhat_mean", 20.0))
    max_component_saturation = float(local_cfg.get("max_component_saturation", 110.0))

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        if aspect_ratio > max_aspect_ratio:
            continue

        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if not (min_vertices <= len(approx) <= max_vertices):
            continue

        pad = int(round(max(w, h) * pad_ratio)) + pad_px
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        roi_gray = gray[y1:y2, x1:x2]
        roi_hsv = hsv[y1:y2, x1:x2]
        roi_mask = np.zeros_like(roi_gray, dtype=np.uint8)
        shifted_contour = contour - np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, thickness=-1)
        inside_values = roi_gray[roi_mask > 0]
        outside_values = roi_gray[roi_mask == 0]
        if inside_values.size == 0 or outside_values.size == 0:
            continue

        intensity_gap = float(np.mean(outside_values) - np.mean(inside_values))
        inside_dark_ratio = float(np.mean(inside_values < inside_dark_threshold))
        blackhat_mean = float(np.mean(blackhat[y : y + h, x : x + w]))
        component_saturation = float(np.mean(roi_hsv[:, :, 1][roi_mask > 0]))
        if blackhat_mean < min_blackhat_mean:
            continue
        if component_saturation > max_component_saturation:
            continue
        if intensity_gap < min_gap and inside_dark_ratio < min_inside_dark_ratio:
            continue

        support_score = (
            min(1.0, blackhat_mean / max(min_blackhat_mean, 1e-6)) * 0.35
            + min(1.0, max(0.0, intensity_gap) / max(min_gap, 1e-6)) * 0.35
            + min(1.0, inside_dark_ratio / max(min_inside_dark_ratio, 1e-6)) * 0.20
            + min(1.0, solidity / max(min_solidity, 1e-6)) * 0.10
        )
        components.append(
            {
                "bbox": (x, y, w, h),
                "center": (x + w / 2.0, y + h / 2.0),
                "area": area,
                "solidity": solidity,
                "support_score": float(support_score),
            }
        )

    if not components:
        return []

    cell_size = max(12, int(local_cfg.get("cell_px", 24)))
    max_per_cell = max(1, int(local_cfg.get("max_per_cell", 3)))
    max_components = max(1, int(local_cfg.get("max_components", 180)))
    limited: list[dict[str, Any]] = []
    per_cell_counts: dict[tuple[int, int], int] = {}

    for component in sorted(
        components,
        key=lambda item: (
            item.get("support_score", 0.0),
            item["area"],
            item["solidity"],
        ),
        reverse=True,
    ):
        cell = (
            int(component["center"][0] // cell_size),
            int(component["center"][1] // cell_size),
        )
        current = per_cell_counts.get(cell, 0)
        if current >= max_per_cell:
            continue
        limited.append(component)
        per_cell_counts[cell] = current + 1
        if len(limited) >= max_components:
            break

    return limited


def _extract_component_template_seed_components(
    image_bgr: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    component_cfg = detector_config(config).get("component_template", {})
    if not bool(component_cfg.get("enabled", False)):
        return []

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sigma = float(component_cfg.get("seed_local_contrast_sigma", 5.0))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    local_contrast = blurred.astype(np.int16) - gray.astype(np.int16)
    mask = (
        (local_contrast >= int(component_cfg.get("seed_min_local_contrast", 18)))
        | (gray <= int(component_cfg.get("seed_max_gray", 145)))
    ) & (hsv[:, :, 1] <= int(component_cfg.get("seed_max_saturation", 190)))
    mask_u8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(component_cfg.get("seed_min_area_px", 3))
    max_area = float(component_cfg.get("seed_max_area_px", 180))
    max_aspect_ratio = float(component_cfg.get("seed_max_aspect_ratio", 3.2))
    min_solidity = float(component_cfg.get("seed_min_solidity", 0.45))

    seeds: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        if aspect_ratio > max_aspect_ratio:
            continue
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        component_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        ys, xs = np.where(component_mask > 0)
        if len(xs) == 0:
            continue

        mean_contrast = float(np.mean(local_contrast[ys, xs]))
        mean_darkness = float(255.0 - np.mean(gray[ys, xs]))
        seed_score = (
            mean_contrast * 0.65 + mean_darkness * 0.35 + min(area, 60.0) * 0.15
        )
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        else:
            center = (x + w / 2.0, y + h / 2.0)

        seeds.append(
            {
                "bbox": (x, y, w, h),
                "center": center,
                "area": area,
                "solidity": solidity,
                "seed_score": float(seed_score),
                "seed_source": "lenient_component_template",
            }
        )

    return seeds


def _group_components(
    components: list[dict[str, Any]], config: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    if not components:
        return []

    grouping_cfg = detector_config(config)["grouping"]
    merge_distance = float(grouping_cfg["merge_distance_px"])
    touch_margin = float(grouping_cfg["bbox_touch_margin_px"])
    groups: list[list[dict[str, Any]]] = []

    for component in sorted(components, key=lambda item: item["area"], reverse=True):
        assigned = False
        for group in groups:
            x1 = min(item["bbox"][0] for item in group)
            y1 = min(item["bbox"][1] for item in group)
            x2 = max(item["bbox"][0] + item["bbox"][2] for item in group)
            y2 = max(item["bbox"][1] + item["bbox"][3] for item in group)
            center_x = float(np.mean([item["center"][0] for item in group]))
            center_y = float(np.mean([item["center"][1] for item in group]))

            center_distance = float(
                np.hypot(
                    component["center"][0] - center_x, component["center"][1] - center_y
                )
            )
            touches_group = (x1 - touch_margin) <= component["center"][0] <= (
                x2 + touch_margin
            ) and (y1 - touch_margin) <= component["center"][1] <= (y2 + touch_margin)
            if center_distance <= merge_distance or touches_group:
                group.append(component)
                assigned = True
                break

        if not assigned:
            groups.append([component])

    return groups


def _normalized_square_match_gray(
    gray: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    square_cfg = detector_config(config).get("square_component", {})
    sigma = float(square_cfg.get("local_contrast_sigma", 5.0))
    alpha = float(square_cfg.get("local_contrast_alpha", 1.8))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    normalized = cv2.addWeighted(gray, alpha, blurred, -(alpha - 1.0), 0.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _square_vote_bbox(
    vote: dict[str, Any],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    square_cfg = detector_config(config).get("square_component", {})
    bbox_scale = float(square_cfg.get("bbox_scale", 1.35))
    bbox_pad_px = int(square_cfg.get("bbox_pad_px", 2))
    center_x = float(vote["cx"])
    center_y = float(vote["cy"])
    half_side = float(vote["w"]) * bbox_scale * 0.5 + bbox_pad_px
    image_h, image_w = image_shape[:2]
    x1 = max(0, int(round(center_x - half_side)))
    y1 = max(0, int(round(center_y - half_side)))
    x2 = min(image_w, int(round(center_x + half_side)))
    y2 = min(image_h, int(round(center_y + half_side)))
    return x1, y1, x2 - x1, y2 - y1


def _extract_square_candidates(
    image_bgr: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    square_cfg = detector_config(config).get("square_component", {})
    if not bool(square_cfg.get("enabled", True)):
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    normalized_gray = _normalized_square_match_gray(gray, config)
    local_max_kernel = _odd_kernel(int(square_cfg.get("local_max_kernel_px", 7)))
    peak_kernel = np.ones((local_max_kernel, local_max_kernel), dtype=np.uint8)
    cell_size = max(local_max_kernel, int(square_cfg.get("vote_cell_px", 24)))
    max_votes_per_cell = max(1, int(square_cfg.get("max_votes_per_cell", 2)))
    max_votes_total = max(1, int(square_cfg.get("max_votes_total", 120)))

    votes: list[dict[str, Any]] = []
    for size in square_cfg.get("sizes_px", [12, 16, 20, 24]):
        template = _marker_template(int(size))
        threshold = float(
            square_cfg.get("small_match_threshold", 0.58)
            if int(size) <= int(square_cfg.get("small_size_limit_px", 22))
            else square_cfg.get("large_match_threshold", 0.60)
        )

        for angle in square_cfg.get("angles_deg", [0, 45, 90, 135]):
            matrix = cv2.getRotationMatrix2D(
                (size / 2.0, size / 2.0), float(angle), 1.0
            )
            rotated = cv2.warpAffine(
                template,
                matrix,
                (int(size), int(size)),
                flags=cv2.INTER_LINEAR,
                borderValue=255,
            )
            response = cv2.matchTemplate(normalized_gray, rotated, cv2.TM_CCOEFF_NORMED)
            if response.size == 0:
                continue

            local_max = cv2.dilate(response, peak_kernel)
            ys, xs = np.where((response >= threshold) & (response == local_max))
            if len(xs) == 0:
                continue

            scores = response[ys, xs]
            order = np.argsort(scores)[::-1]
            cell_counts: dict[tuple[int, int], int] = {}

            for index in order.tolist():
                y = int(ys[index])
                x = int(xs[index])
                center_x = float(x + size / 2.0)
                center_y = float(y + size / 2.0)
                cell = (int(center_x // cell_size), int(center_y // cell_size))
                if cell_counts.get(cell, 0) >= max_votes_per_cell:
                    continue

                votes.append(
                    {
                        "x": x,
                        "y": y,
                        "w": int(size),
                        "h": int(size),
                        "cx": center_x,
                        "cy": center_y,
                        "center_x": center_x,
                        "center_y": center_y,
                        "side_px": float(size),
                        "score": float(response[y, x]),
                        "angle": float(angle),
                    }
                )
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
                if len(votes) >= max_votes_total:
                    break
            if len(votes) >= max_votes_total:
                break
        if len(votes) >= max_votes_total:
            break

    return votes


def _marker_template(size: int) -> np.ndarray:
    size = max(8, int(size))
    template = np.full((size, size), 255, dtype=np.uint8)
    center = size // 2
    triangles = [
        np.array([[0, 0], [center, center], [size - 1, 0]], dtype=np.int32),
        np.array(
            [[size - 1, 0], [center, center], [size - 1, size - 1]], dtype=np.int32
        ),
        np.array(
            [[size - 1, size - 1], [center, center], [0, size - 1]], dtype=np.int32
        ),
        np.array([[0, size - 1], [center, center], [0, 0]], dtype=np.int32),
    ]
    cv2.fillConvexPoly(template, triangles[0], 0)
    cv2.fillConvexPoly(template, triangles[2], 0)
    cv2.line(template, (0, 0), (size - 1, size - 1), 180, 1)
    cv2.line(template, (size - 1, 0), (0, size - 1), 180, 1)
    return template


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = a.astype(np.float32).reshape(-1)
    b_vec = b.astype(np.float32).reshape(-1)
    a_vec -= float(np.mean(a_vec))
    b_vec -= float(np.mean(b_vec))
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-6:
        return -1.0
    return float(np.dot(a_vec, b_vec) / denom)


def _square_patch_from_vote(
    gray: np.ndarray,
    center_xy: tuple[float, float],
    side_px: float,
    angle_deg: float,
    output_size: int,
) -> np.ndarray:
    side_px = max(8.0, float(side_px))
    output_size = max(16, int(output_size))
    half_side = side_px / 2.0
    center = np.array(center_xy, dtype=np.float32)

    theta = np.deg2rad(float(angle_deg))
    axis_x = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    axis_y = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    src = np.array(
        [
            center - axis_x * half_side - axis_y * half_side,
            center + axis_x * half_side - axis_y * half_side,
            center - axis_x * half_side + axis_y * half_side,
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [0.0, 0.0],
            [float(output_size - 1), 0.0],
            [0.0, float(output_size - 1)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(
        gray,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


@lru_cache(maxsize=16)
def _marker_template_variants(size: int) -> list[np.ndarray]:
    base = _marker_template(size).astype(np.float32)
    return [np.ascontiguousarray(np.rot90(base, k)) for k in range(4)]


@lru_cache(maxsize=16)
def _triangle_region_masks(size: int) -> list[np.ndarray]:
    yy, xx = np.indices((size, size))
    top = (yy <= xx) & (yy <= (size - 1 - xx))
    right = (yy <= xx) & (yy >= (size - 1 - xx))
    bottom = (yy >= xx) & (yy >= (size - 1 - xx))
    left = (yy >= xx) & (yy <= (size - 1 - xx))
    center_margin = max(1, size // 10)
    center = size // 2
    center_cutout = (np.abs(xx - center) <= center_margin) | (
        np.abs(yy - center) <= center_margin
    )
    return [
        np.ascontiguousarray(top & ~center_cutout),
        np.ascontiguousarray(right & ~center_cutout),
        np.ascontiguousarray(bottom & ~center_cutout),
        np.ascontiguousarray(left & ~center_cutout),
    ]


def _template_patch_stats(
    gray: np.ndarray,
    vote: dict[str, Any],
    config: dict[str, Any],
    *,
    patch_size_override: int | None = None,
    side_scale_override: float | None = None,
) -> dict[str, Any]:
    rescue_cfg = detector_config(config)["template_rescue"]
    patch_size = max(
        20,
        int(
            patch_size_override
            if patch_size_override is not None
            else rescue_cfg.get("verify_patch_px", 40)
        ),
    )
    side_scale = float(
        side_scale_override
        if side_scale_override is not None
        else rescue_cfg.get("verify_side_scale", 1.2)
    )
    patch = _square_patch_from_vote(
        gray=gray,
        center_xy=(float(vote["cx"]), float(vote["cy"])),
        side_px=float(vote["w"]) * side_scale,
        angle_deg=float(vote.get("angle", 0.0)),
        output_size=patch_size,
    )
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    _, binary_patch = cv2.threshold(patch, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_stats = {
        "template_corr": -1.0,
        "binary_agreement": 0.0,
        "contrast_separation": 0.0,
        "structure_score": 0.0,
        "mean_dark_value": 255.0,
        "mean_light_value": 0.0,
    }
    best_score = -1.0
    triangle_masks = _triangle_region_masks(patch_size)

    for rotation_idx, template in enumerate(_marker_template_variants(patch_size)):
        template_white = (template >= 128).astype(np.uint8)
        template_dark = 1 - template_white

        corr = _normalized_correlation(patch, template)
        agreement = float(np.mean(binary_patch == template_white))
        dark_values = patch[template_dark.astype(bool)]
        light_values = patch[template_white.astype(bool)]
        mean_dark = float(np.mean(dark_values)) if dark_values.size else 255.0
        mean_light = float(np.mean(light_values)) if light_values.size else 0.0
        contrast_separation = max(0.0, (mean_light - mean_dark) / 255.0)

        region_means: list[float] = []
        for mask in triangle_masks:
            values = patch[mask]
            region_means.append(float(np.mean(values)) if values.size else 127.5)

        dark_indices = [(0 - rotation_idx) % 4, (2 - rotation_idx) % 4]
        light_indices = [(1 - rotation_idx) % 4, (3 - rotation_idx) % 4]
        dark_region_means = [region_means[index] for index in dark_indices]
        light_region_means = [region_means[index] for index in light_indices]
        dark_light_gap = max(
            0.0,
            (float(np.mean(light_region_means)) - float(np.mean(dark_region_means)))
            / 255.0,
        )
        opposite_similarity = max(
            0.0,
            1.0
            - (
                abs(dark_region_means[0] - dark_region_means[1])
                + abs(light_region_means[0] - light_region_means[1])
            )
            / (2.0 * 255.0),
        )
        structure_score = float(0.65 * dark_light_gap + 0.35 * opposite_similarity)
        score = (
            max(0.0, corr) * 0.42
            + agreement * 0.18
            + contrast_separation * 0.15
            + structure_score * 0.25
        )

        if score > best_score:
            best_score = score
            best_stats = {
                "template_corr": float(corr),
                "binary_agreement": float(agreement),
                "contrast_separation": float(contrast_separation),
                "structure_score": float(structure_score),
                "mean_dark_value": float(mean_dark),
                "mean_light_value": float(mean_light),
            }

    best_stats["template_patch_score"] = float(best_score)
    return best_stats


def _refine_template_vote(
    gray: np.ndarray,
    cluster: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rescue_cfg = detector_config(config)["template_rescue"]
    weights = np.array(
        [max(1e-3, float(item["score"])) for item in cluster], dtype=np.float32
    )
    seed_vote = max(cluster, key=lambda item: item["score"])
    center_x = float(np.average([item["cx"] for item in cluster], weights=weights))
    center_y = float(np.average([item["cy"] for item in cluster], weights=weights))
    base_size = float(np.median([max(item["w"], item["h"]) for item in cluster]))
    base_angle = float(seed_vote.get("angle", 0.0))
    offset_px = base_size * float(rescue_cfg.get("refine_offset_scale", 0.35))
    angle_steps = [
        float(delta)
        for delta in rescue_cfg.get("refine_angle_deltas_deg", [-15.0, 0.0, 15.0])
    ]
    size_scales = [
        float(scale) for scale in rescue_cfg.get("refine_size_scales", [0.9, 1.0, 1.1])
    ]

    best_vote = dict(seed_vote)
    best_stats = _template_patch_stats(gray, best_vote, config)

    for dx in (-offset_px, 0.0, offset_px):
        for dy in (-offset_px, 0.0, offset_px):
            for angle_delta in angle_steps:
                for size_scale in size_scales:
                    trial_vote = {
                        "cx": center_x + dx,
                        "cy": center_y + dy,
                        "w": base_size * size_scale,
                        "h": base_size * size_scale,
                        "score": float(seed_vote["score"]),
                        "angle": base_angle + angle_delta,
                    }
                    trial_stats = _template_patch_stats(gray, trial_vote, config)
                    if (
                        trial_stats["template_patch_score"]
                        > best_stats["template_patch_score"]
                    ):
                        best_vote = trial_vote
                        best_stats = trial_stats

    return best_vote, best_stats


def _template_match_votes(
    image_bgr: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    rescue_cfg = detector_config(config)["template_rescue"]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(rescue_cfg["canny_low"]), int(rescue_cfg["canny_high"]))

    votes: list[dict[str, Any]] = []
    local_max_kernel = _odd_kernel(int(rescue_cfg["local_max_kernel_px"]))
    peak_kernel = np.ones((local_max_kernel, local_max_kernel), dtype=np.uint8)

    for size in rescue_cfg["sizes_px"]:
        template = _marker_template(int(size))
        template_edges = cv2.Canny(
            template, int(rescue_cfg["canny_low"]), int(rescue_cfg["canny_high"])
        )
        threshold = float(
            rescue_cfg["small_match_threshold"]
            if int(size) <= 22
            else rescue_cfg["large_match_threshold"]
        )

        for angle in rescue_cfg["angles_deg"]:
            matrix = cv2.getRotationMatrix2D(
                (size / 2.0, size / 2.0), float(angle), 1.0
            )
            rotated = cv2.warpAffine(
                template_edges,
                matrix,
                (int(size), int(size)),
                flags=cv2.INTER_LINEAR,
                borderValue=0,
            )
            response = cv2.matchTemplate(edges, rotated, cv2.TM_CCOEFF_NORMED)
            if response.size == 0:
                continue

            local_max = cv2.dilate(response, peak_kernel)
            ys, xs = np.where((response >= threshold) & (response == local_max))
            if len(xs) == 0:
                continue

            scores = response[ys, xs]
            order = np.argsort(scores)[::-1]
            cell_size = max(local_max_kernel, int(rescue_cfg.get("vote_cell_px", 24)))
            max_votes_per_cell = max(
                1, int(rescue_cfg.get("max_votes_per_cell_per_template", 2))
            )
            max_votes_per_template = max(
                1, int(rescue_cfg.get("max_votes_per_template", 90))
            )
            cell_counts: dict[tuple[int, int], int] = {}
            kept_votes = 0

            for index in order.tolist():
                y = int(ys[index])
                x = int(xs[index])
                cell = (
                    int((x + size / 2.0) // cell_size),
                    int((y + size / 2.0) // cell_size),
                )
                if cell_counts.get(cell, 0) >= max_votes_per_cell:
                    continue

                votes.append(
                    {
                        "x": x,
                        "y": y,
                        "w": int(size),
                        "h": int(size),
                        "cx": float(x + size / 2.0),
                        "cy": float(y + size / 2.0),
                        "score": float(response[y, x]),
                        "angle": float(angle),
                    }
                )
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
                kept_votes += 1
                if kept_votes >= max_votes_per_template:
                    break

    return votes


def _cluster_template_votes(
    votes: list[dict[str, Any]], config: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    if not votes:
        return []

    rescue_cfg = detector_config(config)["template_rescue"]
    groups: list[list[dict[str, Any]]] = []
    max_distance = float(rescue_cfg["cluster_distance_px"])

    for vote in sorted(votes, key=lambda item: item["score"], reverse=True):
        assigned = False
        for group in groups:
            center_x = float(np.mean([item["cx"] for item in group]))
            center_y = float(np.mean([item["cy"] for item in group]))
            median_size = float(np.median([item["w"] for item in group]))
            distance = float(np.hypot(vote["cx"] - center_x, vote["cy"] - center_y))
            if distance <= max(max_distance, median_size * 0.9):
                group.append(vote)
                assigned = True
                break
        if not assigned:
            groups.append([vote])

    return groups


def _rescue_cluster_bbox(
    cluster: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    rescue_cfg = detector_config(config)["template_rescue"]
    weights = np.array(
        [max(1e-3, float(item["score"])) for item in cluster], dtype=np.float32
    )
    center_x = float(np.average([item["cx"] for item in cluster], weights=weights))
    center_y = float(np.average([item["cy"] for item in cluster], weights=weights))
    median_size = float(np.median([max(item["w"], item["h"]) for item in cluster]))
    spread = float(
        np.median(
            [np.hypot(item["cx"] - center_x, item["cy"] - center_y) for item in cluster]
        )
    )
    half_side = median_size * (0.5 + float(rescue_cfg["bbox_pad_ratio"])) + spread + 3.0
    image_h, image_w = image_shape[:2]
    x1 = max(0, int(round(center_x - half_side)))
    y1 = max(0, int(round(center_y - half_side)))
    x2 = min(image_w, int(round(center_x + half_side)))
    y2 = min(image_h, int(round(center_y + half_side)))
    return x1, y1, x2 - x1, y2 - y1


def _expand_group_bbox(
    group: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    x1 = min(item["bbox"][0] for item in group)
    y1 = min(item["bbox"][1] for item in group)
    x2 = max(item["bbox"][0] + item["bbox"][2] for item in group)
    y2 = max(item["bbox"][1] + item["bbox"][3] for item in group)

    margin = int(
        round(
            max(x2 - x1, y2 - y1) * float(detector_config(config)["bbox_margin_ratio"])
            + 8
        )
    )
    image_h, image_w = image_shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image_w, x2 + margin)
    y2 = min(image_h, y2 + margin)
    return x1, y1, x2 - x1, y2 - y1


def _local_component_bbox(
    component: dict[str, Any],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    local_cfg = detector_config(config).get("local_dark_component", {})
    x, y, w, h = component["bbox"]
    pad = int(
        round(max(w, h) * float(local_cfg.get("candidate_pad_ratio", 1.0)))
    ) + int(local_cfg.get("candidate_pad_px", 8))
    image_h, image_w = image_shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_w, x + w + pad)
    y2 = min(image_h, y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def _internal_pattern_stats(
    roi_hsv: np.ndarray,
    roi_gray: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    det_cfg = detector_config(config)
    pattern_cfg = det_cfg["pattern"]
    roi_h, roi_w = roi_gray.shape[:2]
    roi_area = max(1, roi_h * roi_w)

    internal_mask = (
        (roi_gray < int(det_cfg["dark_threshold"]))
        & (roi_hsv[:, :, 1] < int(det_cfg["max_dark_saturation"]))
    ).astype(np.uint8) * 255

    area_threshold = max(
        int(pattern_cfg["min_internal_component_area_px"]),
        int(round(float(pattern_cfg["min_internal_component_area_ratio"]) * roi_area)),
    )

    contours, _ = cv2.findContours(
        internal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    significant_count = 0
    compact_count = 0
    max_component_solidity = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_threshold:
            continue

        significant_count += 1
        hull = cv2.convexHull(contour)
        hull_area = max(1.0, float(cv2.contourArea(hull)))
        solidity = area / hull_area
        max_component_solidity = max(max_component_solidity, solidity)

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if solidity >= float(pattern_cfg["compact_min_solidity"]) and int(
            pattern_cfg["compact_min_vertices"]
        ) <= len(approx) <= int(pattern_cfg["compact_max_vertices"]):
            compact_count += 1

    edge_mask = cv2.Canny(roi_gray, 80, 160)
    edge_ratio = float(np.mean(edge_mask > 0))

    return {
        "significant_component_count": significant_count,
        "compact_component_count": compact_count,
        "max_component_solidity": float(max_component_solidity),
        "edge_ratio": edge_ratio,
    }


def _score_square_candidate(
    image_bgr: np.ndarray,
    vote: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    det_cfg = detector_config(config)
    square_cfg = det_cfg.get("square_component", {})
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    patch_size = max(20, int(square_cfg.get("verify_patch_px", 40)))
    patch = _square_patch_from_vote(
        gray=gray,
        center_xy=(float(vote["cx"]), float(vote["cy"])),
        side_px=float(vote["w"]) * float(square_cfg.get("verify_side_scale", 1.05)),
        angle_deg=float(vote.get("angle", 0.0)),
        output_size=patch_size,
    )
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    patch_stats = _template_patch_stats(
        gray,
        vote,
        config,
        patch_size_override=patch_size,
        side_scale_override=float(square_cfg.get("verify_side_scale", 1.05)),
    )

    white_ratio = float(
        np.mean(
            patch
            > int(square_cfg.get("min_white_value", det_cfg["min_white_value"] - 20))
        )
    )
    black_ratio = float(
        np.mean(
            patch
            < int(square_cfg.get("max_black_value", det_cfg["dark_threshold"] + 5))
        )
    )
    contrast_std = float(np.std(patch))

    bbox = _square_vote_bbox(vote, image_bgr.shape[:2], config)
    x, y, w, h = bbox
    roi_hsv = hsv[y : y + h, x : x + w]
    if roi_hsv.size == 0:
        return None
    mean_saturation = float(np.mean(roi_hsv[:, :, 1]))

    strong_template_ok = (
        patch_stats["template_patch_score"]
        >= float(square_cfg.get("min_patch_score", 0.46))
        and patch_stats["template_corr"]
        >= float(square_cfg.get("min_template_corr", 0.42))
        and patch_stats["binary_agreement"]
        >= float(square_cfg.get("min_binary_agreement", 0.63))
        and patch_stats["contrast_separation"]
        >= float(square_cfg.get("min_template_contrast_separation", 0.18))
        and patch_stats["structure_score"]
        >= float(square_cfg.get("min_structure_score", 0.42))
    )
    relaxed_template_ok = (
        float(vote["score"]) >= float(square_cfg.get("relaxed_min_vote_score", 0.74))
        and patch_stats["template_patch_score"]
        >= float(square_cfg.get("relaxed_min_patch_score", 0.40))
        and patch_stats["template_corr"]
        >= float(square_cfg.get("relaxed_min_template_corr", 0.36))
        and patch_stats["binary_agreement"]
        >= float(square_cfg.get("relaxed_min_binary_agreement", 0.66))
        and patch_stats["contrast_separation"]
        >= float(square_cfg.get("relaxed_min_template_contrast_separation", 0.24))
        and patch_stats["structure_score"]
        >= float(square_cfg.get("relaxed_min_structure_score", 0.36))
    )
    high_vote_template_ok = (
        float(vote["score"]) >= float(square_cfg.get("high_vote_min_vote_score", 0.80))
        and patch_stats["template_patch_score"]
        >= float(square_cfg.get("high_vote_min_patch_score", 0.32))
        and patch_stats["template_corr"]
        >= float(square_cfg.get("high_vote_min_template_corr", 0.24))
        and patch_stats["binary_agreement"]
        >= float(square_cfg.get("high_vote_min_binary_agreement", 0.60))
        and patch_stats["contrast_separation"]
        >= float(square_cfg.get("high_vote_min_template_contrast_separation", 0.12))
        and patch_stats["structure_score"]
        >= float(square_cfg.get("high_vote_min_structure_score", 0.33))
    )
    tiny_template_ok = (
        float(vote.get("w", 0.0)) <= float(square_cfg.get("tiny_max_size_px", 10))
        and float(vote["score"]) >= float(square_cfg.get("tiny_min_vote_score", 0.55))
        and patch_stats["template_patch_score"]
        >= float(square_cfg.get("tiny_min_patch_score", 0.42))
        and patch_stats["template_corr"]
        >= float(square_cfg.get("tiny_min_template_corr", 0.45))
        and patch_stats["binary_agreement"]
        >= float(square_cfg.get("tiny_min_binary_agreement", 0.68))
        and patch_stats["contrast_separation"]
        >= float(square_cfg.get("tiny_min_template_contrast_separation", 0.03))
        and patch_stats["structure_score"]
        >= float(square_cfg.get("tiny_min_structure_score", 0.35))
    )
    if (
        not strong_template_ok
        and not relaxed_template_ok
        and not high_vote_template_ok
        and not tiny_template_ok
    ):
        return None
    if white_ratio < float(square_cfg.get("min_white_ratio", 0.26)):
        return None
    if black_ratio < float(square_cfg.get("min_black_ratio", 0.20)):
        return None
    if contrast_std < float(square_cfg.get("min_contrast_std", 26.0)):
        return None
    if mean_saturation > float(square_cfg.get("max_mean_saturation", 115.0)):
        return None

    score = (
        patch_stats["template_patch_score"] * 0.42
        + patch_stats["template_corr"] * 0.18
        + patch_stats["binary_agreement"] * 0.12
        + patch_stats["structure_score"] * 0.16
        + min(
            1.0, white_ratio / max(float(square_cfg.get("min_white_ratio", 0.26)), 1e-6)
        )
        * 0.06
        + min(
            1.0, black_ratio / max(float(square_cfg.get("min_black_ratio", 0.20)), 1e-6)
        )
        * 0.06
    )
    score = min(1.0, float(score))

    return {
        "bbox": bbox,
        "score": score,
        "source": "square_template",
        "white_ratio": white_ratio,
        "black_ratio": black_ratio,
        "red_ratio": 0.0,
        "green_ratio": 0.0,
        "contrast_std": contrast_std,
        "component_count": 1,
        "significant_component_count": 1,
        "compact_component_count": 1,
        "max_component_solidity": 1.0,
        "edge_ratio": 0.0,
        "template_vote_count": 1,
        "template_mean_score": float(vote["score"]),
        "template_patch_score": float(patch_stats["template_patch_score"]),
        "template_corr": float(patch_stats["template_corr"]),
        "binary_agreement": float(patch_stats["binary_agreement"]),
        "contrast_separation": float(patch_stats["contrast_separation"]),
        "structure_score": float(patch_stats["structure_score"]),
        "template_gate": (
            "strong"
            if strong_template_ok
            else (
                "relaxed"
                if relaxed_template_ok
                else "high_vote" if high_vote_template_ok else "tiny"
            )
        ),
        "refined_center_x": float(vote["cx"]),
        "refined_center_y": float(vote["cy"]),
    }


def _component_seeded_template_candidates(
    image_bgr: np.ndarray,
    seed_components: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    det_cfg = detector_config(config)
    component_cfg = det_cfg.get("component_template", {})
    if not bool(component_cfg.get("enabled", False)) or not seed_components:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_shape = image_bgr.shape[:2]
    max_seed_components = max(1, int(component_cfg.get("max_seed_components", 24)))
    max_candidates = max(1, int(component_cfg.get("max_candidates", 10)))
    min_component_area = float(component_cfg.get("min_component_area_px", 5))
    offset_scale = float(component_cfg.get("offset_scale", 0.28))
    offset_dirs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    ]

    candidates: list[dict[str, Any]] = []
    sorted_components = sorted(
        (
            component
            for component in seed_components
            if float(component.get("area", 0.0)) >= min_component_area
        ),
        key=lambda component: (
            0 if component.get("seed_source") == "lenient_component_template" else 1,
            float(component.get("support_score", 0.0)),
            float(component.get("seed_score", 0.0)),
            float(component.get("area", 0.0)),
        ),
        reverse=True,
    )[:max_seed_components]

    for component in sorted_components:
        component_x, component_y = component["center"]
        component_w = float(component["bbox"][2])
        component_h = float(component["bbox"][3])
        component_size = max(component_w, component_h)
        for side in component_cfg.get("sizes_px", [10, 14, 18]):
            side = int(side)
            if side < max(8, int(round(component_size * 0.8))) or side > max(
                10, int(round(component_size * 3.6))
            ):
                continue
            offset_px = max(2.0, float(side) * offset_scale)
            for dir_x, dir_y in offset_dirs:
                center_x = float(component_x) + dir_x * offset_px
                center_y = float(component_y) + dir_y * offset_px
                if (
                    center_x < 0.0
                    or center_y < 0.0
                    or center_x >= image_shape[1]
                    or center_y >= image_shape[0]
                ):
                    continue
                for angle in component_cfg.get("angles_deg", [0, 45, 90, 135]):
                    vote = {
                        "cx": center_x,
                        "cy": center_y,
                        "w": side,
                        "h": side,
                        "score": 0.0,
                        "angle": float(angle),
                    }
                    patch_stats = _template_patch_stats(
                        gray,
                        vote,
                        config,
                        patch_size_override=32,
                        side_scale_override=1.15,
                    )
                    if patch_stats["template_patch_score"] < float(
                        component_cfg.get("min_patch_score", 0.52)
                    ):
                        continue
                    if patch_stats["template_corr"] < float(
                        component_cfg.get("min_template_corr", 0.55)
                    ):
                        continue
                    if patch_stats["binary_agreement"] < float(
                        component_cfg.get("min_binary_agreement", 0.62)
                    ):
                        continue
                    if patch_stats["contrast_separation"] < float(
                        component_cfg.get("min_template_contrast_separation", 0.12)
                    ):
                        continue
                    if patch_stats["structure_score"] < float(
                        component_cfg.get("min_structure_score", 0.39)
                    ):
                        continue

                    bbox = _square_vote_bbox(vote, image_shape, config)
                    x, y, w, h = bbox
                    roi_gray = gray[y : y + h, x : x + w]
                    roi_hsv = hsv[y : y + h, x : x + w]
                    if roi_gray.size == 0 or roi_hsv.size == 0:
                        continue
                    white_ratio = float(
                        np.mean(
                            (
                                roi_hsv[:, :, 1]
                                < int(det_cfg["max_white_saturation"]) + 35
                            )
                            & (roi_hsv[:, :, 2] > 135)
                        )
                    )
                    black_ratio = float(
                        np.mean(roi_gray < int(det_cfg["dark_threshold"]) + 10)
                    )
                    contrast_std = float(np.std(roi_gray))
                    if white_ratio < float(component_cfg.get("min_white_ratio", 0.18)):
                        continue
                    if black_ratio < float(component_cfg.get("min_black_ratio", 0.08)):
                        continue
                    if contrast_std < float(
                        component_cfg.get("min_contrast_std", 18.0)
                    ):
                        continue

                    score = (
                        patch_stats["template_patch_score"] * 0.44
                        + patch_stats["template_corr"] * 0.24
                        + patch_stats["binary_agreement"] * 0.10
                        + patch_stats["contrast_separation"] * 0.10
                        + patch_stats["structure_score"] * 0.12
                    )
                    candidates.append(
                        {
                            "bbox": bbox,
                            "score": min(1.0, float(score)),
                            "source": "component_template",
                            "white_ratio": white_ratio,
                            "black_ratio": black_ratio,
                            "red_ratio": 0.0,
                            "green_ratio": 0.0,
                            "contrast_std": contrast_std,
                            "component_count": 1,
                            "significant_component_count": 1,
                            "compact_component_count": 1,
                            "max_component_solidity": float(
                                component.get("solidity", 1.0)
                            ),
                            "edge_ratio": 0.0,
                            "template_vote_count": 1,
                            "template_mean_score": float(component.get("area", 0.0)),
                            "template_patch_score": float(
                                patch_stats["template_patch_score"]
                            ),
                            "template_corr": float(patch_stats["template_corr"]),
                            "binary_agreement": float(patch_stats["binary_agreement"]),
                            "contrast_separation": float(
                                patch_stats["contrast_separation"]
                            ),
                            "structure_score": float(patch_stats["structure_score"]),
                            "template_gate": "component_seeded",
                            "refined_center_x": center_x,
                            "refined_center_y": center_y,
                        }
                    )

    limited = _dedupe_candidates_by_center(
        candidates,
        center_ratio=float(component_cfg.get("center_dedupe_ratio", 0.85)),
        center_min_px=float(component_cfg.get("center_dedupe_min_px", 14.0)),
    )
    return sorted(limited, key=lambda candidate: candidate["score"], reverse=True)[
        :max_candidates
    ]


def _self_template_candidates(
    image_bgr: np.ndarray,
    anchor_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    self_cfg = detector_config(config).get("self_template", {})
    if not bool(self_cfg.get("enabled", False)):
        return []

    anchors = [
        candidate
        for candidate in anchor_candidates
        if candidate.get("source")
        in set(self_cfg.get("anchor_sources", ["component_template"]))
        and float(candidate.get("score", 0.0))
        >= float(self_cfg.get("min_anchor_score", 0.60))
    ]
    if not anchors:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    normalized_gray = _normalized_square_match_gray(gray, config)
    match_threshold = float(self_cfg.get("match_threshold", 0.74))
    center_min_px = float(self_cfg.get("center_dedupe_min_px", 18.0))
    max_candidates = max(1, int(self_cfg.get("max_candidates", 8)))

    raw_candidates: list[dict[str, Any]] = []
    for anchor in sorted(anchors, key=lambda item: item["score"], reverse=True)[:2]:
        x, y, w, h = anchor["bbox"]
        side = max(w, h)
        pad = max(2, int(round(side * 0.15)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_bgr.shape[1], x + w + pad)
        y2 = min(image_bgr.shape[0], y + h + pad)
        template = normalized_gray[y1:y2, x1:x2]
        if template.size == 0 or min(template.shape[:2]) < 6:
            continue
        anchor_center = _candidate_center(anchor)

        for size in self_cfg.get("sizes_px", [8, 10, 12, 14]):
            size = int(size)
            resized = cv2.resize(template, (size, size), interpolation=cv2.INTER_AREA)
            for rotation_idx in self_cfg.get("rotations_90", [0, 1, 2, 3]):
                rotated = np.ascontiguousarray(np.rot90(resized, int(rotation_idx)))
                response = cv2.matchTemplate(
                    normalized_gray, rotated, cv2.TM_CCOEFF_NORMED
                )
                if response.size == 0:
                    continue
                local_max = cv2.dilate(response, np.ones((7, 7), dtype=np.uint8))
                ys, xs = np.where(
                    (response >= match_threshold) & (response == local_max)
                )
                if len(xs) == 0:
                    continue
                scores = response[ys, xs]
                for index in np.argsort(scores)[::-1][:80].tolist():
                    center_x = float(xs[index] + size / 2.0)
                    center_y = float(ys[index] + size / 2.0)
                    if (
                        float(
                            np.hypot(
                                center_x - anchor_center[0], center_y - anchor_center[1]
                            )
                        )
                        < center_min_px
                    ):
                        continue

                    half_side = size * 0.8
                    bx1 = max(0, int(round(center_x - half_side)))
                    by1 = max(0, int(round(center_y - half_side)))
                    bx2 = min(image_bgr.shape[1], int(round(center_x + half_side)))
                    by2 = min(image_bgr.shape[0], int(round(center_y + half_side)))
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    roi_gray = gray[by1:by2, bx1:bx2]
                    roi_hsv = hsv[by1:by2, bx1:bx2]
                    white_ratio = float(
                        np.mean((roi_hsv[:, :, 1] < 110) & (roi_hsv[:, :, 2] > 150))
                    )
                    black_ratio = float(np.mean(roi_gray < 150))
                    contrast_std = float(np.std(roi_gray))
                    if white_ratio < float(self_cfg.get("min_white_ratio", 0.12)):
                        continue
                    if black_ratio < float(self_cfg.get("min_black_ratio", 0.04)):
                        continue
                    if contrast_std < float(self_cfg.get("min_contrast_std", 12.0)):
                        continue

                    score = min(
                        1.0, float(scores[index]) * 0.75 + float(anchor["score"]) * 0.25
                    )
                    raw_candidates.append(
                        {
                            "bbox": (bx1, by1, bx2 - bx1, by2 - by1),
                            "score": score,
                            "source": "self_template",
                            "white_ratio": white_ratio,
                            "black_ratio": black_ratio,
                            "red_ratio": 0.0,
                            "green_ratio": 0.0,
                            "contrast_std": contrast_std,
                            "component_count": 1,
                            "significant_component_count": 1,
                            "compact_component_count": 1,
                            "max_component_solidity": float(scores[index]),
                            "edge_ratio": 0.0,
                            "template_vote_count": 1,
                            "template_mean_score": float(scores[index]),
                            "template_patch_score": float(scores[index]),
                            "template_corr": float(scores[index]),
                            "binary_agreement": 0.0,
                            "contrast_separation": 0.0,
                            "structure_score": 0.0,
                            "template_gate": "self_template",
                            "refined_center_x": center_x,
                            "refined_center_y": center_y,
                        }
                    )

    limited = _dedupe_candidates_by_center(
        raw_candidates, center_ratio=0.85, center_min_px=center_min_px
    )
    return sorted(limited, key=lambda candidate: candidate["score"], reverse=True)[
        :max_candidates
    ]


def _score_candidate(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    group: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    det_cfg = detector_config(config)
    candidate_cfg = det_cfg["candidates"]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_h, image_w = image_bgr.shape[:2]

    x, y, w, h = bbox
    bbox_area_ratio = (w * h) / float(image_w * image_h)
    if bbox_area_ratio < float(
        candidate_cfg["min_area_ratio"]
    ) or bbox_area_ratio > float(candidate_cfg["max_area_ratio"]):
        return None

    roi_hsv = hsv[y : y + h, x : x + w]
    roi_gray = gray[y : y + h, x : x + w]
    if roi_hsv.size == 0:
        return None

    white_ratio = float(
        np.mean(
            (roi_hsv[:, :, 1] < int(det_cfg["max_white_saturation"]))
            & (roi_hsv[:, :, 2] > int(det_cfg["min_white_value"]))
        )
    )
    black_ratio = float(
        np.mean(
            (roi_gray < int(det_cfg["dark_threshold"]))
            & (roi_hsv[:, :, 1] < int(det_cfg["max_dark_saturation"]))
        )
    )
    red_ratio = float(
        np.mean(
            ((roi_hsv[:, :, 0] <= 14) | (roi_hsv[:, :, 0] >= 166))
            & (roi_hsv[:, :, 1] > 70)
            & (roi_hsv[:, :, 2] > 70)
        )
    )
    green_ratio = float(
        np.mean(
            (roi_hsv[:, :, 0] >= 35)
            & (roi_hsv[:, :, 0] <= 95)
            & (roi_hsv[:, :, 1] > 70)
            & (roi_hsv[:, :, 2] > 70)
        )
    )
    contrast_std = float(np.std(roi_gray))
    pattern_stats = _internal_pattern_stats(roi_hsv, roi_gray, config)

    if white_ratio < float(det_cfg["min_white_ratio"]):
        return None
    if black_ratio < float(det_cfg["min_black_ratio"]):
        return None
    if contrast_std < float(det_cfg["min_contrast_std"]):
        return None
    if red_ratio > float(det_cfg["max_red_ratio"]):
        return None
    if green_ratio > float(det_cfg["max_green_ratio"]):
        return None
    if pattern_stats["significant_component_count"] > int(
        det_cfg["pattern"]["max_significant_components"]
    ):
        return None

    pattern_ok = pattern_stats["compact_component_count"] >= int(
        det_cfg["pattern"]["min_compact_components"]
    )
    clean_partial_ok = (
        white_ratio >= float(det_cfg["pattern"]["strong_white_ratio"])
        and pattern_stats["significant_component_count"] <= 2
        and pattern_stats["edge_ratio"]
        <= float(det_cfg["pattern"]["relaxed_edge_ratio"])
    )
    if not pattern_ok and not clean_partial_ok:
        return None

    component_count = len(group)
    score = (
        min(1.0, white_ratio / max(float(det_cfg["min_white_ratio"]), 1e-6)) * 0.25
        + min(1.0, black_ratio / max(float(det_cfg["min_black_ratio"]), 1e-6)) * 0.30
        + min(1.0, contrast_std / max(float(det_cfg["min_contrast_std"]), 1e-6)) * 0.20
        + min(1.0, component_count / 3.0) * 0.15
        + (1.0 - min(1.0, red_ratio / max(float(det_cfg["max_red_ratio"]), 1e-6)))
        * 0.05
        + (1.0 - min(1.0, green_ratio / max(float(det_cfg["max_green_ratio"]), 1e-6)))
        * 0.05
    )
    score += min(0.15, 0.08 * pattern_stats["compact_component_count"])
    score += max(0.0, 0.05 * (1.0 - min(1.0, pattern_stats["edge_ratio"] / 0.25)))
    score = min(1.0, float(score))

    return {
        "bbox": bbox,
        "score": float(score),
        "source": "dark_component",
        "white_ratio": white_ratio,
        "black_ratio": black_ratio,
        "red_ratio": red_ratio,
        "green_ratio": green_ratio,
        "contrast_std": contrast_std,
        "component_count": component_count,
        "significant_component_count": int(
            pattern_stats["significant_component_count"]
        ),
        "compact_component_count": int(pattern_stats["compact_component_count"]),
        "max_component_solidity": float(pattern_stats["max_component_solidity"]),
        "edge_ratio": float(pattern_stats["edge_ratio"]),
    }


def _score_rescue_candidate(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    cluster: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    det_cfg = detector_config(config)
    rescue_cfg = det_cfg["template_rescue"]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_h, image_w = image_bgr.shape[:2]

    x, y, w, h = bbox
    bbox_area_ratio = (w * h) / float(image_w * image_h)
    if bbox_area_ratio > float(rescue_cfg["max_area_ratio"]):
        return None

    roi_hsv = hsv[y : y + h, x : x + w]
    roi_gray = gray[y : y + h, x : x + w]
    if roi_hsv.size == 0:
        return None

    white_ratio = float(
        np.mean(
            (roi_hsv[:, :, 1] < int(det_cfg["max_white_saturation"]) + 10)
            & (roi_hsv[:, :, 2] > int(det_cfg["min_white_value"]) - 30)
        )
    )
    black_ratio = float(
        np.mean(
            (roi_gray < int(det_cfg["dark_threshold"]) - 25)
            & (roi_hsv[:, :, 1] < int(det_cfg["max_dark_saturation"]) + 50)
        )
    )
    red_ratio = float(
        np.mean(
            ((roi_hsv[:, :, 0] <= 14) | (roi_hsv[:, :, 0] >= 166))
            & (roi_hsv[:, :, 1] > 70)
            & (roi_hsv[:, :, 2] > 70)
        )
    )
    green_ratio = float(
        np.mean(
            (roi_hsv[:, :, 0] >= 35)
            & (roi_hsv[:, :, 0] <= 95)
            & (roi_hsv[:, :, 1] > 70)
            & (roi_hsv[:, :, 2] > 70)
        )
    )
    contrast_std = float(np.std(roi_gray))
    refined_vote, patch_stats = _refine_template_vote(gray, cluster, config)

    if white_ratio < float(rescue_cfg["min_white_ratio"]):
        return None
    if black_ratio < float(rescue_cfg["min_black_ratio"]):
        return None
    if contrast_std < float(rescue_cfg["min_contrast_std"]):
        return None
    if red_ratio > float(rescue_cfg["max_red_ratio"]):
        return None
    if green_ratio > float(rescue_cfg["max_green_ratio"]):
        return None
    if len(cluster) < int(rescue_cfg["min_cluster_votes"]):
        return None
    if patch_stats["template_corr"] < float(rescue_cfg.get("min_template_corr", 0.34)):
        return None
    if patch_stats["binary_agreement"] < float(
        rescue_cfg.get("min_binary_agreement", 0.68)
    ):
        return None
    if patch_stats["contrast_separation"] < float(
        rescue_cfg.get("min_template_contrast_separation", 0.18)
    ):
        return None
    if patch_stats["structure_score"] < float(
        rescue_cfg.get("min_structure_score", 0.48)
    ):
        return None

    score = (
        min(1.0, len(cluster) / 5.0) * 0.35
        + min(1.0, float(np.mean([item["score"] for item in cluster])) / 0.35) * 0.25
        + min(1.0, white_ratio / max(float(rescue_cfg["min_white_ratio"]), 1e-6)) * 0.15
        + min(1.0, black_ratio / max(float(rescue_cfg["min_black_ratio"]), 1e-6)) * 0.15
        + min(1.0, contrast_std / max(float(rescue_cfg["min_contrast_std"]), 1e-6))
        * 0.10
    )
    score = score * 0.70 + patch_stats["template_patch_score"] * 0.30
    score = min(1.0, float(score))

    return {
        "bbox": bbox,
        "score": float(score),
        "source": "template_rescue",
        "white_ratio": white_ratio,
        "black_ratio": black_ratio,
        "red_ratio": red_ratio,
        "green_ratio": green_ratio,
        "contrast_std": contrast_std,
        "component_count": len(cluster),
        "significant_component_count": len(cluster),
        "compact_component_count": len(cluster),
        "max_component_solidity": float(np.mean([item["score"] for item in cluster])),
        "edge_ratio": 0.0,
        "template_vote_count": len(cluster),
        "template_mean_score": float(np.mean([item["score"] for item in cluster])),
        "template_patch_score": float(patch_stats["template_patch_score"]),
        "template_corr": float(patch_stats["template_corr"]),
        "binary_agreement": float(patch_stats["binary_agreement"]),
        "contrast_separation": float(patch_stats["contrast_separation"]),
        "structure_score": float(patch_stats["structure_score"]),
        "refined_center_x": float(refined_vote["cx"]),
        "refined_center_y": float(refined_vote["cy"]),
    }


def _score_relaxed_rescue_candidate(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    cluster: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    det_cfg = detector_config(config)
    rescue_cfg = det_cfg["template_rescue"]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_h, image_w = image_bgr.shape[:2]

    x, y, w, h = bbox
    bbox_area_ratio = (w * h) / float(image_w * image_h)
    if bbox_area_ratio > float(rescue_cfg["max_area_ratio"]):
        return None

    roi_hsv = hsv[y : y + h, x : x + w]
    roi_gray = gray[y : y + h, x : x + w]
    if roi_hsv.size == 0:
        return None

    white_ratio = float(
        np.mean(
            (roi_hsv[:, :, 1] < int(det_cfg["max_white_saturation"]) + 25)
            & (roi_hsv[:, :, 2] > int(det_cfg["min_white_value"]) - 55)
        )
    )
    black_ratio = float(
        np.mean(
            (roi_gray < int(det_cfg["dark_threshold"]) - 35)
            & (roi_hsv[:, :, 1] < int(det_cfg["max_dark_saturation"]) + 70)
        )
    )
    red_ratio = float(
        np.mean(
            ((roi_hsv[:, :, 0] <= 14) | (roi_hsv[:, :, 0] >= 166))
            & (roi_hsv[:, :, 1] > 70)
            & (roi_hsv[:, :, 2] > 70)
        )
    )
    green_ratio = float(
        np.mean(
            (roi_hsv[:, :, 0] >= 35)
            & (roi_hsv[:, :, 0] <= 95)
            & (roi_hsv[:, :, 1] > 70)
            & (roi_hsv[:, :, 2] > 70)
        )
    )
    contrast_std = float(np.std(roi_gray))
    refined_vote, patch_stats = _refine_template_vote(gray, cluster, config)

    if patch_stats["template_patch_score"] < float(
        rescue_cfg.get("fallback_min_patch_score", 0.45)
    ):
        return None
    if patch_stats["template_corr"] < float(
        rescue_cfg.get("fallback_min_template_corr", 0.52)
    ):
        return None
    if patch_stats["binary_agreement"] < float(
        rescue_cfg.get("fallback_min_binary_agreement", 0.70)
    ):
        return None
    if patch_stats["contrast_separation"] < float(
        rescue_cfg.get("fallback_min_contrast_separation", 0.03)
    ):
        return None
    if patch_stats["structure_score"] < float(
        rescue_cfg.get("fallback_min_structure_score", 0.52)
    ):
        return None
    if black_ratio < float(rescue_cfg.get("fallback_min_black_ratio", 0.0)):
        return None
    if contrast_std < float(rescue_cfg.get("fallback_min_contrast_std", 0.0)):
        return None

    vote_count = len(cluster)
    best_vote_score = float(max(item["score"] for item in cluster))
    if vote_count < int(
        rescue_cfg.get("fallback_min_cluster_votes", 1)
    ) and best_vote_score < float(rescue_cfg.get("fallback_min_vote_score", 0.32)):
        return None

    score = (
        patch_stats["template_patch_score"] * 0.45
        + patch_stats["template_corr"] * 0.25
        + patch_stats["binary_agreement"] * 0.15
        + min(1.0, vote_count / 8.0) * 0.10
        + min(1.0, best_vote_score / 0.45) * 0.05
    )
    score = max(score, float(rescue_cfg.get("fallback_min_output_score", 0.52)))
    score = min(1.0, float(score))

    return {
        "bbox": bbox,
        "score": float(score),
        "source": "template_rescue_fallback",
        "white_ratio": white_ratio,
        "black_ratio": black_ratio,
        "red_ratio": red_ratio,
        "green_ratio": green_ratio,
        "contrast_std": contrast_std,
        "component_count": vote_count,
        "significant_component_count": vote_count,
        "compact_component_count": vote_count,
        "max_component_solidity": best_vote_score,
        "edge_ratio": 0.0,
        "template_vote_count": vote_count,
        "template_mean_score": float(np.mean([item["score"] for item in cluster])),
        "template_patch_score": float(patch_stats["template_patch_score"]),
        "template_corr": float(patch_stats["template_corr"]),
        "binary_agreement": float(patch_stats["binary_agreement"]),
        "contrast_separation": float(patch_stats["contrast_separation"]),
        "structure_score": float(patch_stats["structure_score"]),
        "refined_center_x": float(refined_vote["cx"]),
        "refined_center_y": float(refined_vote["cy"]),
    }


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (aw * ah) + (bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def _nms(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    iou_threshold = float(detector_config(config)["nms_iou_threshold"])
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(
            _iou(candidate["bbox"], existing["bbox"]) < iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def _candidate_center(candidate: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = candidate["bbox"]
    return (
        float(candidate.get("refined_center_x", x + w / 2.0)),
        float(candidate.get("refined_center_y", y + h / 2.0)),
    )


def _dedupe_candidates_by_center(
    candidates: list[dict[str, Any]],
    *,
    center_ratio: float,
    center_min_px: float,
) -> list[dict[str, Any]]:
    source_bonus = {
        "square_template": 0.06,
        "template_rescue": 0.04,
        "template_rescue_fallback": 0.03,
        "dark_component": 0.02,
        "local_dark_component": 0.01,
    }
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["score"]) + source_bonus.get(str(item.get("source")), 0.0),
            float(item.get("template_patch_score", 0.0)),
            float(item.get("contrast_separation", 0.0)),
        ),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        center_x, center_y = _candidate_center(candidate)
        w, h = candidate["bbox"][2], candidate["bbox"][3]
        too_close = False
        for existing in kept:
            existing_x, existing_y = _candidate_center(existing)
            existing_w, existing_h = existing["bbox"][2], existing["bbox"][3]
            min_distance = max(
                center_min_px,
                center_ratio * max(w, h, existing_w, existing_h),
            )
            if (
                float(np.hypot(center_x - existing_x, center_y - existing_y))
                < min_distance
            ):
                too_close = True
                break
        if not too_close:
            kept.append(candidate)
    return kept


def _limit_square_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    square_cfg = detector_config(config).get("square_component", {})
    if not candidates:
        return []

    cell_size = max(24, int(square_cfg.get("spatial_cell_px", 96)))
    max_boxes_per_cell = max(1, int(square_cfg.get("max_boxes_per_cell", 1)))
    max_total_boxes = max(1, int(square_cfg.get("max_total_boxes", len(candidates))))
    dense_threshold = max(
        1, int(square_cfg.get("dense_scene_candidate_threshold", max_total_boxes))
    )
    dense_max_total_boxes = max(
        1, int(square_cfg.get("dense_scene_max_total_boxes", max_total_boxes))
    )
    center_ratio = float(square_cfg.get("center_dedupe_ratio", 0.85))
    center_min_px = float(square_cfg.get("center_dedupe_min_px", 20.0))
    effective_max_total = (
        dense_max_total_boxes if len(candidates) >= dense_threshold else max_total_boxes
    )

    limited: list[dict[str, Any]] = []
    per_cell_counts: dict[tuple[int, int], int] = {}
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["score"]),
            float(item.get("template_patch_score", 0.0)),
            float(item.get("contrast_separation", 0.0)),
            float(item.get("template_corr", 0.0)),
        ),
        reverse=True,
    )

    for candidate in ordered:
        center_x, center_y = _candidate_center(candidate)
        w, h = candidate["bbox"][2], candidate["bbox"][3]
        if any(
            float(np.hypot(center_x - kept_x, center_y - kept_y))
            < max(
                center_min_px,
                center_ratio * max(w, h, kept["bbox"][2], kept["bbox"][3]),
            )
            for kept in limited
            for kept_x, kept_y in [_candidate_center(kept)]
        ):
            continue

        cell = (int(center_x // cell_size), int(center_y // cell_size))
        if per_cell_counts.get(cell, 0) >= max_boxes_per_cell:
            continue

        limited.append(candidate)
        per_cell_counts[cell] = per_cell_counts.get(cell, 0) + 1
        if len(limited) >= effective_max_total:
            break

    return limited


def _reject_border_touching_candidates(
    candidates: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    border_px = int(detector_config(config).get("reject_border_touching_px", 0))
    if border_px <= 0:
        return candidates

    image_h, image_w = image_shape[:2]
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        touches_border = (
            x <= border_px
            or y <= border_px
            or x + w >= image_w - border_px
            or y + h >= image_h - border_px
        )
        if not touches_border:
            filtered.append(candidate)
    return filtered


def _limit_template_rescue_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rescue_cfg = detector_config(config)["template_rescue"]
    cell_size = max(24, int(rescue_cfg["spatial_cell_px"]))
    max_boxes_per_cell = max(1, int(rescue_cfg["max_boxes_per_cell"]))
    max_total_boxes = max(1, int(rescue_cfg["max_total_boxes"]))
    dense_threshold = max(
        1, int(rescue_cfg.get("dense_scene_candidate_threshold", max_total_boxes))
    )
    dense_max_total_boxes = max(
        1, int(rescue_cfg.get("dense_scene_max_total_boxes", max_total_boxes))
    )
    dense_min_patch_score = float(rescue_cfg.get("dense_scene_min_patch_score", 0.0))
    dense_min_template_corr = float(
        rescue_cfg.get("dense_scene_min_template_corr", 0.0)
    )
    dense_min_vote_count = max(0, int(rescue_cfg.get("dense_scene_min_vote_count", 0)))
    center_dedupe_ratio = float(rescue_cfg.get("center_dedupe_ratio", 0.8))
    center_dedupe_min_px = float(rescue_cfg.get("center_dedupe_min_px", 18.0))
    is_dense_scene = len(candidates) >= dense_threshold
    effective_max_total_boxes = (
        dense_max_total_boxes if is_dense_scene else max_total_boxes
    )

    limited: list[dict[str, Any]] = []
    per_cell_counts: dict[tuple[int, int], int] = {}
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            item["score"],
            item.get("template_patch_score", 0.0),
            item.get("template_corr", 0.0),
            -max(item["bbox"][2], item["bbox"][3]),
            -item.get("template_vote_count", 0),
        ),
        reverse=True,
    )

    for candidate in sorted_candidates:
        if is_dense_scene:
            if candidate.get("template_patch_score", 0.0) < dense_min_patch_score:
                continue
            if candidate.get("template_corr", 0.0) < dense_min_template_corr:
                continue
            if candidate.get("structure_score", 0.0) < float(
                rescue_cfg.get("dense_scene_min_structure_score", 0.0)
            ):
                continue
            if candidate.get("template_vote_count", 0) < dense_min_vote_count:
                continue

        x, y, w, h = candidate["bbox"]
        center_x = float(candidate.get("refined_center_x", x + w / 2.0))
        center_y = float(candidate.get("refined_center_y", y + h / 2.0))

        too_close = False
        for kept in limited:
            kept_x, kept_y = kept["bbox"][0], kept["bbox"][1]
            kept_w, kept_h = kept["bbox"][2], kept["bbox"][3]
            kept_center_x = float(kept.get("refined_center_x", kept_x + kept_w / 2.0))
            kept_center_y = float(kept.get("refined_center_y", kept_y + kept_h / 2.0))
            min_distance = max(
                center_dedupe_min_px,
                center_dedupe_ratio * max(w, h, kept_w, kept_h),
            )
            if (
                float(np.hypot(center_x - kept_center_x, center_y - kept_center_y))
                < min_distance
            ):
                too_close = True
                break
        if too_close:
            continue

        cell = (int(center_x // cell_size), int(center_y // cell_size))
        current = per_cell_counts.get(cell, 0)
        if current >= max_boxes_per_cell:
            continue
        limited.append(candidate)
        per_cell_counts[cell] = current + 1
        if len(limited) >= effective_max_total_boxes:
            break

    return limited


def _append_recall_fallback_candidates(
    image_bgr: np.ndarray,
    selected: list[dict[str, Any]],
    template_clusters: list[list[dict[str, Any]]],
    anchor_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rescue_cfg = detector_config(config)["template_rescue"]
    trigger_max_selected = max(
        0, int(rescue_cfg.get("fallback_trigger_max_selected", 0))
    )
    if len(selected) > trigger_max_selected or not template_clusters:
        return selected

    fallback_candidates: list[dict[str, Any]] = []
    for cluster in template_clusters:
        bbox = _rescue_cluster_bbox(cluster, image_bgr.shape[:2], config)
        candidate = _score_relaxed_rescue_candidate(image_bgr, bbox, cluster, config)
        if candidate is not None:
            fallback_candidates.append(candidate)

    if not fallback_candidates:
        return selected

    merged = list(selected)
    seed_centers = [
        (
            float(
                candidate.get(
                    "refined_center_x",
                    candidate["bbox"][0] + candidate["bbox"][2] / 2.0,
                )
            ),
            float(
                candidate.get(
                    "refined_center_y",
                    candidate["bbox"][1] + candidate["bbox"][3] / 2.0,
                )
            ),
        )
        for candidate in selected
    ]
    center_ratio = float(
        rescue_cfg.get(
            "fallback_center_dedupe_ratio", rescue_cfg.get("center_dedupe_ratio", 0.85)
        )
    )
    center_min_px = float(
        rescue_cfg.get(
            "fallback_center_dedupe_min_px",
            rescue_cfg.get("center_dedupe_min_px", 20.0),
        )
    )
    attach_radius_px = float(rescue_cfg.get("fallback_attach_radius_px", 0.0))
    anchor_attach_scale = float(rescue_cfg.get("fallback_anchor_attach_scale", 3.2))
    anchor_bonus_radius_px = float(
        rescue_cfg.get("fallback_anchor_bonus_radius_px", 120.0)
    )

    remaining = list(fallback_candidates)
    max_total_boxes = max(
        1, int(rescue_cfg.get("fallback_max_total_boxes", len(merged)))
    )

    while remaining and len(merged) < max_total_boxes:
        ranked: list[tuple[float, float, int, dict[str, Any]]] = []
        for index, candidate in enumerate(remaining):
            center_x = float(
                candidate.get(
                    "refined_center_x",
                    candidate["bbox"][0] + candidate["bbox"][2] / 2.0,
                )
            )
            center_y = float(
                candidate.get(
                    "refined_center_y",
                    candidate["bbox"][1] + candidate["bbox"][3] / 2.0,
                )
            )
            too_close = False
            for existing in merged:
                existing_x = float(
                    existing.get(
                        "refined_center_x",
                        existing["bbox"][0] + existing["bbox"][2] / 2.0,
                    )
                )
                existing_y = float(
                    existing.get(
                        "refined_center_y",
                        existing["bbox"][1] + existing["bbox"][3] / 2.0,
                    )
                )
                existing_distance = float(
                    np.hypot(center_x - existing_x, center_y - existing_y)
                )
                min_distance = max(
                    center_min_px,
                    center_ratio
                    * max(
                        candidate["bbox"][2],
                        candidate["bbox"][3],
                        existing["bbox"][2],
                        existing["bbox"][3],
                    ),
                )
                if existing_distance < min_distance:
                    too_close = True
                    break
            if too_close:
                continue

            if seed_centers:
                min_seed_distance = min(
                    float(np.hypot(center_x - seed_x, center_y - seed_y))
                    for seed_x, seed_y in seed_centers
                )
            else:
                min_seed_distance = 0.0

            min_anchor_distance: float | None = None
            anchor_distance_score = 0.0
            for anchor in anchor_candidates:
                anchor_distance = float(
                    np.hypot(
                        center_x - anchor["center_x"], center_y - anchor["center_y"]
                    )
                )
                if min_anchor_distance is None or anchor_distance < min_anchor_distance:
                    min_anchor_distance = anchor_distance
                anchor_radius_limit = max(
                    anchor_bonus_radius_px,
                    float(anchor["side_px"]) * anchor_attach_scale,
                )
                if anchor_distance <= anchor_radius_limit:
                    anchor_distance_score = max(
                        anchor_distance_score,
                        1.0 - anchor_distance / max(anchor_radius_limit, 1e-6),
                    )

            if (
                merged
                and attach_radius_px > 0.0
                and min_seed_distance > attach_radius_px
            ):
                continue
            ranked.append(
                (
                    float(min_seed_distance)
                    - anchor_distance_score * anchor_bonus_radius_px,
                    -float(candidate["score"]) - anchor_distance_score * 0.15,
                    index,
                    candidate,
                )
            )

        if not ranked:
            break

        ranked.sort(key=lambda item: (item[0], item[1]))
        _, _, chosen_index, chosen_candidate = ranked[0]
        merged.append(chosen_candidate)
        remaining.pop(chosen_index)

    return merged


def _candidate_to_output(
    candidate: dict[str, Any],
    image_shape: tuple[int, int],
    class_id: int,
) -> dict[str, Any]:
    image_h, image_w = image_shape[:2]
    x, y, w, h = candidate["bbox"]
    center_x = float(candidate.get("refined_center_x", x + w / 2.0))
    center_y = float(candidate.get("refined_center_y", y + h / 2.0))

    return {
        "bbox_yolo": (
            class_id,
            float(center_x / image_w),
            float(center_y / image_h),
            float(w / image_w),
            float(h / image_h),
        ),
        "bbox_px": (x, y, w, h),
        "center_xy_px": (int(round(center_x)), int(round(center_y))),
        "center_xy_norm": (float(center_x / image_w), float(center_y / image_h)),
        "score": float(candidate["score"]),
        "debug_info": {
            "source": candidate.get("source", "unknown"),
            "white_ratio": float(candidate["white_ratio"]),
            "black_ratio": float(candidate["black_ratio"]),
            "red_ratio": float(candidate["red_ratio"]),
            "green_ratio": float(candidate["green_ratio"]),
            "contrast_std": float(candidate["contrast_std"]),
            "component_count": int(candidate["component_count"]),
            "significant_component_count": int(
                candidate["significant_component_count"]
            ),
            "compact_component_count": int(candidate["compact_component_count"]),
            "max_component_solidity": float(candidate["max_component_solidity"]),
            "edge_ratio": float(candidate["edge_ratio"]),
            "template_vote_count": int(candidate.get("template_vote_count", 0)),
            "template_mean_score": float(candidate.get("template_mean_score", 0.0)),
            "template_patch_score": float(candidate.get("template_patch_score", 0.0)),
            "template_corr": float(candidate.get("template_corr", 0.0)),
            "binary_agreement": float(candidate.get("binary_agreement", 0.0)),
            "contrast_separation": float(candidate.get("contrast_separation", 0.0)),
            "structure_score": float(candidate.get("structure_score", 0.0)),
            "template_gate": str(candidate.get("template_gate", "")),
        },
    }


def resolve_target_class_ids(config: dict[str, Any]) -> list[int]:
    dataset_cfg = config.get("dataset", {})
    raw_target_class_ids = dataset_cfg.get("target_class_ids")
    if isinstance(raw_target_class_ids, (list, tuple, set)):
        class_ids = sorted({int(class_id) for class_id in raw_target_class_ids})
        if class_ids:
            return class_ids
    return [
        int(dataset_cfg.get("target_class_id", detector_config(config)["class_id"]))
    ]


def load_bw_target_gt(
    label_path: Path,
    class_id: int | list[int] | tuple[int, ...] | set[int],
) -> list[tuple[float, ...]]:
    target_class_ids = (
        {int(class_id)}
        if isinstance(class_id, int)
        else {int(value) for value in class_id}
    )
    gt_boxes: list[tuple[float, ...]] = []
    if label_path.exists() and label_path.stat().st_size > 0:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                values = tuple(float(value) for value in parts)
                if int(values[0]) in target_class_ids:
                    gt_boxes.append(values)
    return gt_boxes


def _iou_yolo(
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


def _draw_debug_image(
    image_bgr: np.ndarray, components: list[dict[str, Any]], boxes: list[dict[str, Any]]
) -> np.ndarray:
    overlay = image_bgr.copy()
    for component in components:
        x, y, w, h = component["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 120, 0), 1)

    for box in boxes:
        x, y, w, h = box["bbox_px"]
        cx, cy = box["center_xy_px"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawMarker(overlay, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            overlay,
            f"{box['score']:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return overlay


def _maybe_save(
    image_path: str,
    result: dict[str, Any] | None,
    output_dir: str | None,
    debug_image: np.ndarray | None = None,
    debug_payload: dict[str, Any] | None = None,
    save_debug_images: bool = True,
) -> None:
    if output_dir is None:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    ext = Path(image_path).suffix

    txt_path = output_path / f"{stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        if result is not None:
            for box in result["boxes"]:
                label = box["bbox_yolo"]
                f.write(
                    f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
                )

    if debug_payload is not None:
        with open(output_path / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2)

    if debug_image is not None and save_debug_images:
        cv2.imwrite(str(output_path / f"detected_{stem}{ext}"), debug_image)


def _find_bw_targets_impl(
    image: np.ndarray | None,
    image_path: str,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any] | None:
    if image is None or image.size == 0:
        return None

    fast_cfg = _fast_runtime_config(config)
    scene_anchors: list[dict[str, Any]] = []
    square_votes = _extract_square_candidates(image, config)
    dark_mask = _dark_mask(image, config)
    components = _extract_dark_components(dark_mask, config)
    local_components = _extract_local_dark_components(image, config)
    groups = _group_components(components, config)

    candidates: list[dict[str, Any]] = []
    square_candidates: list[dict[str, Any]] = []
    for vote in square_votes:
        candidate = _score_square_candidate(image, vote, config)
        if candidate is not None:
            square_candidates.append(candidate)
            candidates.append(candidate)

    for group in groups:
        bbox = _expand_group_bbox(group, image.shape[:2], config)
        candidate = _score_candidate(image, bbox, group, config)
        if candidate is not None:
            candidates.append(candidate)

    for local_component in local_components:
        bbox = _local_component_bbox(local_component, image.shape[:2], config)
        candidate = _score_candidate(image, bbox, [local_component], config)
        if candidate is not None:
            candidate["source"] = "local_dark_component"
            candidate["refined_center_x"] = float(local_component["center"][0])
            candidate["refined_center_y"] = float(local_component["center"][1])
            candidates.append(candidate)

    threshold = float(detector_config(config)["score_threshold"])
    selected = _nms(candidates, config)
    selected = [candidate for candidate in selected if candidate["score"] >= threshold]

    component_template_seeds: list[dict[str, Any]] = []
    component_template_candidates: list[dict[str, Any]] = []
    should_run_component_template = True
    if fast_cfg["enabled"]:
        should_run_component_template = (
            fast_cfg["enable_component_template"]
            and len(selected)
            < fast_cfg["min_base_candidates_for_skip_component_template"]
        )
    if should_run_component_template:
        component_template_seeds = _extract_component_template_seed_components(
            image, config
        )
        component_template_candidates = _component_seeded_template_candidates(
            image,
            components + local_components + component_template_seeds,
            config,
        )
        candidates.extend(component_template_candidates)

        if fast_cfg["enable_self_template"]:
            candidates.extend(
                _self_template_candidates(
                    image, component_template_candidates + square_candidates, config
                )
            )

    template_votes: list[dict[str, Any]] = []
    template_clusters: list[list[dict[str, Any]]] = []
    rescue_cfg = detector_config(config)["template_rescue"]
    selected = _nms(candidates, config)
    selected = [candidate for candidate in selected if candidate["score"] >= threshold]

    should_run_template_rescue = bool(rescue_cfg["enabled"])
    if fast_cfg["enabled"]:
        should_run_template_rescue = (
            fast_cfg["enable_template_rescue"]
            and len(selected)
            < fast_cfg["min_base_candidates_for_skip_template_rescue"]
        )

    if should_run_template_rescue:
        if len(selected) <= int(rescue_cfg["trigger_max_initial_boxes"]):
            template_votes = _template_match_votes(image, config)
            template_clusters = _cluster_template_votes(template_votes, config)
            for cluster in template_clusters:
                bbox = _rescue_cluster_bbox(cluster, image.shape[:2], config)
                candidate = _score_rescue_candidate(image, bbox, cluster, config)
                if candidate is not None:
                    candidates.append(candidate)

    selected = _nms(candidates, config)
    selected = [candidate for candidate in selected if candidate["score"] >= threshold]
    if any(candidate.get("source") == "square_template" for candidate in selected):
        base_selected = [
            candidate
            for candidate in selected
            if candidate.get("source") != "square_template"
        ]
        square_selected = [
            candidate
            for candidate in selected
            if candidate.get("source") == "square_template"
        ]
        square_selected = _limit_square_candidates(square_selected, config)
        selected = base_selected + square_selected
    if any(candidate.get("source") == "template_rescue" for candidate in selected):
        base_selected = [
            candidate
            for candidate in selected
            if candidate.get("source") != "template_rescue"
        ]
        rescue_selected = [
            candidate
            for candidate in selected
            if candidate.get("source") == "template_rescue"
        ]
        rescue_selected = _limit_template_rescue_candidates(rescue_selected, config)
        selected = base_selected + rescue_selected
    if template_clusters:
        if fast_cfg["enable_scene_anchor_for_fallback"]:
            scene_anchors = _scene_anchor_candidates(str(image_path), image)
        selected = _append_recall_fallback_candidates(
            image, selected, template_clusters, scene_anchors, config
        )
    selected = _nms(selected, config)
    selected = _dedupe_candidates_by_center(
        selected,
        center_ratio=float(
            detector_config(config).get("final_center_dedupe_ratio", 0.70)
        ),
        center_min_px=float(
            detector_config(config).get("final_center_dedupe_min_px", 14.0)
        ),
    )
    selected = _reject_border_touching_candidates(selected, image.shape[:2], config)

    result = None
    if selected:
        boxes = [
            _candidate_to_output(
                candidate, image.shape[:2], int(detector_config(config)["class_id"])
            )
            for candidate in selected
        ]
        result = {"boxes": boxes}

    debug_image = None
    debug_payload = None
    if debug or output_dir:
        debug_payload = {
            "image_path": str(image_path),
            "n_square_votes": len(square_votes),
            "n_dark_components": len(components),
            "n_local_dark_components": len(local_components),
            "n_component_template_seeds": len(component_template_seeds),
            "n_groups": len(groups),
            "n_template_votes": len(template_votes),
            "n_template_clusters": len(template_clusters),
            "scene_anchors": scene_anchors,
            "score_threshold": threshold,
            "selected": result,
            "candidates": [
                {
                    "bbox": list(candidate["bbox"]),
                    "score": float(candidate["score"]),
                    "source": candidate.get("source", "unknown"),
                    "white_ratio": float(candidate["white_ratio"]),
                    "black_ratio": float(candidate["black_ratio"]),
                    "red_ratio": float(candidate["red_ratio"]),
                    "green_ratio": float(candidate["green_ratio"]),
                    "contrast_std": float(candidate["contrast_std"]),
                    "component_count": int(candidate["component_count"]),
                    "significant_component_count": int(
                        candidate["significant_component_count"]
                    ),
                    "compact_component_count": int(
                        candidate["compact_component_count"]
                    ),
                    "max_component_solidity": float(
                        candidate["max_component_solidity"]
                    ),
                    "edge_ratio": float(candidate["edge_ratio"]),
                    "template_vote_count": int(candidate.get("template_vote_count", 0)),
                    "template_mean_score": float(
                        candidate.get("template_mean_score", 0.0)
                    ),
                    "template_corr": float(candidate.get("template_corr", 0.0)),
                    "binary_agreement": float(candidate.get("binary_agreement", 0.0)),
                    "contrast_separation": float(
                        candidate.get("contrast_separation", 0.0)
                    ),
                    "structure_score": float(candidate.get("structure_score", 0.0)),
                }
                for candidate in selected
            ],
        }
        debug_image = _draw_debug_image(
            image, components, result["boxes"] if result is not None else []
        )

    _maybe_save(
        image_path=image_path,
        result=result,
        output_dir=output_dir,
        debug_image=debug_image,
        debug_payload=debug_payload if debug else None,
        save_debug_images=bool(benchmark_config(config).get("save_debug_images", True)),
    )
    return result


def find_bw_targets_in_image(
    image: np.ndarray | None,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
    image_path: str = "memory_image.png",
) -> dict[str, Any] | None:
    return _find_bw_targets_impl(
        image=image,
        image_path=image_path,
        config=config,
        output_dir=output_dir,
        debug=debug,
    )


def find_bw_targets(
    image_path: str,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERR] Impossibile caricare: {image_path}")
        return None

    return _find_bw_targets_impl(
        image=image,
        image_path=image_path,
        config=config,
        output_dir=output_dir,
        debug=debug,
    )


def infer_on_image_dir(
    image_dir: Path | str,
    config: dict[str, Any],
    output_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    image_dir = Path(image_dir)
    img_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    detections = 0
    total_boxes = 0

    for idx, img_path in enumerate(img_paths, start=1):
        result = find_bw_targets(
            str(img_path), config=config, output_dir=output_dir, debug=debug
        )
        if result is not None and result["boxes"]:
            detections += 1
            total_boxes += len(result["boxes"])
        if idx % 10 == 0:
            print(
                f"  [{idx}/{len(img_paths)}] images_with_boxes={detections} total_boxes={total_boxes}"
            )

    return {
        "image_dir": str(image_dir),
        "n_images": len(img_paths),
        "images_with_boxes": detections,
        "images_without_boxes": len(img_paths) - detections,
        "total_boxes": total_boxes,
    }


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
    target_class_ids = resolve_target_class_ids(config)
    tp = fp = fn = 0
    errors = 0
    false_positive_images: list[str] = []
    false_negative_images: list[str] = []

    for idx, img_path in enumerate(img_paths, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            errors += 1
            continue
        img_h, img_w = image.shape[:2]

        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_bw_target_gt(label_path, class_id=target_class_ids)
        result = find_bw_targets(
            str(img_path), config=config, output_dir=output_dir, debug=debug
        )
        preds = (
            [tuple(box["bbox_yolo"]) for box in result["boxes"]]
            if result is not None
            else []
        )

        matched_gt: set[int] = set()
        for pred in preds:
            matched = False
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if _iou_yolo(pred, gt_box, img_w, img_h) >= iou_thresh:
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
        "target_class_ids": target_class_ids,
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


def resolve_dataset_paths(
    config: dict[str, Any],
    data_root_override: str = "",
    split_override: str = "",
) -> tuple[Path, Path, str]:
    ds_cfg = config["dataset"]
    data_root = Path(data_root_override or ds_cfg["data_root"]).expanduser()
    split = split_override or ds_cfg["split"]
    return data_root / split / "images", data_root / split / "labels", split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rileva i piccoli target bianchi/neri con CV classica."
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
        help="Cartella piatta di immagini da analizzare.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory output per txt, overlay e debug JSON.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Abilita debug JSON e overlay."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.out_dir or benchmark_config(config)["debug_dir"]

    if args.image:
        result = find_bw_targets(
            args.image, config=config, output_dir=output_dir, debug=args.debug
        )
        if result is None:
            print("Nessun target bianco/nero trovato.")
            return
        print(f"Target trovati: {len(result['boxes'])}")
        for idx, box in enumerate(result["boxes"], start=1):
            label = box["bbox_yolo"]
            print(
                f"  [{idx}] bbox_yolo: {label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f} | score={box['score']:.3f}"
            )
        return

    image_dir = Path(args.image_dir or config["dataset"]["image_dir"]).expanduser()
    if not image_dir.exists():
        raise SystemExit(f"[ERR] Cartella non trovata: {image_dir}")

    summary = infer_on_image_dir(
        image_dir=image_dir, config=config, output_dir=output_dir, debug=args.debug
    )
    print("\n" + "=" * 50)
    print(" RISULTATI DIRECTORY")
    print("=" * 50)
    print(f"  Cartella         : {summary['image_dir']}")
    print(f"  Immagini         : {summary['n_images']}")
    print(f"  Con boxes        : {summary['images_with_boxes']}")
    print(f"  Senza boxes      : {summary['images_without_boxes']}")
    print(f"  Totale box       : {summary['total_boxes']}")
    print("=" * 50)
    print(f"  Output → {output_dir}/")


if __name__ == "__main__":
    main()
