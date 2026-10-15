from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIGIT_LABELS = tuple(str(index) for index in range(10))
LABEL_NAMES = DIGIT_LABELS + ("square_without_number", "background")

_DIGIT_CATEGORY_MAP = {f"Plane.{index}": str(index) for index in range(10)}
_SPECIAL_CATEGORY_MAP = {
    "Plane.010": "square_without_number",
    "X": "background",
    "bulls_eye": "background",
}


@dataclass(frozen=True)
class MatchResult:
    assigned_label: str
    source_category_name: str | None
    matched_annotation_id: int | None
    intersection_area: float


def collapse_category_name(category_name: str) -> str:
    if category_name in _DIGIT_CATEGORY_MAP:
        return _DIGIT_CATEGORY_MAP[category_name]
    if category_name in _SPECIAL_CATEGORY_MAP:
        return _SPECIAL_CATEGORY_MAP[category_name]
    raise ValueError(f"Unsupported category name: {category_name}")


def coco_bbox_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"COCO bbox must have 4 values, got {len(bbox)}")
    x, y, width, height = (float(value) for value in bbox)
    return (x, y, x + width, y + height)


def expand_xyxy(
    box: tuple[float, float, float, float],
    margin_px: float,
    margin_ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    expand_x = float(margin_px) + width * float(margin_ratio)
    expand_y = float(margin_px) + height * float(margin_ratio)
    return (x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y)


def list_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clamp_xyxy(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(int(round(x1)), image_width))
    top = max(0, min(int(round(y1)), image_height))
    right = max(0, min(int(round(x2)), image_width))
    bottom = max(0, min(int(round(y2)), image_height))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return (left, top, right, bottom)


def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0.0
    return float((right - left) * (bottom - top))


def annotation_to_matchable(annotation: dict[str, Any]) -> dict[str, Any]:
    if "bbox_xyxy" not in annotation:
        raise ValueError("Annotation is missing bbox_xyxy")
    if "category_name" not in annotation:
        raise ValueError("Annotation is missing category_name")
    return {
        "id": annotation.get("id"),
        "category_name": str(annotation["category_name"]),
        "collapsed_label": collapse_category_name(str(annotation["category_name"])),
        "bbox_xyxy": tuple(float(value) for value in annotation["bbox_xyxy"]),
    }


def _label_priority(label_name: str) -> int:
    if label_name in DIGIT_LABELS:
        return 2
    if label_name == "square_without_number":
        return 1
    return 0


def match_detection_to_label(
    detection_xyxy: tuple[float, float, float, float],
    annotations: Sequence[dict[str, Any]],
    margin_px: float,
    margin_ratio: float,
) -> MatchResult:
    positive_matches: list[tuple[int, float, dict[str, Any]]] = []

    for raw_annotation in annotations:
        annotation = annotation_to_matchable(raw_annotation)
        expanded_box = expand_xyxy(annotation["bbox_xyxy"], margin_px, margin_ratio)
        overlap_area = intersection_area(detection_xyxy, expanded_box)
        if overlap_area <= 0.0:
            continue
        positive_matches.append(
            (_label_priority(annotation["collapsed_label"]), overlap_area, annotation)
        )

    if not positive_matches:
        return MatchResult(
            assigned_label="background",
            source_category_name=None,
            matched_annotation_id=None,
            intersection_area=0.0,
        )

    positive_matches.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )
    _, overlap_area, annotation = positive_matches[0]
    return MatchResult(
        assigned_label=annotation["collapsed_label"],
        source_category_name=annotation["category_name"],
        matched_annotation_id=annotation["id"],
        intersection_area=overlap_area,
    )
