from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_MATRIX_KEYS = ("px2body_matrix",)
_ERROR_ANGLE_KEYS = ("px2body_error_angle_deg",)


def resolve_px2body_matrix(
    geoloc_config: Mapping[str, Any] | None = None,
    camera_config: Mapping[str, Any] | None = None,
) -> list[list[float]]:
    """Resolve the effective pixel->body matrix from YAML-backed config."""

    geoloc_config = geoloc_config or {}
    camera_config = camera_config or {}

    matrix = _resolve_matrix_value(geoloc_config, camera_config)
    error_angle_deg = _resolve_error_angle_deg(geoloc_config, camera_config)

    if error_angle_deg == 0.0:
        return matrix
    return _matrix_multiply(_rotation_matrix_deg(error_angle_deg), matrix)


def load_px2body_matrix_from_yaml(
    config_yaml_path: str | Path = "sft_config.yaml",
) -> list[list[float]]:
    """Load config YAML and resolve px2body matrix."""

    with Path(config_yaml_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    geoloc_config = config.get("decision_system", {}).get("geoloc", {})
    camera_config = config.get("camera", {})
    return resolve_px2body_matrix(
        geoloc_config,
        camera_config,
    )


def _resolve_matrix_value(
    geoloc_config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> list[list[float]]:
    for key in _MATRIX_KEYS:
        matrix = geoloc_config.get(key)
        if matrix is not None:
            return _normalize_matrix(matrix)

    for key in _MATRIX_KEYS:
        matrix = camera_config.get(key)
        if matrix is not None:
            return _normalize_matrix(matrix)

    raise KeyError("px2body_matrix must be defined in YAML config")


def _resolve_error_angle_deg(
    geoloc_config: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> float:
    for key in _ERROR_ANGLE_KEYS:
        angle = geoloc_config.get(key)
        if angle is not None:
            return float(angle)

    for key in _ERROR_ANGLE_KEYS:
        angle = camera_config.get(key)
        if angle is not None:
            return float(angle)

    return 0.0


def _normalize_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError("px2body matrix must be a 2x2 structure")
    return [
        [float(matrix[0][0]), float(matrix[0][1])],
        [float(matrix[1][0]), float(matrix[1][1])],
    ]


def _rotation_matrix_deg(angle_deg: float) -> list[list[float]]:
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return [
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle],
    ]


def _matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            float(left[row][0] * right[0][column] + left[row][1] * right[1][column])
            for column in range(2)
        ]
        for row in range(2)
    ]
