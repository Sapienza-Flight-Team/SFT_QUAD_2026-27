"""Resolve camera mounting configuration and its angular correction."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from . import check


def resolve_px2body_matrix(
    geoloc_config: Mapping[str, Any] | None = None,
    camera_config: Mapping[str, Any] | None = None,
) -> list[list[float]]:
    """Prefer geolocator settings over camera settings for each configuration key."""
    geoloc_config = geoloc_config or {}
    camera_config = camera_config or {}
    matrix = geoloc_config.get("px2body_matrix")
    if matrix is None:
        matrix = camera_config.get("px2body_matrix")
    check.mounting_required(matrix)
    matrix = np.asarray(matrix, dtype=np.float64)
    check.px2body_matrix(matrix)
    check.finite(matrix, name="px2body_matrix")
    angle = geoloc_config.get("px2body_error_angle_deg")
    if angle is None:
        angle = camera_config.get("px2body_error_angle_deg", 0)
    angle = 0.0 if angle is None else float(angle)
    check.finite_scalar(angle, name="px2body_error_angle_deg")
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    correction = np.array([[cosine, -sine], [sine, cosine]])
    return (correction @ matrix).tolist()


def load_px2body_matrix_from_yaml(
    config_yaml_path: str | Path = "sft_config.yaml",
) -> list[list[float]]:
    """Read mounting configuration from the repository YAML layout."""
    with Path(config_yaml_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return resolve_px2body_matrix(
        config.get("decision_system", {}).get("geoloc", {}), config.get("camera", {})
    )
