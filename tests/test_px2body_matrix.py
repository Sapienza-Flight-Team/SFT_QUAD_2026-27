import math

import pytest
import yaml

from sft_math.cam.px2body_matrix import (
    load_px2body_matrix_from_yaml,
    resolve_px2body_matrix,
)


def test_load_px2body_matrix_reads_repo_yaml() -> None:
    with open("sft_config.yaml", "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    geoloc_config = config["decision_system"]["geoloc"]
    base_matrix = geoloc_config["px2body_matrix"]
    error_angle_deg = geoloc_config["px2body_error_angle_deg"]

    expected_base_matrix = _rotation_matrix_deg(90.0)
    expected_effective_matrix = _rotation_matrix_deg(90.0 + error_angle_deg)

    assert _flatten(base_matrix) == pytest.approx(_flatten(expected_base_matrix))
    assert _flatten(load_px2body_matrix_from_yaml("sft_config.yaml")) == pytest.approx(
        _flatten(expected_effective_matrix)
    )


def test_resolve_px2body_matrix_composes_error_angle() -> None:
    geoloc_config = {
        "px2body_matrix": [[0.0, -1.0], [1.0, 0.0]],
        "px2body_error_angle_deg": 176.0,
    }

    expected = _rotation_matrix_deg(266.0)

    assert _flatten(resolve_px2body_matrix(geoloc_config, {})) == pytest.approx(
        _flatten(expected)
    )


def test_resolve_px2body_matrix_requires_yaml_value() -> None:
    with pytest.raises(KeyError, match="px2body_matrix"):
        resolve_px2body_matrix({}, {})


def _rotation_matrix_deg(angle_deg: float) -> list[list[float]]:
    angle_rad = math.radians(angle_deg)
    return [
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)],
    ]


def _flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]
