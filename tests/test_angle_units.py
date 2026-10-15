import numpy as np
import pytest

from common.homography import metadata_angles as common_metadata_angles
from common.homography import compute_homography as common_compute_homography
from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stages.decision_system.default.decision_system_stage import DecisionSystem
from stuff_i_moved_away.inference.bullseye_homo_cv import (
    compute_homography as bullseye_compute_homography,
    metadata_angles as bullseye_metadata_angles,
)


def _metadata() -> PhotoExif:
    photo = PhotoExif(
        filename="angle-test.npy",
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="",
        pixel_format="RGB8",
        pix_data=PixData(
            lat=41.0,
            lon=12.0,
            alt_rel=50.0,
            roll=1.0,
            pitch=2.0,
            yaw=3.0,
        ),
    )
    photo.bboxes = [{"bbox": [0.0, 0.0, 100.0, 100.0], "class": 2}]
    return photo


def test_decision_input_format_passes_degree_attitude_through() -> None:
    *_, attitude, _, _ = DecisionSystem.input_format(_metadata())

    np.testing.assert_allclose(attitude, (1.0, 2.0, 3.0), atol=1e-12)


@pytest.mark.parametrize(
    "metadata_angles",
    [common_metadata_angles, bullseye_metadata_angles],
)
def test_metadata_angles_always_return_degrees(metadata_angles) -> None:
    roll, pitch, yaw, is_deg = metadata_angles(
        {"roll": 7.0, "pitch": 0.25, "yaw": 400.0},
        {},
    )

    assert (roll, pitch, yaw) == (7.0, 0.25, 400.0)
    assert is_deg is True


@pytest.mark.parametrize(
    "metadata_angles",
    [common_metadata_angles, bullseye_metadata_angles],
)
def test_metadata_angles_require_yaw_without_fallback(
    metadata_angles,
) -> None:
    with pytest.raises(ValueError):
        metadata_angles(
            {"roll": 0.0, "pitch": 0.0},
            {},
        )


@pytest.mark.parametrize(
    "compute_homography",
    [common_compute_homography, bullseye_compute_homography],
)
def test_compute_homography_accepts_removed_legacy_kwargs(compute_homography) -> None:
    computation = compute_homography(
        {"alt_rel": 50.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        {"decision_system": {"geoloc": {"px2body_matrix": [[1.0, 0.0], [0.0, 1.0]]}}},
        64,
        64,
        augment=True,
        heading_conversion_mode="mag_to_true",
    )

    assert computation.matrix.shape == (3, 3)
