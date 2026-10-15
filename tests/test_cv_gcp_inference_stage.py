import json
from pathlib import Path
from typing import Any

import numpy as np

from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stuff_i_moved_away.inference.cv_gcp_inference_stage import CVGCPInferenceStage


def _make_metadata(filename: str) -> PhotoExif:
    pix_data = PixData(
        lat=41.1,
        lon=12.2,
        alt_rel=33.3,
        roll=0.1,
        pitch=0.2,
        yaw=0.3,
    )
    return PhotoExif(
        filename=filename,
        timestamp=1234567890.0,
        exposure_time=10.0,
        gain=1.5,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="Daylight5000K",
        pixel_format="BayerRG8",
        pix_data=pix_data,
        shm_name=None,
        shm_shape=None,
        shm_dtype=None,
        full_image_shape=(100, 200, 3),
    )


def test_run_gcp_detector_converts_detector_boxes_to_pipeline_format(
    monkeypatch,
) -> None:
    from stuff_i_moved_away.inference import cv_gcp_inference_stage

    stage = CVGCPInferenceStage.__new__(CVGCPInferenceStage)
    stage.gcp_config = {}
    stage.gcp_class_id = 10

    def fake_find_bw_targets_in_image(
        image, config, output_dir=None, debug=False, image_path="memory_image.png"
    ):
        return {
            "boxes": [
                {
                    "bbox_yolo": (3, 0.5, 0.5, 0.25, 0.5),
                    "score": 0.91,
                    "bbox_px": (75, 25, 50, 50),
                }
            ]
        }

    monkeypatch.setattr(
        cv_gcp_inference_stage,
        "find_bw_targets_in_image",
        fake_find_bw_targets_in_image,
    )

    boxes = stage._run_gcp_detector(np.zeros((100, 200, 3), dtype=np.uint8))

    assert boxes == [
        {
            "class": 10,
            "conf": 0.91,
            "bbox": [75.0, 25.0, 125.0, 75.0],
        }
    ]


def test_ensure_rgb_image_converts_bayerrg_frame() -> None:
    stage = CVGCPInferenceStage.__new__(CVGCPInferenceStage)

    bayer = np.zeros((8, 8), dtype=np.uint8)
    rgb = stage._ensure_rgb_image(bayer, "BayerRG8")

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8


def test_load_gcp_config_uses_bundled_default_when_not_configured() -> None:
    stage = CVGCPInferenceStage.__new__(CVGCPInferenceStage)
    stage.gcp_config_path = None

    class Configs:
        config_dict: dict[str, Any] = {"inference_engine": {}}

    stage.configs = Configs()

    config = stage._load_gcp_config()

    assert "detector" in config
    assert "dataset" in config
    assert "benchmark" in config


def test_cv_gcp_stage_is_enabled_for_numbers_and_perimeter() -> None:
    stage = CVGCPInferenceStage.__new__(CVGCPInferenceStage)

    stage.current_task = "NUMBERS"
    assert stage._cv_gcp_enabled() is True

    stage.current_task = "PERIMETER"
    assert stage._cv_gcp_enabled() is True

    stage.current_task = "BULLSEYE"
    assert stage._cv_gcp_enabled() is False


def test_save_positive_artifacts_writes_metadata_and_image(tmp_path: Path) -> None:
    stage = CVGCPInferenceStage.__new__(CVGCPInferenceStage)
    stage.save_positive_images = True
    stage.save_positive_overlays = True
    stage.save_positive_metadata = True
    stage.review_dir_name = "cv_gcp_review"

    metadata = _make_metadata(str(tmp_path / "frame_001.npy"))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = [{"class": 10, "conf": 0.88, "bbox": [10.0, 20.0, 40.0, 60.0]}]

    stage._save_positive_artifacts(metadata, image, boxes)

    root = tmp_path / "cv_gcp_review"
    metadata_path = root / "metadata" / "frame_001.json"
    image_path = root / "images" / "frame_001.png"
    overlay_path = root / "overlays" / "frame_001.png"

    assert metadata_path.exists()
    assert image_path.exists()
    assert overlay_path.exists()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["image"]["source_path"] == str(tmp_path / "frame_001.npy")
    assert payload["telemetry"]["lat"] == 41.1
    assert payload["detections"] == boxes
