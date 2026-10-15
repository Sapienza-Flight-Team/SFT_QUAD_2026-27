from unittest.mock import MagicMock

import numpy as np

from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stages.decision_system.default.decision_system_stage import DecisionSystem
from stuff_i_moved_away.inference.find_bullseye import load_config
from stuff_i_moved_away.inference.bullseye_inference_stage import BullseyeInferenceStage
from stages.inference.default.inference_model_stage import InferenceStage


def test_find_bullseye_in_image_returns_none_when_no_target() -> None:
    from stuff_i_moved_away.inference.find_bullseye import find_bullseye_in_image

    image = np.zeros((128, 128, 3), dtype=np.uint8)

    result = find_bullseye_in_image(image, load_config())

    assert result is None


def test_completed_bullseye_detection_is_scaled_and_forwarded() -> None:
    stage = BullseyeInferenceStage.__new__(BullseyeInferenceStage)
    out_q = MagicMock()
    stage.out_q = [out_q]

    metadata = PhotoExif(
        filename="frame.npy",
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="",
        pixel_format="RGB8",
        pix_data=PixData(),
        shm_name="input-shm",
        shm_shape=(100, 200, 3),
        shm_dtype="uint8",
        full_image_shape=(200, 400, 3),
    )
    bboxes = [{"class": 2, "conf": 0.87, "bbox": [75.0, 25.0, 125.0, 75.0]}]

    stage._handle_completed_detection(metadata, bboxes, (100, 200, 3))

    assert metadata.shm_name is None
    assert metadata.bboxes == [
        {
            "class": 2,
            "raw_class": 2,
            "conf": 0.87,
            "bbox": [150.0, 50.0, 250.0, 150.0],
        }
    ]
    out_q.put.assert_called_once_with(metadata, timeout=2)


def test_completed_bullseye_empty_result_is_not_forwarded() -> None:
    stage = BullseyeInferenceStage.__new__(BullseyeInferenceStage)
    stage.out_q = [MagicMock()]
    stage.logger = MagicMock()
    metadata = PhotoExif(
        filename="frame.npy",
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="",
        pixel_format="RGB8",
        pix_data=PixData(),
        shm_name="input-shm",
        shm_shape=(100, 200, 3),
        shm_dtype="uint8",
        full_image_shape=(200, 400, 3),
    )

    stage._handle_completed_detection(metadata, [], (100, 200, 3))

    assert metadata.shm_name is None
    assert metadata.bboxes == []
    stage.out_q[0].put.assert_not_called()


def test_decision_system_sees_both_bullseye_producers_as_class_2() -> None:
    yolo_stage = InferenceStage.__new__(InferenceStage)
    yolo_stage.current_task = "BULLSEYE"
    yolo_stage.yolo_allowed_classes = {0, 1, 2, 3}
    yolo_stage.yolo_class_map = {0: 0, 1: 1, 2: 2, 3: 3}
    yolo_metadata = PhotoExif(
        filename="yolo_frame.npy",
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="",
        pixel_format="RGB8",
        pix_data=PixData(),
    )
    yolo_full_image_bboxes = yolo_stage._prepare_yolo_bboxes(
        [{"class": 2, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}]
    )
    yolo_handoff = yolo_stage._prepare_handoff_metadata(
        yolo_metadata,
        None,
        yolo_full_image_bboxes,
        yolo_full_image_bboxes,
    )

    cv_stage = BullseyeInferenceStage.__new__(BullseyeInferenceStage)
    cv_stage.out_q = [MagicMock()]
    cv_stage.logger = MagicMock()
    cv_metadata = PhotoExif(
        filename="cv_frame.npy",
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="",
        pixel_format="RGB8",
        pix_data=PixData(),
        full_image_shape=(200, 400, 3),
    )
    cv_stage._handle_completed_detection(
        cv_metadata,
        [{"class": 5, "conf": 0.87, "bbox": [75.0, 25.0, 125.0, 75.0]}],
        (100, 200, 3),
    )

    *_, yolo_classes = DecisionSystem.input_format(yolo_handoff)
    *_, cv_classes = DecisionSystem.input_format(cv_metadata)

    np.testing.assert_array_equal(yolo_classes, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(cv_classes, np.array([2], dtype=np.int64))


def test_bullseye_stage_is_enabled_only_for_bullseye_mission() -> None:
    stage = BullseyeInferenceStage.__new__(BullseyeInferenceStage)

    stage.current_task = "BULLSEYE"
    assert stage._bullseye_enabled() is True

    stage.current_task = "NUMBERS"
    assert stage._bullseye_enabled() is False
