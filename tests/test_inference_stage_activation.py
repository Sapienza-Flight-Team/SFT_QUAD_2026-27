from unittest.mock import MagicMock

import numpy as np

from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stages.inference.default.inference_model_stage import InferenceStage


def test_yolo_stage_runs_for_default_bullseye_numbers_and_perimeter_only() -> None:
    stage = InferenceStage.__new__(InferenceStage)

    for mission in ("DEFAULT", "BULLSEYE", "NUMBERS", "PERIMETER"):
        stage.current_task = mission
        assert stage._yolo_enabled() is True

    stage.current_task = "DISABLED"
    assert stage._yolo_enabled() is False


def test_numbers_handoff_does_not_attach_decision_shm() -> None:
    stage = InferenceStage.__new__(InferenceStage)
    stage.current_task = "NUMBERS"

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
        shm_shape=(3, 4, 3),
        shm_dtype="uint8",
        full_image_shape=(6, 8, 3),
    )
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape((3, 4, 3))
    inference_bboxes = [{"class": 0, "conf": 0.7, "bbox": [1.0, 1.0, 3.0, 2.0]}]
    full_image_bboxes = [{"class": 0, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}]

    handoff = stage._prepare_handoff_metadata(
        metadata,
        image,
        inference_bboxes,
        full_image_bboxes,
    )

    assert handoff is not metadata
    assert metadata.shm_name == "input-shm"
    assert handoff.shm_name is None
    assert handoff.shm_shape is None
    assert handoff.shm_dtype is None
    assert handoff.bboxes == full_image_bboxes


def test_non_numbers_handoff_does_not_attach_stale_shm() -> None:
    stage = InferenceStage.__new__(InferenceStage)
    stage.current_task = "DEFAULT"

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
        shm_shape=(3, 4, 3),
        shm_dtype="uint8",
        full_image_shape=(6, 8, 3),
    )
    full_image_bboxes = [{"class": 3, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}]

    handoff = stage._prepare_handoff_metadata(
        metadata,
        np.zeros((3, 4, 3), dtype=np.uint8),
        [{"class": 3, "conf": 0.7, "bbox": [1.0, 1.0, 3.0, 2.0]}],
        full_image_bboxes,
    )

    assert handoff.shm_name is None
    assert handoff.shm_shape is None
    assert handoff.shm_dtype is None
    assert handoff.bboxes == full_image_bboxes


def test_yolo_bboxes_are_filtered_and_remapped_to_decision_classes() -> None:
    stage = InferenceStage.__new__(InferenceStage)
    stage.yolo_allowed_classes = {0, 1, 2, 3}
    stage.yolo_class_map = {0: 0, 1: 1, 2: 2, 3: 3}

    bboxes = stage._prepare_yolo_bboxes(
        [
            {"class": 0, "conf": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"class": 2, "conf": 0.8, "bbox": [1.0, 1.0, 2.0, 2.0]},
            {"class": 3, "conf": 0.7, "bbox": [2.0, 2.0, 3.0, 3.0]},
            {"class": 9, "conf": 0.6, "bbox": [3.0, 3.0, 4.0, 4.0]},
        ]
    )

    assert bboxes == [
        {"class": 0, "raw_class": 0, "conf": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0]},
        {"class": 2, "raw_class": 2, "conf": 0.8, "bbox": [1.0, 1.0, 2.0, 2.0]},
        {"class": 3, "raw_class": 3, "conf": 0.7, "bbox": [2.0, 2.0, 3.0, 3.0]},
    ]


def test_bullseye_handoff_routes_directly_to_decision_queue() -> None:
    stage = InferenceStage.__new__(InferenceStage)
    stage.current_task = "BULLSEYE"
    cnn_q = MagicMock()
    decision_q = MagicMock()
    stage.out_q = [cnn_q, decision_q]

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
        shm_shape=(3, 4, 3),
        shm_dtype="uint8",
        full_image_shape=(6, 8, 3),
    )
    full_image_bboxes = [
        {"class": 2, "raw_class": 2, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}
    ]

    handoff = stage._prepare_handoff_metadata(
        metadata,
        np.zeros((3, 4, 3), dtype=np.uint8),
        [{"class": 2, "raw_class": 2, "conf": 0.7, "bbox": [1.0, 1.0, 3.0, 2.0]}],
        full_image_bboxes,
    )
    stage._send_handoff_metadata(handoff)

    assert handoff.shm_name is None
    assert handoff.bboxes == [
        {"class": 2, "raw_class": 2, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}
    ]
    cnn_q.put.assert_not_called()
    decision_q.put.assert_called_once_with(handoff, timeout=2)


def test_numbers_handoff_is_not_forwarded() -> None:
    stage = InferenceStage.__new__(InferenceStage)
    stage.current_task = "NUMBERS"
    cnn_q = MagicMock()
    decision_q = MagicMock()
    stage.out_q = [cnn_q, decision_q]

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
        shm_shape=(3, 4, 3),
        shm_dtype="uint8",
        full_image_shape=(6, 8, 3),
    )
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    inference_bboxes = [{"class": 0, "conf": 0.7, "bbox": [1.0, 1.0, 3.0, 2.0]}]

    handoff = stage._prepare_handoff_metadata(
        metadata,
        image,
        inference_bboxes,
        [{"class": 0, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}],
    )
    stage._send_handoff_metadata(handoff)

    cnn_q.put.assert_not_called()
    decision_q.put.assert_not_called()


def test_perimeter_handoff_is_not_forwarded() -> None:
    stage = InferenceStage.__new__(InferenceStage)
    stage.current_task = "PERIMETER"
    cnn_q = MagicMock()
    decision_q = MagicMock()
    stage.out_q = [cnn_q, decision_q]

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
        shm_shape=(3, 4, 3),
        shm_dtype="uint8",
        full_image_shape=(6, 8, 3),
    )
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    inference_bboxes = [{"class": 0, "conf": 0.7, "bbox": [1.0, 1.0, 3.0, 2.0]}]

    handoff = stage._prepare_handoff_metadata(
        metadata,
        image,
        inference_bboxes,
        [{"class": 0, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}],
    )
    stage._send_handoff_metadata(handoff)

    assert handoff.bboxes == [{"class": 0, "conf": 0.7, "bbox": [2.0, 2.0, 6.0, 4.0]}]
    assert handoff.shm_name is None
    cnn_q.put.assert_not_called()
    decision_q.put.assert_not_called()
