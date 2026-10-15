from multiprocessing import shared_memory
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from common.mavlink_structs import PixData
from common.structs import PhotoExif
import stuff_i_moved_away.inference.cnn_inference_stage as cnn_stage_module
from stuff_i_moved_away.inference.cnn_inference_stage import CNNInferenceStage, DigitPrediction


def _alloc_shm(array: np.ndarray) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(shm_array, array)
    return shm


def _make_metadata(shm_name: str, shape: tuple[int, ...], dtype: str) -> PhotoExif:
    return PhotoExif(
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
        shm_name=shm_name,
        shm_shape=shape,
        shm_dtype=dtype,
    )


def test_load_rgb_image_for_metadata_reads_and_releases_shm() -> None:
    stage = CNNInferenceStage.__new__(CNNInferenceStage)
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))
    shm = _alloc_shm(image)
    metadata = _make_metadata(shm.name, image.shape, str(image.dtype))
    shm.close()

    loaded = stage._load_rgb_image_for_metadata(metadata)

    np.testing.assert_array_equal(loaded, image)
    assert metadata.shm_name is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm.name)


def test_process_metadata_scales_refined_boxes_and_releases_shm(monkeypatch) -> None:
    stage = CNNInferenceStage.__new__(CNNInferenceStage)
    stage.crop_margin_px = 0.0
    stage.crop_margin_ratio = 0.0
    stage.cnn_checkpoint_path = "unused.pt"
    stage.cnn_imgsz = None
    stage.cnn_batch = 32
    stage.cnn_device = "cpu"
    stage.cnn_workers = 0
    stage.cnn_amp_enabled = False
    stage.no_number_class_id = 10
    stage.target_resolution = [4, 3]
    stage.configs = MagicMock()
    stage.configs.config_dict = {"bbox_logging": {"enabled": True}}
    stage.logger = MagicMock()
    stage._stage_name = "cnn_inference_engine"
    stage._current_task = "NUMBERS"

    image = np.zeros((3, 4, 3), dtype=np.uint8)
    shm = _alloc_shm(image)
    metadata = _make_metadata(shm.name, image.shape, str(image.dtype))
    metadata.full_image_shape = (6, 8, 3)
    metadata.bboxes = [{"class": 0, "conf": 0.9, "bbox": [1.0, 1.0, 3.0, 2.0]}]
    shm.close()

    def fake_predict_digit_crops(**kwargs):
        assert len(kwargs["crops"]) == 1
        np.testing.assert_array_equal(kwargs["crops"][0], image[1:2, 1:3])
        return [DigitPrediction(label="7", confidence=0.8, discard=False)]

    saved_payload = {}

    def fake_save_full_image_bboxes(
        metadata_arg,
        bboxes,
        inference_shape,
        full_image_shape,
        target_resolution,
        *,
        producer,
        mission,
        bbox_logging_config,
        logger,
    ):
        saved_payload["bboxes"] = bboxes
        saved_payload["inference_shape"] = inference_shape
        saved_payload["full_image_shape"] = full_image_shape
        saved_payload["target_resolution"] = target_resolution
        saved_payload["producer"] = producer
        saved_payload["mission"] = mission
        saved_payload["bbox_logging_config"] = bbox_logging_config
        saved_payload["logger"] = logger

    monkeypatch.setattr(
        cnn_stage_module, "predict_digit_crops", fake_predict_digit_crops
    )
    monkeypatch.setattr(
        cnn_stage_module.bbox_utils,
        "save_full_image_bboxes",
        fake_save_full_image_bboxes,
    )

    stage._process_metadata(metadata)

    assert metadata.shm_name is None
    assert metadata.bboxes == [
        {
            "class": 7,
            "conf": 0.9,
            "bbox": [2.0, 2.0, 6.0, 4.0],
            "cnn_label": "7",
            "cnn_conf": 0.8,
        }
    ]
    assert saved_payload["bboxes"] == metadata.bboxes
    assert saved_payload["inference_shape"] == image.shape
    assert saved_payload["full_image_shape"] == (6, 8, 3)
    assert saved_payload["target_resolution"] == [4, 3]
    assert saved_payload["producer"] == "cnn_inference_engine"
    assert saved_payload["mission"] == "NUMBERS"
    assert saved_payload["bbox_logging_config"] == {"enabled": True}
    assert saved_payload["logger"] is stage.logger
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm.name)
