"""
Unit + integration tests for the split Converter/Resizer pipeline stages.

All tests that touch shared memory run within the same process (no mp.Process
spawning) so that SHM blocks are visible and cleanup is reliable.
"""

import gc
from multiprocessing import shared_memory
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stages.resizer.converter import ConverterStage
from stages.resizer.default.resizer import ResizerStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pix_data() -> PixData:
    return PixData()


def _make_photo_exif(shm_name: str, shape: tuple, dtype: str = "uint8") -> PhotoExif:
    return PhotoExif(
        filename="test.raw",
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="BayerBG",
        gamma=1,
        light_source_preset="",
        pixel_format="BayerBG8",
        pix_data=_make_pix_data(),
        shm_name=shm_name,
        shm_shape=shape,
        shm_dtype=dtype,
    )


def _alloc_shm(array: np.ndarray) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    buf = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(buf, array)
    return shm


def _close_unlink(shm: shared_memory.SharedMemory) -> None:
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass


def _make_configs(target_resolution=(640, 480)):
    cfg = MagicMock()
    cfg.config_dict = {
        "preprocessor": {"target_resolution": list(target_resolution)},
        "converter": {},
        "utils": {},
    }
    cfg.config_yaml = "sft_config.yaml"
    return cfg


def _build_stage(cls, out_qs, target_resolution=(640, 480), idx=0):
    """Instantiate a stage without spawning a subprocess."""
    configs = _make_configs(target_resolution)
    logger = MagicMock()
    stage = cls.__new__(cls)
    stage.in_q = None
    stage.out_q = out_qs
    stage.configs = configs
    stage.logger = logger
    stage._stage_name = "converter" if cls is ConverterStage else "preprocessor"
    stage.idx = idx
    stage.in_command_q = None
    stage.out_command_q = None
    stage._current_task = "DEFAULT"
    if cls is ResizerStage:
        stage.target_resolution = list(target_resolution)
    return stage


# ---------------------------------------------------------------------------
# ConverterStage — unit tests
# ---------------------------------------------------------------------------


class TestConverterStage:

    def test_converts_bayer_to_rgb(self):
        """2-D BayerBG input must produce a 3-channel RGB array on both outputs."""
        bayer = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        assert q0.put.call_count == 1
        assert q1.put.call_count == 1
        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            arr = np.ndarray(m.shm_shape, dtype=np.dtype(m.shm_dtype), buffer=s.buf)
            assert arr.ndim == 3 and arr.shape[2] == 3
            _close_unlink(s)

        _close_unlink(shm)

    def test_output_shm_names_are_distinct(self):
        bayer = np.zeros((60, 60), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        m0 = q0.put.call_args[0][0]
        m1 = q1.put.call_args[0][0]
        assert m0.shm_name != m1.shm_name

        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)

    def test_full_image_shape_set_on_both_outputs(self):
        bayer = np.zeros((80, 120), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        for q in (q0, q1):
            m = q.put.call_args[0][0]
            assert m.full_image_shape == (80, 120, 3)
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)

    def test_input_shm_unlinked_after_conversion(self):
        bayer = np.zeros((50, 50), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        name = shm.name
        meta = _make_photo_exif(name, bayer.shape)

        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        shm.close()

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    def test_non_photo_exif_forwarded_to_all_queues(self):
        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        sentinel = object()
        stage._forward_non_frame(sentinel)
        q0.put.assert_called_once_with(sentinel)
        q1.put.assert_called_once_with(sentinel)

    def test_non_2d_array_logs_warning_and_still_fans_out(self):
        """If the camera already sends RGB, the stage must warn but not crash."""
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)

        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        stage.logger.warning.assert_called()
        assert q0.put.call_count == 1
        assert q1.put.call_count == 1

        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)

    def test_missing_shm_logs_error_and_does_not_crash(self):
        meta = _make_photo_exif("nonexistent_shm_converter_xyz", (100, 100))
        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)
        stage.logger.error.assert_called()

    def test_output_pixel_values_match_cv2_conversion(self):
        bayer = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        expected = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)

        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)
        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            arr = np.ndarray(m.shm_shape, dtype=np.dtype(m.shm_dtype), buffer=s.buf)
            np.testing.assert_array_equal(arr, expected)
            _close_unlink(s)
        _close_unlink(shm)

    def test_metadata_filename_preserved_on_both_outputs(self):
        bayer = np.zeros((50, 50), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)
        meta.filename = "img_042.raw"

        q0, q1 = MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1])
        stage._process_frame(meta)

        for q in (q0, q1):
            m = q.put.call_args[0][0]
            assert m.filename == "img_042.raw"
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)


# ---------------------------------------------------------------------------
# ResizerStage — unit tests
# ---------------------------------------------------------------------------


class TestResizerStage:

    def test_resizes_to_target_resolution(self):
        rgb = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)

        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(100, 80))
        stage._process_frame(meta)

        m = out_q.put.call_args[0][0]
        s = shared_memory.SharedMemory(name=m.shm_name)
        arr = np.ndarray(m.shm_shape, dtype=np.dtype(m.shm_dtype), buffer=s.buf)
        assert arr.shape == (80, 100, 3)  # (height, width, channels)
        _close_unlink(s)
        _close_unlink(shm)

    def test_metadata_updated_after_resize(self):
        rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)

        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(200, 150))
        stage._process_frame(meta)

        m = out_q.put.call_args[0][0]
        assert m.shm_shape == (150, 200, 3)
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        _close_unlink(shm)

    def test_input_shm_unlinked_after_resize(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        name = shm.name
        meta = _make_photo_exif(name, rgb.shape)

        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(50, 50))
        stage._process_frame(meta)

        m = out_q.put.call_args[0][0]
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        shm.close()

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    def test_non_photo_exif_forwarded(self):
        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q])
        sentinel = {"type": "command"}
        stage._forward_non_frame(sentinel)
        out_q.put.assert_called_once_with(sentinel, timeout=2)

    def test_pixel_values_match_cv2_resize(self):
        rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        target_w, target_h = 60, 40
        expected = cv2.resize(rgb, (target_w, target_h))

        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)
        out_q = MagicMock()
        stage = _build_stage(
            ResizerStage, [out_q], target_resolution=(target_w, target_h)
        )
        stage._process_frame(meta)

        m = out_q.put.call_args[0][0]
        s = shared_memory.SharedMemory(name=m.shm_name)
        arr = np.ndarray(m.shm_shape, dtype=np.dtype(m.shm_dtype), buffer=s.buf)
        np.testing.assert_array_equal(arr, expected)
        _close_unlink(s)
        _close_unlink(shm)

    def test_missing_input_shm_logs_error(self):
        meta = _make_photo_exif("nonexistent_shm_resizer_abc", (100, 100, 3))
        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q])
        stage._process_frame(meta)
        stage.logger.error.assert_called()

    def test_full_image_shape_preserved(self):
        """full_image_shape set by ConverterStage must survive through ResizerStage."""
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)
        meta.full_image_shape = (100, 100, 3)

        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(50, 50))
        stage._process_frame(meta)

        m = out_q.put.call_args[0][0]
        assert m.full_image_shape == (100, 100, 3)
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        _close_unlink(shm)

    def test_filename_preserved(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)
        meta.filename = "img_007.raw"

        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(50, 50))
        stage._process_frame(meta)

        m = out_q.put.call_args[0][0]
        assert m.filename == "img_007.raw"
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        _close_unlink(shm)


# ---------------------------------------------------------------------------
# Integration: ConverterStage → ResizerStage
# ---------------------------------------------------------------------------


class TestConverterResizerIntegration:

    def test_full_pipeline_produces_correct_resized_output(self):
        """Bayer → ConverterStage → ResizerStage → resized RGB with correct pixels."""
        bayer = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        expected_rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
        target_w, target_h = 100, 80
        expected_resized = cv2.resize(expected_rgb, (target_w, target_h))

        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        conv_bull_q, conv_resz_q = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [conv_bull_q, conv_resz_q])
        converter._process_frame(meta)

        meta_for_resizer = conv_resz_q.put.call_args[0][0]
        resized_out_q = MagicMock()
        resizer = _build_stage(
            ResizerStage, [resized_out_q], target_resolution=(target_w, target_h)
        )
        resizer._process_frame(meta_for_resizer)

        m_resized = resized_out_q.put.call_args[0][0]
        s = shared_memory.SharedMemory(name=m_resized.shm_name)
        arr = np.ndarray(
            m_resized.shm_shape, dtype=np.dtype(m_resized.shm_dtype), buffer=s.buf
        )
        np.testing.assert_array_equal(arr, expected_resized)
        _close_unlink(s)

        m_bull = conv_bull_q.put.call_args[0][0]
        s2 = shared_memory.SharedMemory(name=m_bull.shm_name)
        _close_unlink(s2)
        _close_unlink(shm)

    def test_bullseye_branch_receives_full_resolution(self):
        """The bullseye consumer must see the full-resolution RGB image."""
        bayer = np.zeros((300, 400), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        conv_bull_q, conv_resz_q = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [conv_bull_q, conv_resz_q])
        converter._process_frame(meta)

        m_bull = conv_bull_q.put.call_args[0][0]
        assert m_bull.shm_shape == (300, 400, 3)

        for q in (conv_bull_q, conv_resz_q):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)

    def test_yolo_branch_receives_downscaled_image(self):
        """After the Resizer, the YOLO consumer must see the target resolution."""
        bayer = np.zeros((300, 400), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        conv_bull_q, conv_resz_q = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [conv_bull_q, conv_resz_q])
        converter._process_frame(meta)

        meta_for_resizer = conv_resz_q.put.call_args[0][0]
        resized_out_q = MagicMock()
        resizer = _build_stage(
            ResizerStage, [resized_out_q], target_resolution=(200, 150)
        )
        resizer._process_frame(meta_for_resizer)

        m_resized = resized_out_q.put.call_args[0][0]
        assert m_resized.shm_shape == (150, 200, 3)

        s = shared_memory.SharedMemory(name=m_resized.shm_name)
        _close_unlink(s)
        m_bull = conv_bull_q.put.call_args[0][0]
        s2 = shared_memory.SharedMemory(name=m_bull.shm_name)
        _close_unlink(s2)
        _close_unlink(shm)

    def test_two_branches_hold_independent_shm_blocks(self):
        """Freeing one consumer's SHM block must not affect the other."""
        bayer = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)

        q0, q1 = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [q0, q1])
        converter._process_frame(meta)

        m0 = q0.put.call_args[0][0]
        m1 = q1.put.call_args[0][0]
        assert m0.shm_name != m1.shm_name

        s0 = shared_memory.SharedMemory(name=m0.shm_name)
        _close_unlink(s0)

        # Second block must still be accessible after the first is freed.
        s1 = shared_memory.SharedMemory(name=m1.shm_name)
        arr = np.ndarray(m1.shm_shape, dtype=np.dtype(m1.shm_dtype), buffer=s1.buf)
        assert arr.shape == (64, 64, 3)
        _close_unlink(s1)
        _close_unlink(shm)

    def test_no_orphan_shm_after_full_pipeline(self):
        """After all consumers clean up, every SHM block must be gone."""
        bayer = np.zeros((50, 50), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        original_name = shm.name
        meta = _make_photo_exif(original_name, bayer.shape)

        q0, q1 = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [q0, q1])
        converter._process_frame(meta)

        m_bull = q0.put.call_args[0][0]
        m_resz = q1.put.call_args[0][0]
        bull_name = m_bull.shm_name
        resz_name = m_resz.shm_name

        for name in (bull_name, resz_name):
            s = shared_memory.SharedMemory(name=name)
            _close_unlink(s)

        _close_unlink(shm)
        gc.collect()

        for name in (original_name, bull_name, resz_name):
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_converter_minimal_bayer_frame(self):
        """Smallest valid BayerBG pattern (2×2) must not crash."""
        bayer = np.zeros((2, 2), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)
        q0, q1 = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [q0, q1])
        converter._process_frame(meta)
        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)

    def test_resizer_upscale(self):
        """Resizer must handle upscaling (target > source) without error."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)
        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(200, 200))
        stage._process_frame(meta)
        m = out_q.put.call_args[0][0]
        assert m.shm_shape == (200, 200, 3)
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        _close_unlink(shm)

    @pytest.mark.slow
    def test_converter_large_frame(self):
        """4K-equivalent frame (2160×3840) must process without memory errors."""
        bayer = np.random.randint(0, 256, (2160, 3840), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)
        q0, q1 = MagicMock(), MagicMock()
        converter = _build_stage(ConverterStage, [q0, q1])
        converter._process_frame(meta)
        for q in (q0, q1):
            m = q.put.call_args[0][0]
            s = shared_memory.SharedMemory(name=m.shm_name)
            _close_unlink(s)
        _close_unlink(shm)

    def test_converter_single_output_queue(self):
        """ConverterStage with only one output queue must still work correctly."""
        bayer = np.zeros((40, 40), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)
        q0 = MagicMock()
        stage = _build_stage(ConverterStage, [q0])
        stage._process_frame(meta)
        assert q0.put.call_count == 1
        m = q0.put.call_args[0][0]
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        _close_unlink(shm)

    def test_converter_three_output_queues(self):
        """ConverterStage must allocate one independent SHM block per downstream queue."""
        bayer = np.zeros((32, 32), dtype=np.uint8)
        shm = _alloc_shm(bayer)
        meta = _make_photo_exif(shm.name, bayer.shape)
        q0, q1, q2 = MagicMock(), MagicMock(), MagicMock()
        stage = _build_stage(ConverterStage, [q0, q1, q2])
        stage._process_frame(meta)

        outputs = [q.put.call_args[0][0] for q in (q0, q1, q2)]
        shm_names = {output.shm_name for output in outputs}
        assert len(shm_names) == 3

        for output in outputs:
            s = shared_memory.SharedMemory(name=output.shm_name)
            arr = np.ndarray(
                output.shm_shape, dtype=np.dtype(output.shm_dtype), buffer=s.buf
            )
            assert arr.shape == (32, 32, 3)
            _close_unlink(s)

        shm.close()

    def test_resizer_square_non_standard_resolution(self):
        """Resizer must correctly handle non-standard target resolutions."""
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        shm = _alloc_shm(rgb)
        meta = _make_photo_exif(shm.name, rgb.shape)
        out_q = MagicMock()
        stage = _build_stage(ResizerStage, [out_q], target_resolution=(333, 222))
        stage._process_frame(meta)
        m = out_q.put.call_args[0][0]
        assert m.shm_shape == (222, 333, 3)
        s = shared_memory.SharedMemory(name=m.shm_name)
        _close_unlink(s)
        _close_unlink(shm)
