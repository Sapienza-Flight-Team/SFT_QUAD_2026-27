from multiprocessing import shared_memory

import numpy as np
import pytest

from common.mavlink_structs import PixData
from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif


class DummyStage(PipelineStage):
    def mission(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def get(self, data):
        pass

    def put(self, data):
        pass

    def _dump_data(self) -> None:
        pass


def _metadata_for_shm(
    shm_name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> PhotoExif:
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


def _alloc_shm(array: np.ndarray) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(shm_array, array)
    return shm


def test_metadata_array_consumes_shared_memory_by_default() -> None:
    stage = DummyStage.__new__(DummyStage)
    image = np.arange(3 * 4, dtype=np.uint8).reshape((3, 4))
    shm = _alloc_shm(image)
    metadata = _metadata_for_shm(shm.name, image.shape, str(image.dtype))
    shm.close()

    with stage.metadata_array(metadata) as loaded:
        np.testing.assert_array_equal(loaded, image)

    assert metadata.shm_name is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm.name)


def test_attach_array_to_metadata_creates_new_owned_block() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
    metadata = _metadata_for_shm("old-shm", image.shape, str(image.dtype))

    PipelineStage.attach_array_to_metadata(metadata, image)

    assert metadata.shm_name != "old-shm"
    assert metadata.shm_shape == image.shape
    assert metadata.shm_dtype == str(image.dtype)

    shm = shared_memory.SharedMemory(name=metadata.shm_name)
    try:
        loaded = np.ndarray(
            metadata.shm_shape,
            dtype=np.dtype(metadata.shm_dtype),
            buffer=shm.buf,
        )
        np.testing.assert_array_equal(loaded, image)
    finally:
        shm.close()
        shm.unlink()
