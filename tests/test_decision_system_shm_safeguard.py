from multiprocessing import shared_memory
from unittest.mock import MagicMock

import numpy as np
import pytest

from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stages.decision_system.default.decision_system_stage import DecisionSystem


def test_decision_get_releases_unexpected_image_shm() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
    shm_array = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
    np.copyto(shm_array, image)
    shm_name = shm.name
    shm.close()

    photo = PhotoExif(
        filename="frame.npy",
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
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        ),
        shm_name=shm_name,
        shm_shape=image.shape,
        shm_dtype=str(image.dtype),
    )
    photo.bboxes = []

    stage = DecisionSystem.__new__(DecisionSystem)
    stage.logger = MagicMock()
    stage.receive_from_queue = MagicMock(return_value=photo)

    stage.get()

    stage.logger.warning.assert_called_once()
    assert shm_name in stage.logger.warning.call_args[0][0]
    assert photo.shm_name is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm_name)
