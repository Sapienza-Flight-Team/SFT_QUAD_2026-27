import os
import time
import multiprocessing as mp
from queue import Empty
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import pandas as pd

from stages.cam.camera_engine.camera_engine_stage import CameraEngine
from common.config import SFTConfig
from core.logger import SFTLogger
from common.structs import PhotoExif


class MockCameraEngine(CameraEngine):
    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        in_q: Optional[mp.Queue],
        out_q: Optional[list[mp.Queue]],
        idx: int,
        in_command_q: Optional[list[mp.Queue]] = None,
        out_command_q: Optional[list[mp.Queue]] = None,
        ack_q: Optional[mp.Queue] = None,
        stage_name: str = "camera",
    ) -> None:
        super().__init__(
            configs=configs,
            logger=logger,
            in_q=in_q,
            out_q=out_q,
            ack_q=ack_q,
            idx=idx,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            stage_name=stage_name,
        )
        self.use_random_data = self.config_dict["camera"].get(
            "emulation_use_random_data", False
        )
        self.mock_metadata_csv = self.config_dict["camera"].get("mock_metadata_csv")
        self.ready = False

    def gps_in_queue(self) -> bool:  # helper method
        try:
            if self.receive_from_queue() is not None:
                return True
            else:
                return False
        except Empty:
            return False

    def _load_replay_metadata(self) -> pd.DataFrame:
        replay_csv = pd.read_csv(self.mock_metadata_csv)
        if "filename" not in replay_csv.columns:
            raise ValueError(f"{self.mock_metadata_csv} must contain a filename column")
        if replay_csv.empty:
            raise ValueError(f"{self.mock_metadata_csv} does not contain any rows")
        return replay_csv

    def _row_value(self, row: pd.Series, column: str, default):
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return value

    def _random_image_row(self, img_count: int) -> dict:
        filename = os.path.join(self.out_dir, f"mock_{img_count}.npy")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not os.path.exists(filename):
            np.save(filename, np.zeros((480, 640), dtype=np.uint8), allow_pickle=False)
        return {
            "filename": filename,
            "timestamp": int(time.time() * 1000),
            "exposure_time": 0,
            "gain": 0,
            "black_level": 0,
            "bsl_color_space": "",
            "gamma": 0,
            "light_source_preset": "",
            "pixel_format": "BayerRG8",
        }

    def _photo_exif_from_row(self, row, pix_data) -> PhotoExif:
        return PhotoExif(
            filename=str(self._row_value(row, "filename", "")),
            timestamp=float(self._row_value(row, "timestamp", int(time.time() * 1000))),
            exposure_time=float(self._row_value(row, "exposure_time", 0)),
            gain=float(self._row_value(row, "gain", 0)),
            black_level=int(self._row_value(row, "black_level", 0)),
            bsl_color_space=str(self._row_value(row, "bsl_color_space", "")),
            gamma=int(self._row_value(row, "gamma", 0)),
            light_source_preset=str(self._row_value(row, "light_source_preset", "")),
            pixel_format=str(self._row_value(row, "pixel_format", "BayerRG8")),
            pix_data=pix_data,
        )

    def mission(self) -> bool:
        self.is_initialized.set()

        def wait_for_ready():
            self.check_inbound_command(block=True, timeout=1)

        wait_for_ready()
        while not self.ready and not self.stopped.is_set():
            wait_for_ready()

        while self.ready and not self.stopped.is_set():
            if not self.use_random_data:
                try:
                    replay_metadata = self._load_replay_metadata()
                except Exception as e:
                    self.logger.warning(f"Could not read metadata CSV: {e}")
                    return True
            else:
                replay_metadata = pd.DataFrame()

            img_count = 0
            while self.ready and not self.stopped.is_set():
                self.check_inbound_command()
                try:
                    pix_data = self.receive_from_queue()
                except Empty:
                    pix_data = None

                if pix_data in (None, "GLHF"):
                    time.sleep(0.01)
                    continue
                if pix_data == "FF15":
                    return False

                if self.use_random_data:
                    row = self._random_image_row(img_count)
                else:
                    row = replay_metadata.iloc[img_count % len(replay_metadata)]

                metadata = self._photo_exif_from_row(row, pix_data)

                if not os.path.exists(metadata.filename):
                    self.logger.warning(
                        f"Mock replay image does not exist: {metadata.filename}"
                    )
                    img_count += 1
                    continue

                # image not actually loaded in RAM here, we are only loading information about its size, so we avoid
                # committing double the RAM for a single image (loading locally and then putting in shared memory)
                try:
                    mmap_data = np.load(metadata.filename, mmap_mode="r")
                except Exception as e:
                    self.logger.error(f"Failed to load image {metadata.filename}: {e}")
                    img_count += 1
                    continue

                shm = shared_memory.SharedMemory(create=True, size=mmap_data.nbytes)

                shm_array: np.ndarray = np.ndarray(
                    mmap_data.shape, dtype=mmap_data.dtype, buffer=shm.buf
                )

                # image loaded here
                shm_array[:] = mmap_data[:]

                metadata.shm_name = shm.name
                metadata.shm_shape = mmap_data.shape
                metadata.shm_dtype = str(mmap_data.dtype)

                shm.close()

                self.put(metadata)
                self.logger.debug(f"mock camera sent image {img_count}")
                img_count += 1

            if self.stopped.is_set():
                self.logger.info(f"Sent a total of {img_count} fake images")
                self.logger.info("Quitting camera simulation")
                return True

            return False

        return self.stopped.is_set()

    def glhf(self) -> None:
        self.logger.info("All clear to start CameraEngine!")
        self.ready = True

    def bullseye_stop(self) -> None:
        self.ready = False

    def perimeter_stop(self) -> None:
        self.ready = False

    def numbers_stop(self) -> None:
        self.ready = False

    def update(self) -> None:
        pass
