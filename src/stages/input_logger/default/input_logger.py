"""Persist the images and telemetry already paired by the aggregator."""

import csv
import datetime
import os
import time
from dataclasses import replace
from enum import Enum
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any, TextIO

import numpy as np

from common.config import SFTConfig
from core.logger import SFTLogger
from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif


class MetadataSaver(PipelineStage):
    """Save each buffered ``(image, PhotoExif)`` pair without matching or filtering.

    ``in_q`` is a dedicated FIFO from the aggregator. It may alternatively carry
    a ``PhotoExif`` describing a shared-memory image; that buffer is acknowledged
    only after saving finishes. The aggregator must keep it alive until then.

    Images use the camera engines' NumPy format and naming convention::

        <camera.img_out_dir>/<YYYYMMDD-HHMMSS>/<index>_<initial_timestamp>.npy
        <metadata_db parent>/<YYYYMMDD-HHMMSS>/metadata_<YYYYMMDD-HHMMSS>.csv

    With the default configuration both live in the same run folder. A custom
    ``metadata_db`` basename is retained, with the run timestamp appended.
    Send ``None`` after the last frame for FIFO shutdown, or stop the producer
    before calling ``stop()`` to drain the remaining input. Files open in the
    consuming process, so constructing the stage is safe with multiprocessing
    spawn. ``save()`` also supports direct use; call ``close()`` when finished.
    """

    METADATA_COLUMNS = {
        "filename": "filename",
        "timestamp": "timestamp",
        "exposure_time": "exposure_time",
        "gain": "gain",
        "black_level": "black_level",
        "bsl_color_space": "bsl_color_space",
        "gamma": "gamma",
        "light_source_preset": "light_source_preset",
        "pixel_format": "pixel_format",
        "camera_trigger_seq": "pix_data.camera_trigger_seq",
        "teensy_timestamp": "pix_data.teensy_timestamp",
        "lat": "pix_data.lat",
        "lon": "pix_data.lon",
        "alt_msl": "pix_data.alt_msl",
        "alt_rel": "pix_data.alt_rel",
        "roll": "pix_data.roll",
        "pitch": "pix_data.pitch",
        "yaw": "pix_data.yaw",
        "mission_mode": "pix_data.mission_mode",
        "mission_state": "pix_data.mission_state",
    }

    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        in_q: Queue | None = None,
        out_q: list[Queue] | None = None,
        in_command_q: list[Queue] | None = None,
        out_command_q: list[Queue] | None = None,
        ack_q: Queue | None = None,
        stage_name: str = "picture_saver",
        idx: int = 0,
    ) -> None:
        super().__init__(
            configs=configs,
            logger=logger,
            in_q=in_q,
            out_q=out_q,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            ack_q=ack_q,
            stage_name=stage_name,
            idx=idx,
        )
        saver_config = configs.config_dict["picture_saver"]

        now = time.time()
        self.db_creation_time = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        self.initial_timestamp = (
            datetime.datetime.fromtimestamp(now)
            .isoformat(timespec="milliseconds", sep="-")
            .replace(":", "-")
        )
        self.out_dir = os.path.join(
            configs.config_dict["camera"]["img_out_dir"], self.db_creation_time
        )
        metadata_path = saver_config["metadata_db"]
        metadata_stem = os.path.splitext(os.path.basename(metadata_path))[0]
        self.out_metadata = os.path.join(
            os.path.dirname(metadata_path),
            self.db_creation_time,
            f"{metadata_stem}_{self.db_creation_time}.csv",
        )
        self.saved_img_count = 0
        self.f: TextIO | None = None
        self.writer: Any = None
        self._metadata_initialized = False
        self._closed = False

    def _initialize_metadata(self) -> None:
        """Create the run directories and CSV header in the consuming process."""
        if self._closed:
            raise RuntimeError("MetadataSaver is closed")
        if self._metadata_initialized:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.out_metadata), exist_ok=True)
        # Refuse to truncate an earlier run if two runs have the same timestamp.
        self.f = open(self.out_metadata, mode="x", newline="", encoding="utf-8")
        try:
            self.writer = csv.writer(self.f)
            self.writer.writerow(self.METADATA_COLUMNS)
            self.f.flush()
        except Exception:
            self.close()
            raise
        self._metadata_initialized = True

    @staticmethod
    def _resolve_path(obj: object, path: str) -> Any:
        """Read a camera field or a nested PixData field; missing fields are blank."""
        value = obj
        for part in path.split("."):
            value = getattr(value, part, None)
        return value.value if isinstance(value, Enum) else value

    def _metadata_row(self, metadata: PhotoExif) -> list[Any]:
        row = []
        for path in self.METADATA_COLUMNS.values():
            value = self._resolve_path(metadata, path)
            if path == "timestamp":
                # PhotoExif.timestamp is Unix seconds; retain the legacy CSV format.
                try:
                    value = datetime.datetime.fromtimestamp(value).isoformat(
                        timespec="seconds", sep=" "
                    )
                except (TypeError, ValueError, OverflowError, OSError):
                    value = ""
            row.append(value)
        return row

    def save(self, image: np.ndarray, metadata: PhotoExif) -> PhotoExif:
        """Write one image and row, returning a copy with the saved filename.

        The saver owns disk naming. The aggregator's metadata is left untouched
        because other stages may be consuming the same object. Write errors
        propagate so missing images or metadata cannot silently go unnoticed.
        """
        self._initialize_metadata()
        filename = os.path.join(
            self.out_dir, f"{self.saved_img_count}_{self.initial_timestamp}.npy"
        )
        saved_metadata = replace(metadata, filename=filename)
        row = self._metadata_row(saved_metadata)
        with open(filename, "xb") as image_file:
            try:
                np.save(image_file, image, allow_pickle=False)
            except Exception:
                os.unlink(filename)
                raise

        # Write the row only after the corresponding image has been written.
        assert self.f is not None
        self.writer.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())
        self.saved_img_count += 1
        self.logger.debug(f"Saved image and metadata: {filename}")
        return saved_metadata

    def get(self, data: tuple[np.ndarray, PhotoExif] | PhotoExif) -> None:
        """Consume one already matched pair or a shared-memory descriptor."""
        if isinstance(data, PhotoExif):
            with self.consume_shm_array(data) as image:
                self.save(image, data)
        else:
            image, metadata = data
            self.save(image, metadata)

    def mission(self) -> bool:
        """Persist buffered frames until the sentinel or stop-and-drain completes."""
        if self.in_q is None:
            raise ValueError("MetadataSaver requires an aggregator input queue")
        try:
            self._initialize_metadata()
            while True:
                self.check_inbound_command()
                try:
                    data = self.receive_from_queue(timeout=0.1)
                except Empty:
                    if self.stopped.is_set():
                        break
                    continue
                if data is None:
                    break
                self.get(data)
            return True
        finally:
            self.close()

    def stop(self) -> None:
        """Request shutdown after the producer stops and buffered frames drain."""
        self.stopped.set()

    def close(self) -> None:
        """Flush and close the CSV in the process that owns it; safe to repeat."""
        try:
            if self.f is not None:
                self.f.close()
        finally:
            self.f = None
            self.writer = None
            self._closed = True

    def put(self, data: Any) -> None:
        """This is a terminal stage; the aggregator feeds other consumers directly."""
        raise NotImplementedError("MetadataSaver has no downstream output")

    def _dump_data(self) -> None:
        self.close()
