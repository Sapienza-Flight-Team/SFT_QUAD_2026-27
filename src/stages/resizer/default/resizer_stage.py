import time
from queue import Empty
from typing import Any

import cv2
import numpy as np

from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif


class ResizerStage(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        ack_q=None,
        stage_name="preprocessor",
        idx=0,
    ):
        super().__init__(
            configs=configs,
            logger=logger,
            out_q=out_q,
            in_q=in_q,
            idx=idx,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            ack_q=ack_q,
            stage_name=stage_name,
        )
        self.stopped.clear()

        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution", [2656, 1504]
        )

    def _process_frame(self, metadata: PhotoExif) -> None:
        try:
            t_start = time.perf_counter()

            with self.consume_shm_array(metadata) as in_array:
                target_w, target_h = self.target_resolution
                out_shape = (target_h, target_w) + in_array.shape[2:]
                out_dtype = in_array.dtype

                # as usual skip the intermediate copy and just put the resized image into outbound shm
                out_nbytes = int(np.prod(out_shape)) * out_dtype.itemsize
                out_shm = self.allocate_and_register_shm(out_nbytes)
                try:
                    out_array = np.ndarray(
                        out_shape, dtype=out_dtype, buffer=out_shm.buf
                    )
                    cv2.resize(in_array, (target_w, target_h), dst=out_array)
                finally:
                    out_shm.close()

            metadata.shm_name = out_shm.name
            metadata.shm_shape = out_shape
            metadata.shm_dtype = str(out_dtype)

            self.send_to_queues(metadata)
            self.logger.debug("Sent resized image downstream.")

            elapsed = (time.perf_counter() - t_start) * 1000
            self.logger.debug(
                f"ResizerStage: {metadata.filename} | Resize: {elapsed:.2f} ms"
            )

        except FileNotFoundError:
            self.logger.error(
                f"ResizerStage: shared memory '{metadata.shm_name}' not found"
            )
        except Exception as e:
            self.logger.error(f"ResizerStage error: {e}")

    def _forward_non_frame(self, data: Any) -> None:
        self.send_to_queues(data)

    def mission(self) -> bool:
        while not self.stopped.is_set():
            while not self.stopped.is_set():
                self.check_inbound_command()

                try:
                    data: Any = self.receive_from_queue(timeout=1)

                except Empty:
                    continue

                if not isinstance(data, PhotoExif):
                    self.logger.warning("why the fuck is this not a PhotoExif??")
                    continue

                self._process_frame(data)

        return True

    def stop(self) -> None:
        self.logger.info("Stopping ResizerStage")
        self.stopped.set()

    def get(self, data: Any) -> None:
        raise NotImplementedError

    def put(self, data: Any) -> None:
        raise NotImplementedError

    def _dump_data(self) -> None:
        pass

    def ff15(self) -> None:
        pass

    def glhf(self) -> None:
        pass

    def bullseye_stop(self) -> None:
        pass

    def perimeter_stop(self) -> None:
        pass

    def numbers_stop(self) -> None:
        pass
