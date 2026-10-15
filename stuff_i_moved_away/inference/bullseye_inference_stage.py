import time
import multiprocessing as mp
from typing import Any

import numpy as np
from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif
from common.homography import (
    image_array_to_uint8_rgb,
    warp_image_to_north_up,
    inverse_transform_points,
    bbox_xyxy_to_corners,
    corners_to_bbox_xyxy,
    clip_bbox_to_shape,
    yolo_bbox_to_xyxy,
    parse_output_size,
)
from stages.inference.default.utils import bbox_utils
from stuff_i_moved_away.inference.find_bullseye import (
    find_bullseye_in_image,
    load_config as load_bullseye_config,
)
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor


def _metadata_row_from_photo(metadata: PhotoExif) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pixel_format": getattr(metadata, "pixel_format", ""),
    }
    pix_data = getattr(metadata, "pix_data", None)
    if pix_data is not None:
        for field in ("lat", "lon", "alt_rel", "roll", "pitch", "yaw"):
            if hasattr(pix_data, field):
                row[field] = getattr(pix_data, field)
    return row


def _bullseye_worker_task(
    shm_name: str,
    shm_shape: tuple,
    shm_dtype: Any,
    sft_config: dict,
    bullseye_config: dict,
    metadata_row: dict,
    output_dir: str | None,
    debug: bool,
    output_size: tuple | None,
    meters_per_pixel: float | None,
    padding_px: int,
    max_output_side: int,
    altitude_offset_m: float,
) -> list[dict[str, Any]]:
    """Worker running in a separate process.

    Attaches to shared memory, runs homography correction on the raw image,
    performs bullseye detection on the north-up warped image, then maps
    all detections back to original-image pixel coordinates before returning.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    try:
        image = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)
        image_rgb = image_array_to_uint8_rgb(
            image, metadata_row.get("pixel_format", "")
        )

        warped, computation = warp_image_to_north_up(
            image_rgb,
            metadata_row,
            sft_config,
            output_size=output_size,
            meters_per_pixel=meters_per_pixel,
            padding_px=padding_px,
            max_output_side=max_output_side,
            altitude_offset_m=altitude_offset_m,
        )

        result = find_bullseye_in_image(
            image=warped,
            config=bullseye_config,
            output_dir=output_dir,
            debug=debug,
        )
        if result is None:
            return []

        bbox_yolo = result.get("bbox_yolo")
        if bbox_yolo is None or len(bbox_yolo) != 5:
            raise ValueError("Invalid YOLO bbox returned by bullseye detector")

        warped_bbox = yolo_bbox_to_xyxy(tuple(bbox_yolo), warped.shape)
        warped_corners = bbox_xyxy_to_corners(warped_bbox)
        source_corners = inverse_transform_points(warped_corners, computation)
        source_bbox = clip_bbox_to_shape(
            corners_to_bbox_xyxy(source_corners), image_rgb.shape
        )

        return [
            {
                "class": int(bbox_yolo[0]),
                "conf": float(result.get("score", 1.0)),
                "bbox": source_bbox,
            }
        ]
    finally:
        existing_shm.close()


class BullseyeInferenceStage(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        ack_q=None,
        stage_name="bullseye_cv_engine",
        idx=3,
    ):
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

        self.stopped = mp.Event()
        self.stopped.clear()

        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        homography_cfg = self.configs.config_dict.get("homography_stage", {})

        self.bullseye_config_path = inference_cfg.get("bullseye_config")
        self.bullseye_output_dir = inference_cfg.get("bullseye_output_dir")
        self.bullseye_debug = bool(inference_cfg.get("bullseye_debug", False))
        self.bullseye_config = self._load_bullseye_config()
        self.bullseye_decision_class_id = bbox_utils.resolve_target_class_id(
            self.configs.config_dict,
            "BULLSEYE",
            default=2,
        )

        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution", [2656, 1504]
        )

        self.homography_output_size = parse_output_size(
            homography_cfg.get("output_size")
        )
        meters_per_pixel = homography_cfg.get("meters_per_pixel")
        self.homography_meters_per_pixel: float | None = (
            float(meters_per_pixel) if meters_per_pixel not in (None, "") else None
        )
        self.homography_padding_px = max(0, int(homography_cfg.get("padding_px", 0)))
        self.homography_max_output_side = max(
            0, int(homography_cfg.get("max_output_side", 0))
        )
        self.altitude_offset_m = float(homography_cfg.get("altitude_offset_m", 0.0))

        self.max_workers = inference_cfg.get("max_workers", 8)
        self.executor = None
        self.active_futures: dict = {}

        self.logger.info(
            f"BullseyeInferenceStage initialized: config_path={self.bullseye_config_path}, "
            f"debug={self.bullseye_debug}"
        )

    def _load_bullseye_config(self) -> dict[str, Any]:
        return load_bullseye_config(self.bullseye_config_path)

    def _bullseye_enabled(self) -> bool:
        return self.current_task == "BULLSEYE"

    def _handle_completed_detection(
        self,
        metadata: PhotoExif,
        bboxes: list[dict[str, Any]],
        image_shape: tuple[int, ...] | None,
    ) -> None:
        full_image_shape = getattr(metadata, "full_image_shape", None)
        full_image_bboxes = bbox_utils.scale_bboxes_to_full_image(
            bboxes,
            image_shape,
            full_image_shape,
        )
        full_image_bboxes = bbox_utils.with_class_id(
            full_image_bboxes,
            getattr(self, "bullseye_decision_class_id", 2),
        )
        metadata.bboxes = full_image_bboxes
        metadata.shm_name = None
        metadata.shm_shape = None
        metadata.shm_dtype = None
        if full_image_bboxes:
            self.send_to_queues(metadata)
        else:
            self.logger.debug(
                f"Bullseye produced no detections for {metadata.filename}; "
                "not forwarding to decision system"
            )

    def mission(self) -> bool:
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.logger.info("Process pool for bullseye inference instantiated.")

        while not self.stopped.is_set():
            self.check_inbound_command()

            # ==========================================
            # PHASE 1: Process Completed Futures
            # ==========================================
            done_futures = [f for f in self.active_futures if f.done()]
            for future in done_futures:
                done_metadata, t_start, shm_shape = self.active_futures.pop(future)

                try:
                    result = future.result()
                    self.fast_ack(done_metadata)

                    m_bboxes = result if isinstance(result, list) else []
                    if result is not None and not isinstance(result, list):
                        self.logger.warning(
                            "Bullseye detector returned invalid results"
                        )

                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    self.logger.info(
                        f"Bullseye processed {done_metadata.filename}: "
                        f"detections={len(m_bboxes)}, elapsed_ms={elapsed_ms:.1f}"
                    )

                    self._handle_completed_detection(
                        done_metadata,
                        m_bboxes,
                        shm_shape,
                    )

                except Exception as e:
                    self.logger.error(f"BullseyeInference worker failed: {e}")
                    self.fast_ack(done_metadata)

            # ==========================================
            # PHASE 2: Check for New Data & Submit Tasks
            # ==========================================
            try:
                metadata: Any = self.receive_from_queue(timeout=1)

                if not isinstance(metadata, PhotoExif):
                    self.send_to_queues(metadata)
                    continue

                if not self._bullseye_enabled():
                    self.logger.debug(f"Bullseye skipped: {metadata.filename}")
                    self.fast_ack(metadata)
                    continue

                t_start = time.perf_counter()

                try:
                    future = self.executor.submit(
                        _bullseye_worker_task,
                        metadata.shm_name,
                        metadata.shm_shape,
                        metadata.shm_dtype,
                        self.configs.config_dict,
                        self.bullseye_config,
                        _metadata_row_from_photo(metadata),
                        self.bullseye_output_dir,
                        self.bullseye_debug,
                        self.homography_output_size,
                        self.homography_meters_per_pixel,
                        self.homography_padding_px,
                        self.homography_max_output_side,
                        self.altitude_offset_m,
                    )
                    self.active_futures[future] = (
                        metadata,
                        t_start,
                        metadata.shm_shape,
                    )
                    self.logger.info("process assigned")
                except Exception as e:
                    self.logger.warning(f"Failed to submit to ProcessPoolExecutor: {e}")

            except Exception:
                pass

            if not self.stopped.is_set():
                time.sleep(0.01)

        return True

    def update(self) -> None:
        super().update()
        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        homography_cfg = self.configs.config_dict.get("homography_stage", {})

        self.bullseye_config_path = inference_cfg.get(
            "bullseye_config", self.bullseye_config_path
        )
        self.bullseye_output_dir = inference_cfg.get(
            "bullseye_output_dir", self.bullseye_output_dir
        )
        self.bullseye_debug = bool(
            inference_cfg.get("bullseye_debug", self.bullseye_debug)
        )
        self.bullseye_config = self._load_bullseye_config()
        self.bullseye_decision_class_id = bbox_utils.resolve_target_class_id(
            self.configs.config_dict,
            "BULLSEYE",
            default=self.bullseye_decision_class_id,
        )

        self.homography_output_size = parse_output_size(
            homography_cfg.get("output_size", self.homography_output_size)
        )
        meters_per_pixel = homography_cfg.get("meters_per_pixel")
        if meters_per_pixel not in (None, ""):
            self.homography_meters_per_pixel = float(meters_per_pixel)
        self.homography_padding_px = max(
            0, int(homography_cfg.get("padding_px", self.homography_padding_px))
        )
        self.homography_max_output_side = max(
            0,
            int(homography_cfg.get("max_output_side", self.homography_max_output_side)),
        )
        self.altitude_offset_m = float(
            homography_cfg.get("altitude_offset_m", self.altitude_offset_m)
        )

    def stop(self) -> None:
        self.logger.info("Stopping BullseyeInferenceStage")
        self.stopped.set()

    def get(self, data: Any) -> None:
        raise NotImplementedError()

    def put(self, data: Any) -> None:
        raise NotImplementedError()

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

    def error_handler(self) -> None:
        pass
