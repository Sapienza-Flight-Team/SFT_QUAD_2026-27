import copy
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import numpy as np
import torch
from typing import Any, List

from stages.inference.default.utils import bbox_utils

from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif
from ultralytics import YOLO
from queue import Empty


class InferenceStage(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        ack_q=None,
        stage_name="inference_engine",
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

        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution", [2656, 1504]
        )
        self.imgsz = self.configs.config_dict.get("inference_engine", {}).get(
            "imgsz", self.target_resolution
        )
        self.images_per_batch = self.configs.config_dict.get(
            "inference_engine", {}
        ).get("images_per_batch", 1)
        self.batch_size = self.images_per_batch
        self._load_detection_mapping()
        self.bullseye_decision_class_id = bbox_utils.resolve_target_class_id(
            self.configs.config_dict,
            "BULLSEYE",
            default=2,
        )

        device_id = 0
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )

        weights_path = self.configs.config_dict["inference_engine"]["weights"]

        self.logger.info(
            f"Initializing YOLO Inference Stage on {self.device} with {weights_path}"
        )
        self.yolo = YOLO(str(weights_path), task="detect")

    def _yolo_enabled(self) -> bool:
        return self.current_task in ("DEFAULT", "BULLSEYE", "NUMBERS", "PERIMETER")

    def _load_detection_mapping(self) -> None:
        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        raw_classes = inference_cfg.get("classes")
        self.yolo_allowed_classes = (
            {int(class_id) for class_id in raw_classes}
            if isinstance(raw_classes, list)
            else None
        )
        self.yolo_class_map = bbox_utils.resolve_class_id_map(
            self.configs.config_dict,
            inference_cfg.get("yolo_class_map", {}),
        )

    def _prepare_yolo_bboxes(self, bboxes: list[dict]) -> list[dict]:
        yolo_allowed_classes = getattr(self, "yolo_allowed_classes", None)
        if yolo_allowed_classes is not None:
            bboxes = [
                bbox
                for bbox in bboxes
                if int(bbox.get("class", -1)) in yolo_allowed_classes
            ]
        return bbox_utils.remap_bbox_classes(
            bboxes,
            getattr(self, "yolo_class_map", {}),
        )

    def _prepare_handoff_metadata(
        self,
        metadata: PhotoExif,
        image: np.ndarray | None,
        inference_bboxes: list[dict],
        full_image_bboxes: list[dict],
    ) -> PhotoExif:
        handoff_metadata = copy.copy(metadata)

        handoff_metadata.shm_name = None
        handoff_metadata.shm_shape = None
        handoff_metadata.shm_dtype = None
        if self.current_task == "BULLSEYE":
            bullseye_class_id = getattr(self, "bullseye_decision_class_id", 2)
            handoff_metadata.bboxes = [
                bbox
                for bbox in full_image_bboxes
                if int(bbox.get("class", -1)) == bullseye_class_id
            ]
        else:
            handoff_metadata.bboxes = full_image_bboxes
        return handoff_metadata

    def _should_forward_to_decision(self, metadata: PhotoExif) -> bool:
        return self.current_task == "BULLSEYE" and bool(getattr(metadata, "bboxes", []))

    def _send_handoff_metadata(self, metadata: PhotoExif) -> None:
        if not self._should_forward_to_decision(metadata):
            return

        if not self.out_q:
            return

        try:
            self.out_q[-1].put(metadata, timeout=2)
        except Exception as e:
            self.logger.error(
                f"[{self._stage_name}] Failed to send bullseye detection to decision queue: {e}"
            )

    def mission(self) -> bool:
        metadata_buffer = []
        active_shm_blocks = []
        while not self.stopped.is_set():
            while not self.stopped.is_set():
                self.check_inbound_command()
                try:
                    metadata: Any = self.receive_from_queue(timeout=1)

                    if not isinstance(metadata, PhotoExif):
                        self.send_to_queues(metadata)
                        continue

                except Empty:
                    continue

                metadata_buffer.append(metadata)

                if len(metadata_buffer) < self.images_per_batch:
                    continue

                try:
                    if not self._yolo_enabled():
                        self.logger.debug(
                            f"YOLO skipped batch of {len(metadata_buffer)} "
                            f"(task={self.current_task})"
                        )
                        continue
                    start_time = time.time()

                    inference_images = []
                    inference_image_indexes = []
                    image_shapes: List[tuple[int, ...] | None] = []

                    t_load = 0.0

                    for metadata_index, m in enumerate(metadata_buffer):
                        t0 = time.time()
                        try:
                            in_shm = shared_memory.SharedMemory(name=m.shm_name)

                            # we keep track of the file descriptor we create to access the information in RAM
                            active_shm_blocks.append(in_shm)

                        except FileNotFoundError:
                            self.logger.warning(
                                f"Inference: Shared memory {m.shm_name} not found!"
                            )
                            image_shapes.append(None)
                            continue

                        in_array: np.ndarray = np.ndarray(
                            m.shm_shape, dtype=np.dtype(m.shm_dtype), buffer=in_shm.buf
                        )
                        image_shapes.append(in_array.shape)
                        t_load += time.time() - t0

                        inference_images.append(in_array)
                        inference_image_indexes.append(metadata_index)

                    t2 = time.time()
                    all_bboxes_by_image: List[List[dict]] = [
                        [] for _ in metadata_buffer
                    ]

                    if inference_images:
                        for i in range(0, len(inference_images), self.batch_size):
                            batch_images = inference_images[i : i + self.batch_size]
                            batch_indexes = inference_image_indexes[
                                i : i + self.batch_size
                            ]

                            pad_needed = self.batch_size - len(batch_images)
                            if pad_needed > 0:
                                batch_images = (
                                    batch_images + [batch_images[-1]] * pad_needed
                                )

                            results = self.yolo(
                                batch_images, imgsz=self.imgsz, verbose=False
                            )

                            for res, metadata_index in zip(
                                results[: len(batch_indexes)], batch_indexes
                            ):
                                raw_image_boxes = []
                                for box in res.boxes:
                                    cls = int(box.cls[0].item())
                                    conf = float(box.conf[0].item())
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                                    raw_image_boxes.append(
                                        {
                                            "class": cls,
                                            "conf": conf,
                                            "bbox": [x1, y1, x2, y2],
                                        }
                                    )
                                image_boxes = self._prepare_yolo_bboxes(raw_image_boxes)
                                all_bboxes_by_image[metadata_index] = image_boxes
                                self.logger.debug(
                                    f"YOLO: image {m.filename} → {len(image_boxes)} detections"
                                )

                    t_infer = time.time() - t2

                    for metadata_index, (m, image_shape) in enumerate(
                        zip(metadata_buffer, image_shapes)
                    ):
                        m_bboxes = all_bboxes_by_image[metadata_index]
                        full_image_shape = getattr(m, "full_image_shape", None)
                        full_image_bboxes = bbox_utils.scale_bboxes_to_full_image(
                            m_bboxes, image_shape, full_image_shape
                        )
                        m.bboxes = full_image_bboxes
                        # TODO: make this optional, we might want to just send the bboxes through the queue without saving them to disk
                        bbox_utils.save_full_image_bboxes(
                            m,
                            full_image_bboxes,
                            image_shape,
                            full_image_shape,
                            self.target_resolution,
                            producer=self._stage_name,
                            mission=self.current_task,
                            bbox_logging_config=self.configs.config_dict.get(
                                "bbox_logging", {}
                            ),
                            logger=self.logger,
                        )
                        handoff_metadata = self._prepare_handoff_metadata(
                            m,
                            None,
                            m_bboxes,
                            full_image_bboxes,
                        )
                        self._send_handoff_metadata(handoff_metadata)
                        if self._should_forward_to_decision(handoff_metadata):
                            self.logger.debug(
                                "Sent bullseye detection to decision system"
                            )

                    elapsed_time = time.time() - start_time
                    self.logger.debug(
                        f"Inference Latency ({self.images_per_batch} images) - Total: {elapsed_time:.4f}s | Load: {t_load:.4f}s | Infer ({len(inference_images)} full images): {t_infer:.4f}s"
                    )

                except Exception as e:
                    self.logger.error(f"InferenceStage error: {e}")
                finally:
                    # close local file descriptors (the GC handles unlinking)
                    for shm in active_shm_blocks:
                        shm.close()
                    active_shm_blocks.clear()

                    # ACK every metadata entry so the GC knows we're done with it
                    for m in metadata_buffer:
                        self.fast_ack(m)

                    # now we can clear the metadata buffer
                    metadata_buffer.clear()

            if not self.stopped.is_set():
                time.sleep(0.05)

        return True

    def update(self) -> None:
        updated = super().update()
        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution", self.target_resolution
        )
        if updated:
            self.images_per_batch = self.configs.config_dict.get(
                "inference_engine", {}
            ).get("images_per_batch", self.images_per_batch)
            self.imgsz = self.configs.config_dict.get("inference_engine", {}).get(
                "imgsz", self.target_resolution
            )
            self.batch_size = self.images_per_batch
            self._load_detection_mapping()
            self.bullseye_decision_class_id = bbox_utils.resolve_target_class_id(
                self.configs.config_dict,
                "BULLSEYE",
                default=self.bullseye_decision_class_id,
            )
            self.logger.info(
                f"Inference configured with target_resolution={self.target_resolution}, "
                f"batch_size={self.batch_size}, imgsz={self.imgsz}"
            )

    def stop(self) -> None:
        self.logger.info("Stopping InferenceStage")
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
