import json
import time
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Any

import cv2
import numpy as np

from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif
from stuff_i_moved_away.inference.find_bw_targets import (
    find_bw_targets_in_image,
    load_config as load_bw_targets_config,
)
from stages.inference.default.utils import bbox_utils


class CVGCPInferenceStage(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        ack_q=None,
        stage_name="cv_gcp_engine",
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
        self.gcp_class_id = int(inference_cfg.get("cv_gcp_class_id", 10))
        self.gcp_config_path = inference_cfg.get("bw_targets_config")
        self.gcp_inline_config = inference_cfg.get("cv_gcp_config")
        self.gcp_config = self._load_gcp_config()

        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution", [2656, 1504]
        )

        self.review_dir_name = str(
            inference_cfg.get("cv_gcp_review_dir", "cv_gcp_review")
        )
        self.save_positive_images = bool(inference_cfg.get("cv_gcp_save_images", True))
        self.save_positive_overlays = bool(
            inference_cfg.get("cv_gcp_save_overlays", True)
        )
        self.save_positive_metadata = bool(
            inference_cfg.get("cv_gcp_save_metadata", True)
        )
        self.gcp_debug = bool(inference_cfg.get("cv_gcp_debug", False))

    def _cv_gcp_enabled(self) -> bool:
        return self.current_task in ("NUMBERS", "PERIMETER")

    def _load_gcp_config(self) -> dict[str, Any]:
        gcp_inline_config = getattr(self, "gcp_inline_config", None)
        if isinstance(gcp_inline_config, dict):
            return gcp_inline_config
        if self.gcp_config_path:
            return load_bw_targets_config(self.gcp_config_path)
        return load_bw_targets_config()

    @staticmethod
    def _yolo_bbox_to_xyxy(
        bbox_yolo: tuple[int, float, float, float, float],
        image_shape: tuple[int, ...],
    ) -> list[float]:
        _, x_center, y_center, width_norm, height_norm = bbox_yolo
        image_height, image_width = image_shape[:2]

        width = float(width_norm) * image_width
        height = float(height_norm) * image_height
        center_x = float(x_center) * image_width
        center_y = float(y_center) * image_height

        x1 = max(0.0, center_x - (width / 2.0))
        y1 = max(0.0, center_y - (height / 2.0))
        x2 = min(float(image_width), center_x + (width / 2.0))
        y2 = min(float(image_height), center_y + (height / 2.0))
        return [x1, y1, x2, y2]

    @staticmethod
    def _ensure_rgb_image(image: np.ndarray, pixel_format: str = "") -> np.ndarray:
        if image.ndim == 3:
            return image
        if image.ndim != 2:
            raise ValueError(f"Unsupported frame shape for GCP stage: {image.shape}")

        if "bayerrg" in pixel_format.lower() or not pixel_format:
            return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)

    def _run_gcp_detector(self, image: np.ndarray) -> list[dict[str, Any]]:
        result = find_bw_targets_in_image(
            image=image,
            config=self.gcp_config,
            output_dir=None,
            debug=bool(getattr(self, "gcp_debug", False)),
        )
        if result is None:
            return []

        boxes = []
        for box in result.get("boxes", []):
            bbox_yolo = box.get("bbox_yolo")
            if bbox_yolo is None or len(bbox_yolo) != 5:
                continue
            boxes.append(
                {
                    "class": self.gcp_class_id,
                    "conf": float(box.get("score", 1.0)),
                    "bbox": self._yolo_bbox_to_xyxy(bbox_yolo, image.shape),
                }
            )
        return boxes

    def _draw_overlay(self, image_rgb: np.ndarray, boxes: list[dict]) -> np.ndarray:
        overlay = image_rgb.copy()
        for box in boxes:
            x1, y1, x2, y2 = [int(round(coord)) for coord in box["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                overlay,
                f"{box['conf']:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return overlay

    # TODO: _save_positive_artifacts va rifatta per eventuale modalità di debug e visualizzazione
    #  Da guardare in una prossima iterazione.
    def _save_positive_artifacts(
        self,
        metadata: PhotoExif,
        image_rgb: np.ndarray,
        boxes: list[dict],
    ) -> None:
        review_root = Path(metadata.filename).resolve().parent / self.review_dir_name
        stem = Path(metadata.filename).stem

        images_dir = review_root / "images"
        overlays_dir = review_root / "overlays"
        metadata_dir = review_root / "metadata"

        if self.save_positive_images:
            images_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(images_dir / f"{stem}.png"),
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            )

        if self.save_positive_overlays:
            overlays_dir.mkdir(parents=True, exist_ok=True)
            overlay = self._draw_overlay(image_rgb, boxes)
            cv2.imwrite(
                str(overlays_dir / f"{stem}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        if self.save_positive_metadata:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "image": {
                    "source_path": metadata.filename,
                    "review_image_path": str(images_dir / f"{stem}.png"),
                    "overlay_path": str(overlays_dir / f"{stem}.png"),
                    "timestamp": metadata.timestamp,
                    "pixel_format": metadata.pixel_format,
                    "full_image_shape": getattr(metadata, "full_image_shape", None),
                },
                "telemetry": {
                    "lat": getattr(metadata.pix_data, "lat", None),
                    "lon": getattr(metadata.pix_data, "lon", None),
                    "alt_rel": getattr(metadata.pix_data, "alt_rel", None),
                    "roll": getattr(metadata.pix_data, "roll", None),
                    "pitch": getattr(metadata.pix_data, "pitch", None),
                    "yaw": getattr(metadata.pix_data, "yaw", None),
                },
                "detections": boxes,
            }
            with (metadata_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")

    def mission(self) -> bool:
        while not self.stopped.is_set():
            while not self.stopped.is_set():
                self.check_inbound_command()
                try:
                    metadata: Any = self.receive_from_queue(timeout=1)
                except Empty:
                    time.sleep(0.01)
                    continue
                except Exception as e:
                    self.logger.error(f"Error reading from queue: {e}")
                    time.sleep(0.01)
                    continue

                if metadata is None:
                    time.sleep(0.01)
                    continue

                if not isinstance(metadata, PhotoExif):
                    self.send_to_queues(metadata)
                    continue

                try:
                    if not self._cv_gcp_enabled():
                        self.fast_ack(metadata)
                        continue

                    shm_name = metadata.shm_name
                    try:
                        with self.consume_shm_array(metadata) as in_array:
                            image_rgb = self._ensure_rgb_image(
                                in_array, getattr(metadata, "pixel_format", "")
                            )
                            image_shape = image_rgb.shape

                            m_bboxes = self._run_gcp_detector(image_rgb)
                            full_image_shape = (
                                getattr(metadata, "full_image_shape", None)
                                or image_shape
                            )
                            full_image_bboxes = bbox_utils.scale_bboxes_to_full_image(
                                m_bboxes,
                                image_shape,
                                full_image_shape,
                            )
                            metadata.bboxes = full_image_bboxes
                            bbox_utils.save_full_image_bboxes(
                                metadata,
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

                            if full_image_bboxes:
                                self._save_positive_artifacts(
                                    metadata, image_rgb, full_image_bboxes
                                )
                        metadata.shm_name = None
                    except FileNotFoundError:
                        self.logger.warning(
                            f"CVGCPInferenceStage: shared memory {shm_name} not found"
                        )
                        continue

                    self.send_to_queues(metadata)
                    self.logger.debug("Sent GCP CV data to decision system")
                except Exception as e:
                    self.logger.error(f"CVGCPInferenceStage error: {e}")
                    self.fast_ack(metadata)

            if not self.stopped.is_set():
                time.sleep(0.05)

        return True

    def update(self) -> None:
        super().update()
        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        self.gcp_class_id = int(inference_cfg.get("cv_gcp_class_id", self.gcp_class_id))
        self.gcp_config_path = inference_cfg.get(
            "bw_targets_config", self.gcp_config_path
        )
        self.gcp_config = self._load_gcp_config()

    def stop(self) -> None:
        self.logger.info("Stopping CVGCPInferenceStage")
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
