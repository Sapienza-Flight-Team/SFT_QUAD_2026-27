import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif
from stages.inference.default.utils import bbox_utils
from stages.inference.default.utils.dataset_utils import LABEL_NAMES, clamp_xyxy, expand_xyxy


@dataclass(frozen=True)
class DigitPrediction:
    label: str
    confidence: float
    discard: bool


class CropDataset(Dataset):
    def __init__(self, crops: Sequence[np.ndarray], *, transform: Any) -> None:
        self.crops = [np.asarray(crop, dtype=np.uint8) for crop in crops]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int) -> torch.Tensor:
        crop = self.crops[index]
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise RuntimeError(f"Expected RGB crop with shape HxWx3, got {crop.shape}")
        return self.transform(Image.fromarray(crop, mode="RGB"))


def build_cnn_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_digit_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_digit_cnn_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
) -> tuple[nn.Module, list[str], int]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    label_names = payload.get("label_names", list(LABEL_NAMES))
    if list(label_names) != list(LABEL_NAMES):
        raise ValueError(
            "Checkpoint label_names do not match the expected digit CNN label order. "
            f"Expected {list(LABEL_NAMES)}, got {list(label_names)}."
        )

    imgsz = int(payload.get("imgsz", 128))
    model = build_digit_model(num_classes=len(label_names)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, list(label_names), imgsz


def predict_digit_crops(
    *,
    checkpoint_path: Path | str,
    crops: Sequence[np.ndarray],
    imgsz: int | None = None,
    batch: int = 64,
    device: str = "auto",
    workers: int = 0,
    amp_enabled: bool = True,
) -> list[DigitPrediction]:
    if not crops:
        return []

    device_obj = resolve_device(device)
    model, label_names, checkpoint_imgsz = load_digit_cnn_checkpoint(
        checkpoint_path,
        device=device_obj,
    )
    effective_imgsz = int(imgsz) if imgsz is not None else checkpoint_imgsz
    loader = DataLoader(
        CropDataset(crops, transform=build_cnn_transforms(effective_imgsz)),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    predictions: list[DigitPrediction] = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device_obj, non_blocking=True)
            with torch.amp.autocast(
                device_type=device_obj.type,
                enabled=amp_enabled and device_obj.type == "cuda",
            ):
                probabilities = torch.softmax(model(images), dim=1)

            confidences, predicted_indices = probabilities.max(dim=1)
            for confidence, predicted_index in zip(confidences, predicted_indices):
                label = label_names[int(predicted_index.item())]
                predictions.append(
                    DigitPrediction(
                        label=label,
                        confidence=float(confidence.item()),
                        discard=label == "background",
                    )
                )

    return predictions


class CNNInferenceStage(PipelineStage):
    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        ack_q=None,
        stage_name="cnn_inference_engine",
        idx=6,
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
        self._load_settings()
        self.logger.info(
            f"CNNInferenceStage initialized: checkpoint={self.cnn_checkpoint_path}, "
            f"device={self.cnn_device}, batch={self.cnn_batch}"
        )

    def _load_settings(self) -> None:
        inference_cfg = self.configs.config_dict.get("inference_engine", {})
        cnn_cfg = self.configs.config_dict.get("cnn_inference", {})
        self.no_number_class_id = int(inference_cfg.get("cv_gcp_class_id", 10))
        self.crop_margin_px = float(inference_cfg.get("cnn_crop_margin_px", 0.0))
        self.crop_margin_ratio = float(inference_cfg.get("cnn_crop_margin_ratio", 0.0))
        self.cnn_checkpoint_path = inference_cfg.get(
            "digit_cnn_checkpoint",
            cnn_cfg.get("weights", "models/digit_cnn/best.pt"),
        )
        self.cnn_imgsz = inference_cfg.get("digit_cnn_imgsz")
        self.cnn_batch = int(inference_cfg.get("digit_cnn_batch", 32))
        self.cnn_device = str(inference_cfg.get("digit_cnn_device", "auto"))
        self.cnn_workers = int(inference_cfg.get("digit_cnn_workers", 0))
        self.cnn_amp_enabled = bool(inference_cfg.get("digit_cnn_amp", True))
        self.target_resolution = self.configs.config_dict.get("preprocessor", {}).get(
            "target_resolution", [2656, 1504]
        )

    def _cnn_enabled(self) -> bool:
        return self.current_task == "NUMBERS" or self.current_task == "PERIMETER"

    @staticmethod
    def _ensure_rgb_image(image: np.ndarray, pixel_format: str = "") -> np.ndarray:
        if image.ndim == 3:
            return image.astype(np.uint8)
        if image.ndim != 2:
            raise ValueError(f"Unsupported frame shape for CNN stage: {image.shape}")

        if "bayerrg" in pixel_format.lower() or not pixel_format:
            return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)

    def _load_rgb_image_for_metadata(self, metadata: PhotoExif) -> np.ndarray:
        with self.consume_shm_array(metadata) as in_array:
            image_rgb = self._ensure_rgb_image(
                in_array, getattr(metadata, "pixel_format", "")
            )
            ret = np.array(image_rgb, copy=True)

        metadata.shm_name = None
        return ret

    def _crop_from_bbox(
        self, image_rgb: np.ndarray, bbox: list[float] | tuple[float, ...]
    ) -> np.ndarray:
        image_height, image_width = image_rgb.shape[:2]
        expanded_box = expand_xyxy(
            tuple(float(value) for value in bbox),
            self.crop_margin_px,
            self.crop_margin_ratio,
        )
        left, top, right, bottom = clamp_xyxy(expanded_box, image_width, image_height)
        return image_rgb[top:bottom, left:right].copy()

    def _classify_bboxes(
        self, image_rgb: np.ndarray, bboxes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not bboxes:
            return []

        crops = [self._crop_from_bbox(image_rgb, bbox["bbox"]) for bbox in bboxes]
        predictions = predict_digit_crops(
            checkpoint_path=self.cnn_checkpoint_path,
            crops=crops,
            imgsz=self.cnn_imgsz,
            batch=self.cnn_batch,
            device=self.cnn_device,
            workers=self.cnn_workers,
            amp_enabled=self.cnn_amp_enabled,
        )

        refined: list[dict[str, Any]] = []
        for bbox, prediction in zip(bboxes, predictions):
            refined_bbox = self._bbox_with_prediction(bbox, prediction)
            if refined_bbox is not None:
                refined.append(refined_bbox)

        return refined

    def _bbox_with_prediction(
        self,
        bbox: dict[str, Any],
        prediction: DigitPrediction,
    ) -> dict[str, Any] | None:
        if prediction.discard:
            return None

        predicted_label = prediction.label
        if not predicted_label.isdigit() and predicted_label != "square_without_number":
            return None

        refined_bbox = dict(bbox)
        refined_bbox["cnn_label"] = predicted_label
        refined_bbox["cnn_conf"] = prediction.confidence

        if predicted_label.isdigit():
            refined_bbox["class"] = int(predicted_label)
        else:
            refined_bbox["class"] = self.no_number_class_id
        return refined_bbox

    def _should_process(self, data: Any) -> bool:
        return (
            isinstance(data, PhotoExif)
            and bool(getattr(data, "bboxes", []))
            and self._cnn_enabled()
        )

    def _process_metadata(self, metadata: PhotoExif) -> None:
        image_rgb = self._load_rgb_image_for_metadata(metadata)
        inference_shape = image_rgb.shape
        refined_bboxes = self._classify_bboxes(image_rgb, list(metadata.bboxes))
        full_image_shape = (
            getattr(metadata, "full_image_shape", None) or inference_shape
        )
        metadata.bboxes = bbox_utils.scale_bboxes_to_full_image(
            refined_bboxes,
            inference_shape,
            full_image_shape,
        )
        bbox_utils.save_full_image_bboxes(
            metadata,
            metadata.bboxes,
            inference_shape,
            full_image_shape,
            self.target_resolution,
            producer=self._stage_name,
            mission=self.current_task,
            bbox_logging_config=self.configs.config_dict.get("bbox_logging", {}),
            logger=self.logger,
        )
        metadata.shm_name = None

    def mission(self) -> bool:
        while not self.stopped.is_set():
            self.check_inbound_command()
            try:
                metadata = self.receive_from_queue(timeout=1)
            except Empty:
                continue

            if not isinstance(metadata, PhotoExif):
                self.send_to_queues(metadata)
                continue

            image_name = Path(metadata.filename).name
            input_bbox_count = len(getattr(metadata, "bboxes", []) or [])
            self.logger.info(
                f"CNN received image {image_name}: task={self.current_task}, "
                f"input_bboxes={input_bbox_count}"
            )

            if not self._should_process(metadata):
                reasons = []
                if not self._cnn_enabled():
                    reasons.append(f"task={self.current_task} is not enabled")
                if not getattr(metadata, "bboxes", []):
                    reasons.append("no input bboxes")
                reason_text = "; ".join(reasons) if reasons else "not processable"
                self.logger.info(
                    f"CNN skipped image {image_name}: {reason_text}; forwarded=True"
                )
                # ACK without opening shm — the GC handles cleanup
                self.fast_ack(metadata)
                metadata.shm_name = None
                self.send_to_queues(metadata)
                continue

            try:
                start_time = time.perf_counter()
                # _process_metadata opens shm internally via consume_shm_array
                self._process_metadata(metadata)
                output_bbox_count = len(getattr(metadata, "bboxes", []) or [])
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.logger.info(
                    f"CNN analyzed image {image_name}: "
                    f"input_bboxes={input_bbox_count}, "
                    f"output_bboxes={output_bbox_count}, "
                    f"elapsed_ms={elapsed_ms:.1f}, forwarded_to_decision=True"
                )
            except Exception as e:
                self.logger.error(f"CNNInferenceStage error on {image_name}: {e}")
                inference_shape = getattr(metadata, "shm_shape", None)
                full_image_shape = (
                    getattr(metadata, "full_image_shape", None) or inference_shape
                )
                metadata.bboxes = bbox_utils.scale_bboxes_to_full_image(
                    list(getattr(metadata, "bboxes", []) or []),
                    inference_shape,
                    full_image_shape,
                )
                self.fast_ack(metadata)

            metadata.shm_name = None
            self.send_to_queues(metadata)

        return True

    def update(self) -> None:
        super().update()
        self._load_settings()

    def stop(self) -> None:
        self.logger.info("Stopping CNNInferenceStage")
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
