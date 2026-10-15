# CNN Inference Stage

`CNNInferenceStage` is the refinement stage that runs after YOLO.

## Purpose

The stage receives a `PhotoExif` whose `bboxes` were already produced by YOLO.
For each image:

1. load the image from `metadata.shm_name`
2. crop every YOLO bbox
3. run the digit CNN on those crops
4. rewrite the bbox classes and scale them to full-image coordinates before sending the frame downstream

This stage is intentionally image-by-image and crop-by-crop. It does not run object detection on the full image.

## Input Contract

- input object: `PhotoExif`
- required field: `metadata.bboxes`
- bbox format: `xyxy` in the coordinate space of the shared-memory image
- source image: loaded from `metadata.shm_name`
- cleanup: this stage owns and unlinks the received shared-memory image, including skipped frames

When chained after YOLO, `InferenceStage` passes inference-resolution boxes with an inference-resolution image. `CNNInferenceStage` scales the refined boxes to `metadata.full_image_shape` before forwarding to the decision system.

## Class Mapping

CNN output is mapped as follows:

- digit `"0"` … `"9"` -> classes `0` … `9`
- `square_without_number` -> class `10`
- `background` -> detection dropped

The refined bbox keeps the original geometry and adds:

- `cnn_label`
- `cnn_conf`

## Recommended Model Location

Put the checkpoint here:

`models/digit_cnn/best.pt`

That is also the default path expected by the stage if no explicit config is provided.

If you want a different location, set:

```yaml
inference_engine:
  digit_cnn_checkpoint: /absolute/or/relative/path/to/best.pt
```

## Optional Runtime Config

```yaml
inference_engine:
  digit_cnn_checkpoint: models/digit_cnn/best.pt
  digit_cnn_imgsz: 128
  digit_cnn_batch: 32
  digit_cnn_device: auto
  digit_cnn_workers: 0
  digit_cnn_amp: true
  cnn_crop_margin_px: 0.0
  cnn_crop_margin_ratio: 0.0
```

## Intended Placement In Pipeline

The intended chain is:

`inference_engine -> cnn_inference_engine -> decision_system`

For the `NUMBERS` mission this can coexist with a parallel `cv_gcp_engine` branch.
