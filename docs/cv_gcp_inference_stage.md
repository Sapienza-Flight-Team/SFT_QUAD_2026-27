# CVGCPInferenceStage

`CVGCPInferenceStage` is a standalone pipeline stage for classical-CV GCP detection.

## Scope

- No dependency on YOLO
- Input: `PhotoExif` + shared-memory frame
- Output: `PhotoExif.bboxes` in the standard pipeline format

## Input Contract

The stage expects:

- a `PhotoExif` object on `in_q`
- `shm_name`, `shm_shape`, `shm_dtype` set
- `pixel_format` set, typically `BayerRG8`

If the frame is still Bayer (2-D), the stage demosaicizes it internally using a BayerRG to RGB conversion.

## Output Contract

For every processed frame, the stage attaches:

```python
metadata.bboxes = [
  {
    "class": 10,
    "conf": <float>,
    "bbox": [x1, y1, x2, y2],
  },
  ...
]
```

Default class id is `10`, which matches the current `no_number_GCP` target configuration.

## Required Runtime Config

The stage requires one of:

- `inference_engine.bw_targets_config`
  - path to a YAML config for the GCP detector
- `inference_engine.cv_gcp_config`
  - inline config dictionary passed through the runtime config object

Optional keys:

- `inference_engine.cv_gcp_class_id`
  - default `10`
- `inference_engine.cv_gcp_review_dir`
  - default `cv_gcp_review`
- `inference_engine.cv_gcp_save_images`
  - default `true`
- `inference_engine.cv_gcp_save_overlays`
  - default `true`
- `inference_engine.cv_gcp_save_metadata`
  - default `true`
- `inference_engine.cv_gcp_debug`
  - default `false`

## Positive Detections Review Artifacts

When at least one GCP is found, the stage writes review artifacts next to the original camera output directory:

- `cv_gcp_review/images/<stem>.png`
- `cv_gcp_review/overlays/<stem>.png`
- `cv_gcp_review/metadata/<stem>.json`

The metadata JSON includes:

- original image path
- timestamp
- telemetry from `pix_data`
- full-image shape
- detected GCP bounding boxes

This is intended for manual number assignment on the detected GCP frames.

## Integration Notes

`pipeline.py` instantiates `CVGCPInferenceStage` directly on the full-resolution
converter branch. The stage is active for `NUMBERS` and `PERIMETER` missions and
consumes/releases its branch without emitting detections for other missions.
