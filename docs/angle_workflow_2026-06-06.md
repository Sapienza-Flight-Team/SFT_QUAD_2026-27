# Angle Workflow - 2026-06-06

This is the yaw-only angle path after the cleanup. The rule is:

```text
External telemetry may arrive in degrees.
PixData stores roll, pitch, yaw in degrees.
DecisionSystem passes those degrees through.
Geoloc tells Camera.Configuration that those values are degrees.
Homography treats roll, pitch, yaw as degrees and never guesses from magnitude.
```

## 1. Telemetry Ingestion

Source: `src/sft_cv/communicator/mavsdk_communicator.py`

MAVSDK provides attitude as:

```text
att.roll_deg
att.pitch_deg
att.yaw_deg
```

The communicator stores them directly:

```text
PixData.roll = att.roll_deg
PixData.pitch = att.pitch_deg
PixData.yaw = att.yaw_deg
```

There is no alternate yaw field in `PixData`.

## 2. Camera Metadata

Source: `src/sft_cv/common/mavlink_structs.py`, `src/sft_cv/cam/camera_engine.py`

`PixData` stores:

```text
roll: degrees
pitch: degrees
yaw: degrees
```

`CameraEngine` writes image metadata with `roll`, `pitch`, and `yaw` only.

## 3. Preprocess Stages

Source: `src/sft_cv/preprocess/converter.py`, `src/sft_cv/preprocess/resizer.py`

Converter and resizer stages do not modify attitude. They pass the same `PhotoExif.pix_data` object forward with angles still in degrees.

## 4. Bullseye Homography Path

Source: `src/sft_cv/inference/bullseye_inference_stage.py`, `src/sft_cv/common/homography.py`

`BullseyeInferenceStage` copies only these angle fields into its worker metadata row:

```text
roll
pitch
yaw
```

`common.homography.metadata_angles()` requires yaw and returns degree values. The homography path calls `Camera.Configuration(..., is_deg=True)`.

There is no unit config and no `abs(angle) > 2*pi` heuristic.

## 5. YOLO Path

Source: `src/sft_cv/inference/inference_model.py`

YOLO attaches bounding boxes to the same image metadata. It does not alter roll, pitch, or yaw.

## 6. Decision Input Formatting

Source: `src/sft_cv/decision_system/decision_system.py`

`DecisionSystem.input_format()` requires degree values:

```text
pix_data.roll
pix_data.pitch
pix_data.yaw
```

It passes them through:

```text
roll_deg = pix_data.roll
pitch_deg = pix_data.pitch
yaw_deg = pix_data.yaw
```

The returned attitude tuple is:

```text
(roll_deg, pitch_deg, yaw_deg)
```

No fallback yaw source exists.

## 7. Geolocation

Source: `src/sft_cv/decision_system/geoloc/geoloc.py`

`Geoloc.__call__()` receives degrees from `DecisionSystem`. It adds configured Monte Carlo errors in degrees, then calls:

```text
Camera.Configuration(..., is_deg=True)
```

`Camera.Configuration` performs the final internal conversion to radians for camera math.

## Unit Contract

| Boundary | Unit |
|---|---|
| MAVSDK attitude input | degrees |
| `PixData.roll/pitch/yaw` | degrees |
| saved camera metadata | degrees |
| homography live path | degrees |
| `DecisionSystem.input_format()` output attitude tuple | degrees |
| `Geoloc.__call__()` input attitude | degrees |
| `Camera.Configuration` internal storage | radians |

## Regression Tests

The angle contract is covered by `tests/test_angle_units.py`:

- Decision uses yaw and passes degree values through.
- Homography preserves large degree values instead of guessing units.
- Homography has no angle-unit config.
- Homography raises if yaw is missing.
