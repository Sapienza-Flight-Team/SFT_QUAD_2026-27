# Basler a2A5320 configuration

Run the generator from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m tools.config_file_generation.gui
```

Select **cam → camera_engine**, adjust the fields, and save as
`mission_config.yaml`. Regenerate older mission files to include all camera
settings. The GUI converts `Options` into plain YAML values.

`Pipeline` loads that YAML and passes the selected camera's settings dictionary
to `CameraEngine` as `configs`. Inside the camera process, the stage creates and
opens `self.cam`, then calls:

```python
setup_camera(self.cam, self.configs)
```

`camera_setup.py` applies the supplied values and contains the counter reset
helper. Counter/chunk enumeration settings are checked against the opened
camera's available symbols. The caller owns opening, grabbing, and closing the
camera. Configuration errors propagate; unsupported optional color space is
skipped with a warning.

- Resolution is `[width, height]`; reduce it before adding offsets to a
  full-sensor image. Restart the pipeline after changing the image layout.
- `exposure_time` applies when exposure is `Timed` and `exposure_auto` is `Off`.
  Auto exposure uses the limits in `profile`. Exposure and trigger times are in
  microseconds.
- `gain` applies when `auto_gain` is `Off`; otherwise the profile's gain limits
  apply. Gain is in dB and target brightness is normalized from 0 to 1.
- `TriggerWidth` requires auto exposure off and an enabled hardware trigger.
  With `trigger_mode: 'On'` and `trigger_source: Software`, the stage issues
  software triggers after checking the opened camera's trigger nodes.
- Frame rate, in Hz, applies when `acquisition_frame_rate_enable` is true.
  `line_source` applies to output lines; chunk selection applies when chunk mode
  is enabled. `output_queue_size` must not exceed `max_num_buffer`.
- For a2A5320-23ucBAS with BayerRG8, leave `bsl_color_space: null`. This feature
  is only available with RGB8 / YCbCr422 on this model. Supported enum values
  use device spelling, so `sRgb` is normalized to `sRGB` when available.
- `channels`, `start_delay`, and `acquisition_time` are application settings,
  rather than camera properties.

To test without a Pixhawk, select `sensor_generator.external_clock: true` with
the software trigger settings above and an enabled `CounterValue` chunk. Select
the desired counter with `chunk_counter_selector` (default `Counter2`). Pipeline
connects camera → sensors with a one-way pipe; no readiness or acknowledgement
messages are exchanged. `sensor_generator.generation_rate_hz` sets the target
trigger rate, limited by camera readiness. Internal sensor timing remains the
default when `external_clock` is false or omitted.

All camera parameters, including counter configuration, reset, and startup, are
applied in `setup_camera()`. The selected counter counts `ExposureStart` events.
`CounterReset.Execute()` resets it and `SoftwareSignalPulse.Execute()` re-arms it.
Each successful frame supplies its own `ChunkCounterValue` snapshot to both
`PhotoExif.frame_counter` and `PixData.camera_trigger_seq`; gaps are preserved.
Camera capture requires the `CounterValue` chunk and rejects missing, repeated,
or decreasing IDs before publishing or notifying sensors.

`CameraEngine.reset_frame_counter()` can reset and re-arm the counter without
closing or rebooting the camera. Pause triggers, drain old results, and start a
fresh registry/PX4 capture session before IDs restart. It does not reset the
registry's high-water marks. See [capture matching](basler_capture.md).

The capture stage leases registry memory before starting a single-frame grab.
A pypylon buffer factory supplies the address of that lease's image field to
the SDK. Image pixels are acquired there directly; no NumPy image copy is made.
The structure reserves 4096 trailing bytes for chunks, and the factory rejects
larger payloads. Grabbing stops and SDK references are released before publishing
the lease. This adds start/stop overhead per frame; hardware throughput still
needs to be measured on the camera.

Basler references: [exposure auto](https://docs.baslerweb.com/exposure-auto),
[triggered acquisition](https://docs.baslerweb.com/triggered-image-acquisition),
[image ROI](https://docs.baslerweb.com/image-roi), and
[gamma/color space](https://docs.baslerweb.com/gamma).
