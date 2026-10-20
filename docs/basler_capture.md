# Basler a2A5320-23ucBAS capture and matching

The camera uses a software-resettable hardware counter. `chunk_counter_selector`
chooses `Counter1` or `Counter2`; the current mission selects `Counter2`.
`FrameCounter` is not a valid `CounterSelector` symbol for this model.

Setup configures:

| Feature | Value |
| --- | --- |
| CounterSelector | Selected Counter1 or Counter2 |
| CounterEventSource | ExposureStart |
| CounterTriggerSource | SoftwareSignal1 |
| CounterResetSource | Off |
| CounterDuration | Device maximum |
| ChunkModeActive | true |
| ChunkSelector | CounterValue |
| ChunkCounterSelector | Same selected counter |
| ChunkEnable | true |

Setup executes `CounterReset`, then selects `SoftwareSignal1` and executes
`SoftwareSignalPulse` to start/re-arm counting, before enabling acquisition
triggers. `CameraEngine.reset_frame_counter()` repeats the reset and re-arm
without closing or rebooting the camera.

Pause triggers, drain old grabs, and coordinate a fresh registry/PX4 capture
session before resetting IDs. This method does not reset registry high-water
marks or release reader-held data. Arbitrary mid-session ID resets are not
supported.

The stage reads `ChunkCounterValue` from each grab result after selecting its
`ChunkCounterSelector`. It never falls back to `ChunkFrameID`, BlockID, or the
live camera counter. A first zero or nonzero value is preserved; subsequent
values must increase. Gaps remain gaps. Missing/repeated/decreasing counters
raise an error and return the unpublished lease without notifying sensors.

The current `mission_config.yaml` already enables software triggers and the
existing `sensor_generator/default` external clock. The camera sends the exact
captured ID through its pipe only after publishing the image; the generator
puts that ID in `PixData`. This carries small notifications, not image payloads.
`generation_rate_hz` controls the requested software trigger rate. Slow telemetry
can still exceed the registry's matching window; the pipe does not pin images.
These packets are synthetic telemetry, not PX4 measurements.

The shared-memory buffer factory remains in use: pylon receives the leased
image address directly. The stage does not create an intermediate pixel array
or copy the image. Grab references are released before publication. The
emulator test checks that the SDK writes to the supplied address; physical
camera timing, counter nodes, and trigger behavior require hardware validation.

For BayerRG8, `bsl_color_space` is `null` in both the defaults and mission.
Basler exposes this feature only with RGB8 / YCbCr422 on this model. Explicitly
requested values are checked against available device enum symbols; case
differences such as `sRgb` are accepted. Unavailable/non-writable color space
is skipped, and unreadable color metadata is stored as an empty string.
Quote `"On"` and `"Off"` in YAML to prevent boolean conversion.

The registry discards duplicate `(datatype, capture ID)` publications, preserves
the first immutable payload, and frees the incoming slot. This fixes the
18-slot exhaustion caused by repeated ID zero, including while readers hold
the original payload.

Run the configured pipeline with
`PYTHONPATH=src .venv/bin/python -m core.pipeline`. See
[camera configuration](camera_configuration.md) for the full settings.

References:

- [Basler Counter](https://docs.baslerweb.com/counter)
- [Basler CounterEventSource](https://docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera_CounterEventSourceEnum)
- [Basler Data Chunks](https://docs.baslerweb.com/data-chunks#counter-value-chunk)
- [Basler Software Signal Pulse](https://docs.baslerweb.com/software-signal-pulse)
- [Basler Gamma / BslColorSpace](https://docs.baslerweb.com/gamma)
