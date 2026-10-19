from core.options import Options

# Basler ace 2 / a2A5320 defaults. Tuples become GUI dropdowns, ranges become
# integer sliders, and scalar Options remain editable (including float values).
# Feature availability, numeric limits, and increments depend on the exact model
# and ROI; pylon validates them against the opened camera during setup.
CONFIG = {
    # [width, height]. Restart the pipeline after changing the image layout.
    "resolution": [5320, 3032],
    "channels": 1,
    # The shared-memory image layout currently stores one uint8 per pixel.
    "pixel_format": Options(("BayerRG8", "Mono8"), default=0),
    "offset_x": Options(range(0, 5320), default=0),
    "offset_y": Options(range(0, 3032), default=0),
    "reverse_x": Options((False, True), default=0),
    "reverse_y": Options((False, True), default=0),
    "start_delay": Options(
        range(0, 300), 0
    ),  # delay before starting the camera in seconds
    "acquisition_time": Options(range(0, 60 * 30), 300),  # acquisition time in seconds
    "acquisition_frame_rate_enable": Options((False, True), default=0),
    "acquisition_frame_rate": Options(5.0, default=5.0),  # Hz; used when enabled
    "exposure_mode": Options(("Timed", "TriggerWidth"), default=0),
    "exposure_auto": Options(("Off", "Once", "Continuous"), default=2),
    "exposure_time": Options(1000.0, default=1000.0),  # us; Timed + auto Off
    "auto_gain": Options(("Off", "Once", "Continuous"), default=2),
    "gain": Options(0.0, default=0.0),  # dB; used when auto_gain is Off
    "profile": {
        "min_exposure_time": Options(range(1, 1000001), default=100),  # us
        "max_exposure_time": Options(range(1, 1000001), default=2000),  # us
        "min_auto_gain": Options(0.0, default=0.0),  # dB
        "max_auto_gain": Options(5.0, default=5.0),  # dB
        "auto_target_brightness": Options(0.2, default=0.2),  # normalized [0, 1]
    },
    "auto_function_profile": Options(
        ("MinimizeGain", "MinimizeExposureTime"), default=0
    ),
    "gamma": Options(1.0, default=1.0),
    "black_level": Options(0.0, default=0.0),
    "bsl_color_space": Options(("sRGB", "RGB"), default=0),
    "balance_white_auto": Options(("Off", "Once", "Continuous"), default=2),
    "light_source_preset": Options(
        ("Off", "Daylight5000K", "Daylight6500K", "Tungsten2800K"), default=1
    ),
    "trigger_selector": Options(("FrameStart", "FrameBurstStart"), default=0),
    "trigger_mode": Options(("Off", "On"), default=1),
    "trigger_source": Options(
        ("Line1", "Line2", "Line3", "Line4", "Software"), default=0
    ),
    "trigger_activation": Options(("RisingEdge", "FallingEdge", "AnyEdge"), default=0),
    "trigger_delay": Options(0.0, default=0.0),  # us
    "line_selector": Options(("Line1", "Line2", "Line3", "Line4"), default=1),
    "line_mode": Options(("Input", "Output"), default=1),
    "line_source": Options(("ExposureActive", "FlashWindow", "UserOutput1"), default=0),
    "line_inverter": Options((False, True), default=0),
    "max_num_buffer": Options(range(1, 101), default=10),
    "output_queue_size": Options(range(1, 101), default=10),
    "chunk_mode_active": Options((False, True), default=1),
    "chunk_selector": Options(
        ("CounterValue", "FrameID", "Timestamp", "ExposureTime", "Gain"), default=0
    ),
    "chunk_counter_selector": Options(("Counter1", "Counter2"), default=1),
    "chunk_enable": Options((False, True), default=1),
}
