import logging

from pypylon import genicam


def set_enum(nodemap, name, value, *, optional=False):
    """Use device enum symbols, with useful errors for unsupported settings."""
    node = nodemap.GetNode(name)
    if node is None or not genicam.IsWritable(node):
        if optional:
            logging.getLogger(__name__).warning(
                "Skipping unavailable/non-writable %s", name
            )
            return
        raise RuntimeError(
            f"Required camera feature {name} is unavailable/non-writable"
        )
    symbols = [
        symbol
        for symbol in node.GetSymbolics()
        if genicam.IsAvailable(node.GetEntryByName(symbol))
    ]
    selected = next(
        (
            symbol
            for symbol in symbols
            if isinstance(value, str) and symbol.casefold() == value.casefold()
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"{name}={value!r} is invalid; available values: {', '.join(symbols)}. "
            'Quote "On" and "Off" in YAML to prevent boolean conversion.'
        )
    node.SetValue(selected)


def reset_frame_counter(cam, counter):
    """Reset and re-arm an ace 2 counter without closing/rebooting the camera.

    The caller must pause triggers, drain old results, and coordinate a fresh
    registry/PX4 capture session before IDs restart.
    """
    nodemap = cam.GetNodeMap()
    set_enum(nodemap, "CounterSelector", counter)
    cam.CounterReset.Execute()
    set_enum(nodemap, "SoftwareSignalSelector", "SoftwareSignal1")
    cam.SoftwareSignalPulse.Execute()


def setup_camera(cam, config: dict) -> None:
    """Apply Pipeline's camera settings to an open camera before grabbing."""
    profile = config["profile"]
    nodemap = cam.GetNodeMap()

    # Clear the previous crop before changing the image dimensions.
    cam.PixelFormat.Value = config["pixel_format"]
    cam.OffsetX.Value = 0
    cam.OffsetY.Value = 0
    cam.Width.Value, cam.Height.Value = config["resolution"]
    cam.OffsetX.Value = config["offset_x"]
    cam.OffsetY.Value = config["offset_y"]
    cam.ReverseX.Value = config["reverse_x"]
    cam.ReverseY.Value = config["reverse_y"]
    cam.AcquisitionMode.Value = "Continuous"

    cam.TriggerSelector.Value = config["trigger_selector"]
    cam.TriggerMode.Value = "Off"
    cam.ExposureMode.Value = "Timed"
    cam.ExposureAuto.Value = "Off"
    cam.GainAuto.Value = "Off"
    cam.ExposureMode.Value = config["exposure_mode"]

    if config["exposure_mode"] == "Timed":
        if config["exposure_auto"] == "Off":
            cam.ExposureTime.Value = config["exposure_time"]
        else:
            # Reset the lower bound so an earlier profile cannot block the upper.
            cam.AutoExposureTimeLowerLimit.Value = cam.AutoExposureTimeLowerLimit.Min
            cam.AutoExposureTimeUpperLimit.Value = profile["max_exposure_time"]
            cam.AutoExposureTimeLowerLimit.Value = profile["min_exposure_time"]
        cam.ExposureAuto.Value = config["exposure_auto"]

    if config["auto_gain"] == "Off":
        cam.Gain.Value = config["gain"]
    else:
        cam.AutoGainLowerLimit.Value = cam.AutoGainLowerLimit.Min
        cam.AutoGainUpperLimit.Value = profile["max_auto_gain"]
        cam.AutoGainLowerLimit.Value = profile["min_auto_gain"]
    cam.GainAuto.Value = config["auto_gain"]
    if config["exposure_auto"] != "Off" or config["auto_gain"] != "Off":
        cam.AutoTargetBrightness.Value = profile["auto_target_brightness"]
        cam.AutoFunctionProfile.Value = config["auto_function_profile"]

    cam.BalanceWhiteAuto.Value = "Off"
    cam.BslLightSourcePreset.Value = config["light_source_preset"]
    cam.Gamma.Value = config["gamma"]
    cam.BlackLevel.Value = config["black_level"]
    if config["bsl_color_space"] is not None:
        set_enum(nodemap, "BslColorSpace", config["bsl_color_space"], optional=True)
    cam.BalanceWhiteAuto.Value = config["balance_white_auto"]

    cam.AcquisitionFrameRateEnable.Value = config["acquisition_frame_rate_enable"]
    if config["acquisition_frame_rate_enable"]:
        cam.AcquisitionFrameRate.Value = config["acquisition_frame_rate"]

    cam.LineSelector.Value = config["line_selector"]
    # LineMode may be fixed on dedicated input/output lines.
    if cam.LineMode.Value != config["line_mode"]:
        cam.LineMode.Value = config["line_mode"]
    if config["line_mode"] == "Output":
        cam.LineSource.Value = config["line_source"]
    cam.LineInverter.Value = config["line_inverter"]

    if config["trigger_mode"] == "On":
        cam.TriggerSource.Value = config["trigger_source"]
        if config["trigger_source"] != "Software":
            cam.TriggerActivation.Value = config["trigger_activation"]
        cam.TriggerDelay.Value = config["trigger_delay"]
    # Keep the queue within the buffer count while changing either value.
    cam.OutputQueueSize.Value = 1
    cam.MaxNumBuffer.Value = config["max_num_buffer"]
    cam.OutputQueueSize.Value = config["output_queue_size"]
    cam.ChunkModeActive.Value = config["chunk_mode_active"]
    if config["chunk_mode_active"]:
        set_enum(nodemap, "ChunkSelector", config["chunk_selector"])
        if config["chunk_enable"] and config["chunk_selector"] == "CounterValue":
            counter = config["chunk_counter_selector"]
            set_enum(nodemap, "CounterSelector", counter)
            set_enum(nodemap, "CounterEventSource", "ExposureStart")
            set_enum(nodemap, "CounterTriggerSource", "SoftwareSignal1")
            set_enum(nodemap, "CounterResetSource", "Off")
            cam.CounterDuration.Value = cam.CounterDuration.Max
            set_enum(nodemap, "ChunkCounterSelector", counter)
        cam.ChunkEnable.Value = config["chunk_enable"]
        if config["chunk_enable"] and config["chunk_selector"] == "CounterValue":
            reset_frame_counter(cam, counter)
    # Only enable acquisition triggers after the counter has been armed.
    cam.TriggerMode.Value = config["trigger_mode"]
