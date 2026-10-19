import ctypes as ct
from logging import getLogger
from time import monotonic
from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
import pytest
import yaml

from common.config import MISSION_CONFIG_PATH
from common.new_structs import IMAGE_SIZE, PhotoExif, PixData
from core import pipeline as pipeline_module


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "SFTLogger", lambda config: Mock())
    pipeline = pipeline_module.Pipeline(
        yaml.safe_load(MISSION_CONFIG_PATH.read_text()), {"logger": {}}
    )
    pipeline.build()
    try:
        yield pipeline
    finally:
        pipeline.close_registry()


def fake_camera(camera):
    camera.cam = Mock()
    for feature, value in {
        "ExposureTime": 350.5,
        "Gain": 4.5,
        "Gamma": 0.75,
        "BlackLevel": 1.5,
        "BslColorSpace": "sRGB",
        "BslLightSourcePreset": "Daylight5000K",
        "TriggerMode": "On",
        "TriggerSource": "Software",
        "ChunkModeActive": True,
        "ChunkSelector": "CounterValue",
        "CounterSelector": "Counter2",
    }.items():
        getattr(camera.cam, feature).Value = value
    camera.next_frame_at, camera.capture_period = monotonic(), 0
    result = MagicMock()
    result.__enter__.return_value = result
    result.GrabSucceeded.return_value = True
    type(result).Array = PropertyMock(
        side_effect=AssertionError("Image must be grabbed directly into the lease")
    )
    camera.cam.RetrieveResult.return_value = result
    factory, address = None, None

    def bind(value):
        nonlocal factory
        if value is None and factory is not None:
            factory.FreeBuffer(address, None)
            factory.OnReleased()
        factory = value

    def start(*args):
        nonlocal address
        address = factory.AllocateBuffer(IMAGE_SIZE + 64)

    def retrieve(*args):
        ct.memset(address, 123, IMAGE_SIZE)
        ct.memset(address + IMAGE_SIZE, 204, 64)  # Chunk trailer follows pixels.
        return camera.cam.RetrieveResult.return_value

    camera.cam.SetBufferFactory.side_effect = bind
    camera.cam.StartGrabbingMax.side_effect = start
    camera.cam.RetrieveResult.side_effect = retrieve
    return result


def test_received_hardware_counter_matches_image_and_sensors(pipeline):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    for counter in (17, 21):
        result.ChunkCounterValue.Value = counter
        assert camera.mission() is False
        assert sensor.mission() is False
        assert sensor.capture_id == counter
        with pipeline.registry.lease_slot_for_reading(
            {PhotoExif, PixData}, "viewer"
        ) as buffers:
            photo = np.ndarray((), np.dtype(PhotoExif), buffer=buffers[PhotoExif])
            pix = np.ndarray((), np.dtype(PixData), buffer=buffers[PixData])
            assert photo["frame_counter"] == pix["camera_trigger_seq"] == counter
            assert photo["gamma"] == 0.75 and photo["black_level"] == 1.5
            assert photo["image"][0] == photo["image"][-1] == 123
            assert photo["_camera_chunks"][0] == 204
            del photo, pix
    assert camera.cam.ExecuteSoftwareTrigger.call_count == 2
    assert result.ChunkCounterSelector.Value == "Counter2"


@pytest.mark.parametrize(
    "mode, source, triggered",
    [("On", "Software", True), ("Off", "Software", False), ("On", "Line1", False)],
)
def test_trigger_decision_uses_open_camera_nodes(pipeline, mode, source, triggered):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    result.ChunkCounterValue.Value = 1
    camera.cam.TriggerMode.Value, camera.cam.TriggerSource.Value = mode, source
    camera.mission()
    assert camera.cam.ExecuteSoftwareTrigger.called is triggered


def test_failed_grab_does_not_notify_sensors(pipeline):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    result.GrabSucceeded.return_value = False
    assert camera.mission() is False
    assert not sensor.capture_connection.poll()
    assert pipeline.registry.entries == {}
    camera.cam.RetrieveResult.return_value = None
    assert camera.mission() is False
    assert not sensor.capture_connection.poll()


def test_external_clock_wait_and_eof_are_interruptible(pipeline):
    _, camera, sensor, _ = pipeline.processes
    assert sensor.mission() is False
    assert sensor.packets == 0 and pipeline.registry.entries == {}
    camera.capture_connection.close()
    assert sensor.mission() is True
    sensor.stop()
    assert sensor.mission() is True


def test_sensor_process_receives_camera_notifications_without_acknowledgements(
    pipeline,
):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    sensor.logger = getLogger("test.camera_clock")
    sensor.start()
    sensor.capture_connection.close()  # Parent copy; the child owns its endpoint.
    try:
        result.ChunkCounterValue.Value = 37
        camera.mission()
        required = frozenset({PhotoExif, PixData})
        assert pipeline.registry.events[required]["viewer"].wait(5)
        assert pipeline.registry.latest[required] == 37
        assert sensor.is_alive()
    finally:
        sensor.stop()
        sensor.join(3)
        if sensor.is_alive():
            sensor.terminate()
            sensor.join()
    assert sensor.exitcode == 0


@pytest.mark.parametrize(
    "change, message",
    [
        ({"trigger_source": "Line1"}, "trigger_source=Software"),
        ({"trigger_mode": "Off"}, "trigger_mode=On"),
        ({"chunk_selector": "FrameID"}, "CounterValue"),
    ],
)
def test_external_clock_rejects_incompatible_camera_settings(pipeline, change, message):
    mission = pipeline._mission
    next(
        entry["cam"][0]["camera_engine"] for entry in mission if "cam" in entry
    ).append(change)
    pipeline.close_registry()
    with pytest.raises(ValueError, match=message):
        pipeline.build()
    assert pipeline.registry is None


def test_internal_clock_still_advances_without_a_pipe(pipeline):
    _, camera, sensor, _ = pipeline.processes
    sensor.configs.pop("external_clock")  # Older YAML uses Options' False default.
    sensor.configs.update(generation_rate_hz=10000)
    sensor.update()
    sensor.capture_id = 0
    assert sensor.mission() is False
    assert sensor.mission() is False
    assert sensor.capture_id == sensor.packets == 2
