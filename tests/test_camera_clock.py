import ctypes as ct
from logging import getLogger
from time import monotonic
from unittest.mock import MagicMock, Mock, PropertyMock
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from common.config import MISSION_CONFIG_PATH
from common.new_structs import IMAGE_SIZE, PhotoExif, PixData
from core import pipeline as pipeline_module


@pytest.fixture
def pipeline(monkeypatch):
    from pypylon import genicam

    monkeypatch.setattr(genicam, "IsWritable", lambda node: node.writable)
    monkeypatch.setattr(genicam, "IsReadable", lambda node: node.readable)
    monkeypatch.setattr(genicam, "IsAvailable", lambda node: node.available)
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
        node = getattr(camera.cam, feature)
        node.Value, node.readable = value, True
        node.GetValue.side_effect = lambda node=node: node.Value
    camera.cam.GetNodeMap.return_value.GetNode.side_effect = lambda name: getattr(
        camera.cam, name
    )
    camera.next_frame_at, camera.capture_period = monotonic(), 0
    result = MagicMock()
    result.__enter__.return_value = result
    result.GrabSucceeded.return_value = True
    result.GetBlockID.side_effect = AssertionError("Use the resettable counter only")
    result.GetChunkDataNodeMap.return_value.GetNode.side_effect = lambda name: getattr(
        result, name
    )
    result.ChunkCounterValue.readable = True
    result.ChunkCounterValue.GetValue.side_effect = (
        lambda: result.ChunkCounterValue.Value
    )
    selector = result.ChunkCounterSelector
    selector.writable = True
    selector.GetSymbolics.return_value = ("Counter1", "Counter2")
    selector.GetEntryByName.return_value = SimpleNamespace(available=True)
    selector.SetValue.side_effect = lambda value: setattr(selector, "Value", value)
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


def test_counter_and_sensor_matching_remain_bounded_past_pool_capacity(pipeline):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    for counter in range(100, 200, 2):
        result.ChunkCounterValue.Value = counter
        camera.mission()
        sensor.mission()
        assert sensor.capture_id == counter
        required = frozenset({PhotoExif, PixData})
        assert pipeline.registry.latest[required] == counter
        with pipeline.registry.lease_slot_for_reading(required, "viewer") as buffers:
            for kind, view in buffers.items():
                assert (
                    pipeline.registry.data_array[kind].id_reader.unpack_from(
                        view, kind.ID.offset
                    )[0]
                    == counter
                )
        for kind, helper in pipeline.registry.data_array.items():
            retained = sum(
                kind in entry for entry in pipeline.registry.entries.values()
            )
            assert retained + len(helper.free_slots) == len(helper.slots)
    assert sensor.packets == 50


@pytest.mark.parametrize("first", [0, 5])
def test_first_counter_is_not_renumbered(pipeline, first):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    result.ChunkCounterValue.Value = first
    camera.mission()
    sensor.mission()
    assert sensor.capture_id == camera.last_frame_id == first


@pytest.mark.parametrize("bad", [None, 0, -1, 6])
def test_missing_repeated_or_decreasing_counter_reclaims_lease_without_notification(
    pipeline, bad
):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    result.ChunkCounterValue.Value = 6
    camera.mission()
    sensor.mission()
    helper = pipeline.registry.data_array[PhotoExif]
    before = len(helper.free_slots), dict(pipeline.registry.entries[6])
    result.ChunkCounterValue.Value = bad
    if bad is None:
        result.ChunkCounterValue.readable = False
    with pytest.raises(RuntimeError, match="ChunkCounterValue|counter did not advance"):
        camera.mission()
    assert not sensor.capture_connection.poll()
    assert (len(helper.free_slots), dict(pipeline.registry.entries[6])) == before
    assert list(pipeline.registry.entries) == [6]
    assert camera.last_frame_id == 6
    assert result.__exit__.call_count == 2
    camera.cam.SetBufferFactory.assert_called_with(None)


def test_missing_chunk_node_is_fatal_without_falling_back_to_block_id(pipeline):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    result.GetChunkDataNodeMap.return_value.GetNode.side_effect = lambda name: (
        result.ChunkCounterSelector if name == "ChunkCounterSelector" else None
    )
    with pytest.raises(RuntimeError, match="no readable ChunkCounterValue"):
        camera.mission()
    assert not pipeline.registry.entries
    assert not sensor.capture_connection.poll()
    result.GetBlockID.assert_not_called()


def test_unavailable_bayer_colorspace_does_not_prevent_capture(pipeline):
    _, camera, sensor, _ = pipeline.processes
    result = fake_camera(camera)
    result.ChunkCounterValue.Value = 1
    camera.cam.BslColorSpace.readable = False
    camera.cam.BslColorSpace.GetValue.side_effect = AssertionError(
        "Unavailable feature"
    )
    camera.mission()
    sensor.mission()
    with pipeline.registry.lease_slot_for_reading(
        {PhotoExif, PixData}, "viewer"
    ) as buffers:
        photo = np.ndarray((), np.dtype(PhotoExif), buffer=buffers[PhotoExif])
        assert photo["bsl_color_space"].tobytes().split(b"\0", 1)[0] == b""
        del photo


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
