import ctypes as ct
import struct
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import pytest
import yaml

from common.new_structs import DATA_TYPE_REGISTRY, PhotoExif, PixData
from common.pipeline_stage import PipelineStage
from core.central_registry import CentralRegistry
from core import central_registry as registry_module
from core import pipeline as pipeline_module
from core.stage_catalog import build_stage_catalog
from stages.image_generator.default.image_generator_stage import ImageGenerator
from stages.sensor_generator.default.sensor_generator_stage import SensorGenerator
from stages.dummy_consumer.default import dummy_consumer_stage
from stages.viewer.default.viewer_stage import Viewer

STRIPES = (PixData, PhotoExif)
PACKET = DATA_TYPE_REGISTRY["main_camera"]["data_type"]


def test_compact_registry_references_preserve_leases_across_live_growth():
    class Pix(ct.Structure):
        _fields_ = [("value", ct.c_double), ("ID", ct.c_uint64)]

    class Photo(ct.Structure):
        _fields_ = [("ID", ct.c_uint64), ("image", ct.c_uint8 * 4)]

    packet = frozenset({Pix, Photo})
    registry = CentralRegistry()
    registry.register_consumer("background", packet)
    registry.register_producer("controller", Pix)
    registry.register_producer("camera", Photo)
    stopped = Event()
    entered = [Event(), Event()]
    release = [Event(), Event()]
    received = Queue()
    client = registry.for_stage("background", stopped)

    def consume():
        try:
            for index in range(2):
                with client.lease_slot_for_reading(packet, "background") as buffers:
                    assert all(view.readonly for view in buffers.values())
                    received.put(struct.unpack_from("@Q", buffers[Pix], Pix.ID.offset)[0])
                    received.put(buffers[Photo][Photo.image.offset])
                    entered[index].set()
                    assert release[index].wait(3)
                    if index == 1:
                        stopped.set()
        finally:
            client.close()

    def publish(capture_id):
        for primitive in (Pix, Photo):
            with registry.lease_slot_for_writing(primitive) as view:
                struct.pack_into("@Q", view, primitive.ID.offset, capture_id)
                if primitive is Photo:
                    view[Photo.image.offset] = capture_id

    workers = [Thread(target=consume, daemon=True)]
    for worker in workers:
        worker.start()
    try:
        publish(0)
        assert entered[0].wait(3)
        assert [received.get_nowait(), received.get_nowait()] == [0, 0]
        locations = {
            kind: registry.data_array[kind].slot_location(registry.entries[0][kind])
            for kind in packet
        }
        publish(1)
        registry.register_consumer("added", packet)
        publish(2)
        assert registry.entries[0].readers == 1
        assert all(
            registry.data_array[kind].slot_location(registry.entries[0][kind]) == location
            for kind, location in locations.items()
        )
        release[0].set()
        assert entered[1].wait(3)
        assert [received.get_nowait(), received.get_nowait()] == [2, 2]
        assert registry.entries[2].readers == 1
        stopped.set()
        release[1].set()
        for worker in workers:
            worker.join(3)
        for worker, _, _ in registry._connections:
            worker.join(3)
        assert not any(worker.is_alive() for worker in workers)
        assert all(entry.readers == 0 for entry in registry.entries.values())
    finally:
        stopped.set()
        for event in release:
            event.set()
        for worker in workers:
            worker.join(3)
        registry.close()


def test_remote_writer_publishes_only_after_release_and_rolls_back_failures(monkeypatch):
    class Data(ct.Structure):
        _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]

    registry = CentralRegistry()
    registry.register_producer("writer", Data)
    stopped = Event()
    finished = Queue()
    original_lease = registry.lease_slot_for_writing

    @contextmanager
    def observed_lease(datatype):
        try:
            with original_lease(datatype) as view:
                yield view
        finally:
            finished.put(True)

    monkeypatch.setattr(registry, "lease_slot_for_writing", observed_lease)
    client = registry.for_stage("writer", stopped)
    server = registry._connections[0][0]
    try:
        with client.lease_slot_for_writing(Data) as view:
            struct.pack_into("@QQ", view, 0, 42, 123)
            assert not registry.entries
            assert struct.unpack_from("@QQ", registry.data_array[Data].buffers[0].buf) == (42, 123)
        finished.get(timeout=3)
        assert list(registry.entries) == [42]
        with pytest.raises(RuntimeError, match="write failed"):
            with client.lease_slot_for_writing(Data) as view:
                struct.pack_into("@QQ", view, 0, 43, 456)
                raise RuntimeError("write failed")
        finished.get(timeout=3)
        assert list(registry.entries) == [42]
        assert len(registry.data_array[Data].free_slots) == registry.buffer_length - 1
    finally:
        client.close()
        server.join(3)
        assert not server.is_alive()
        registry.close()


def test_stage_client_routes_primitive_group_and_output_leases_without_prefetch():
    class Image(ct.Structure):
        _fields_ = [("ID", ct.c_uint64)]

    class Sensors(ct.Structure):
        _fields_ = [("ID", ct.c_uint64)]

    packet = frozenset({Image, Sensors})
    registry = CentralRegistry()
    registry.register_consumer("stage", Image)
    registry.register_consumer("stage", packet)
    registry.register_producer("camera", Image)
    registry.register_producer("stage", Sensors)
    stopped = Event()
    stage = registry.for_stage("stage", stopped)
    try:
        with registry.lease_slot_for_writing(Image) as view:
            struct.pack_into("@Q", view, 0, 42)
        assert registry.entries[42].readers == 0
        with pytest.raises(RuntimeError, match="processing failed"):
            with stage.lease_slot_for_reading(Image, "stage"):
                raise RuntimeError("processing failed")
        with stage.lease_slot_for_reading(Image, "stage") as buffers:
            assert struct.unpack_from("@Q", buffers[Image])[0] == 42
            assert registry.entries[42].readers == 1
            with stage.lease_slot_for_writing(Sensors) as view:
                struct.pack_into("@Q", view, 0, 42)
                assert registry.entries[42].readers == 1
        with stage.lease_slot_for_reading(packet, "stage") as buffers:
            assert all(view.readonly for view in buffers.values())
            assert {struct.unpack_from("@Q", view)[0] for view in buffers.values()} == {42}
    finally:
        stopped.set()
        stage.close()
        registry.close()
        assert not any(worker.is_alive() for worker, _, _ in registry._connections)
        assert registry.entries[42].readers == 0


def test_demo_stages_follow_catalog_structure_and_group_contracts():
    catalog = build_stage_catalog()
    for name in ("image_generator", "sensor_generator", "viewer", "dummy_consumer"):
        spec = catalog[name]["default"]
        assert spec["config"].is_file() and spec["path"].is_file()
        contract = yaml.safe_load(spec["contract"].read_text())
        for stream in contract["input"] + contract["output"]:
            assert stream in DATA_TYPE_REGISTRY
        if name in ("viewer", "dummy_consumer"):
            assert DATA_TYPE_REGISTRY[contract["input"][0]]["data_type"] == PACKET


def test_demo_stages_use_the_normal_pipeline_constructor_and_run_loop(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(pipeline_module, "SFTLogger", lambda config: MagicMock())
    configs = {
        "image_generator": {"generation_rate_hz": 10000},
        "sensor_generator": {"generation_rate_hz": 10000},
        "viewer": {"display": False, "max_frames": 1},
    }
    profile = [
        {name: [{"default": [{key: value} for key, value in configs.get(name, {}).items()]}]}
        for name in ("image_generator", "sensor_generator", "viewer", "dummy_consumer")
    ]
    pipeline = pipeline_module.Pipeline(profile, {"logger": {}})
    try:
        image, sensor, viewer, dummy = pipeline.build()
        for stage in (image, sensor, viewer, dummy):
            assert type(stage).__bases__ == (PipelineStage,)
            assert type(stage).run is PipelineStage.run
            assert stage.registry is pipeline.registry and stage.pid is None
        assert viewer.configs == {"display": False, "max_frames": 1}
        assert dummy.options["release_delay_range"] == [3.0, 4.0]
        image.mission()
        sensor.mission()
        viewer.run()
        assert viewer.frames == 1 and viewer.stopped.is_set()
        assert pipeline.registry.last_read["viewer", PACKET] == 0
        assert all(entry.readers == 0 for entry in pipeline.registry.entries.values())
    finally:
        pipeline.close_registry()


def test_generator_configs_control_shared_image_and_sensor_payloads():
    registry = CentralRegistry()
    registry.register_consumer("reader", PACKET)
    registry.register_producer("image_generator", PhotoExif)
    registry.register_producer("sensor_generator", PixData)
    configs = {
        "image_generator": {
            "generation_rate_hz": 10000,
            "first_capture_id": 7,
            "bar_width": 3,
            "bar_speed_px": 10,
            "exposure_time": 2.5,
        },
        "sensor_generator": {
            "generation_rate_hz": 10000,
            "first_capture_id": 7,
            "latitude": 1.0,
            "longitude": 2.0,
            "coordinate_step": 0.5,
        },
    }
    try:
        for cls, name in (
            (ImageGenerator, "image_generator"),
            (SensorGenerator, "sensor_generator"),
        ):
            stage = cls(configs[name], None, registry, name)
            stage.update()
            stage.mission()
        with registry.lease_slot_for_reading(PACKET, "reader") as buffers:
            photo, pix = buffers[PhotoExif], buffers[PixData]
            assert struct.unpack_from("@Q", photo, PhotoExif.ID.offset)[0] == 7
            assert struct.unpack_from("@Q", pix, PixData.ID.offset)[0] == 7
            assert struct.unpack_from("@d", photo, PhotoExif.exposure_time.offset)[0] == 2.5
            assert photo[PhotoExif.image.offset + 69] == 7
            assert photo[PhotoExif.image.offset + 70] == 255
            assert photo[PhotoExif.image.offset + 73] == 7
            assert photo[-1] == 7
            assert struct.unpack_from("@d", pix, PixData.lat.offset)[0] == 4.5
            assert struct.unpack_from("@d", pix, PixData.lon.offset)[0] == 5.5
    finally:
        for helper in registry.data_array.values():
            helper.close()


def test_dummy_delay_is_configurable_and_keeps_the_capture_pinned(monkeypatch):
    registry = CentralRegistry()
    registry.register_consumer("dummy", PACKET)
    for datatype in STRIPES:
        registry.register_producer("writer", datatype)
        with registry.lease_slot_for_writing(datatype) as view:
            struct.pack_into("@Q", view, datatype.ID.offset, 0)
    stage = dummy_consumer_stage.DummyConsumer(
        {"release_delay_range": [0.1, 0.2]}, None, registry, "dummy"
    )
    observed = []

    def random_delay(low, high):
        observed.append((low, high))
        return 0.15

    def hold(delay):
        assert registry.entries[0].readers == 1
        observed.append(delay)

    monkeypatch.setattr(dummy_consumer_stage, "uniform", random_delay)
    monkeypatch.setattr(stage.stopped, "wait", hold)
    try:
        stage.mission()
        assert observed == [(0.1, 0.2), 0.15]
        assert registry.entries[0].readers == 0
    finally:
        for helper in registry.data_array.values():
            helper.close()


def test_waiting_registry_subscription_can_stop_without_another_packet(monkeypatch):
    class Data(ct.Structure):
        _fields_ = [("ID", ct.c_uint64)]

    registry = CentralRegistry()
    registry.register_consumer("waiting", Data)
    registry.register_producer("writer", Data)
    stopped = Event()
    waiting = Event()
    original_wait = registry_module._ConsumerEvent.wait

    def observed_wait(event, timeout=None):
        waiting.set()
        return original_wait(event, timeout)

    monkeypatch.setattr(registry_module._ConsumerEvent, "wait", observed_wait)
    client = registry.for_stage("waiting", stopped)
    remote = client.readers[frozenset({Data})][0]
    server = registry._connections[0][0]
    try:
        remote.send_bytes(b"R")
        assert waiting.wait(3)
        stopped.set()
        server.join(3)
        assert not server.is_alive()
        with pytest.raises(EOFError):
            remote.recv_bytes()
        assert not registry.entries
    finally:
        stopped.set()
        server.join(3)
        client.close()
        registry.close()


def test_separate_generator_processes_complete_a_headless_demo():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.central_registry_demo import main; "
            "raise SystemExit(main({"
            "'image_generator': {'generation_rate_hz': 30}, "
            "'sensor_generator': {'generation_rate_hz': 30}, "
            "'viewer': {'display': False, 'max_frames': 8}, "
            "'dummy_consumer': {'instances': 2, 'release_delay_range': [0.01, 0.02]}}))",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resource_tracker" not in result.stderr
