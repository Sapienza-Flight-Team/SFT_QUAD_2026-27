"""Tests for the contract-based pipeline and complete ctypes messages."""

import ctypes as ct
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from multiprocessing import Process, get_context, shared_memory
from pathlib import Path
from queue import Empty
from time import monotonic, sleep
from types import SimpleNamespace

import pytest
import yaml

from common import new_structs
from common.data_connection import DataConnection
from common.new_structs import AggregatedMetadata, PhotoExif, PixData
from core import pipeline as pipeline_module
from core.pipeline import Pipeline


def load_structs(monkeypatch, config):
    original_open = Path.open

    def open_config(path, *args, **kwargs):
        if path == new_structs.MISSION_CONFIG_PATH:
            return StringIO(yaml.safe_dump(config))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_config)
    spec = spec_from_file_location("configured_test_structs", new_structs.__file__)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_no_argument_image_structures_use_mission_config(
    monkeypatch, tmp_path, channels
):
    # A different cwd must not change which mission file is read.
    monkeypatch.chdir(tmp_path)
    structs = load_structs(
        monkeypatch, {"image": {"resolution": [5, 3], "channels": channels}}
    )
    for cls in (structs.PhotoExif, structs.AggregatedMetadata):
        first, second = cls(), cls()
        assert type(first) is type(second) is cls
        assert cls.resolution == (5, 3)
        assert cls.channels == channels
        assert len(first.image) == 15 * channels
        first.image[-1] = 201
        assert second.image[-1] == 0
        restored = cls.from_buffer_copy(bytes(first))
        assert restored.image[-1] == 201
        assert cls.image.size == 15 * channels
    assert len(structs.FIELD_REGISTRY["PhotoExif"]) == 7
    assert structs.PixData().camera_trigger_seq == -1
    assert structs.PixData(camera_trigger_seq=42).camera_trigger_seq == 42
    assert structs.AggregatedMetadata().camera_trigger_seq == -1


@pytest.mark.parametrize(
    "config, message",
    [
        ({}, "missing 'image'"),
        ({"image": {"resolution": [0, 3], "channels": 1}}, "image.resolution"),
        ({"image": {"resolution": [5], "channels": 1}}, "image.resolution"),
        ({"image": {"resolution": [True, 3], "channels": 1}}, "image.resolution"),
        ({"image": {"resolution": [5, 3], "channels": 0}}, "image.channels"),
        ({"image": {"resolution": [5, 3], "channels": 1.5}}, "image.channels"),
    ],
)
def test_invalid_image_configuration_fails_before_allocation(
    monkeypatch, config, message
):
    with pytest.raises(ValueError, match=message):
        load_structs(monkeypatch, config)


def receive(connection, consumer):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        try:
            return connection.get_from_connection(consumer)
        except Empty:
            sleep(0.01)
    pytest.fail("No message received")


def publish_telemetry(connection):
    connection.push_onto_connection("camera", PixData(camera_trigger_seq=42))


@pytest.mark.parametrize("consumer_first", [False, True])
def test_fifo_broadcasts_typed_messages_between_processes(consumer_first):
    connection = DataConnection("fifo", False, PixData)
    try:
        if consumer_first:
            connection.register_consumer("aggregator")
            connection.register_consumer("logger")
        connection.register_producer("camera")
        connection.register_producer("other_camera")
        if not consumer_first:
            connection.register_consumer("aggregator")
            connection.register_consumer("logger")
        worker = Process(target=publish_telemetry, args=(connection,))
        worker.start()
        try:
            for consumer in ("aggregator", "logger"):
                [message] = receive(connection, consumer)
                assert type(message) is PixData
                assert message.camera_trigger_seq == 42
            worker.join(5)
            assert worker.exitcode == 0
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
        with pytest.raises(Empty):
            connection.get_from_connection("aggregator")
    finally:
        connection.close()


def publish_image(connection):
    photo = PhotoExif(frame_counter=42)
    photo.image[0], photo.image[-1] = 17, 201
    connection.push_onto_connection("camera", photo)
    connection.close()


def test_spawned_producer_uses_the_same_concrete_image_class():
    # Allocate synchronization objects in the same context as the spawned child.
    import common.data_connection as connections_module

    ctx = get_context("spawn")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(connections_module, "Event", ctx.Event)
        monkeypatch.setattr(connections_module, "Lock", ctx.Lock)
        connection = DataConnection("shm", False, PhotoExif)
        connection.register_producer("camera")
        connection.register_consumer("aggregator")
    resource = connection.CONNECTION["camera"]
    name = resource.name
    worker = ctx.Process(target=publish_image, args=(connection,))
    try:
        assert resource.size == ct.sizeof(PhotoExif)
        worker.start()
        [photo] = receive(connection, "aggregator")
        worker.join(5)
        assert worker.exitcode == 0
        assert type(photo) is PhotoExif
        assert photo.frame_counter == 42
        assert (photo.image[0], photo.image[-1]) == (17, 201)
    finally:
        if worker.pid is not None and worker.is_alive():
            worker.terminate()
            worker.join(5)
        connection.close(unlink=True)
    assert photo.image[-1] == 201  # The read owns its data after SHM is unlinked.
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


class RecordingStage(Process):
    def __init__(self, configs, logger, connections, stage_name):
        super().__init__(name=stage_name)
        self.CONNECTIONS = connections
        self._stage_name = stage_name
        for connection in connections.values():
            assert connection.producers  # Registration precedes construction.


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    def configure(contracts, cls=RecordingStage):
        entries = {}
        for name, contract in contracts.items():
            contract_path = tmp_path / f"{name}_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
            entries[name] = {
                "default": {"contract": contract_path, "module": f"stages.{name}"}
            }
        profile_path = tmp_path / "pipeline_config.yaml"
        profile_path.write_text(
            yaml.safe_dump([{name: "default"} for name in entries]), encoding="utf-8"
        )
        monkeypatch.setattr(pipeline_module, "build_stage_catalog", lambda: entries)
        monkeypatch.setattr(
            pipeline_module,
            "import_module",
            lambda name: SimpleNamespace(__name__=cls.__module__, Stage=cls),
        )
        return Pipeline("DEFAULT", SimpleNamespace(), profile_path=profile_path)

    return configure


def test_assembly_keeps_streams_with_identical_schemas_separate(catalog):
    pipeline = catalog(
        {
            "camera": {"input": [], "output": ["tagged_image", "tagged_image"]},
            "telemetry": {"input": [], "output": ["sensors"]},
            "aggregator": {
                "input": ["tagged_image", "sensors"],
                "output": ["aggregator_output"],
            },
            "resizer": {
                "input": ["aggregator_output"],
                "output": ["resized_aggregator_output"],
            },
            "inference": {"input": ["resized_aggregator_output"], "output": []},
        }
    )
    try:
        pipeline.build()
        assert len(pipeline.processes) == 5
        original = pipeline.connections["aggregator_output"]
        resized = pipeline.connections["resized_aggregator_output"]
        assert original is not resized
        assert original.data_type is resized.data_type is AggregatedMetadata
        assert original.producers == {"aggregator"}
        assert original.consumers == {"resizer"}
        assert resized.producers == {"resizer"}
        assert resized.consumers == {"inference"}
        for stage in pipeline.processes:
            assert stage.current_task == "DEFAULT"
            assert stage.pid is None
            for stream, connection in stage.CONNECTIONS.items():
                assert pipeline.connections[stream] is connection
        with pytest.raises(RuntimeError, match="already built"):
            pipeline.build()
    finally:
        pipeline.close_connections()


@pytest.mark.parametrize(
    "contract, message",
    [
        ({"input": ["px4_commands"]}, "unregistered input stream 'px4_commands'"),
        ({"input": ["sensors"]}, "without a producer"),
        ({"input": "sensors"}, "must be a list"),
    ],
)
def test_bad_contracts_fail_without_allocating_connections(catalog, contract, message):
    pipeline = catalog({"consumer": contract})
    with pytest.raises(ValueError, match=message):
        pipeline.build()
    assert pipeline.connections == {}
    assert pipeline.processes == []


class LegacyStage(Process):
    def __init__(self, configs, logger, in_q):
        super().__init__()


def test_incompatible_stage_constructor_fails_before_allocation(catalog):
    pipeline = catalog({"camera": {"output": ["tagged_image"]}}, LegacyStage)
    with pytest.raises(TypeError, match="must accept configs, logger, connections"):
        pipeline.build()
    assert pipeline.connections == {}


class FailingStage(RecordingStage):
    buffers = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).buffers = [
            resource.name
            for connection in self.CONNECTIONS.values()
            for resource in connection.CONNECTION.values()
        ]
        raise RuntimeError("construction failed")


def test_stage_construction_failure_unlinks_allocated_buffers(catalog):
    pipeline = catalog({"camera": {"output": ["tagged_image"]}}, FailingStage)
    with pytest.raises(RuntimeError, match="construction failed"):
        pipeline.build()
    assert pipeline.connections == {}
    assert pipeline.processes == []
    assert FailingStage.buffers
    for name in FailingStage.buffers:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_run_starts_all_stages_and_releases_the_connections(catalog):
    pipeline = catalog(
        {
            "camera": {"output": ["tagged_image"]},
            "consumer": {"input": ["tagged_image"]},
        }
    )
    pipeline.run()
    assert len(pipeline.processes) == 2
    assert all(process.exitcode == 0 for process in pipeline.processes)
    assert pipeline.connections == {}


class CrashingStage(RecordingStage):
    def run(self):
        raise RuntimeError("worker failed")


def test_run_reports_child_failure_and_releases_the_connections(catalog):
    pipeline = catalog({"camera": {"output": ["tagged_image"]}}, CrashingStage)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        pipeline.run()
    assert pipeline.connections == {}
    assert not pipeline.processes[0].is_alive()


def test_stage_helpers_route_by_stream_and_endpoint_direction():
    from common.pipeline_stage import PipelineStage

    incoming = DataConnection("shm", False, PixData)
    outgoing = DataConnection("shm", False, PixData)
    try:
        incoming.register_producer("upstream")
        incoming.register_consumer("transform")
        outgoing.register_producer("transform")
        outgoing.register_consumer("downstream")
        stage = SimpleNamespace(
            _stage_name="transform", CONNECTIONS={"input": incoming, "output": outgoing}
        )
        incoming.push_onto_connection("upstream", PixData(camera_trigger_seq=37))
        [data] = PipelineStage.receive_from_connection(stage, "input")
        PipelineStage.send_to_connections(stage, data)
        [result] = outgoing.get_from_connection("downstream")
        assert result.camera_trigger_seq == 37
        with pytest.raises(Empty):
            incoming.get_from_connection("transform")
        with pytest.raises(ValueError, match="cannot send"):
            PipelineStage.send_to_connections(stage, data, stream="input")
        # A type still selects the unique input even when outputs share its schema.
        incoming.push_onto_connection("upstream", PixData(camera_trigger_seq=42))
        [data] = PipelineStage.receive_from_connection(stage, PixData)
        assert data.camera_trigger_seq == 42
    finally:
        incoming.close(unlink=True)
        outgoing.close(unlink=True)
