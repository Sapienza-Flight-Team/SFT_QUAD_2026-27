import ctypes as ct
import struct
from unittest.mock import MagicMock

import pytest

from common import pipeline_stage as stage_module
from common.pipeline_stage import PipelineStage
from core.central_registry import CentralRegistry


class Image(ct.Structure):
    _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]


class Sensors(ct.Structure):
    _fields_ = [("value", ct.c_uint64), ("ID", ct.c_uint64)]


ALL = frozenset({Image, Sensors})


class Stage(PipelineStage):
    def mission(self):
        return True

    def stop(self):
        self.stopped.set()

    def _dump_data(self):
        pass


@pytest.fixture
def stages(monkeypatch):
    monkeypatch.setattr(stage_module, "DATA_TYPE_REGISTRY", {
        "image": {"data_type": Image}, "all": {"data_type": ALL},
    })
    registry = CentralRegistry()
    registry.register_consumer("reader", Image)
    registry.register_consumer("reader", ALL)
    registry.register_producer("writer", Image)
    registry.register_producer("writer", Sensors)
    writer = Stage(None, MagicMock(), registry, "writer")
    reader = Stage(None, MagicMock(), registry, "reader")
    try:
        yield writer, reader, registry
    finally:
        for helper in registry.data_array.values():
            helper.close()


@pytest.mark.parametrize("datatype", [Image, "image"])
def test_primitive_stage_leases_preserve_shared_views_and_reader_counts(stages, datatype):
    writer, reader, registry = stages
    with writer.lease_slot_for_writing(datatype) as writable:
        struct.pack_into("@QQ", writable, 0, 42, 123)
    with pytest.raises(ValueError):
        len(writable)
    with reader.lease_slot_for_reading(datatype) as buffers:
        view = buffers[Image]
        assert struct.unpack_from("@QQ", view) == (42, 123)
        entry = registry.entries[42]
        block, _ = registry.data_array[Image].slots[entry[Image]]
        assert view.obj is block.buf.obj
        assert view.readonly and entry.readers == 1
        assert registry.last_read["reader", frozenset({Image})] == 42
    assert entry.readers == 0
    with pytest.raises(ValueError):
        len(view)


@pytest.mark.parametrize("datatype", [ALL, {Image, Sensors}, "all"])
def test_group_stage_reads_match_capture_ids_and_retry_failed_processing(stages, datatype):
    writer, reader, registry = stages
    event = registry.events[ALL]["reader"]
    with writer.lease_slot_for_writing("image") as view:
        struct.pack_into("@Q", view, Image.ID.offset, 11)
    with writer.lease_slot_for_writing(Sensors) as view:
        struct.pack_into("@Q", view, Sensors.ID.offset, 10)
    assert not event.is_set()
    with writer.lease_slot_for_writing(Sensors) as view:
        struct.pack_into("@Q", view, Sensors.ID.offset, 11)
    assert event.is_set()
    with pytest.raises(RuntimeError, match="processing failed"):
        with reader.lease_slot_for_reading(datatype) as buffers:
            assert {struct.unpack_from("@Q", buffer, kind.ID.offset)[0]
                    for kind, buffer in buffers.items()} == {11}
            raise RuntimeError("processing failed")
    assert event.is_set() and registry.entries[11].readers == 0
    assert registry.last_read["reader", ALL] == -1
    with reader.lease_slot_for_reading(datatype) as buffers:
        assert set(buffers) == ALL
    assert not event.is_set()


@pytest.mark.parametrize("error", [EOFError, BrokenPipeError, RuntimeError])
@pytest.mark.parametrize("stopped", [False, True])
def test_stage_run_always_cleans_up_and_only_suppresses_shutdown_disconnects(error, stopped):
    stage = Stage(None, MagicMock(), None, "stage")
    stage.update = MagicMock()
    stage.mission = MagicMock(side_effect=error("disconnected"))
    stage._dump_data = MagicMock()
    if stopped:
        stage.stop()
    if stopped and error in (EOFError, BrokenPipeError):
        stage.run()
    else:
        with pytest.raises(error, match="disconnected"):
            stage.run()
    stage._dump_data.assert_called_once_with()
