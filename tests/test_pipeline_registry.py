import ctypes as ct
import struct
from multiprocessing import Process, shared_memory
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from core import pipeline as pipeline_module
from core.central_registry import MATCHING_WINDOW


class Image(ct.Structure):
    _fields_ = [("ID", ct.c_uint64), ("value", ct.c_uint64)]


class Sensors(ct.Structure):
    _fields_ = [("value", ct.c_uint64), ("ID", ct.c_uint64)]


ALL = frozenset({Image, Sensors})


class RecordingStage(Process):
    def __init__(self, configs, logger, registry, stage_name):
        super().__init__(name=stage_name)
        self.configs = configs
        self.registry = registry
        self.snapshot = (
            dict(registry.producers),
            {group: set(consumers) for group, consumers in registry.events.items()},
            {datatype: len(helper.slots) for datatype, helper in registry.data_array.items()},
        )


@pytest.fixture
def configure(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "SFTLogger", lambda config: MagicMock())
    monkeypatch.setattr(
        pipeline_module,
        "DATA_TYPE_REGISTRY",
        {
            "image": {"data_type": Image},
            "sensors": {"data_type": Sensors},
            "all": {"data_type": ALL},
        },
    )

    def make(contracts, cls=RecordingStage):
        catalog = {}
        for name, contract in contracts.items():
            contract_path = tmp_path / f"{name}_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
            catalog[name] = {
                "default": {
                    "module": f"stages.{name}",
                    "contract": contract_path,
                }
            }
        monkeypatch.setattr(pipeline_module, "build_stage_catalog", lambda: catalog)
        monkeypatch.setattr(
            pipeline_module,
            "import_module",
            lambda name: SimpleNamespace(
                __name__=cls.__module__,
                Stage=cls,
            ),
        )
        profile = [{name: [{"default": [{"option": name}]}]} for name in contracts]
        return pipeline_module.Pipeline(profile, {"logger": {}})

    return make


def test_stages_share_registry_and_register_before_construction(configure):
    pipeline = configure(
        {
            "camera": {"output": ["image", "image"]},
            "viewer": {"input": ["image", "image"]},
            "matched": {"input": ["all"]},
            "controller": {"output": ["sensors"]},
        }
    )
    try:
        camera, viewer, matched, controller = pipeline.build()
        registry = pipeline.registry
        assert all(stage.registry is registry and stage.pid is None for stage in pipeline.processes)
        assert [stage.configs for stage in pipeline.processes] == [
            {"option": name} for name in ("camera", "viewer", "matched", "controller")
        ]
        assert camera.snapshot == ({Image: "camera"}, {}, {Image: MATCHING_WINDOW + 1})
        assert viewer.snapshot[1] == {frozenset({Image}): {"viewer"}}
        assert matched.snapshot[2] == {Image: MATCHING_WINDOW + 3}
        assert controller.snapshot[0] == {Image: "camera", Sensors: "controller"}
        assert controller.snapshot[2] == {Image: MATCHING_WINDOW + 3, Sensors: MATCHING_WINDOW + 3}
        assert registry.array_length == MATCHING_WINDOW + 2
        assert registry.buffer_length == registry.array_length + 1
        assert not registry.events[ALL]["matched"].is_set()

        for datatype in (Image, Sensors):
            with registry.lease_slot_for_writing(datatype) as view:
                struct.pack_into("@Q", view, datatype.ID.offset, 42)
            if datatype is Image:
                assert registry.events[frozenset({Image})]["viewer"].is_set()
                assert not registry.events[ALL]["matched"].is_set()
        assert registry.events[ALL]["matched"].is_set()
        with registry.lease_slot_for_reading(ALL, "matched") as buffers:
            assert {
                struct.unpack_from("@Q", view, kind.ID.offset)[0] for kind, view in buffers.items()
            } == {42}
        with pytest.raises(RuntimeError, match="already built"):
            pipeline.build()
    finally:
        pipeline.close_registry()


class FailingStage(RecordingStage):
    names = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).names = [
            block.name for helper in self.registry.data_array.values() for block in helper.buffers
        ]
        raise RuntimeError("construction failed")


def test_failed_stage_construction_reports_error_and_unlinks_buffers(configure):
    pipeline = configure({"camera": {"output": ["image"]}}, FailingStage)
    with pytest.raises(RuntimeError, match="construction failed"):
        pipeline.build()
    assert pipeline.registry is None
    assert pipeline.processes == []
    assert FailingStage.names
    for name in FailingStage.names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_two_producers_cannot_replace_the_same_stripe(configure):
    pipeline = configure(
        {
            "camera": {"output": ["image"]},
            "other_camera": {"output": ["image"]},
        }
    )
    with pytest.raises(ValueError, match="already has a producer"):
        pipeline.build()
    assert pipeline.registry is None
    assert pipeline.processes == []


def test_run_releases_registry_after_failed_construction(configure):
    pipeline = configure({"camera": {"output": ["image"]}}, FailingStage)
    with pytest.raises(RuntimeError, match="construction failed"):
        pipeline.run()
    assert pipeline.registry is None
    assert pipeline.processes == []


def test_mock_camera_accepts_the_pipeline_registry(configure):
    from stages.cam.mock_camera_engine.mock_camera_engine_stage import CameraEngine

    pipeline = configure({"camera": {"output": ["image"]}}, CameraEngine)
    try:
        [camera] = pipeline.build()
        assert camera.registry is pipeline.registry
        assert pipeline.registry.producers == {Image: "camera"}
    finally:
        pipeline.close_registry()
