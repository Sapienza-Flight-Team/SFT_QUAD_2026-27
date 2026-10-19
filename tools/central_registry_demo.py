"""PYTHONPATH=src python tools/central_registry_demo.py

Two generator processes, a viewer, and configurable dummy consumer processes.
Edit stage overrides in mission_config.yaml; Q or Esc stops the viewer.
"""

import multiprocessing as mp

import yaml

from common.config import MISSION_CONFIG_PATH, UTILITY_CONFIG_PATH
from common.new_structs import DATA_TYPE_REGISTRY, PhotoExif, PixData
from core.central_registry import CentralRegistry
from core.logger import SFTLogger
from core.stage_catalog import build_stage_catalog, read_profile
from stages.dummy_consumer.default.dummy_consumer_config import CONFIG as DUMMY_CONFIG
from stages.dummy_consumer.default.dummy_consumer_stage import DummyConsumer
from stages.image_generator.default.image_generator_stage import ImageGenerator
from stages.sensor_generator.default.sensor_generator_stage import SensorGenerator
from stages.viewer.default.viewer_stage import Viewer


def main(configs=None):
    mp.set_start_method("spawn", force=True)
    with UTILITY_CONFIG_PATH.open(encoding="utf-8") as file:
        logger = SFTLogger(yaml.safe_load(file)["logger"])
    overrides = configs or {}
    with MISSION_CONFIG_PATH.open(encoding="utf-8") as file:
        profile = read_profile(build_stage_catalog(), yaml.safe_load(file))
    configs = {
        name: {**profile.get(name, {}).get("settings", {}), **overrides.get(name, {})}
        for name in ("image_generator", "sensor_generator", "viewer", "dummy_consumer")
    }
    configs["sensor_generator"]["external_clock"] = False  # This demo uses fake images.
    packet = DATA_TYPE_REGISTRY["main_camera"]["data_type"]
    count = {**DUMMY_CONFIG, **configs.get("dummy_consumer", {})}["instances"]
    specifications = [
        (ImageGenerator, "image_generator", (PhotoExif,), True),
        (SensorGenerator, "sensor_generator", (PixData,), True),
        (Viewer, "viewer", packet, False),
    ]
    specifications += [(DummyConsumer, f"dummy_consumer_{i}", packet, False) for i in range(count)]
    registry = CentralRegistry(logger=logger)
    stages, started = [], []

    try:
        # Known consumers first: allocate one block per stripe at startup.
        for _, name, _, producer in specifications:
            if not producer:
                registry.register_consumer(name, packet)
        for _, name, primitives, producer in specifications:
            if producer:
                registry.register_producer(name, primitives[0])
        for cls, name, primitives, producer in specifications:
            stage = cls(
                configs["dummy_consumer" if cls is DummyConsumer else name],
                logger,
                registry,
                name,
            )
            stage.name = name
            stages.append(stage)
        for stage in stages:
            stage.start()
            started.append(stage)
        next(stage for stage in stages if stage.name == "viewer").join()
    finally:
        for stage in started:
            stage.stop()
        for stage in started:
            stage.join(3)
            if stage.is_alive():
                stage.terminate()
                stage.join()
        registry.close()
    return next((stage.exitcode for stage in started if stage.exitcode), 0)


if __name__ == "__main__":
    raise SystemExit(main())
