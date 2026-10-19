from __future__ import annotations

from importlib import import_module
from multiprocessing import Pipe, Process, set_start_method

import yaml

from common.config import MISSION_CONFIG_PATH, UTILITY_CONFIG_PATH
from common.new_structs import DATA_TYPE_REGISTRY
from core.central_registry import CentralRegistry
from core.logger import SFTLogger
from core.stage_catalog import build_stage_catalog, read_profile


class Pipeline(Process):
    """Construct stages from the mission profile, with separate utility settings."""

    _config: dict
    _mission: list[dict]

    def __init__(self, mission: list[dict], config: dict):
        super().__init__()
        self._mission = mission
        self._config = config
        self.registry = None
        self.processes = []
        self.logger = SFTLogger(config=self._config["logger"])

    def import_classes(self, catalog):
        """Import the single class defined in each selected stage module."""
        for stage in catalog.values():
            self.logger.debug("Importing stage implementation %s", stage["module"])
            module = import_module(stage["module"])
            stage["class"] = next(
                cls
                for cls in vars(module).values()
                if isinstance(cls, type) and cls.__module__ == module.__name__
            )

    def build(self):
        """Register each stage's inputs and outputs before constructing it."""
        if self.registry is not None:
            raise RuntimeError("Pipeline already built")
        self.logger.info("Building pipeline")
        self.registry = CentralRegistry(logger=self.logger)
        self.logger.info("Created central registry")
        self.processes.clear()
        try:
            catalog = build_stage_catalog()
            self.logger.debug("Discovered %d stage types", len(catalog))
            catalog = read_profile(catalog, self._mission)
            self.logger.info("Selected stages: %s", ", ".join(catalog))
            self.import_classes(catalog)

            for stage_name, specs in catalog.items():
                self.logger.debug(
                    "Reading contract for %s: %s", stage_name, specs["contract"]
                )
                with open(specs["contract"], encoding="utf-8") as file:
                    contract = yaml.safe_load(file)

                for data in dict.fromkeys(contract.get("input", [])):
                    datatype = DATA_TYPE_REGISTRY[data]["data_type"]
                    self.logger.info(
                        "Registering %s as consumer of %s", stage_name, data
                    )
                    self.registry.register_consumer(stage_name, datatype)
                for data in dict.fromkeys(contract.get("output", [])):
                    datatype = DATA_TYPE_REGISTRY[data]["data_type"]
                    if datatype in self.registry.producers:
                        raise ValueError(f"{data!r} already has a producer")
                    self.logger.info(
                        "Registering %s as producer of %s", stage_name, data
                    )
                    self.registry.register_producer(stage_name, datatype)

                self.logger.info(
                    "Constructing stage %s (%s)", stage_name, specs["class"].__name__
                )
                self.logger.debug("Settings for %s: %s", stage_name, specs["settings"])
                self.processes.append(
                    specs["class"](
                        configs=specs["settings"],
                        logger=self.logger,
                        registry=self.registry,
                        stage_name=stage_name,
                    )
                )
                self.processes[-1].name = stage_name

            stages = {stage.name: stage for stage in self.processes}
            sensor, camera = stages.get("sensor_generator"), stages.get("cam")
            if sensor is not None and sensor.options["external_clock"]:
                if (
                    camera is None
                    or camera.configs.get("trigger_mode") != "On"
                    or camera.configs.get("trigger_source") != "Software"
                ):
                    raise ValueError(
                        "external_clock requires cam trigger_mode=On and trigger_source=Software"
                    )
                if not (
                    camera.configs["chunk_mode_active"]
                    and camera.configs["chunk_enable"]
                    and camera.configs["chunk_selector"] == "CounterValue"
                ):
                    raise ValueError(
                        "external_clock requires an enabled CounterValue chunk"
                    )
                sensor.capture_connection, camera.capture_connection = Pipe(
                    duplex=False
                )
                camera.capture_period = 1 / sensor.options["generation_rate_hz"]

            self.logger.info("Pipeline build complete: %d stages", len(self.processes))
        except Exception:
            self.logger.exception("Pipeline build failed")
            self.close_registry()
            self.processes.clear()
            raise
        return self.processes

    def close_registry(self):
        for stage in self.processes:
            if getattr(stage, "capture_connection", None) is not None:
                stage.capture_connection.close()
        if self.registry is not None:
            self.logger.info("Closing central registry")
            self.registry.close()
            self.registry = None
            self.logger.info("Central registry closed")

    def run(self):
        """Build and start independent stages; wait until they exit."""
        self.logger.info("Pipeline starting")
        started = []
        try:
            self.build()
            for stage in self.processes:
                self.logger.info("Starting stage %s", stage.name)
                stage.start()
                started.append(stage)
            for stage in started:
                if getattr(stage, "capture_connection", None) is not None:
                    stage.capture_connection.close()
            for stage in started:
                stage.join()
        except KeyboardInterrupt:
            self.logger.info("Pipeline interrupted")
        finally:
            for stage in started:
                stage.stop()
            for stage in started:
                stage.join(3)
                if stage.is_alive():
                    self.logger.warning(
                        "Terminating stage %s after shutdown timeout", stage.name
                    )
                    stage.terminate()
                    stage.join()
                self.logger.info(
                    "Stage %s exited with code %s", stage.name, stage.exitcode
                )
            self.close_registry()
            self.logger.info("Pipeline stopped")
        for stage in started:
            if stage.exitcode:
                raise RuntimeError(
                    f"Stage {stage.name} exited with code {stage.exitcode}"
                )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    with UTILITY_CONFIG_PATH.open(encoding="utf-8") as file:
        utility_config = yaml.safe_load(file)

    with MISSION_CONFIG_PATH.open(encoding="utf-8") as file:
        mission_config = yaml.safe_load(file)

    Pipeline(mission=mission_config, config=utility_config).run()
