"""Hold the latest complete packet for a configurable random processing delay."""

from random import uniform
from struct import unpack_from

from common.new_structs import PixData
from common.pipeline_stage import PipelineStage
from .dummy_consumer_config import CONFIG


class DummyConsumer(PipelineStage):
    def __init__(self, configs, logger, registry, stage_name):
        super().__init__(configs, logger, registry, stage_name)
        self.update()
        self.packets = 0

    def update(self):
        self.options = {**CONFIG, **self.configs}

    def mission(self):
        if self.stopped.is_set():
            return True
        if self.packets == 0:
            self.logger.info("[%s] Waiting for matched captures; hold delay=%s seconds",
                             self._stage_name, self.options["release_delay_range"])
        with self.lease_slot_for_reading("main_camera") as buffers:
            capture_id = unpack_from("@q", buffers[PixData], PixData.ID.offset)[0]
            delay = uniform(*self.options["release_delay_range"])
            log = self.logger.info if self.packets == 0 else self.logger.debug
            log("[%s] Holding matched capture=%d for %.3f seconds", self._stage_name, capture_id, delay)
            self.stopped.wait(delay)
        self.packets += 1
        self.logger.debug("[%s] Released capture=%d; packets=%d", self._stage_name, capture_id, self.packets)
        return self.stopped.is_set()

    def stop(self):
        self.logger.info("[%s] Stop requested", self._stage_name)
        self.stopped.set()

    def _dump_data(self):
        self.logger.info("[%s] Consumed %d matched captures", self._stage_name, self.packets)
