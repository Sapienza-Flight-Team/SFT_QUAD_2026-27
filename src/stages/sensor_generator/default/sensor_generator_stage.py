"""Generate fake flight-controller data with matching capture IDs."""

from time import monotonic

from common.new_structs import PixData
from common.pipeline_stage import PipelineStage
from .sensor_generator_config import CONFIG


class SensorGenerator(PipelineStage):
    def __init__(self, configs, logger, registry, stage_name):
        super().__init__(configs, logger, registry, stage_name)
        self.update()
        self.capture_id = self.options["first_capture_id"]
        self.next_frame_at = None

    def update(self):
        self.options = {**CONFIG, **self.configs}

    def mission(self):
        if self.stopped.is_set():
            return True
        if self.next_frame_at is None:
            self.logger.info("[%s] Generating sensor data: rate=%g Hz, first capture=%d",
                             self._stage_name, self.options["generation_rate_hz"], self.capture_id)
        with self.lease_slot_for_writing("sensors") as buffer:
            if self.next_frame_at is None:
                self.next_frame_at = monotonic()
            pix = PixData.from_buffer(buffer)
            try:
                pix.ID = self.capture_id
                step = self.capture_id * self.options["coordinate_step"]
                pix.lat, pix.lon = self.options["latitude"] + step, self.options["longitude"] + step
                pix.alt_msl, pix.alt_rel = (
                    self.options["altitude_msl"],
                    self.options["altitude_relative"],
                )
                pix.roll, pix.pitch, pix.yaw = 0, 0, self.capture_id % 360
                pix.mission_current_wp = self.capture_id % self.options["waypoints"]
                pix.mission_total_wp = self.options["waypoints"]
            finally:
                del pix
        self.logger.debug("[%s] Submitted sensor capture=%d", self._stage_name, self.capture_id)
        self.capture_id += 1
        self.next_frame_at += 1 / self.options["generation_rate_hz"]
        return self.stopped.wait(max(0, self.next_frame_at - monotonic()))

    def stop(self):
        self.logger.info("[%s] Stop requested", self._stage_name)
        self.stopped.set()

    def _dump_data(self):
        self.logger.info("[%s] Generated %d sensor packets", self._stage_name,
                         self.capture_id - self.options["first_capture_id"])
