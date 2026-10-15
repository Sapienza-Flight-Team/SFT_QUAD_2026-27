"""Generate fake full-resolution images directly in leased registry buffers."""

from time import monotonic

import numpy as np

from common.new_structs import IMAGE_SHAPE, PhotoExif
from common.pipeline_stage import PipelineStage
from .image_generator_config import CONFIG


class ImageGenerator(PipelineStage):
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
            self.logger.info("[%s] Generating images: shape=%s, rate=%g Hz, first capture=%d",
                             self._stage_name, IMAGE_SHAPE, self.options["generation_rate_hz"], self.capture_id)
        with self.lease_slot_for_writing("tagged_image") as buffer:
            if self.next_frame_at is None:
                self.next_frame_at = monotonic()
            photo = PhotoExif.from_buffer(buffer)
            image = np.ctypeslib.as_array(photo.image).reshape(IMAGE_SHAPE)
            try:
                photo.ID = self.capture_id
                photo.exposure_time, photo.gain = (
                    self.options["exposure_time"],
                    self.options["gain"],
                )
                image.fill(self.capture_id % 256)
                x = (self.capture_id * self.options["bar_speed_px"]) % IMAGE_SHAPE[1]
                image[:, x : x + self.options["bar_width"]] = 255
            finally:
                del image, photo
        self.logger.debug("[%s] Submitted image capture=%d", self._stage_name, self.capture_id)
        self.capture_id += 1
        self.next_frame_at += 1 / self.options["generation_rate_hz"]
        return self.stopped.wait(max(0, self.next_frame_at - monotonic()))

    def stop(self):
        self.logger.info("[%s] Stop requested", self._stage_name)
        self.stopped.set()

    def _dump_data(self):
        self.logger.info("[%s] Generated %d images", self._stage_name,
                         self.capture_id - self.options["first_capture_id"])
