"""Display the latest matching image and flight-controller data."""

import cv2
import numpy as np

from common.new_structs import IMAGE_SHAPE, PhotoExif, PixData
from common.pipeline_stage import PipelineStage
from .viewer_config import CONFIG


class Viewer(PipelineStage):
    def __init__(self, configs, logger, registry, stage_name):
        super().__init__(configs, logger, registry, stage_name)
        self.update()
        self.frames = 0

    def update(self):
        self.options = {**CONFIG, **self.configs}

    def mission(self):
        if self.stopped.is_set():
            return True
        if self.frames == 0:
            self.logger.info("[%s] Waiting for matched captures: display=%s, max_frames=%d",
                             self._stage_name, self.options["display"], self.options["max_frames"])
        with self.lease_slot_for_reading("main_camera") as buffers:
            pix = np.ndarray((), np.dtype(PixData), buffer=buffers[PixData])
            image = np.ndarray(
                IMAGE_SHAPE, np.uint8, buffer=buffers[PhotoExif], offset=PhotoExif.image.offset
            )
            try:
                capture_id = pix["camera_trigger_seq"].item()
                if self.options["display"]:
                    preview = cv2.resize(image, tuple(self.options["preview_resolution"]))
                    cv2.rectangle(preview, (0, 0), (390, 295), 0, -1)
                    for row, (field, _) in enumerate(PixData._fields_):
                        cv2.putText(
                            preview,
                            f"{field}: {pix[field].item()}",
                            (10, 25 + 27 * row),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                        )
                    cv2.imshow(self.options["window_name"], preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self.stop()
                self.frames += 1
                if self.options["max_frames"] and self.frames >= self.options["max_frames"]:
                    self.stop()
            finally:
                del pix, image
        log = self.logger.info if self.frames == 1 else self.logger.debug
        log("[%s] Processed matched capture=%d; frames=%d", self._stage_name, capture_id, self.frames)
        return self.stopped.is_set()

    def stop(self):
        self.logger.info("[%s] Stop requested", self._stage_name)
        self.stopped.set()

    def _dump_data(self):
        self.logger.info("[%s] Processed %d matched captures", self._stage_name, self.frames)
        if self.options["display"]:
            cv2.destroyAllWindows()
